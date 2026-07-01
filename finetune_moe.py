"""
Finetuning of the Frequency-Aware MoE MAE encoder for deepfake classification.

Loads the encoder pretrained by pretrain_moe.py, attaches a classification head
(sequence pooling + MLP), and finetunes on CelebDF + FaceForensics++.

Key differences from finetune_mae.py:
  - Imports MoEMAEClassifier / load_moe_mae_classifier from model.mae_model_moe
  - Separate AdamW parameter groups: router LR = 0.1 × base LR, no weight decay
  - forward() of MoEMAEClassifier does NOT return aux_loss (encode() path, no masking)
    so the training loop is identical to finetune_mae.py

Usage:
    python finetune_moe.py \\
        --checkpoint experiments/MAE_MoE_CelebDF_FFPP/encoder.pth \\
        --exp-name MAE_MoE_ft_CelebDF_FFPP \\
        --epochs 15 \\
        --batch-size 8 \\
        --gradient-accumulation-steps 2
"""

import os
import shutil
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import F1Score
from torchvision.transforms import v2 as T
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from datasets.combined_dataset import CombinedVideoDataset
from model.mae_model_moe import MoEMAEClassifier, load_moe_mae_classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_neg1_to_1(x: torch.Tensor) -> torch.Tensor:
    """Scale tensor from [0, 1] to [-1, 1]."""
    return x * 2 - 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune MoE MAE encoder for deepfake classification"
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to encoder checkpoint produced by pretrain_moe.py',
    )
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--frames-per-video', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2)

    parser.add_argument('--lr0', type=float, default=3e-4)
    parser.add_argument('--lrN', type=float, default=1e-5)

    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder weights during finetuning')

    parser.add_argument('--exp-name', type=str, default="MAE_MoE_ft_CelebDF_FFPP")
    parser.add_argument('--output-dir', type=str, default='experiments')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    DEVICE = 'cuda:0'
    EPOCHS = args.epochs
    LR_0 = args.lr0
    LR_N = args.lrN
    BATCH_SIZE = args.batch_size
    FRAMES_PER_VIDEO = args.frames_per_video
    NUM_CLASSES = 2
    IMG_SIZE = args.img_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps

    EXP_NAME = args.exp_name
    OUTPUT_DIR = args.output_dir
    exp_dir = os.path.join(OUTPUT_DIR, EXP_NAME)

    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model: MoEMAEClassifier = load_moe_mae_classifier(
        args.checkpoint,
        num_classes=NUM_CLASSES,
        freeze_encoder=args.freeze_encoder,
    )
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, trainable: {trainable_params:,}')

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.JPEG((60, 100))], p=0.3),
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(normalize_neg1_to_1),
    ])

    test_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(normalize_neg1_to_1),
    ])

    # ── Dataset ──────────────────────────────────────────────────────────────
    celebdf_path = '../datasets/Celeb-DF-v2'
    ffpp_path = '../datasets/ffpp'

    train_dataset = CombinedVideoDataset(
        celebdf_path=celebdf_path,
        ff_path=ffpp_path,
        transforms=train_transforms,
        frames_per_video=FRAMES_PER_VIDEO,
        split='train',
        split_into_smaller_segments_mul=-1,
        supersample_reals=True,
    )
    val_dataset = CombinedVideoDataset(
        celebdf_path=celebdf_path,
        ff_path=ffpp_path,
        transforms=test_transforms,
        frames_per_video=FRAMES_PER_VIDEO,
        split='validation',
        split_into_smaller_segments_mul=-1,
    )
    test_dataset = CombinedVideoDataset(
        celebdf_path=celebdf_path,
        ff_path=ffpp_path,
        transforms=test_transforms,
        frames_per_video=FRAMES_PER_VIDEO,
        split='test',
        split_into_smaller_segments_mul=-1,
    )

    train_loader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True,
        num_workers=8, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, BATCH_SIZE, shuffle=False,
        num_workers=4, drop_last=True, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True,
    )

    # ── Optimizer — separate param groups for router ──────────────────────────
    router_params = [p for n, p in model.named_parameters() if 'router' in n]
    other_params  = [p for n, p in model.named_parameters() if 'router' not in n]

    optimizer = torch.optim.AdamW(
        [
            {'params': router_params, 'lr': LR_0 * 0.1, 'weight_decay': 0.0},
            {'params': other_params,  'lr': LR_0,        'weight_decay': 0.05},
        ],
    )

    # ── LR scheduler ─────────────────────────────────────────────────────────
    warmup_iters = int(len(train_loader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS * 0.1)
    regular_iters = int(len(train_loader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS * 0.9)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters
    )
    regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=regular_iters, eta_min=LR_N
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, regular_scheduler], milestones=[warmup_iters]
    )

    # ── Loss & metrics ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    f1_score_fn = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

    # ── Dataloader warmup ─────────────────────────────────────────────────────
    start = time.time()
    iterator = iter(train_loader)
    for _ in range(10):
        next(iterator)
    print(f'[TEST] Fetched 10 batches in {round(time.time() - start, 4)}s')

    # ── Training history ──────────────────────────────────────────────────────
    train_loss_history: list[float] = []
    val_loss_history:   list[float] = []
    val_f1_history:     list[float] = []
    step_losses:        list[float] = []
    step_lrs:           list[float] = []
    step_grad_norms:    list[float] = []

    best_f1 = -1.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
        epoch_loss_history:      list[float] = []
        epoch_grad_norm_history: list[float] = []

        start = time.time()

        for step, (x, attention_masks, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            attention_masks = attention_masks.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            # MoEMAEClassifier.forward() uses encode() — no aux_loss returned
            y_pred = model(x, attention_masks)
            loss = criterion(y_pred, y)
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            total_norm = nn.utils.get_total_norm(
                [p.grad for p in model.parameters() if p.grad is not None]
            ).mean().item()

            epoch_loss_history.append(loss.item())
            epoch_grad_norm_history.append(total_norm)
            step_losses.append(loss.item())
            step_grad_norms.append(total_norm)
            step_lrs.append(lr_scheduler.get_last_lr()[0])

            train_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 49:
                window = min(100, step + 1)
                print(
                    f'step: {step} | '
                    f'loss: {sum(epoch_loss_history[-window:]) / window:.4f} | '
                    f'grad_norm: {sum(epoch_grad_norm_history[-window:]) / window:.4f} | '
                    f'lr: {lr_scheduler.get_last_lr()[0]:.2e}'
                )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()

        val_loss = 0.0
        val_y_target: list[torch.Tensor] = []
        val_y_pred:   list[torch.Tensor] = []

        for x, attention_masks, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            attention_masks = attention_masks.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                y_pred = model(x, attention_masks)

            val_loss += criterion(y_pred, y).item()
            val_y_pred.append(y_pred.argmax(-1))
            val_y_target.append(y)

        val_y_pred_cat   = torch.cat(val_y_pred)
        val_y_target_cat = torch.cat(val_y_target)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss   = val_loss / len(val_loader)
        epoch_val_f1     = f1_score_fn(val_y_pred_cat, val_y_target_cat).item()

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        val_f1_history.append(epoch_val_f1)

        print(
            f'Epoch {epoch + 1}/{EPOCHS} | '
            f'time: {round(time.time() - start, 2)}s | '
            f'train_loss: {epoch_train_loss:.6f} | '
            f'val_loss: {epoch_val_loss:.6f} | '
            f'val_f1: {epoch_val_f1:.6f}'
        )

        # Save best checkpoint by val F1
        if epoch_val_f1 > best_f1:
            best_f1 = epoch_val_f1
            torch.save(
                model.state_dict(),
                os.path.join(exp_dir, "classifier.pth"),
            )
            print(f'  -> Saved best classifier (val_f1={best_f1:.6f})')

    # ── Plots ─────────────────────────────────────────────────────────────────
    smoothing_ksize = 100

    def smooth(arr: list[float]) -> np.ndarray:
        k = min(smoothing_ksize, len(arr))
        return np.convolve(arr, np.ones(k) / k, mode='same')

    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))

    axes[0].plot(step_losses, alpha=0.3, label='loss')
    axes[0].plot(smooth(step_losses), label='smoothed')
    axes[0].legend()
    axes[0].set_title("Train Loss / step")

    axes[1].plot(step_lrs)
    axes[1].set_title("LR / step")

    axes[2].plot(step_grad_norms, alpha=0.3, label='grad_norm')
    axes[2].plot(smooth(step_grad_norms), label='smoothed')
    axes[2].legend()
    axes[2].set_title("Grad Norm / step")

    fig.savefig(os.path.join(exp_dir, 'losses_n_grads.png'))

    fig2, axes2 = plt.subplots(ncols=2, figsize=(12, 4))
    axes2[0].plot(train_loss_history, label='train')
    axes2[0].plot(val_loss_history, label='val')
    axes2[0].legend()
    axes2[0].set_title("Loss / epoch")
    axes2[1].plot(val_f1_history)
    axes2[1].set_title("Val F1 / epoch")
    fig2.savefig(os.path.join(exp_dir, 'epoch_metrics.png'))

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    test_y_target: list[torch.Tensor] = []
    test_y_pred:   list[torch.Tensor] = []

    for x, attention_masks, y in test_loader:
        x = x.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            y_pred = model(x, attention_masks)

        test_y_pred.append(y_pred)
        test_y_target.append(y)

    test_y_pred_cat   = torch.cat(test_y_pred, dim=0)
    test_y_target_cat = torch.cat(test_y_target)

    test_loss = criterion(test_y_pred_cat, test_y_target_cat).item()
    test_f1   = f1_score_fn(test_y_pred_cat.argmax(-1), test_y_target_cat).item()

    print(f'\nTest loss: {test_loss:.6f}')
    print(f'Test F1:   {test_f1:.6f}')
    print(
        'Test classification report:\n'
        + classification_report(
            test_y_target_cat.cpu().numpy(),
            test_y_pred_cat.argmax(-1).cpu().numpy(),
        )
    )
