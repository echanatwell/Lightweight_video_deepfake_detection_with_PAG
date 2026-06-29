"""
Finetuning of the Frequency-Aware MAE encoder for deepfake classification.

Loads the encoder pretrained by pretrain_mae.py, attaches a classification head
(sequence pooling + MLP), and finetunes on CelebDF — replicating Experiment 4
hyperparameters (5 epochs, LR 0.0003->0.00001, batch 12, grad accum 4).

Usage:
    python finetune_mae.py --checkpoint MAE_CelebDF_FreqAware_encoder.pth
"""
import os
import shutil
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from torchvision.transforms import v2 as T

import numpy as np
import time
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from datasets.celebdf import CelebDFDataset
from datasets.combined_dataset import CombinedVideoDataset
from model.mae_model import FrequencyAwareMAE, patchify, MAEClassifier, load_mae_classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to encoder checkpoint produced by pretrain_mae.py',
    )
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--frames-per-video', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2)

    parser.add_argument('--lr0', type=float, default=0.0003)
    parser.add_argument('--lrN', type=float, default=0.00001)

    parser.add_argument('--exp-name', type=str, default="MAE_pt_CDF_FFPP_ft_CDF_FPP_clsw_rgbtarget")
    parser.add_argument('--output-dir', type=str, default='experiments')
    return parser.parse_args()

# Named function instead of lambda to support multiprocessing pickling on Windows
def normalize_neg1_to_1(x):
    """Scale tensor from [0, 1] to [-1, 1]."""
    return x * 2 - 1

if __name__ == '__main__':
    args = parse_args()

    print(args)

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
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, EXP_NAME, "encoder.pth")
    FULL_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, EXP_NAME, "full.pth")

    if os.path.exists(os.path.join(OUTPUT_DIR, EXP_NAME)):
        shutil.rmtree(os.path.join(OUTPUT_DIR, EXP_NAME))
    os.mkdir(os.path.join(OUTPUT_DIR, EXP_NAME))

    train_loss_history = list()
    val_loss_history = list()
    val_f1_history = list()

    # ---- Model ----
    model = load_mae_classifier(args.checkpoint, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, trainable: {trainable_params:,}')

    # ---- Transforms ----
   
    train_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
        # T.RandomChoice([
        #     T.GaussianBlur(3),
        #     T.ColorJitter(brightness=0.15, hue=0.1, saturation=0.15),
        # ]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.JPEG((60, 100))], p=0.5),
        # T.RandomChannelPermutation(),
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(normalize_neg1_to_1),
    ])

    test_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(normalize_neg1_to_1),
    ])

    celebdf_path = '../datasets/Celeb-DF-v2'
    ffpp_path = '../datasets/ffpp'
    
    train_dataset = CombinedVideoDataset(
        celebdf_path=celebdf_path,
        ff_path=ffpp_path,
        transforms=train_transforms,
        frames_per_video=FRAMES_PER_VIDEO,
        split='train',
        split_into_smaller_segments_mul=-1,
        supersample_reals=True
    )

    val_dataset = CombinedVideoDataset(
        celebdf_path=celebdf_path,
        ff_path=ffpp_path,
        transforms=test_transforms,
        frames_per_video=FRAMES_PER_VIDEO,
        split='validation',
        split_into_smaller_segments_mul=-1
    )

    test_dataset = CombinedVideoDataset(
        celebdf_path=celebdf_path,
        ff_path=ffpp_path,
        transforms=test_transforms,
        frames_per_video=FRAMES_PER_VIDEO,
        split='test',
        split_into_smaller_segments_mul=-1
    )

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    # ---- Optimizer & scheduler ----
    # Считаем веса обратно пропорционально частоте классов
    # n_real = sum(1 for _, lbl in train_dataset.celebdf_dataset.entries if lbl == 0) + \
    #     sum(1 for _, lbl in train_dataset.ff_dataset.entries if lbl == 0)
    # n_fake = sum(1 for _, lbl in train_dataset.celebdf_dataset.entries if lbl == 1) + \
    #     sum(1 for _, lbl in train_dataset.ff_dataset.entries if lbl == 1)
    # n_total = n_real + n_fake
    # class_weights = torch.tensor([n_total / (2 * n_real), n_total / (2 * n_fake)], device=DEVICE) # sklearn compute_class_weight
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.05, weight=class_weights)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    f1_score_fn = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_0, weight_decay=0.05)

    warmup_iters = int(len(train_loader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS * 0.1)
    regular_iters = int(len(train_loader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS * 0.9)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1, total_iters=warmup_iters)
    regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=regular_iters, eta_min=LR_N)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, regular_scheduler], [warmup_iters]
    )

    step_losses = list()
    step_lrs = list()
    step_grad_norms = list()

    # Dataloader warmup test
    start = time.time()
    iterator = iter(train_loader)
    for _ in range(10):
        next(iterator)
    print(f'[TEST] Fetched 10 batches in {round(time.time() - start, 4)} seconds')

    best_f1 = -1

    # ---- Training loop ----
    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0
        val_loss = 0

        epoch_loss_history = list()
        epoch_grad_norm_history = list()

        start = time.time()

        for step, (x, attention_masks, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            attention_masks = attention_masks.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            y_pred = model(x, attention_masks)

            loss = criterion(y_pred, y)
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            total_norm = nn.utils.get_total_norm(
                [p.grad for p in model.parameters() if p.grad is not None]
            ).mean().item()

            epoch_loss_history.append(loss.item())
            epoch_grad_norm_history.append(total_norm)
            step_grad_norms.append(total_norm)
            step_lrs.append(lr_scheduler.get_last_lr()[0])
            step_losses.append(loss.item())

            train_loss += loss.item()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 49:
                print(
                    f'step: {step}, '
                    f'loss smoothed: {round(sum(epoch_loss_history[-100:]) / min(100, step + 1), 4)}, '
                    f'grad_norm smoothed: {round(sum(epoch_grad_norm_history[-100:]) / min(100, step + 1), 4)}, '
                    f'lr: {"{:0.2e}".format(lr_scheduler.get_last_lr()[0])}'
                )

        model.eval()

        val_y_target = list()
        val_y_pred = list()

        for x, attention_masks, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            attention_masks = attention_masks.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                y_pred = model(x, attention_masks)

            loss = criterion(y_pred, y)
            val_loss += loss.item()
            val_y_pred.append(y_pred.argmax(-1))
            val_y_target.append(y)

        val_y_pred = torch.cat(val_y_pred)
        val_y_target = torch.cat(val_y_target)

        train_loss_history.append(train_loss / len(train_loader))
        val_loss_history.append(val_loss / len(val_loader))
        val_f1_history.append(f1_score_fn(val_y_pred, val_y_target).item())

        print(f'Epoch {epoch + 1}/{EPOCHS}, epoch time: {round(time.time() - start, 2)}.', end='')
        print(f' Train loss: {round(train_loss_history[-1], 6)},', end='')
        print(f' val loss: {round(val_loss_history[-1], 6)}, val f1: {round(val_f1_history[-1], 6)}')

        if val_f1_history[-1] > best_f1:
            best_f1 = val_f1_history[-1]
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, EXP_NAME, "best.pth"))
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, EXP_NAME, "last.pth"))

    # ---- Plots ----
    smoothing_ksize = 100
    step_losses_smoothed = np.convolve(step_losses, np.ones(smoothing_ksize) / smoothing_ksize, mode='same')
    step_grad_norms_smoothed = np.convolve(step_grad_norms, np.ones(smoothing_ksize) / smoothing_ksize, mode='same')

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    axes[0].plot(step_losses, alpha=0.3, label='loss')
    axes[0].plot(step_losses_smoothed, label='loss smoothed')
    axes[0].legend()
    axes[0].set_title("Train Loss / step")
    axes[1].plot(step_lrs)
    axes[1].set_title("LR / step")
    axes[2].plot(step_grad_norms, alpha=0.3, label='grad norms')
    axes[2].plot(step_grad_norms_smoothed, label='grad norms smoothed')
    axes[2].legend()
    axes[2].set_title("Grad Norm / step")
    fig.savefig(os.path.join(OUTPUT_DIR, EXP_NAME, 'losses_n_grads.png'))

    # ---- Test evaluation ----
    model.eval()
    test_y_target = list()
    test_y_pred = list()

    for x, attention_masks, y in test_loader:
        x = x.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            y_pred = model(x, attention_masks)

        test_y_pred.append(y_pred)
        test_y_target.append(y)

    test_y_pred = torch.cat(test_y_pred, dim=0)
    test_y_target = torch.cat(test_y_target)

    print(f'Test loss: {round(criterion(test_y_pred, test_y_target).item(), 6)}')
    print(f'Test f1: {round(f1_score_fn(test_y_pred.argmax(-1), test_y_target).item(), 6)}')
    print(
        f'Test classification report:\n'
        f'{classification_report(test_y_target.cpu().numpy(), test_y_pred.argmax(-1).cpu().numpy())}'
    )
