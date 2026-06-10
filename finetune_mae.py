"""
Finetuning of the Frequency-Aware MAE encoder for deepfake classification.

Loads the encoder pretrained by pretrain_mae.py, attaches a classification head
(sequence pooling + MLP), and finetunes on CelebDF — replicating Experiment 4
hyperparameters (5 epochs, LR 0.0003->0.00001, batch 12, grad accum 4).

Usage:
    python finetune_mae.py --checkpoint MAE_CelebDF_FreqAware_encoder.pth
"""

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
from model.mae_model import FrequencyAwareMAE


# ---------------------------------------------------------------------------
# Classifier built on top of the pretrained MAE encoder
# ---------------------------------------------------------------------------

class MAEClassifier(nn.Module):
    """
    Wraps the encoder part of FrequencyAwareMAE and adds:
      - sequence pooling (attention-weighted, same as Model in model.py)
      - classification head (Linear -> BN -> LeakyReLU -> Dropout -> Linear)

    Interface is compatible with the custom Model class in model.py:
        forward(x, attention_mask=None) -> logits (B, num_classes)
    """

    def __init__(
        self,
        mae: FrequencyAwareMAE,
        num_classes: int = 2,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        d_model = mae.d_model

        # Encoder components (shared reference — no copy)
        self.patch_embedding = mae.patch_embedding
        self.positional_encoding = mae.positional_encoding
        self.encoder_layers = mae.encoder_layers
        self.encoder_norm = mae.encoder_norm

        # Sequence pooling weight
        self.seq_pool_weight = nn.Linear(d_model, 1)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )

        if freeze_encoder:
            for p in self.encoder_layers.parameters():
                p.requires_grad_(False)
            for p in self.patch_embedding.parameters():
                p.requires_grad_(False)
            for p in self.positional_encoding.parameters():
                p.requires_grad_(False)

    def sequence_pooling(self, seq: torch.Tensor, attention_mask=None) -> torch.Tensor:
        weights = self.seq_pool_weight(seq).permute(0, 2, 1)  # (B, 1, N)
        if attention_mask is not None:
            weights = weights.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        weights = weights.softmax(dim=-1)
        return (weights @ seq).squeeze(1)  # (B, d_model)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Args:
            x:              (B, C, T, H, W) video frames in [-1, 1]
            attention_mask: (B, T) optional — not used in current encoder,
                            kept for API compatibility with train.py / train_mvit.py

        Returns:
            logits: (B, num_classes)
        """
        from model.mae_model import patchify

        patches, grids = patchify(x, patch_size=16)  # (B, N, 768)
        hidden = self.patch_embedding(patches)         # (B, N, d_model)

        hidden, cu_seqlens, position_embeddings = self.positional_encoding(hidden, grids)

        for layer in self.encoder_layers:
            hidden = layer(hidden, cu_seqlens, None, position_embeddings)
        hidden = self.encoder_norm(hidden)             # (B, N, d_model)

        pooled = self.sequence_pooling(hidden)         # (B, d_model)
        return self.classification_head(pooled)        # (B, num_classes)

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_mae_classifier(checkpoint_path: str, num_classes: int = 2) -> MAEClassifier:
    """
    Load a pretrained MAE encoder checkpoint and wrap it in MAEClassifier.

    The checkpoint is produced by pretrain_mae.py and contains:
        {
            'encoder_state_dict': {...},
            'hparams': {encoder_depth, d_model, num_heads, patch_size, ...},
            'epoch': int,
            'loss': float,
        }
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = ckpt['hparams']

    # Reconstruct the full MAE model with the same encoder architecture
    mae = FrequencyAwareMAE(
        encoder_depth=hparams['encoder_depth'],
        d_model=hparams['d_model'],
        num_heads=hparams['num_heads'],
        patch_size=hparams.get('patch_size', 16),
        max_frames=hparams.get('max_frames', 16),
        img_size=hparams.get('img_size', 224),
    )

    # Load only encoder weights (decoder weights are discarded)
    missing, unexpected = mae.load_state_dict(ckpt['encoder_state_dict'], strict=False)
    print(f'Loaded encoder from {checkpoint_path} (epoch {ckpt["epoch"]}, loss {ckpt["loss"]:.6f})')
    if missing:
        print(f'  Missing keys (decoder — expected): {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys: {unexpected}')

    classifier = MAEClassifier(mae, num_classes=num_classes)
    return classifier


# ---------------------------------------------------------------------------
# Main finetuning script
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='MAE_CelebDF_FreqAware_encoder.pth',
        help='Path to encoder checkpoint produced by pretrain_mae.py',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DEVICE = 'cuda:0'
    EPOCHS = 5
    LR_0 = 0.0003
    LR_N = 0.00001
    BATCH_SIZE = 12
    MAX_FRAMES_PER_VIDEO = 16
    NUM_CLASSES = 2
    IMG_SIZE = 224
    GRADIENT_ACCUMULATION_STEPS = 4

    EXP_NAME = "MAE_Finetune_CelebDF"

    train_loss_history = list()
    val_loss_history = list()
    val_f1_history = list()

    # ---- Model ----
    model = load_mae_classifier(args.checkpoint, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}, trainable: {trainable_params:,}')

    # Named function instead of lambda to support multiprocessing pickling on Windows
    def normalize_neg1_to_1(x):
        """Scale tensor from [0, 1] to [-1, 1]."""
        return x * 2 - 1

    # ---- Transforms ----
    train_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
        T.RandomChoice([
            T.GaussianBlur(3),
            T.ColorJitter(brightness=0.15, hue=0.1, saturation=0.15),
        ]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.JPEG((60, 100))], p=0.5),
        T.RandomChannelPermutation(),
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(normalize_neg1_to_1),
    ])

    test_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(normalize_neg1_to_1),
    ])

    # ---- Dataset ----
    dataset_path = '/home/peter/faigc/data/Celeb-DF-v2'

    train_dataset = CelebDFDataset(
        dataset_path=dataset_path, transforms=train_transforms,
        frames_per_video=MAX_FRAMES_PER_VIDEO, split='train',
    )
    val_dataset = CelebDFDataset(
        dataset_path=dataset_path, transforms=test_transforms,
        frames_per_video=MAX_FRAMES_PER_VIDEO, split='validation',
    )
    test_dataset = CelebDFDataset(
        dataset_path=dataset_path, transforms=test_transforms,
        frames_per_video=MAX_FRAMES_PER_VIDEO, split='test',
    )

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=10, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

    # ---- Optimizer & scheduler ----
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
    fig.savefig(f'{EXP_NAME}_losses_n_grads.png')

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
