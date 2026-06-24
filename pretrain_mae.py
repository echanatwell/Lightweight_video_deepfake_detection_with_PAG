"""
Frequency-Aware MAE pretraining on CelebDF train split.

Labels are NOT used — this is pure self-supervised pretraining.
After training, the encoder weights are saved to mae_encoder_checkpoint.pth
for use in finetune_mae.py.
"""

import os
import argparse
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

import numpy as np
import time
import matplotlib.pyplot as plt

from datasets.combined_dataset import CombinedVideoDataset
from model.mae_model import FrequencyAwareMAE


# ---------------------------------------------------------------------------
# Constants (module-level is fine — no side-effects)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--frames_per_video', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

    return parser.parse_args()

DEVICE = 'cuda:0'
LR_0 = 1.5e-4
LR_N = 1e-5

GRADIENT_ACCUMULATION_STEPS = 2

# MAE hyperparameters
MASK_RATIO = 0.75
ENCODER_DEPTH = 4
D_MODEL = 128
NUM_HEADS = 8
DECODER_DEPTH = 2
D_DEC = 128
DECODER_NUM_HEADS = 4
BLUR_KERNEL = 5
BLUR_SIGMA = 1.0

OUTPUT_DIR = "/scratch/users/k25137033/interpretability/Lightweight_video_deepfake_detection_with_PAG"
EXP_NAME = "MAE_CelebDF_FFPP_FreqAware_rgbtarget"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, EXP_NAME, "encoder.pth")
FULL_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, EXP_NAME, "full.pth")

if os.path.exists(os.path.join(OUTPUT_DIR, EXP_NAME)):
    shutil.rmtree(os.path.join(OUTPUT_DIR, EXP_NAME))
os.mkdir(os.path.join(OUTPUT_DIR, EXP_NAME))

# Named function instead of lambda to support multiprocessing pickling on Windows
def normalize_neg1_to_1(x):
    """Scale tensor from [0, 1] to [-1, 1]."""
    return x * 2 - 1


# ---------------------------------------------------------------------------
# Entry point — required on Windows (spawn) to prevent recursive worker launch
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    FRAMES_PER_VIDEO = args.frames_per_video
    IMG_SIZE = args.img_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps

    train_loss_history = list()

    model = FrequencyAwareMAE(
        encoder_depth=ENCODER_DEPTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        decoder_depth=DECODER_DEPTH,
        d_dec=D_DEC,
        decoder_num_heads=DECODER_NUM_HEADS,
        mask_ratio=MASK_RATIO,
        patch_size=16,
        max_frames=FRAMES_PER_VIDEO,
        img_size=IMG_SIZE,
        blur_kernel=BLUR_KERNEL,
        blur_sigma=BLUR_SIGMA,
    ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(
        p.numel() for name, p in model.named_parameters()
        if any(k in name for k in ('patch_embedding', 'positional_encoding', 'encoder_layers', 'encoder_norm'))
    )
    print(f'Total params: {total_params:,}')
    print(f'Encoder params: {encoder_params:,}')
    print(f'Decoder params: {total_params - encoder_params:,}')

    # Transforms: same as train.py custom transforms (no MViT-specific normalization)
    # Input is normalized to [-1, 1] as expected by the model
    train_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.15, hue=0.1, saturation=0.15)], p=0.5),
        T.RandomApply([T.JPEG((60, 100))], p=0.3),
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(normalize_neg1_to_1),  # [0,1] -> [-1,1]
    ])

    celebdf_path = '../datasets/Celeb-DF-v2'
    ffpp_path = '../datasets/ffpp'

    # Labels are loaded but not used — CelebDFDataset returns (x, attention_mask, y)
    # train_dataset = CelebDFDataset(
    #     dataset_path=dataset_path,
    #     transforms=train_transforms,
    #     frames_per_video=FRAMES_PER_VIDEO,
    #     split='train',
    # )

    train_dataset = CombinedVideoDataset(
        celebdf_path=celebdf_path,
        ff_path=ffpp_path,
        transforms=train_transforms,
        frames_per_video=FRAMES_PER_VIDEO,
        split='train',
    )

    train_loader = DataLoader(
        train_dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_0, weight_decay=0.05, betas=(0.9, 0.95))

    # Linear warmup (5% of total steps) + cosine decay
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * 0.05)
    regular_steps = total_steps - warmup_steps

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=warmup_steps)
    regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=regular_steps, eta_min=LR_N)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, regular_scheduler], [warmup_steps]
    )

    step_losses = list()
    step_lrs = list()
    step_grad_norms = list()

    # Dataloader warmup test
    start = time.time()
    iterator = iter(train_loader)
    for _ in range(5):
        next(iterator)
    print(f'[TEST] Fetched 5 batches in {round(time.time() - start, 4)} seconds')

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0
        epoch_loss_history = list()
        epoch_grad_norm_history = list()

        start = time.time()

        for step, (x, attention_masks, _y) in enumerate(train_loader):
            # Labels (_y) are intentionally ignored — SSL pretraining
            x = x.to(DEVICE, non_blocking=True)

            loss = model(x)
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
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 49:
                print(
                    f'step: {step}, '
                    f'loss smoothed: {round(sum(epoch_loss_history[-100:]) / min(100, step + 1), 6)}, '
                    f'grad_norm smoothed: {round(sum(epoch_grad_norm_history[-100:]) / min(100, step + 1), 4)}, '
                    f'lr: {"{:0.2e}".format(lr_scheduler.get_last_lr()[0])}'
                )

        avg_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        print(
            f'Epoch {epoch + 1}/{EPOCHS}, '
            f'epoch time: {round(time.time() - start, 2)}s, '
            f'train loss: {round(avg_loss, 6)}, '
            f'lr: {"{:0.2e}".format(lr_scheduler.get_last_lr()[0])}'
        )

        # Save best encoder checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    'epoch': epoch + 1,
                    'encoder_state_dict': {
                        k: v for k, v in model.state_dict().items()
                        if any(part in k for part in (
                            'patch_embedding', 'positional_encoding',
                            'encoder_layers', 'encoder_norm',
                        ))
                    },
                    'loss': best_loss,
                    'hparams': {
                        'encoder_depth': ENCODER_DEPTH,
                        'd_model': D_MODEL,
                        'num_heads': NUM_HEADS,
                        'patch_size': 16,
                        'max_frames': FRAMES_PER_VIDEO,
                        'img_size': IMG_SIZE,
                    },
                },
                CHECKPOINT_PATH,
            )
            print(f'  -> Saved best encoder checkpoint (loss={round(best_loss, 6)})')

        # Save full model checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, EXP_NAME, f'epoch{epoch + 1}.pth'))
            print(f'  -> Saved full checkpoint at epoch {epoch + 1}')

    # Save final full model
    torch.save(model.state_dict(), FULL_CHECKPOINT_PATH)
    print(f'Saved final full model to {FULL_CHECKPOINT_PATH}')

    # Plot training curves
    smoothing_ksize = 100
    step_losses_smoothed = np.convolve(step_losses, np.ones(smoothing_ksize) / smoothing_ksize, mode='same')
    step_grad_norms_smoothed = np.convolve(step_grad_norms, np.ones(smoothing_ksize) / smoothing_ksize, mode='same')

    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
    axes[0].plot(step_losses, alpha=0.3, label='loss')
    axes[0].plot(step_losses_smoothed, label='loss smoothed')
    axes[0].legend()
    axes[0].set_title("Pretrain Loss / step")
    axes[1].plot(step_lrs)
    axes[1].set_title("LR / step")
    axes[2].plot(step_grad_norms, alpha=0.3, label='grad norms')
    axes[2].plot(step_grad_norms_smoothed, label='grad norms smoothed')
    axes[2].legend()
    axes[2].set_title("Grad Norm / step")

    fig.savefig(os.path.join(OUTPUT_DIR, EXP_NAME, 'training_curves.png'))
    print(f'Saved training curves to training_curves.png')

    # Epoch-level loss curve
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(train_loss_history)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Frequency-Aware MAE Pretraining Loss')
    fig2.savefig(os.path.join(OUTPUT_DIR, EXP_NAME, 'epoch_loss.png'))
    print(f'Saved epoch loss curve to epoch_loss.png')
