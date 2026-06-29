"""
Frequency-Aware MAE with Sparse MoE Encoder — pretraining script.

Based on pretrain_mae.py; key differences:
  - Uses FrequencyAwareMoEMAE instead of FrequencyAwareMAE
  - Adds MoE-specific CLI args: --num-experts, --top-k, --capacity-factor, --aux-loss-alpha
  - Separate AdamW parameter groups: router LR = 0.1 × base LR, no weight decay on router
  - Longer warmup: 10% of total steps (vs 5% in pretrain_mae.py)
  - forward() returns (total_loss, mse_loss, aux_loss) — all three are logged
  - MoE overflow fraction is logged every 50 steps
  - hparams in checkpoint includes MoE fields for load_moe_mae_classifier()

Usage:
    python pretrain_moe.py \\
        --exp-name MAE_MoE_CelebDF_FFPP \\
        --epochs 200 \\
        --batch-size 8 \\
        --gradient-accumulation-steps 2 \\
        --encoder-depth 4 \\
        --d-model 128 \\
        --num-heads-encoder 8 \\
        --decoder-depth 4 \\
        --d-decoder 128 \\
        --mask-ratio 0.65 \\
        --num-experts 8 \\
        --top-k 2 \\
        --capacity-factor 1.5 \\
        --aux-loss-alpha 0.01
"""

import os
import sys
import logging
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchinfo import summary
import matplotlib.pyplot as plt

from datasets.combined_dataset import CombinedVideoDataset
from model.mae_model_moe import FrequencyAwareMoEMAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_neg1_to_1(x: torch.Tensor) -> torch.Tensor:
    """Scale tensor from [0, 1] to [-1, 1]."""
    return x * 2 - 1


def setup_logger(log_path: str) -> logging.Logger:
    """Configure and return a logger that writes to both console and a log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("pretrain_moe")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_moe_overflow(model: FrequencyAwareMoEMAE) -> float:
    """Return mean overflow fraction across all MoE layers."""
    fracs = [
        layer.moe.last_overflow_frac
        for layer in model.encoder_layers
    ]
    return sum(fracs) / len(fracs) if fracs else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain Frequency-Aware MAE with Sparse MoE encoder"
    )

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--frames-per-video', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2)

    # Encoder
    parser.add_argument('--encoder-depth', type=int, default=4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads-encoder', type=int, default=8)

    # Decoder
    parser.add_argument('--decoder-depth', type=int, default=4)
    parser.add_argument('--d-decoder', type=int, default=128)
    parser.add_argument('--num-heads-decoder', type=int, default=8)

    # MAE
    parser.add_argument('--mask-ratio', type=float, default=0.65)

    # MoE
    parser.add_argument('--num-experts', type=int, default=8,
                        help='Number of experts per MoE layer')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Number of experts each token is routed to')
    parser.add_argument('--capacity-factor', type=float, default=1.5,
                        help='Expert capacity = capacity_factor * tokens / num_experts')
    parser.add_argument('--aux-loss-alpha', type=float, default=0.01,
                        help='Weight of MoE load-balancing loss')

    # Experiment
    parser.add_argument('--exp-name', type=str, default="MAE_MoE_CelebDF_FFPP")
    parser.add_argument('--output-dir', type=str, default='experiments')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEVICE = 'cuda:0'
LR_0 = 1.5e-4
LR_N = 1e-5
BLUR_KERNEL = 5
BLUR_SIGMA = 1.0


if __name__ == '__main__':
    args = parse_args()
    print(args)

    EXP_NAME = args.exp_name
    OUTPUT_DIR = args.output_dir
    exp_dir = os.path.join(OUTPUT_DIR, EXP_NAME)
    CHECKPOINT_PATH = os.path.join(exp_dir, "encoder.pth")
    FULL_CHECKPOINT_PATH = os.path.join(exp_dir, "full.pth")
    LOG_PATH = os.path.join("logs", f"{EXP_NAME}.log")

    if os.path.exists(exp_dir) and os.listdir(exp_dir):
        print(f'WARNING! {exp_dir} already exists and is non-empty — stopping.')
        sys.exit(1)
    os.makedirs(exp_dir, exist_ok=True)

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    FRAMES_PER_VIDEO = args.frames_per_video
    IMG_SIZE = args.img_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    MASK_RATIO = args.mask_ratio

    ENCODER_DEPTH = args.encoder_depth
    D_MODEL = args.d_model
    NUM_HEADS = args.num_heads_encoder
    DECODER_DEPTH = args.decoder_depth
    D_DEC = args.d_decoder
    DECODER_NUM_HEADS = args.num_heads_decoder

    NUM_EXPERTS = args.num_experts
    TOP_K = args.top_k
    CAPACITY_FACTOR = args.capacity_factor
    AUX_LOSS_ALPHA = args.aux_loss_alpha

    logger = setup_logger(LOG_PATH)
    logger.info("=" * 70)
    logger.info(f"Starting MoE pretrain experiment: {EXP_NAME}")
    logger.info(
        f"Training: EPOCHS={EPOCHS}, LR_0={LR_0}, LR_N={LR_N}, "
        f"BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM={GRADIENT_ACCUMULATION_STEPS}"
    )
    logger.info(
        f"Encoder: depth={ENCODER_DEPTH}, d_model={D_MODEL}, heads={NUM_HEADS}, "
        f"mask_ratio={MASK_RATIO}"
    )
    logger.info(
        f"MoE: num_experts={NUM_EXPERTS}, top_k={TOP_K}, "
        f"capacity_factor={CAPACITY_FACTOR}, aux_loss_alpha={AUX_LOSS_ALPHA}"
    )
    logger.info(
        f"Decoder: depth={DECODER_DEPTH}, d_dec={D_DEC}, heads={DECODER_NUM_HEADS}"
    )
    logger.info("=" * 70)

    # ── Model ────────────────────────────────────────────────────────────────
    model = FrequencyAwareMoEMAE(
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
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        capacity_factor=CAPACITY_FACTOR,
        aux_loss_alpha=AUX_LOSS_ALPHA,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(
        p.numel() for name, p in model.named_parameters()
        if any(k in name for k in ('patch_embedding', 'positional_encoding',
                                   'encoder_layers', 'encoder_norm'))
    )
    logger.info(f"Total params:   {total_params:,}")
    logger.info(f"Encoder params: {encoder_params:,}")
    logger.info(f"Decoder params: {total_params - encoder_params:,}")

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.JPEG((60, 100))], p=0.3),
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
        real_fake_split='real_only',
        split_into_smaller_segments_mul=2,
    )

    train_loader = DataLoader(
        train_dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=False,
    )

    # ── Optimizer — separate param groups for router ──────────────────────────
    # Router weights need lower LR and no weight decay to stay stable
    router_params = [p for n, p in model.named_parameters() if 'router' in n]
    other_params  = [p for n, p in model.named_parameters() if 'router' not in n]

    optimizer = torch.optim.AdamW(
        [
            {'params': router_params, 'lr': LR_0 * 0.1, 'weight_decay': 0.0},
            {'params': other_params,  'lr': LR_0,        'weight_decay': 0.05},
        ],
        betas=(0.9, 0.95),
    )

    # ── LR scheduler — 10% warmup (vs 5% in pretrain_mae.py) ─────────────────
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = min(int(total_steps * 0.10), 500)
    regular_steps = total_steps - warmup_steps

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=regular_steps, eta_min=LR_N
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, regular_scheduler], milestones=[warmup_steps]
    )

    logger.info(
        f"Scheduler: warmup_steps={warmup_steps}, total_steps={total_steps}"
    )

    # ── Dataloader warmup ─────────────────────────────────────────────────────
    start = time.time()
    iterator = iter(train_loader)
    for _ in range(5):
        next(iterator)
    logger.info(f"[TEST] Fetched 5 batches in {round(time.time() - start, 4)}s")

    # ── Training history ──────────────────────────────────────────────────────
    train_loss_history: list[float] = []
    step_total_losses: list[float] = []
    step_mse_losses:   list[float] = []
    step_aux_losses:   list[float] = []
    step_lrs:          list[float] = []
    step_grad_norms:   list[float] = []

    best_loss = float('inf')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
        epoch_loss_history:      list[float] = []
        epoch_mse_history:       list[float] = []
        epoch_aux_history:       list[float] = []
        epoch_grad_norm_history: list[float] = []

        start = time.time()

        for step, (x, _attention_masks, _y) in enumerate(train_loader):
            # Labels are intentionally ignored — SSL pretraining
            x = x.to(DEVICE, non_blocking=True)

            # forward() returns (total_loss, mse_loss, aux_loss)
            total_loss, mse_loss, aux_loss = model(x)
            (total_loss / GRADIENT_ACCUMULATION_STEPS).backward()

            total_norm = nn.utils.get_total_norm(
                [p.grad for p in model.parameters() if p.grad is not None]
            ).mean().item()

            epoch_loss_history.append(total_loss.item())
            epoch_mse_history.append(mse_loss.item())
            epoch_aux_history.append(aux_loss.item())
            epoch_grad_norm_history.append(total_norm)

            step_total_losses.append(total_loss.item())
            step_mse_losses.append(mse_loss.item())
            step_aux_losses.append(aux_loss.item())
            step_grad_norms.append(total_norm)
            step_lrs.append(lr_scheduler.get_last_lr()[0])

            train_loss += total_loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 49:
                window = min(100, step + 1)
                overflow = get_moe_overflow(model)
                logger.info(
                    f"step: {step} | "
                    f"total_loss: {sum(epoch_loss_history[-window:]) / window:.6f} | "
                    f"mse_loss: {sum(epoch_mse_history[-window:]) / window:.6f} | "
                    f"aux_loss: {sum(epoch_aux_history[-window:]) / window:.4f} | "
                    f"overflow: {overflow:.3f} | "
                    f"grad_norm: {sum(epoch_grad_norm_history[-window:]) / window:.4f} | "
                    f"lr: {lr_scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"time: {round(time.time() - start, 2)}s | "
            f"avg_total_loss: {avg_loss:.6f} | "
            f"lr: {lr_scheduler.get_last_lr()[0]:.2e}"
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
                        # MoE fields — required by load_moe_mae_classifier()
                        'num_experts': NUM_EXPERTS,
                        'top_k': TOP_K,
                        'capacity_factor': CAPACITY_FACTOR,
                        'moe': True,
                    },
                },
                CHECKPOINT_PATH,
            )
            logger.info(
                f"  -> Saved best encoder checkpoint "
                f"(loss={best_loss:.6f}) to {CHECKPOINT_PATH}"
            )

        # Save full model every 50 epochs
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(exp_dir, f'epoch{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  -> Saved full checkpoint at epoch {epoch + 1}")

    # Save final full model
    torch.save(model.state_dict(), FULL_CHECKPOINT_PATH)
    logger.info(f"Saved final full model to {FULL_CHECKPOINT_PATH}")

    # ── Training curves ───────────────────────────────────────────────────────
    smoothing_ksize = 100

    def smooth(arr: list[float]) -> np.ndarray:
        k = min(smoothing_ksize, len(arr))
        return np.convolve(arr, np.ones(k) / k, mode='same')

    fig, axes = plt.subplots(ncols=4, figsize=(20, 4))

    axes[0].plot(step_total_losses, alpha=0.3, label='total_loss')
    axes[0].plot(smooth(step_total_losses), label='smoothed')
    axes[0].set_title("Total Loss / step")
    axes[0].legend()

    axes[1].plot(step_mse_losses, alpha=0.3, label='mse_loss')
    axes[1].plot(smooth(step_mse_losses), label='smoothed')
    axes[1].set_title("MSE Loss / step")
    axes[1].legend()

    axes[2].plot(step_aux_losses, alpha=0.3, label='aux_loss')
    axes[2].plot(smooth(step_aux_losses), label='smoothed')
    axes[2].set_title("Aux (MoE) Loss / step")
    axes[2].legend()

    axes[3].plot(step_lrs)
    axes[3].set_title("LR / step")

    fig.savefig(os.path.join(exp_dir, 'training_curves.png'))
    logger.info("Saved training curves.")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(train_loss_history)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('MoE MAE Pretraining Loss (per epoch)')
    fig2.savefig(os.path.join(exp_dir, 'epoch_loss.png'))
    logger.info("Saved epoch loss curve.")
    logger.info("Training complete.")
