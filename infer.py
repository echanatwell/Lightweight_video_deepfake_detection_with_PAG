"""
Inference script for MAEClassifier on a test video dataset.

Expected checkpoint layout:
    checkpoints/last.pth  — MAEClassifier fine-tuned weights (state_dict),
                            saved directly via torch.save(model.state_dict(), ...)

Dataset layout:
    <dataset_path>/
        real/   *.mp4 (or any video format supported by OpenCV)
        fake/   *.mp4

Strategy:
    Each video is split into non-overlapping segments of `frames_per_video` frames.
    The model predicts a class for every segment; the final video-level prediction
    is the mode (majority vote) across all segment predictions.

Usage:
    python infer.py --dataset_path /path/to/dataset [options]
"""

import argparse
import logging
import os
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import av
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score
from torchvision.transforms import v2 as T

from model.mae_model import FrequencyAwareMAE, MAEClassifier


# ---------------------------------------------------------------------------
# Transforms helper (top-level so DataLoader workers can pickle it)
# ---------------------------------------------------------------------------

def _normalize_neg1_to_1(x: Tensor) -> Tensor:
    """Scale float tensor from [0, 1] to [-1, 1]."""
    return x * 2 - 1


# ---------------------------------------------------------------------------
# Video I/O (PyAV)
# ---------------------------------------------------------------------------

def _get_total_frames(video_path: str) -> int:
    container = av.open(video_path)
    n = container.streams.video[0].frames
    container.close()
    return n


def _read_segment(video_path: str, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
    """Read frames [start_frame, end_frame) from a video via PyAV.

    Returns (T, H, W, 3) uint8 BGR array, or None on failure.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "SLICE"

    framerate = stream.average_rate
    time_base = stream.time_base
    frames_to_read = end_frame - start_frame

    sec = int(start_frame / framerate)
    container.seek(int(sec / time_base))

    frames: List[np.ndarray] = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="bgr24"))
        if len(frames) >= frames_to_read:
            break

    container.close()
    return np.array(frames) if frames else None


# ---------------------------------------------------------------------------
# Dataset — one item per video segment
# ---------------------------------------------------------------------------

def _build_segment_entries(
    video_entries: List[Tuple[str, int]],
    frames_per_segment: int,
) -> List[Tuple[str, int, Tuple[int, int], int]]:
    """Split each video into non-overlapping segments.

    Returns list of (video_path, video_idx, (start_frame, end_frame), label).
    """
    result: List[Tuple[str, int, Tuple[int, int], int]] = []
    for video_idx, (video_path, label) in enumerate(video_entries):
        total = _get_total_frames(video_path)
        n_segments = total // frames_per_segment - 1
        for i in range(max(n_segments, 1)):
            start = i * frames_per_segment
            end = start + frames_per_segment
            if end > total:
                break
            result.append((video_path, video_idx, (start, end), label))
    return result


class SegmentDataset(Dataset):
    """Each item is one fixed-length segment of a video.

    Returns (x, attention_mask, label, video_idx) so that segment predictions
    can be grouped back to video level.
    """

    def __init__(
        self,
        dataset_path: str,
        transforms: Optional[Callable] = None,
        frames_per_segment: int = 16,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.frames_per_segment = frames_per_segment

        video_entries: List[Tuple[str, int]] = (
            [(os.path.join(dataset_path, "fake", fn), 1)
             for fn in sorted(os.listdir(os.path.join(dataset_path, "fake")))] +
            [(os.path.join(dataset_path, "real", fn), 0)
             for fn in sorted(os.listdir(os.path.join(dataset_path, "real")))]
        )

        # Store per-video labels for final aggregation
        self.video_labels: List[int] = [label for _, label in video_entries]

        self.entries = _build_segment_entries(video_entries, frames_per_segment)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        video_path, video_idx, (start, end), label = self.entries[idx]

        frames = _read_segment(video_path, start, end)

        if frames is not None:
            n_valid = len(frames)
        else:
            frames = np.zeros((self.frames_per_segment, 224, 224, 3), dtype=np.uint8)
            n_valid = 0

        attention_mask = torch.zeros(self.frames_per_segment, dtype=torch.bool)
        attention_mask[:n_valid] = True

        frame_tensors: List[Tensor] = []
        for frame in frames:
            t = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W) uint8
            if self.transforms is not None:
                t = self.transforms(t)
            frame_tensors.append(t)

        # Pad if segment is shorter than expected
        while len(frame_tensors) < self.frames_per_segment:
            frame_tensors.append(torch.zeros_like(frame_tensors[0]))

        # (T, C, H, W) → (C, T, H, W)
        x: Tensor = torch.stack(frame_tensors[:self.frames_per_segment], dim=0).permute(1, 0, 2, 3)
        return x, attention_mask, torch.tensor(label, dtype=torch.long), video_idx


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    classifier_ckpt: str,
    encoder_depth: int = 4,
    d_model: int = 128,
    num_heads: int = 8,
    patch_size: int = 16,
    max_frames: int = 16,
    img_size: int = 224,
    num_classes: int = 2,
) -> MAEClassifier:
    """Instantiate MAEClassifier and load weights directly from classifier checkpoint.

    Builds a bare FrequencyAwareMAE (random weights) with the given architecture
    hparams, wraps it in MAEClassifier, then overwrites all weights from the
    classifier checkpoint — no encoder.pth required.
    """
    mae = FrequencyAwareMAE(
        encoder_depth=encoder_depth,
        d_model=d_model,
        num_heads=num_heads,
        patch_size=patch_size,
        max_frames=max_frames,
        img_size=img_size,
    )
    model = MAEClassifier(mae, num_classes=num_classes)

    ckpt = torch.load(classifier_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        logging.warning("Missing keys in checkpoint: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys in checkpoint: %s", unexpected)
    logging.info("Loaded MAEClassifier from %s", classifier_ckpt)
    return model


# ---------------------------------------------------------------------------
# Inference: segment-level → video-level via majority vote
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: MAEClassifier,
    loader: DataLoader,
    device: torch.device,
    n_videos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and aggregate segment predictions to video level.

    Returns:
        y_true: (n_videos,) ground-truth labels
        y_pred: (n_videos,) majority-vote predictions
    """
    model.eval()

    # Accumulate per-video segment predictions
    video_preds: Dict[int, List[int]] = {i: [] for i in range(n_videos)}
    video_labels: Dict[int, int] = {}

    for x, attention_masks, y, video_indices in loader:
        x = x.to(device)
        attention_masks = attention_masks.to(device)

        logits = model(x, attention_masks)                  # (B, num_classes)
        preds = logits.argmax(dim=-1).cpu().tolist()        # (B,)
        labels = y.tolist()
        indices = video_indices.tolist()

        for pred, label, vid_idx in zip(preds, labels, indices):
            video_preds[vid_idx].append(pred)
            video_labels[vid_idx] = label

    # Majority vote per video
    y_true, y_pred = [], []
    for vid_idx in sorted(video_labels.keys()):
        y_true.append(video_labels[vid_idx])
        segs = video_preds[vid_idx]
        majority = Counter(segs).most_common(1)[0][0] if segs else 0
        y_pred.append(majority)

    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAEClassifier video-level inference")
    p.add_argument("--dataset_path", required=True,
                   help="Root dir with real/ and fake/ subdirectories")
    p.add_argument("--classifier_ckpt", default="checkpoints/last.pth",
                   help="Path to MAEClassifier fine-tuned weights (state_dict)")
    # encoder architecture (must match the checkpoint)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--frames_per_segment", type=int, default=16,
                   help="Frames per video segment fed to the model")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_file", default="classification_report.log",
                   help="File to write the classification report to")
    return p.parse_args()


def setup_logging(log_file: str) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    logging.info("=" * 60)
    logging.info("MAEClassifier video-level inference")
    logging.info("  dataset      : %s", args.dataset_path)
    logging.info("  classifier   : %s", args.classifier_ckpt)
    logging.info("  device       : %s", args.device)
    logging.info("  seg frames   : %d", args.frames_per_segment)
    logging.info("=" * 60)

    device = torch.device(args.device)

    # ── transforms ──────────────────────────────────────────────────────────
    test_transforms = T.Compose([
        T.Resize((args.img_size, args.img_size), T.InterpolationMode.BICUBIC),
        T.ToDtype(torch.float32, scale=True),
        T.Lambda(_normalize_neg1_to_1),
    ])

    # ── dataset / loader ────────────────────────────────────────────────────
    dataset = SegmentDataset(
        dataset_path=args.dataset_path,
        transforms=test_transforms,
        frames_per_segment=args.frames_per_segment,
    )
    n_videos = len(dataset.video_labels)
    logging.info("Videos: %d  |  Segments: %d", n_videos, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── model ───────────────────────────────────────────────────────────────
    model = load_model(
        classifier_ckpt=args.classifier_ckpt,
        encoder_depth=args.encoder_depth,
        d_model=args.d_model,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        max_frames=args.max_frames,
        img_size=args.img_size,
        num_classes=2,
    )
    model.to(device)

    # ── inference ───────────────────────────────────────────────────────────
    y_true, y_pred = run_inference(model, loader, device, n_videos)

    # ── metrics ─────────────────────────────────────────────────────────────
    f1_fn = F1Score(task="multiclass", num_classes=2)
    f1 = f1_fn(torch.from_numpy(y_pred), torch.from_numpy(y_true)).item()
    report = classification_report(
        y_true, y_pred,
        target_names=["real", "fake"],
        digits=4,
    )

    logging.info("Video-level F1 : %.6f", f1)
    logging.info("Classification report:\n%s", report)
    logging.info("Report saved to: %s", args.log_file)


if __name__ == "__main__":
    main()
