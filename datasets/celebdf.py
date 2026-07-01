"""
CelebDFDataset — PyTorch Dataset for Celeb-DF-v2.

Expected dataset layout (official Celeb-DF-v2 structure):

    <dataset_path>/
        Celeb-real/                     # real celebrity videos  (*.mp4)
        Celeb-synthesis/                # fake celebrity videos  (*.mp4)
        YouTube-real/                   # real YouTube videos    (*.mp4)
        List_of_testing_videos.txt      # official test-split list

Format of List_of_testing_videos.txt (one entry per line):
    <label> <subfolder>/<video_filename>
    e.g.  1 Celeb-synthesis/id0_id1_0000.mp4
          0 Celeb-real/id0_0000.mp4

Labels: 0 = real, 1 = fake  (consistent with the file convention above).

Splits
------
* 'test'       — videos listed in List_of_testing_videos.txt
* 'train'      — 90 % of the remaining videos (deterministic, seed-fixed shuffle)
* 'validation' — 10 % of the remaining videos
"""

import os
import random
from typing import List, Tuple, Optional, Callable, Literal
import av

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ── constants ────────────────────────────────────────────────────────────────
_REAL_DIRS        = ("Celeb-real", "YouTube-real")
_FAKE_DIRS        = ("Celeb-synthesis",)
_TEST_LIST        = "List_of_testing_videos.txt"
_TRAIN_VAL_SEED   = 42
_TRAIN_RATIO      = 0.9


# ── helpers ──────────────────────────────────────────────────────────────────

def _collect_videos(dataset_path: str, real_fake_split: Literal['all', 'real_only', 'fake_only'] = 'all') -> List[Tuple[str, int]]:
    """Return a list of (absolute_video_path, label) for every video found."""
    entries: List[Tuple[str, int]] = []

    if real_fake_split in ['all', 'real_only']:
        for subdir in _REAL_DIRS:
            folder = os.path.join(dataset_path, subdir)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(".mp4"):
                    entries.append((os.path.join(folder, fname), 0))
    if real_fake_split in ['all', 'fake_only']:
        for subdir in _FAKE_DIRS:
            folder = os.path.join(dataset_path, subdir)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(".mp4"):
                    entries.append((os.path.join(folder, fname), 1))
    return entries


def _read_test_set(dataset_path: str) -> set:
    """
    Parse List_of_testing_videos.txt and return a set of normalised relative
    paths, e.g. {'Celeb-synthesis/id0_id1_0000.mp4', ...}.
    """
    test_list_path = os.path.join(dataset_path, _TEST_LIST)
    test_paths: set = set()
    if not os.path.isfile(test_list_path):
        return test_paths
    with open(test_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # format: "<label> <relative_path>"
            rel_path = parts[1] if len(parts) >= 2 else parts[0]
            test_paths.add(rel_path.replace("\\", "/"))
    return test_paths


def _split_dataset(
    dataset_path: str,
    split: str,
    real_fake_split: Literal['all', 'real_only', 'fake_only'] = 'all'
) -> List[Tuple[str, int]]:
    """
    Return the (path, label) list for the requested split.

    split ∈ {'train', 'validation', 'test'}
    """
    all_videos = _collect_videos(dataset_path, real_fake_split)
    test_rel_paths = _read_test_set(dataset_path)

    test_entries:     List[Tuple[str, int]] = []
    non_test_entries: List[Tuple[str, int]] = []

    for abs_path, label in all_videos:
        rel = os.path.relpath(abs_path, dataset_path).replace("\\", "/")
        if rel in test_rel_paths:
            test_entries.append((abs_path, label))
        else:
            non_test_entries.append((abs_path, label))

    if split == "test":
        return test_entries

    # deterministic train / val split
    rng = random.Random(_TRAIN_VAL_SEED)
    shuffled = non_test_entries[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * _TRAIN_RATIO)

    if split == "train":
        return shuffled[:n_train]
    elif split == "validation":
        return shuffled[n_train:]
    else:
        raise ValueError(
            f"Unknown split '{split}'. "
            "Expected one of: 'train', 'validation', 'test'."
        )


def _split_videos_into_smaller_segments(video_entries: List[Tuple[str, int]], frames_per_video: int, multiplier: int = 4):
    #  video entries: list of pair video_path-label

    videos_with_segments = list()

    for video_path, label in video_entries:
        total_frames = _get_number_of_frames_pyav(video_path)
        segment_frames = frames_per_video * multiplier
        for i in range(total_frames // segment_frames - 1):
            videos_with_segments.append((video_path, (i * segment_frames, (i + 1) * segment_frames), label))

    return videos_with_segments


def _get_frame_ranges(video_entries: List[Tuple[str, int]]):
    #  video entries: list of pair video_path-label

    videos_with_segments = list()

    for video_path, label in video_entries:
        total_frames = _get_number_of_frames_pyav(video_path)

        if total_frames < 16:
            print(f'{video_path}: {total_frames}')
            continue

        videos_with_segments.append((video_path, (0, total_frames), label))

    return videos_with_segments


def _get_number_of_frames_pyav(video_path: str):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    container.close()

    return total_frames


def _read_frames_pyav(video_path: str, start_frame: int, end_frame: int, frames_per_video: int) -> Optional[np.ndarray]:
    container = av.open(video_path)
    container.streams.video[0].thread_type = "SLICE"
    total_frames = container.streams.video[0].frames
    framerate = container.streams.video[0].average_rate
    time_base = container.streams.video[0].time_base

    indices = sorted(random.sample(range(start_frame, end_frame), frames_per_video))
    frames = list()

    for idx in indices:
        sec = int(idx / framerate)
        container.seek(int(sec / time_base))
        frame = next(container.decode(video=0)).to_ndarray(format='bgr24')

        frames.append(frame)

    # for i, frame in enumerate(container.decode(video=0)):
    #     if i + start_frame not in indices:
    #         continue
    #     frame = frame.to_ndarray(format='bgr24')
    #     frames.append(frame)

    frames = np.array(frames)

    container.close()

    return frames


# ── dataset class ─────────────────────────────────────────────────────────────

class CelebDFDataset(Dataset):
    """
    Celeb-DF-v2 video dataset.

    Parameters
    ----------
    dataset_path : str
        Root directory of the Celeb-DF-v2 dataset.
    transforms : callable, optional
        Torchvision transform applied to each frame tensor of shape (C, H, W)
        *before* stacking into the video clip.  The MViT default weights
        transform (``MViT_V2_S_Weights.DEFAULT.transforms()``) is compatible.
    frames_per_video : int
        Number of frames to uniformly sample from each video clip.
    split : str
        One of ``'train'``, ``'validation'``, or ``'test'``.

    Returns (from __getitem__)
    --------------------------
    x : Tensor  shape (C, T, H, W)
        Video clip ready for MViT / 3-D CNN input.
    attention_mask : Tensor  shape (T,)  dtype bool
        ``True`` for every valid (non-padded) frame.  All frames are valid
        when the video has at least ``frames_per_video`` frames; padded frames
        are marked ``False``.
    label : Tensor  scalar int64
        0 = real, 1 = fake.
    """

    def __init__(
        self,
        dataset_path: str,
        transforms: Optional[Callable] = None,
        frames_per_video: int = 16,
        split: str = "train",
        img_size: int = 224,
        real_fake_split: Literal['all', 'real_only', 'fake_only'] = 'all',
        split_into_smaller_segments_mul: int = -1
    ) -> None:
        super().__init__()
        self.dataset_path     = dataset_path
        self.transforms       = transforms
        self.frames_per_video = frames_per_video
        self.split            = split
        self.img_size         = img_size
        self.real_fake_split  = real_fake_split
        self.split_into_smaller_segments_mul = split_into_smaller_segments_mul

        self.entries: List[Tuple[str, int]] = _split_dataset(dataset_path, split, self.real_fake_split)
        if split_into_smaller_segments_mul > 1:
            self.entries = _split_videos_into_smaller_segments(self.entries, self.frames_per_video, self.split_into_smaller_segments_mul)
        else:
            self.entries = _get_frame_ranges(self.entries)

        if len(self.entries) == 0:
            raise RuntimeError(
                f"No videos found for split='{split}' in '{dataset_path}'. "
                "Check that the path points to a valid Celeb-DF-v2 directory "
                "and that List_of_testing_videos.txt is present."
            )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        video_path, (start_frame, end_frame), label = self.entries[idx]

        if self.split_into_smaller_segments_mul < 2:
            start_frame = random.randint(start_frame, end_frame - self.frames_per_video)
            end_frame = min(start_frame + self.frames_per_video * 2, end_frame)

        frames = _read_frames_pyav(video_path, start_frame, end_frame, self.frames_per_video)

        # ── build attention mask ──────────────────────────────────────────────
        # _read_frames always returns exactly frames_per_video frames (padding
        # with the last valid frame on failure), so the mask is all-True in the
        # normal case.  We keep the mask for API compatibility with models that
        # accept it (e.g. the custom Transformer in model.py).
        if frames is not None:
            n_valid = len(frames)
        else:
            # complete failure: synthesise a black clip
            frames = np.zeros(
                (self.frames_per_video, self.img_size, self.img_size, 3), dtype=np.uint8
            )
            n_valid = 0

        attention_mask = torch.zeros(self.frames_per_video, dtype=torch.bool)
        attention_mask[:n_valid] = True

        # ── convert frames to tensors and apply transforms ────────────────────
        # frames: (T, H, W, C) uint8  →  list of (C, H, W) uint8 tensors
        frame_tensors: List[Tensor] = []
        for frame in frames:
            t = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W) uint8
            if self.transforms is not None:
                t = self.transforms(t)
            frame_tensors.append(t)

        # stack: (T, C, H, W) → permute → (C, T, H, W)
        x: Tensor = torch.stack(frame_tensors, dim=0).permute(1, 0, 2, 3)

        if n_valid < self.frames_per_video:
            x = torch.cat([x, torch.zeros((3, self.frames_per_video - n_valid, self.img_size, self.img_size), dtype=torch.float32)], dim=1)

        return x, attention_mask, torch.tensor(label, dtype=torch.long)

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_real = sum(1 for _, lbl in self.entries if lbl == 0)
        n_fake = sum(1 for _, lbl in self.entries if lbl == 1)
        return (
            f"CelebDFDataset("
            f"split='{self.split}', "
            f"videos={len(self.entries)} "
            f"[real={n_real}, fake={n_fake}], "
            f"frames_per_video={self.frames_per_video})"
        )
