"""
FaceForensicsDataset — PyTorch Dataset for FaceForensics++ (FF++).

Expected dataset layout (official FaceForensics++ structure):

    <dataset_path>/
        original_sequences/
            youtube/
                <compression>/
                    videos/             # real videos (*.mp4)
            actors/
                <compression>/
                    videos/             # real DFD videos (*.mp4)
        manipulated_sequences/
            Deepfakes/
                <compression>/
                    videos/             # fake videos (*.mp4)
            Face2Face/
                <compression>/
                    videos/             # fake videos (*.mp4)
            FaceSwap/
                <compression>/
                    videos/             # fake videos (*.mp4)
            NeuralTextures/
                <compression>/
                    videos/             # fake videos (*.mp4)
            FaceShifter/
                <compression>/
                    videos/             # fake videos (*.mp4)
            DeepFakeDetection/
                <compression>/
                    videos/             # fake DFD videos (*.mp4)
        splits/                         # optional official splits
            train.json
            val.json
            test.json

Labels: 0 = real, 1 = fake.
"""

import os
import json
import random
from typing import List, Tuple, Optional, Callable
import av

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_video_id(filename: str) -> str:
    """
    Extract the target video ID or actor ID from the filename.
    e.g. "000.mp4" -> "000"
         "000_001.mp4" -> "000"
         "01__talking_against_wall.mp4" -> "01"
         "01_02__talking_against_wall__12345678.mp4" -> "01"
    """
    name = os.path.splitext(filename)[0]
    if "_" in name:
        if "__" in name:
            # DFD: "01_02__scene" -> "01" or "01__scene" -> "01"
            return name.split("__")[0].split("_")[0]
        else:
            # FF++: "000_001" -> "000"
            return name.split("_")[0]
    return name


def _collect_ff_videos(
    dataset_path: str,
    compression: str,
    methods: List[str],
    include_dfd: bool
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Collect all FF++ videos and return youtube and DFD entries separately."""
    youtube_entries: List[Tuple[str, int]] = []
    dfd_entries: List[Tuple[str, int]] = []

    # 1. Real videos
    # Youtube real
    yt_real_dir = os.path.join(dataset_path, "original_sequences", "youtube", compression, "videos")
    if os.path.isdir(yt_real_dir):
        for fname in sorted(os.listdir(yt_real_dir)):
            if fname.lower().endswith(".mp4"):
                youtube_entries.append((os.path.join(yt_real_dir, fname), 0))

    # DFD real
    if include_dfd:
        dfd_real_dir = os.path.join(dataset_path, "original_sequences", "actors", compression, "videos")
        if os.path.isdir(dfd_real_dir):
            for fname in sorted(os.listdir(dfd_real_dir)):
                if fname.lower().endswith(".mp4"):
                    dfd_entries.append((os.path.join(dfd_real_dir, fname), 0))

    # 2. Fake videos
    # Youtube fake (manipulated methods)
    for method in methods:
        method_dir = os.path.join(dataset_path, "manipulated_sequences", method, compression, "videos")
        if os.path.isdir(method_dir):
            for fname in sorted(os.listdir(method_dir)):
                if fname.lower().endswith(".mp4"):
                    youtube_entries.append((os.path.join(method_dir, fname), 1))

    # DFD fake
    if include_dfd:
        dfd_fake_dir = os.path.join(dataset_path, "manipulated_sequences", "DeepFakeDetection", compression, "videos")
        if os.path.isdir(dfd_fake_dir):
            for fname in sorted(os.listdir(dfd_fake_dir)):
                if fname.lower().endswith(".mp4"):
                    dfd_entries.append((os.path.join(dfd_fake_dir, fname), 1))

    return youtube_entries, dfd_entries


def _load_official_split(dataset_path: str, split: str) -> Optional[set]:
    """Try to load official split JSON file if it exists."""
    # Map 'validation' to 'val' as FF++ splits are usually named 'val.json'
    split_name = "val" if split == "validation" else split
    possible_paths = [
        os.path.join(dataset_path, "splits", f"{split_name}.json"),
        os.path.join(dataset_path, f"{split_name}.json"),
    ]
    for path in possible_paths:
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                video_ids = set()
                for item in data:
                    if isinstance(item, list):
                        for subitem in item:
                            video_ids.add(str(subitem))
                    else:
                        video_ids.add(str(item))
                return video_ids
            except Exception:
                pass
    return None


def _split_ff_dataset(
    dataset_path: str,
    split: str,
    compression: str,
    methods: List[str],
    include_dfd: bool,
) -> List[Tuple[str, int]]:
    """Return the (path, label) list for the requested split."""
    youtube_entries, dfd_entries = _collect_ff_videos(dataset_path, compression, methods, include_dfd)

    # Split youtube entries
    official_ids = _load_official_split(dataset_path, split)
    if official_ids is not None:
        selected_youtube = []
        for path, label in youtube_entries:
            vid_id = _get_video_id(os.path.basename(path))
            if vid_id in official_ids:
                selected_youtube.append((path, label))
    else:
        # Generate deterministic split of 1000 IDs (720 train, 140 val, 140 test)
        all_ff_ids = [f"{i:03d}" for i in range(1000)]
        rng = random.Random(42)
        rng.shuffle(all_ff_ids)
        if split == "train":
            ff_split_ids = set(all_ff_ids[:720])
        elif split == "validation":
            ff_split_ids = set(all_ff_ids[720:860])
        elif split == "test":
            ff_split_ids = set(all_ff_ids[860:])
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'validation', or 'test'.")

        selected_youtube = []
        for path, label in youtube_entries:
            vid_id = _get_video_id(os.path.basename(path))
            if vid_id in ff_split_ids:
                selected_youtube.append((path, label))

    # Split DFD entries if included
    selected_dfd = []
    if include_dfd and dfd_entries:
        all_actor_ids = set()
        for path, _ in dfd_entries:
            all_actor_ids.add(_get_video_id(os.path.basename(path)))
        
        shuffled_actors = sorted(list(all_actor_ids))
        rng = random.Random(42)
        rng.shuffle(shuffled_actors)
        
        n_actors = len(shuffled_actors)
        n_train = int(n_actors * 0.7)
        n_val = int(n_actors * 0.15)
        
        if split == "train":
            dfd_split_ids = set(shuffled_actors[:n_train])
        elif split == "validation":
            dfd_split_ids = set(shuffled_actors[n_train:n_train+n_val])
        elif split == "test":
            dfd_split_ids = set(shuffled_actors[n_train+n_val:])
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'validation', or 'test'.")

        for path, label in dfd_entries:
            actor_id = _get_video_id(os.path.basename(path))
            if actor_id in dfd_split_ids:
                selected_dfd.append((path, label))

    return selected_youtube + selected_dfd


def _read_frames(video_path: str, frames_per_video: int) -> Optional[np.ndarray]:
    """
    Read `frames_per_video` frames uniformly sampled from the video.

    Returns an ndarray of shape (T, H, W, C) in uint8 RGB, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # uniform indices, clamped to valid range
    indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # fall back to the last successfully read frame (or a black frame)
            if frames:
                frames.append(frames[-1].copy())
            else:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 224
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W, C)


def _read_frames_pyav(video_path: str, frames_per_video: int) -> Optional[np.ndarray]:
    container = av.open(video_path)
    container.streams.video[0].thread_type = "SLICE"
    total_frames = container.streams.video[0].frames
    framerate = container.streams.video[0].average_rate
    time_base = container.streams.video[0].time_base

    start_frame = random.randint(total_frames - frames_per_video - 1, total_frames)
    indices = list(range(start_frame, start_frame + frames_per_video + 1))

    sec = int(start_frame / framerate)
    container.seek(int(sec / time_base))

    frames = [] 

    for frame in container.decode(video=0):
        frame = frame.to_ndarray(format='bgr24')
        frames.append(frame)

    frames = np.array(frames)

    return frames

# ── dataset class ─────────────────────────────────────────────────────────────

class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ (FF++) video dataset.

    Parameters
    ----------
    dataset_path : str
        Root directory of the FaceForensics++ dataset.
    transforms : callable, optional
        Torchvision transform applied to each frame tensor of shape (C, H, W)
        *before* stacking into the video clip.
    frames_per_video : int
        Number of frames to uniformly sample from each video clip.
    split : str
        One of ``'train'``, ``'validation'``, or ``'test'``.
    compression : str
        One of ``'raw'``, ``'c23'``, or ``'c40'``.
    methods : list of str, optional
        List of manipulation methods to include. Defaults to:
        ``['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']``.
    include_dfd : bool
        Whether to include DeepFakeDetection (DFD) dataset videos.

    Returns (from __getitem__)
    --------------------------
    x : Tensor  shape (C, T, H, W)
        Video clip ready for MViT / 3-D CNN input.
    attention_mask : Tensor  shape (T,)  dtype bool
        ``True`` for every valid (non-padded) frame.
    label : Tensor  scalar int64
        0 = real, 1 = fake.
    """

    def __init__(
        self,
        dataset_path: str,
        transforms: Optional[Callable] = None,
        frames_per_video: int = 16,
        split: str = "train",
        compression: str = "c23",
        methods: Optional[List[str]] = None,
        include_dfd: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path     = dataset_path
        self.transforms       = transforms
        self.frames_per_video = frames_per_video
        self.split            = split
        self.compression      = compression
        self.methods          = methods or ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        self.include_dfd      = include_dfd

        self.entries: List[Tuple[str, int]] = _split_ff_dataset(
            dataset_path=self.dataset_path,
            split=self.split,
            compression=self.compression,
            methods=self.methods,
            include_dfd=self.include_dfd
        )

        if len(self.entries) == 0:
            raise RuntimeError(
                f"No videos found for split='{split}' in '{dataset_path}'. "
                "Check that the path points to a valid FaceForensics++ directory."
            )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        video_path, label = self.entries[idx]

        frames = _read_frames_pyav(video_path, self.frames_per_video)

        # ── build attention mask ──────────────────────────────────────────────
        if frames is not None:
            n_valid = self.frames_per_video
        else:
            # complete failure: synthesise a black clip
            frames = np.zeros(
                (self.frames_per_video, 224, 224, 3), dtype=np.uint8
            )
            n_valid = 0

        attention_mask = torch.zeros(self.frames_per_video, dtype=torch.bool)
        attention_mask[:n_valid] = True

        # ── convert frames to tensors and apply transforms ────────────────────
        frame_tensors: List[Tensor] = []
        for frame in frames:
            t = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W) uint8
            if self.transforms is not None:
                t = self.transforms(t)
            frame_tensors.append(t)

        # stack: (T, C, H, W) → permute → (C, T, H, W)
        x: Tensor = torch.stack(frame_tensors, dim=0).permute(1, 0, 2, 3)

        return x, attention_mask, torch.tensor(label, dtype=torch.long)

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_real = sum(1 for _, lbl in self.entries if lbl == 0)
        n_fake = sum(1 for _, lbl in self.entries if lbl == 1)
        return (
            f"FaceForensicsDataset("
            f"split='{self.split}', "
            f"compression='{self.compression}', "
            f"videos={len(self.entries)} "
            f"[real={n_real}, fake={n_fake}], "
            f"frames_per_video={self.frames_per_video})"
        )
