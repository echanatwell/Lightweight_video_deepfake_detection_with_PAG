"""
CombinedVideoDataset — PyTorch Dataset combining Celeb-DF-v2 and FaceForensics++ (FF++).
"""

from typing import List, Tuple, Optional, Callable
import torch
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset

from .celebdf import CelebDFDataset
from .faceforensics import FaceForensicsDataset


class CombinedVideoDataset(Dataset):
    """
    A combined dataset for Celeb-DF-v2 and FaceForensics++ (FF++).

    Parameters
    ----------
    celebdf_path : str, optional
        Root directory of the Celeb-DF-v2 dataset. If None, Celeb-DF-v2 is omitted.
    ff_path : str, optional
        Root directory of the FaceForensics++ dataset. If None, FF++ is omitted.
    transforms : callable, optional
        Torchvision transform applied to each frame tensor of shape (C, H, W)
        *before* stacking into the video clip.
    frames_per_video : int
        Number of frames to uniformly sample from each video clip.
    split : str
        One of ``'train'``, ``'validation'``, or ``'test'``.
    ff_compression : str
        One of ``'raw'``, ``'c23'``, or ``'c40'`` for FF++.
    ff_methods : list of str, optional
        List of manipulation methods to include for FF++.
    ff_include_dfd : bool
        Whether to include DeepFakeDetection (DFD) dataset videos in FF++.

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
        celebdf_path: Optional[str] = None,
        ff_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        frames_per_video: int = 16,
        split: str = "train",
        ff_compression: str = "c23",
        ff_methods: Optional[List[str]] = None,
        ff_include_dfd: bool = False,
    ) -> None:
        super().__init__()
        self.celebdf_path     = celebdf_path
        self.ff_path          = ff_path
        self.transforms       = transforms
        self.frames_per_video = frames_per_video
        self.split            = split
        self.ff_compression   = ff_compression
        self.ff_methods        = ff_methods
        self.ff_include_dfd   = ff_include_dfd

        self.datasets: List[Dataset] = []

        if celebdf_path is not None:
            self.celebdf_dataset = CelebDFDataset(
                dataset_path=celebdf_path,
                transforms=transforms,
                frames_per_video=frames_per_video,
                split=split,
            )
            self.datasets.append(self.celebdf_dataset)
        else:
            self.celebdf_dataset = None

        if ff_path is not None:
            self.ff_dataset = FaceForensicsDataset(
                dataset_path=ff_path,
                transforms=transforms,
                frames_per_video=frames_per_video,
                split=split,
                compression=ff_compression,
                methods=ff_methods,
                include_dfd=ff_include_dfd,
            )
            self.datasets.append(self.ff_dataset)
        else:
            self.ff_dataset = None

        if not self.datasets:
            raise ValueError("At least one of 'celebdf_path' or 'ff_path' must be provided.")

        # Use ConcatDataset internally to handle indexing and length
        self.concat_dataset = ConcatDataset(self.datasets)

    # ── dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.concat_dataset[idx]

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        repr_str = f"CombinedVideoDataset(split='{self.split}', total_videos={len(self)})"
        if self.celebdf_dataset is not None:
            repr_str += f"\n  └─ {repr(self.celebdf_dataset)}"
        if self.ff_dataset is not None:
            repr_str += f"\n  └─ {repr(self.ff_dataset)}"
        return repr_str
