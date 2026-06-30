from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable
import numpy as np
import cv2
from torch import Tensor
import torch
import pandas as pd
import argparse
from finetune_mae import MAEClassifier
from model.mae_model import FrequencyAwareMAE
from torch import nn
from torchmetrics import F1Score
from sklearn.metrics import classification_report
import os
import av


def _read_frames_pyav(video_path: str, start_frame: int, end_frame: int) -> Optional[np.ndarray]:
    container = av.open(video_path)
    container.streams.video[0].thread_type = "SLICE"
    total_frames = container.streams.video[0].frames
    framerate = container.streams.video[0].average_rate
    time_base = container.streams.video[0].time_base

    frames_to_read = end_frame - start_frame
    sec = int(start_frame / framerate)
    container.seek(int(sec / time_base))

    frames = list()

    for frame in container.decode(video=0):
        frame = frame.to_ndarray(format='bgr24')
        frames.append(frame)

        if len(frames) >= frames_to_read:
            break

    frames = np.array(frames)

    container.close()

    return frames


def _get_number_of_frames_pyav(video_path: str):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    container.close()

    return total_frames


def _split_videos_into_smaller_segments(video_entries: List[Tuple[str, int]], frames_per_video: int):
    #  video entries: list of pair video_path-label

    videos_with_segments = list()

    for video_path, label in video_entries:
        total_frames = _get_number_of_frames_pyav(video_path)
        segment_frames = frames_per_video
        for i in range(total_frames // segment_frames - 1):
            videos_with_segments.append((video_path, (i * segment_frames, (i + 1) * segment_frames), label))

    return videos_with_segments


class TestDataset(Dataset):

    def __init__(
        self,
        labels_path: str,
        videos_path: str,
        transforms: Optional[Callable] = None,
        frames_per_video: int = 16,
    ) -> None:
        super().__init__()

        self.transforms = transforms
        self.frames_per_video = frames_per_video
        
        labels_df = pd.read_csv(labels_path)
        self.entries = list(zip(labels_df.obj_id.apply(lambda x: os.path.join(videos_path, x + '.mp4')), labels_df.label))
        self.entries = _split_videos_into_smaller_segments(self.entries, self.frames_per_video)


    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        video_path, (start_frame, end_frame), label = self.entries[idx]

        frames = _read_frames_pyav(video_path, start_frame, end_frame)

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


def normalize_neg1_to_1(x):
    """Scale tensor from [0, 1] to [-1, 1]."""
    return x * 2 - 1

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('pretrained_encoder_checkpoint', type=str)
    parser.add_argument('classifier_checkpoint', type=str)
    parser.add_argument('--frames-per-video', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)

    return parser.parse_args()


labels_path = "../datasets/huawei_test_part1/d83a0ce6-dc87-46a6-9679-98a71cf91886.csv"
videos_path = "../datasets/huawei_test_part1/videos"
DEVICE = 'cuda:0'

if __name__ == '__main__':
    args = parse_args()

    pretrained_encoder_checkpoint = args.pretrained_encoder_checkpoint
    classifier_checkpoint = args.classifier_checkpoint
    FRAMES_PER_VIDEO = args.frames_per_video
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size

    test_transforms = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
            T.ToDtype(torch.float32, scale=True),
            T.Lambda(normalize_neg1_to_1),
        ])


    test_dataset = TestDataset(
            labels_path=labels_path, 
            videos_path=videos_path,
            transforms=test_transforms,
            frames_per_video=FRAMES_PER_VIDEO
        )
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, drop_last=False)

    model = load_mae_classifier(pretrained_encoder_checkpoint, num_classes=2)
    model.load_state_dict(torch.load(classifier_checkpoint, map_location='cpu'))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    f1_score_fn = F1Score(task="multiclass", num_classes=2).to(DEVICE)

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