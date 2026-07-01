import sys
import time
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from celebdf import CelebDFDataset
from faceforensics import FaceForensicsDataset
from combined_dataset import CombinedVideoDataset

celebdf_path = '../datasets/Celeb-DF-v2'
ffpp_path = '../datasets/ffpp'

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_BATCHES_TO_FETCH = 10
def normalize_neg1_to_1(x):
    """Scale tensor from [0, 1] to [-1, 1]."""
    return x * 2 - 1


train_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.15, hue=0.1, saturation=0.15)], p=0.5),
    T.RandomApply([T.JPEG((60, 100))], p=0.3),
    T.ToDtype(torch.float32, scale=True),
    T.Lambda(normalize_neg1_to_1),  # [0,1] -> [-1,1]
])

ffpp_dataset = FaceForensicsDataset(dataset_path=ffpp_path, transforms=train_transforms)
celebdf_dataset = CelebDFDataset(dataset_path=celebdf_path, transforms=train_transforms)
combined_dataset = CombinedVideoDataset(celebdf_path=celebdf_path, ff_path=ffpp_path, transforms=train_transforms)

print(f'fetching {NUM_BATCHES_TO_FETCH} with batch_size={BATCH_SIZE} from each dataset with num_workers=0')

for dataset in [ffpp_dataset, celebdf_dataset, combined_dataset]:
    loader = DataLoader(
        dataset, 
        BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    start = time.time()

    for i, (x, attention_masks, _y) in enumerate(loader):
        if i >= NUM_BATCHES_TO_FETCH:
            break

    print(time.time() - start)

    del loader

print(f'fetching {NUM_BATCHES_TO_FETCH} with batch_size={BATCH_SIZE} from each dataset with num_workers=8')

for dataset in [ffpp_dataset, celebdf_dataset, combined_dataset]:
    loader = DataLoader(
        dataset, 
        BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    start = time.time()

    for i, (x, attention_masks, _y) in enumerate(loader):
        if i >= NUM_BATCHES_TO_FETCH:
            break

    print(time.time() - start)

    del loader
