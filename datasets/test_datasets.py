"""
Basic test script to verify imports and class structures.
"""

import sys
from celebdf import CelebDFDataset
from faceforensics import FaceForensicsDataset
from combined_dataset import CombinedVideoDataset

print("Successfully imported all dataset classes!")
print("CelebDFDataset:", CelebDFDataset)
print("FaceForensicsDataset:", FaceForensicsDataset)
print("CombinedVideoDataset:", CombinedVideoDataset)
