"""
Dataset interfaces and implementations for depth estimation evaluation and training.
"""

from .base import BaseDataset, DatasetItem, DatasetConfig
from .cityscapes import CityscapesDataset
from .drivingstereo import DrivingStereoDataset
from .middlebury import MiddleburyDataset
from .vkitti import VKITTIDataset
from .training_datasets import VKITTI2TrainingDataset, KITTITrainingDataset

__all__ = [
    'BaseDataset',
    'DatasetItem',
    'DatasetConfig',
    'CityscapesDataset',
    'DrivingStereoDataset',
    'MiddleburyDataset',
    'VKITTIDataset',
    'VKITTI2TrainingDataset',
    'KITTITrainingDataset',
]

