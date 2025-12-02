"""
Dataset interfaces and implementations for depth estimation evaluation.
"""

from .base import BaseDataset, DatasetItem, DatasetConfig
from .cityscapes import CityscapesDataset
from .drivingstereo import DrivingStereoDataset
from .middlebury import MiddleburyDataset
from .vkitti import VKITTIDataset

__all__ = [
    'BaseDataset',
    'DatasetItem',
    'DatasetConfig',
    'CityscapesDataset',
    'DrivingStereoDataset',
    'MiddleburyDataset',
    'VKITTIDataset',
]

