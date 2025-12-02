"""
Dataset interfaces and implementations for depth estimation evaluation.
"""

from .base import BaseDataset, DatasetItem, DatasetConfig
from .cityscapes import CityscapesDataset
from .drivingstereo import DrivingStereoDataset
from .middlebury import MiddleburyDataset

__all__ = [
    'BaseDataset',
    'DatasetItem',
    'DatasetConfig',
    'CityscapesDataset',
    'DrivingStereoDataset',
    'MiddleburyDataset',
]

