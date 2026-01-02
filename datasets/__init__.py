"""
Dataset interfaces and implementations for depth estimation evaluation and training.
"""

from .base import BaseDataset, DatasetItem, DatasetConfig
from .cityscapes import CityscapesDataset
from .drivingstereo import DrivingStereoDataset
from .middlebury import MiddleburyDataset
from .vkitti import VKITTIDataset
from .diode import DIODEDataset
from .nyu import NYUDataset
from .kitti import KITTIDataset

# Lazy import for training datasets to avoid import errors when not needed
def __getattr__(name):
    if name == 'VKITTI2TrainingDataset':
        from .training_datasets import VKITTI2TrainingDataset
        return VKITTI2TrainingDataset
    elif name == 'KITTITrainingDataset':
        from .training_datasets import KITTITrainingDataset
        return KITTITrainingDataset
    elif name == 'NYUTrainingDataset':
        from .training_datasets import NYUTrainingDataset
        return NYUTrainingDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BaseDataset',
    'DatasetItem',
    'DatasetConfig',
    'CityscapesDataset',
    'DrivingStereoDataset',
    'MiddleburyDataset',
    'VKITTIDataset',
    'DIODEDataset',
    'NYUDataset',
    'KITTIDataset',
    'VKITTI2TrainingDataset',
    'KITTITrainingDataset',
    'NYUTrainingDataset',
]

