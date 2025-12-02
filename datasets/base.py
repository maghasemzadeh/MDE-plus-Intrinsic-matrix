"""
Base dataset interface for depth estimation evaluation.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class DatasetItem:
    """
    Represents a single item in a dataset.
    
    Attributes:
        item_id: Unique identifier for this item (e.g., image name, scene name)
        image_path: Path to the input image
        gt_path: Path to ground truth depth/disparity
        camera_id: Optional camera identifier (e.g., '0', '1' for stereo pairs)
        metadata: Optional dictionary with additional metadata (e.g., calibration params)
    """
    item_id: str
    image_path: str
    gt_path: str
    camera_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DatasetConfig:
    """
    Configuration for dataset processing.
    
    Attributes:
        dataset_path: Root path to the dataset (None to use dataset's default path)
        split: Dataset split ('train', 'val', 'test')
        max_items: Maximum number of items to process (None for all)
        force_evaluate: If True, re-process even if output exists
        regex_filter: Optional regex pattern to filter items by name (works for all datasets)
    """
    dataset_path: Optional[str] = None
    split: str = 'train'
    max_items: Optional[int] = None
    force_evaluate: bool = False
    regex_filter: Optional[str] = None


class BaseDataset(ABC):
    """
    Abstract base class for depth estimation datasets.
    
    To implement a new dataset, subclass this and implement:
    - find_items(): Find all items in the dataset
    - load_gt_depth(): Load ground truth depth from file
    - get_output_subdir(): Get subdirectory name for output
    - get_default_path(): Get default dataset path
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize dataset.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        # Use provided path or default path
        self.dataset_path = config.dataset_path if config.dataset_path else self.get_default_path()
        self.split = config.split
        self.max_items = config.max_items
        self.force_evaluate = config.force_evaluate
        self.regex_filter = config.regex_filter
    
    @abstractmethod
    def get_default_path(self) -> str:
        """
        Get the default path to the dataset.
        
        Returns:
            Default path to the dataset directory
        """
        pass
    
    @abstractmethod
    def find_items(self) -> List[DatasetItem]:
        """
        Find all items in the dataset.
        
        Returns:
            List of DatasetItem objects
        """
        pass
    
    @abstractmethod
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """
        Load ground truth depth map from file.
        
        Args:
            gt_path: Path to ground truth file
            item: DatasetItem with metadata (may contain calibration info)
        
        Returns:
            Ground truth depth map in meters (invalid pixels as NaN)
        """
        pass
    
    @abstractmethod
    def get_output_subdir(self) -> str:
        """
        Get subdirectory name for output (e.g., 'cityscapes', 'drivingstereo', 'middlebury').
        
        Returns:
            Subdirectory name
        """
        pass
    
    def get_item_output_dir(self, base_output_dir: str, item: DatasetItem) -> str:
        """
        Get output directory for a specific item.
        
        Args:
            base_output_dir: Base output directory
            item: DatasetItem
        
        Returns:
            Full path to item output directory
        """
        import os
        return os.path.join(base_output_dir, self.get_output_subdir(), item.item_id)
    
    def supports_multiple_cameras(self) -> bool:
        """
        Whether this dataset supports multiple cameras per item (e.g., stereo pairs).
        
        Returns:
            True if multiple cameras are supported, False otherwise
        """
        return False
    
    def get_camera_ids(self, item: DatasetItem) -> List[str]:
        """
        Get list of camera IDs for an item (for multi-camera datasets).
        
        Args:
            item: DatasetItem
        
        Returns:
            List of camera IDs (e.g., ['0', '1'] for stereo)
        """
        if item.camera_id is not None:
            return [item.camera_id]
        return ['0']  # Default single camera

