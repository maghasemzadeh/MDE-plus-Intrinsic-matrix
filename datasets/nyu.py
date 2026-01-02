"""
NYU Depth V2 dataset implementation.

This dataset contains indoor scenes with RGB images and depth maps
from a Microsoft Kinect sensor. The labeled subset contains 1,449
densely labeled pairs of aligned RGB and depth images.

The data is stored in a MATLAB .mat file (nyu_depth_v2_labeled.mat) containing:
    - images: RGB images (shape: [3, height, width, n_images] in HDF5 format)
    - depths: Depth maps (shape: [height, width, n_images] in HDF5 format)
    
Note: The .mat file uses HDF5 format (MATLAB v7.3), so data is transposed.
    - images: actual shape [n_images, width, height, 3] when loaded with h5py
    - depths: actual shape [n_images, width, height] when loaded with h5py

Camera Intrinsics (RGB camera, 640×480 resolution):
    fx = 518.8579
    fy = 519.4696
    cx = 325.5824
    cy = 253.7362

Reference:
    Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012).
    Indoor Segmentation and Support Inference from RGBD Images.
    In ECCV 2012.
    https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""

import os
import re
from typing import List, Optional
import numpy as np

from .base import BaseDataset, DatasetItem, DatasetConfig


# NYU Depth V2 camera intrinsic parameters (640x480 resolution)
NYU_INTRINSIC_FX = 518.8579
NYU_INTRINSIC_FY = 519.4696
NYU_INTRINSIC_CX = 325.5824
NYU_INTRINSIC_CY = 253.7362
NYU_BASE_WIDTH = 640
NYU_BASE_HEIGHT = 480


class NYUDataset(BaseDataset):
    """
    NYU Depth V2 dataset for depth estimation evaluation.
    
    This dataset provides RGB images with corresponding depth maps from the
    NYU Depth V2 labeled subset. The depth values are in meters, captured
    using a Microsoft Kinect sensor.
    
    The dataset is stored as a single .mat file (nyu_depth_v2_labeled.mat)
    containing all 1,449 labeled image pairs.
    
    Attributes:
        max_depth: Maximum valid depth value in meters (default: 10.0 for indoor scenes)
        _images: Cached RGB images from the .mat file
        _depths: Cached depth maps from the .mat file
        _mat_loaded: Flag indicating if the .mat file has been loaded
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize NYU Depth V2 dataset.
        
        Args:
            config: Dataset configuration
        """
        super().__init__(config)
        self.max_depth = 10.0  # meters (NYU indoor typical max ~10m)
        self._images = None
        self._depths = None
        self._mat_loaded = False
        self._mat_file_path = None
    
    def get_default_path(self) -> str:
        """Get default NYU Depth V2 dataset path."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'nyu')
    
    def _load_mat_file(self) -> None:
        """
        Load the NYU Depth V2 .mat file.
        
        Uses h5py to read the HDF5-format .mat file (MATLAB v7.3).
        Caches the images and depths arrays for efficient access.
        """
        if self._mat_loaded:
            return
        
        # Try to import h5py
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required to read NYU Depth V2 .mat files. "
                "Install it with: pip install h5py"
            )
        
        # Find the .mat file
        mat_file = os.path.join(self.dataset_path, 'nyu_depth_v2_labeled.mat')
        
        if not os.path.exists(mat_file):
            raise FileNotFoundError(
                f"NYU Depth V2 .mat file not found: {mat_file}\n"
                f"Please download the labeled subset from:\n"
                f"https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html\n"
                f"and place nyu_depth_v2_labeled.mat in: {self.dataset_path}"
            )
        
        self._mat_file_path = mat_file
        
        print(f"Loading NYU Depth V2 dataset from {mat_file}...")
        print("(This may take a moment for the first load)")
        
        # Load the .mat file using h5py
        with h5py.File(mat_file, 'r') as f:
            # HDF5/MATLAB v7.3 format stores data transposed
            # images: shape [3, height, width, n_images] in MATLAB
            # becomes [n_images, width, height, 3] when loaded with h5py
            # We need to transpose to get [n_images, height, width, 3]
            self._images = np.array(f['images'])
            self._depths = np.array(f['depths'])
        
        # Transpose from HDF5 format to standard format
        # images: [3, height, width, n_images] -> [n_images, height, width, 3]
        # depths: [height, width, n_images] -> [n_images, height, width]
        self._images = np.transpose(self._images, (3, 2, 1, 0))
        self._depths = np.transpose(self._depths, (2, 1, 0))
        
        self._mat_loaded = True
        print(f"Loaded {len(self._images)} image pairs from NYU Depth V2 dataset")
    
    def find_items(self) -> List[DatasetItem]:
        """
        Find all NYU Depth V2 image pairs.
        
        Loads the .mat file and creates DatasetItem objects for each image pair.
        The image data is stored in memory as cached arrays.
        
        Returns:
            List of DatasetItem objects, each representing an RGB/depth pair
        """
        # Load the .mat file if not already loaded
        self._load_mat_file()
        
        items = []
        n_images = len(self._images)
        
        for idx in range(n_images):
            # Create item ID with zero-padded index
            item_id = f"nyu_{idx:05d}"
            
            # For this dataset, we use indices instead of file paths
            # The image_path and gt_path will be special markers that indicate
            # to use the cached arrays
            image_path = f"__NYU_CACHE__:{idx}:image"
            gt_path = f"__NYU_CACHE__:{idx}:depth"
            
            # Create metadata with intrinsics
            metadata = {
                'index': idx,
                'intrinsic': self.get_default_intrinsic()
            }
            
            items.append(DatasetItem(
                item_id=item_id,
                image_path=image_path,
                gt_path=gt_path,
                camera_id='0',  # NYU uses a single RGB-D camera (Kinect)
                metadata=metadata
            ))
        
        # Apply regex filter if provided
        if self.regex_filter is not None:
            try:
                pattern = re.compile(self.regex_filter)
                original_count = len(items)
                items = [item for item in items if pattern.search(item.item_id)]
                if len(items) > 0:
                    print(f"Regex filter '{self.regex_filter}': {len(items)}/{original_count} items match")
                else:
                    print(f"Warning: Regex filter '{self.regex_filter}' matched 0 items")
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{self.regex_filter}': {e}")
                print("Proceeding without regex filter...")
        
        # Apply max_items limit
        if self.max_items is not None:
            items = items[:self.max_items]
        
        print(f"Found {len(items)} NYU Depth V2 image pairs")
        return items
    
    def load_image(self, item: DatasetItem) -> np.ndarray:
        """
        Load RGB image from the cached data.
        
        Args:
            item: DatasetItem containing the index in metadata
        
        Returns:
            RGB image as numpy array with shape [H, W, 3] and dtype uint8
        """
        # Ensure mat file is loaded
        self._load_mat_file()
        
        # Get index from metadata
        idx = item.metadata['index']
        
        # Return the cached image
        return self._images[idx].astype(np.uint8)
    
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """
        Load NYU Depth V2 depth map from cached data.
        
        The depth values are in meters. Invalid pixels (depth <= 0 or > max_depth)
        are marked as NaN.
        
        Args:
            gt_path: Path identifier (used to extract index for cached data)
            item: DatasetItem with metadata containing the index
        
        Returns:
            Ground truth depth map in meters (invalid pixels as NaN)
        """
        # Ensure mat file is loaded
        self._load_mat_file()
        
        # Get index from metadata
        idx = item.metadata['index']
        
        # Get depth from cached array
        depth = self._depths[idx].astype(np.float32)
        
        # Mark invalid pixels as NaN
        # Zero or negative depth is invalid
        depth[depth <= 0] = np.nan
        
        # Clip to max depth (values beyond max are unreliable for Kinect)
        depth[depth > self.max_depth] = np.nan
        
        return depth
    
    def get_output_subdir(self) -> str:
        """Get output subdirectory name."""
        return 'nyu'
    
    def load_intrinsic(self, item: DatasetItem) -> Optional[np.ndarray]:
        """
        Load camera intrinsic matrix for a NYU Depth V2 image.
        
        NYU Depth V2 uses fixed camera intrinsics for all images:
            fx = 518.8579
            fy = 519.4696
            cx = 325.5824
            cy = 253.7362
        
        for the default 640x480 resolution.
        
        Args:
            item: DatasetItem containing the image path and metadata
        
        Returns:
            3x3 numpy array with camera intrinsic matrix K:
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
        # Check if intrinsics are in metadata (set during find_items)
        if item.metadata and 'intrinsic' in item.metadata:
            intrinsic = item.metadata['intrinsic']
            if isinstance(intrinsic, np.ndarray) and intrinsic.shape == (3, 3):
                return intrinsic.astype(np.float32)
        
        # Return default NYU intrinsics
        return self.get_default_intrinsic()
    
    def get_default_intrinsic(self) -> np.ndarray:
        """
        Get default NYU Depth V2 intrinsic matrix.
        
        NYU Depth V2 camera parameters (640x480 resolution):
            fx = 518.8579 pixels
            fy = 519.4696 pixels
            cx = 325.5824 pixels
            cy = 253.7362 pixels
        
        Returns:
            3x3 numpy array with default camera intrinsic matrix
        """
        intrinsic = np.array([
            [NYU_INTRINSIC_FX, 0.0, NYU_INTRINSIC_CX],
            [0.0, NYU_INTRINSIC_FY, NYU_INTRINSIC_CY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return intrinsic
    
    def get_intrinsic_for_size(self, width: int, height: int, 
                                base_intrinsic: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get camera intrinsic matrix scaled to a specific image size.
        
        Scales the intrinsic parameters proportionally based on the ratio
        between the target size and the base NYU resolution (640x480).
        
        Args:
            width: Target image width
            height: Target image height
            base_intrinsic: Optional base intrinsic to scale from 
                          (default: use get_default_intrinsic())
        
        Returns:
            3x3 numpy array with scaled camera intrinsic matrix
        """
        if base_intrinsic is None:
            base_intrinsic = self.get_default_intrinsic()
        
        # Scale from base resolution
        scale_x = width / NYU_BASE_WIDTH
        scale_y = height / NYU_BASE_HEIGHT
        
        scaled_intrinsic = base_intrinsic.copy()
        scaled_intrinsic[0, 0] *= scale_x  # fx
        scaled_intrinsic[1, 1] *= scale_y  # fy
        scaled_intrinsic[0, 2] *= scale_x  # cx
        scaled_intrinsic[1, 2] *= scale_y  # cy
        
        return scaled_intrinsic
    
    def has_intrinsics(self) -> bool:
        """NYU Depth V2 provides camera intrinsic parameters."""
        return True
    
    def is_cached_dataset(self) -> bool:
        """
        Indicates that this dataset uses cached data rather than file paths.
        
        Returns:
            True, indicating images should be loaded via load_image method
        """
        return True

