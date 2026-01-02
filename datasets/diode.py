"""
DIODE dataset implementation.

This dataset contains indoor and outdoor scenes with RGB images and depth maps
stored as numpy arrays. The data is organized hierarchically by scene type,
scene number, and scan number.

Folder structure:
    diode/
    └── val/
        ├── indoors/
        │   └── scene_XXXXX/
        │       └── scan_XXXXX/
        │           ├── XXXXX_XXXXX_{type}_{angle1}_{angle2}.png
        │           ├── XXXXX_XXXXX_{type}_{angle1}_{angle2}_depth.npy
        │           └── XXXXX_XXXXX_{type}_{angle1}_{angle2}_depth_mask.npy
        └── outdoor/
            └── ... (same structure)

Camera Intrinsics (RGB camera, 1024×768 resolution):
    fx = 886.81
    fy = 927.06
    cx = 512
    cy = 384
"""

import os
import re
from typing import List, Optional
from glob import glob
import numpy as np

from .base import BaseDataset, DatasetItem, DatasetConfig


# DIODE camera intrinsic parameters (1024x768 resolution)
DIODE_INTRINSIC_FX = 886.81
DIODE_INTRINSIC_FY = 927.06
DIODE_INTRINSIC_CX = 512.0
DIODE_INTRINSIC_CY = 384.0
DIODE_BASE_WIDTH = 1024
DIODE_BASE_HEIGHT = 768


class DIODEDataset(BaseDataset):
    """
    DIODE dataset for depth estimation evaluation.
    
    This dataset provides RGB images with corresponding depth maps stored as
    numpy arrays. The depth values are in meters. A depth mask is also provided
    to indicate valid depth pixels.
    
    Dataset structure:
    - RGB: diode/val/{indoors,outdoor}/scene_XXXXX/scan_XXXXX/{name}.png
    - Depth: diode/val/{indoors,outdoor}/scene_XXXXX/scan_XXXXX/{name}_depth.npy
    - Mask: diode/val/{indoors,outdoor}/scene_XXXXX/scan_XXXXX/{name}_depth_mask.npy
    
    Attributes:
        max_depth: Maximum valid depth value in meters (default: 10.0 for indoor scenes)
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize DIODE dataset.
        
        Args:
            config: Dataset configuration
        """
        super().__init__(config)
        self.max_depth = 10.0  # meters (DIODE indoor typical max ~10m)
    
    def get_default_path(self) -> str:
        """Get default DIODE dataset path."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'diode')
    
    def find_items(self) -> List[DatasetItem]:
        """
        Find all DIODE image pairs.
        
        Searches through the dataset directory structure to find all RGB images
        with corresponding depth maps.
        
        Returns:
            List of DatasetItem objects, each representing an RGB/depth pair
        """
        items = []
        
        # Determine which split to use (default to 'val' since that's what's available)
        split_dir = os.path.join(self.dataset_path, self.split)
        
        if not os.path.exists(split_dir):
            # Fall back to 'val' if specified split doesn't exist
            split_dir = os.path.join(self.dataset_path, 'val')
            if not os.path.exists(split_dir):
                print(f"Error: DIODE dataset split directory not found: {split_dir}")
                print(f"  Dataset path: {self.dataset_path}")
                raise ValueError(f"DIODE dataset split directory not found: {split_dir}")
        
        # Find all scene type directories (indoors, outdoor)
        scene_type_dirs = sorted([d for d in os.listdir(split_dir) 
                                  if os.path.isdir(os.path.join(split_dir, d))])
        
        for scene_type in scene_type_dirs:
            scene_type_path = os.path.join(split_dir, scene_type)
            
            # Find all scene directories (scene_XXXXX)
            scene_dirs = sorted([d for d in os.listdir(scene_type_path)
                               if os.path.isdir(os.path.join(scene_type_path, d)) 
                               and d.startswith('scene_')])
            
            for scene_name in scene_dirs:
                scene_path = os.path.join(scene_type_path, scene_name)
                
                # Find all scan directories (scan_XXXXX)
                scan_dirs = sorted([d for d in os.listdir(scene_path)
                                   if os.path.isdir(os.path.join(scene_path, d))
                                   and d.startswith('scan_')])
                
                for scan_name in scan_dirs:
                    scan_path = os.path.join(scene_path, scan_name)
                    
                    # Find all RGB images (*.png, excluding depth files)
                    png_files = sorted(glob(os.path.join(scan_path, '*.png')))
                    
                    for rgb_path in png_files:
                        base_name = os.path.basename(rgb_path)
                        
                        # Skip if this is not an RGB image (check for depth in name)
                        if '_depth' in base_name:
                            continue
                        
                        # Get the base name without extension
                        name_without_ext = os.path.splitext(base_name)[0]
                        
                        # Find corresponding depth file
                        depth_file = os.path.join(scan_path, f'{name_without_ext}_depth.npy')
                        
                        if not os.path.exists(depth_file):
                            # Try with .npy* (might be gzipped or similar)
                            depth_files = glob(os.path.join(scan_path, f'{name_without_ext}_depth.npy*'))
                            if len(depth_files) > 0:
                                depth_file = depth_files[0]
                            else:
                                continue
                        
                        # Extract scene number from scene_name (scene_XXXXX -> XXXXX)
                        scene_num = scene_name.replace('scene_', '')
                        # Extract scan number from scan_name (scan_XXXXX -> XXXXX)
                        scan_num = scan_name.replace('scan_', '')
                        
                        # Create item ID: scene_type_sceneNum_scanNum_imageName
                        item_id = f"{scene_type}_{scene_num}_{scan_num}_{name_without_ext}"
                        
                        # Check if depth mask exists
                        mask_file = os.path.join(scan_path, f'{name_without_ext}_depth_mask.npy')
                        mask_files = glob(os.path.join(scan_path, f'{name_without_ext}_depth_mask.npy*'))
                        if len(mask_files) > 0:
                            mask_file = mask_files[0]
                        else:
                            mask_file = None
                        
                        # Create metadata with intrinsics and scene info
                        metadata = {
                            'scene_type': scene_type,
                            'scene_name': scene_name,
                            'scan_name': scan_name,
                            'image_name': name_without_ext,
                            'mask_path': mask_file,
                            'intrinsic': self.get_default_intrinsic()
                        }
                        
                        items.append(DatasetItem(
                            item_id=item_id,
                            image_path=rgb_path,
                            gt_path=depth_file,
                            camera_id='0',  # DIODE uses a single RGB-D camera
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
        
        print(f"Found {len(items)} DIODE image pairs")
        return items
    
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """
        Load DIODE depth map from numpy file.
        
        The depth values are stored in meters. Invalid pixels are marked using
        the depth mask if available, otherwise values <= 0 or > max_depth are
        considered invalid.
        
        Args:
            gt_path: Path to the depth .npy file
            item: DatasetItem with metadata (may contain mask_path)
        
        Returns:
            Ground truth depth map in meters (invalid pixels as NaN)
        """
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Depth file not found: {gt_path}")
        
        # Load depth from numpy file
        depth = np.load(gt_path)
        
        # Ensure float32
        depth = depth.astype(np.float32)
        
        # Load mask if available
        mask = None
        if item.metadata and 'mask_path' in item.metadata and item.metadata['mask_path']:
            mask_path = item.metadata['mask_path']
            if os.path.exists(mask_path):
                try:
                    mask = np.load(mask_path)
                except Exception as e:
                    print(f"Warning: Could not load mask from {mask_path}: {e}")
                    mask = None
        
        # Apply mask to mark invalid pixels as NaN
        if mask is not None:
            # Mask is typically True for valid pixels, False for invalid
            # If mask is inverted, check the data
            if mask.dtype == bool:
                depth[~mask] = np.nan
            else:
                # Assume non-zero values indicate valid pixels
                depth[mask == 0] = np.nan
        else:
            # Without mask, mark invalid based on depth values
            # Zero or negative depth is invalid
            depth[depth <= 0] = np.nan
        
        # Clip to max depth (values beyond max are unreliable)
        depth[depth > self.max_depth] = np.nan
        
        return depth
    
    def get_output_subdir(self) -> str:
        """Get output subdirectory name."""
        return 'diode'
    
    def load_intrinsic(self, item: DatasetItem) -> Optional[np.ndarray]:
        """
        Load camera intrinsic matrix for a DIODE image.
        
        DIODE uses fixed camera intrinsics for all images:
            fx = 886.81
            fy = 927.06
            cx = 512
            cy = 384
        
        for the default 1024x768 resolution.
        
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
        
        # Return default DIODE intrinsics
        return self.get_default_intrinsic()
    
    def get_default_intrinsic(self) -> np.ndarray:
        """
        Get default DIODE intrinsic matrix.
        
        DIODE camera parameters (1024x768 resolution):
            fx = 886.81 pixels
            fy = 927.06 pixels
            cx = 512 pixels
            cy = 384 pixels
        
        Returns:
            3x3 numpy array with default camera intrinsic matrix
        """
        intrinsic = np.array([
            [DIODE_INTRINSIC_FX, 0.0, DIODE_INTRINSIC_CX],
            [0.0, DIODE_INTRINSIC_FY, DIODE_INTRINSIC_CY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return intrinsic
    
    def get_intrinsic_for_size(self, width: int, height: int, 
                                base_intrinsic: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get camera intrinsic matrix scaled to a specific image size.
        
        Scales the intrinsic parameters proportionally based on the ratio
        between the target size and the base DIODE resolution (1024x768).
        
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
        scale_x = width / DIODE_BASE_WIDTH
        scale_y = height / DIODE_BASE_HEIGHT
        
        scaled_intrinsic = base_intrinsic.copy()
        scaled_intrinsic[0, 0] *= scale_x  # fx
        scaled_intrinsic[1, 1] *= scale_y  # fy
        scaled_intrinsic[0, 2] *= scale_x  # cx
        scaled_intrinsic[1, 2] *= scale_y  # cy
        
        return scaled_intrinsic
    
    def has_intrinsics(self) -> bool:
        """DIODE provides camera intrinsic parameters."""
        return True

