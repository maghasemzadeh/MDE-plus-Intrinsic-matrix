"""
DrivingStereo dataset implementation.
"""

import os
import re
from typing import List, Optional
from glob import glob
import numpy as np
import cv2

from .base import BaseDataset, DatasetItem, DatasetConfig


class DrivingStereoDataset(BaseDataset):
    """
    DrivingStereo dataset for depth estimation evaluation.
    
    DrivingStereo provides depth maps directly (no conversion needed).
    
    Camera Intrinsics:
        DrivingStereo uses calibrated stereo cameras with known intrinsic parameters.
        Based on official DrivingStereo documentation:
        
        Half-resolution (881x400):
        - fx = fy = 879.03 pixels
        - cx = 484.9 pixels
        - cy = 280.85 pixels
        
        Full-resolution (1762x800):
        - fx = fy = 1758.06 pixels
        - cx = 969.8 pixels
        - cy = 561.7 pixels
        
        The intrinsic matrix K is:
            K = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
    """
    
    # Reference resolutions for intrinsic scaling
    HALF_RES_WIDTH = 881
    HALF_RES_HEIGHT = 400
    FULL_RES_WIDTH = 1762
    FULL_RES_HEIGHT = 800
    
    # Half-resolution intrinsic parameters (base values)
    HALF_RES_FX = 879.03
    HALF_RES_FY = 879.03
    HALF_RES_CX = 484.9
    HALF_RES_CY = 280.85
    
    # Full-resolution intrinsic parameters
    FULL_RES_FX = 1758.06
    FULL_RES_FY = 1758.06
    FULL_RES_CX = 969.8
    FULL_RES_CY = 561.7
    
    def get_default_path(self) -> str:
        """Get default DrivingStereo dataset path."""
        import os
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'DrivingStereo')
    
    def find_items(self) -> List[DatasetItem]:
        """Find all DrivingStereo image pairs."""
        items = []
        
        # Find all sequence folders
        all_dirs = [d for d in os.listdir(self.dataset_path) 
                   if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        sequence_folders = []
        for d in all_dirs:
            # Skip depth folders
            if d.endswith(' - depth'):
                continue
            # Clean folder name
            clean_name = d.rstrip(' /')
            if clean_name:
                sequence_folders.append((clean_name, d))
        
        for clean_seq_name, orig_seq_folder in sorted(sequence_folders):
            seq_path = os.path.join(self.dataset_path, orig_seq_folder)
            depth_folder = clean_seq_name + ' - depth'
            depth_path = os.path.join(self.dataset_path, depth_folder)
            
            if not os.path.exists(depth_path):
                # Try with original folder name
                depth_folder_alt = orig_seq_folder.rstrip(' /') + ' - depth'
                depth_path = os.path.join(self.dataset_path, depth_folder_alt)
                if not os.path.exists(depth_path):
                    continue
            
            # Find all images in sequence folder
            image_files = sorted(glob(os.path.join(seq_path, '*.jpg'))) + \
                         sorted(glob(os.path.join(seq_path, '*.png')))
            
            for img_path in image_files:
                # Extract base name
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Find corresponding depth file
                depth_file = os.path.join(depth_path, base_name + '.png')
                
                if os.path.exists(depth_file):
                    items.append(DatasetItem(
                        item_id=base_name,
                        image_path=img_path,
                        gt_path=depth_file,
                        camera_id=None,
                        metadata={'sequence': clean_seq_name}
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
        
        if self.max_items is not None:
            items = items[:self.max_items]
        
        print(f"Found {len(items)} DrivingStereo image pairs")
        return items
    
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """Load DrivingStereo depth map directly."""
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Depth file not found: {gt_path}")
        
        depth_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise ValueError(f"Could not read depth image: {gt_path}")
        
        # Convert uint16 to float and divide by 256 to get depth in meters
        if depth_img.dtype == np.uint16:
            depth = depth_img.astype(np.float32) / 256.0
        else:
            depth = depth_img.astype(np.float32)
        
        # Zero values indicate invalid pixels
        depth[depth == 0] = np.nan
        
        return depth
    
    def get_output_subdir(self) -> str:
        """Get output subdirectory name."""
        return 'drivingstereo'
    
    def load_intrinsic(self, item: DatasetItem) -> Optional[np.ndarray]:
        """
        Load camera intrinsic matrix for a DrivingStereo image.
        
        DrivingStereo uses calibrated cameras with known intrinsic parameters.
        The intrinsics are scaled based on the actual image resolution.
        
        Args:
            item: DatasetItem containing the image path
        
        Returns:
            3x3 numpy array with camera intrinsic matrix K:
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
            Returns None if intrinsics cannot be determined.
        """
        # Get image dimensions to determine which resolution we're working with
        try:
            image = cv2.imread(item.image_path)
            if image is None:
                return None
            img_height, img_width = image.shape[:2]
        except Exception:
            # Use half-resolution as default if we can't read the image
            img_width = self.HALF_RES_WIDTH
            img_height = self.HALF_RES_HEIGHT
        
        # Determine base intrinsics and scale to actual image size
        # Use half-resolution as reference since most DrivingStereo images are half-res
        scale_x = img_width / self.HALF_RES_WIDTH
        scale_y = img_height / self.HALF_RES_HEIGHT
        
        fx = self.HALF_RES_FX * scale_x
        fy = self.HALF_RES_FY * scale_y
        cx = self.HALF_RES_CX * scale_x
        cy = self.HALF_RES_CY * scale_y
        
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return intrinsic
    
    def get_intrinsic_for_size(self, width: int, height: int) -> np.ndarray:
        """
        Get camera intrinsic matrix scaled to a specific image size.
        
        Args:
            width: Target image width
            height: Target image height
        
        Returns:
            3x3 numpy array with camera intrinsic matrix
        """
        # Scale from half-resolution reference
        scale_x = width / self.HALF_RES_WIDTH
        scale_y = height / self.HALF_RES_HEIGHT
        
        fx = self.HALF_RES_FX * scale_x
        fy = self.HALF_RES_FY * scale_y
        cx = self.HALF_RES_CX * scale_x
        cy = self.HALF_RES_CY * scale_y
        
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return intrinsic
    
    def has_intrinsics(self) -> bool:
        """DrivingStereo provides camera intrinsic parameters."""
        return True

