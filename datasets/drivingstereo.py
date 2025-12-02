"""
DrivingStereo dataset implementation.
"""

import os
import re
from typing import List
from glob import glob
import numpy as np
import cv2

from .base import BaseDataset, DatasetItem, DatasetConfig


class DrivingStereoDataset(BaseDataset):
    """
    DrivingStereo dataset for depth estimation evaluation.
    
    DrivingStereo provides depth maps directly (no conversion needed).
    """
    
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

