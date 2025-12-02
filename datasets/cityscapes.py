"""
Cityscapes dataset implementation.
"""

import os
import re
from typing import List, Optional
from glob import glob
import numpy as np
import cv2

from .base import BaseDataset, DatasetItem, DatasetConfig


class CityscapesDataset(BaseDataset):
    """
    Cityscapes dataset for depth estimation evaluation.
    
    Cityscapes provides disparity maps that need to be converted to depth.
    """
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        # Cityscapes stereo parameters
        self.baseline = 0.22  # meters
        self.focal_length = 2262.52  # pixels
        self.min_disparity = 0.1  # pixels
        self.max_depth = 200.0  # meters
    
    def get_default_path(self) -> str:
        """Get default Cityscapes dataset path."""
        import os
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'CityScapes')
    
    def find_items(self) -> List[DatasetItem]:
        """Find all Cityscapes image pairs."""
        gt_dir = os.path.join(self.dataset_path, 'disparity', self.split)
        
        if not os.path.exists(gt_dir):
            print(f"Error: Cityscapes disparity directory not found: {gt_dir}")
            print(f"  Dataset path: {self.dataset_path}")
            print(f"  Split: {self.split}")
            raise ValueError(f"Cityscapes disparity directory not found: {gt_dir}")
        
        # Try to find left images directory
        possible_left_dirs = [
            os.path.join(self.dataset_path, 'leftImg8bit', self.split),
            os.path.join(self.dataset_path, 'leftImg8bit', self.split.lower()),
            os.path.join(self.dataset_path, 'images', self.split),
            os.path.join(self.dataset_path, 'left', self.split),
        ]
        
        left_img_dir = None
        for possible_dir in possible_left_dirs:
            if os.path.exists(possible_dir):
                left_img_dir = possible_dir
                break
        
        if left_img_dir is None:
            print(f"Warning: Cityscapes left images directory not found.")
            print(f"  Dataset path: {self.dataset_path}")
            print(f"  Split: {self.split}")
            print(f"  Tried paths:")
            for pd in possible_left_dirs:
                print(f"    - {pd} (exists: {os.path.exists(pd)})")
            return []
        
        items = []
        
        # Walk through city directories
        for city_dir in sorted(os.listdir(gt_dir)):
            city_gt_path = os.path.join(gt_dir, city_dir)
            
            if not os.path.isdir(city_gt_path):
                continue
            
            # Find all disparity files
            disparity_files = sorted(glob(os.path.join(city_gt_path, '*_disparity.png')))
            
            for disp_path in disparity_files:
                # Extract base name
                base_name = os.path.basename(disp_path).replace('_disparity.png', '')
                
                # Try to find corresponding left image
                city_left_path = os.path.join(left_img_dir, city_dir)
                if not os.path.isdir(city_left_path):
                    city_left_path = left_img_dir
                
                # Try multiple naming patterns
                possible_image_names = [
                    f'{base_name}_leftImg8bit.png',
                    f'{base_name}_left.png',
                    f'{base_name}.png',
                ]
                
                left_img_path = None
                for img_name in possible_image_names:
                    candidate = os.path.join(city_left_path, img_name)
                    if os.path.exists(candidate):
                        left_img_path = candidate
                        break
                
                if left_img_path is None:
                    continue
                
                items.append(DatasetItem(
                    item_id=base_name,
                    image_path=left_img_path,
                    gt_path=disp_path,
                    camera_id=None,
                    metadata={'city': city_dir}
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
        
        print(f"Found {len(items)} Cityscapes image pairs in {self.split} split")
        return items
    
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """Load Cityscapes disparity and convert to depth in meters."""
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Disparity file not found: {gt_path}")
        
        ext = os.path.splitext(gt_path)[1].lower()
        
        if ext == '.png':
            # Cityscapes disparity is stored as uint16 PNG
            disp_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if disp_img is None:
                raise ValueError(f"Could not read disparity image: {gt_path}")
            
            # Convert uint16 to float and divide by 256
            if disp_img.dtype == np.uint16:
                disparity = disp_img.astype(np.float32) / 256.0
            else:
                disparity = disp_img.astype(np.float32)
            
            # Zero values indicate invalid pixels
            disparity[disparity == 0] = np.nan
        else:
            raise ValueError(f"Unsupported disparity format: {ext}")
        
        # Convert disparity to depth
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid_mask = np.isfinite(disparity) & (disparity >= self.min_disparity)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            depth[valid_mask] = (self.baseline * self.focal_length) / disparity[valid_mask]
            depth[valid_mask] = np.clip(depth[valid_mask], 0.0, self.max_depth)
        
        depth[~valid_mask] = np.nan
        return depth
    
    def get_output_subdir(self) -> str:
        """Get output subdirectory name."""
        return 'cityscapes'

