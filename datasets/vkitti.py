"""
VKITTI dataset implementation.
"""

import os
import re
from typing import List, Optional
from glob import glob
import numpy as np
import cv2

from .base import BaseDataset, DatasetItem, DatasetConfig


class VKITTIDataset(BaseDataset):
    """
    VKITTI dataset for depth estimation evaluation.
    
    VKITTI provides depth maps directly (stored in cm, need to convert to meters).
    Dataset structure:
    - RGB: vkitti/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg
    - Depth: vkitti/vkitti_2.0.3_depth/Scene01/15-deg-left/frames/depth/Camera_0/depth_00000.png
    - Intrinsics: vkitti/vkitti_2.0.3_textgt/Scene01/15-deg-left/intrinsic.txt
    """
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.max_depth = 80.0  # meters (VKITTI max depth)
    
    def get_default_path(self) -> str:
        """Get default VKITTI dataset path."""
        import os
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'vkitti')
    
    def find_items(self) -> List[DatasetItem]:
        """Find all VKITTI image pairs."""
        items = []
        
        # Paths to different VKITTI components
        rgb_base = os.path.join(self.dataset_path, 'vkitti_2.0.3_rgb')
        depth_base = os.path.join(self.dataset_path, 'vkitti_2.0.3_depth')
        textgt_base = os.path.join(self.dataset_path, 'vkitti_2.0.3_textgt')
        
        # Check if directories exist
        if not os.path.exists(rgb_base):
            print(f"Error: VKITTI RGB directory not found: {rgb_base}")
            print(f"  Dataset path: {self.dataset_path}")
            raise ValueError(f"VKITTI RGB directory not found: {rgb_base}")
        
        if not os.path.exists(depth_base):
            print(f"Error: VKITTI depth directory not found: {depth_base}")
            print(f"  Dataset path: {self.dataset_path}")
            raise ValueError(f"VKITTI depth directory not found: {depth_base}")
        
        # Find all scene directories
        scene_dirs = sorted([d for d in os.listdir(rgb_base) 
                           if os.path.isdir(os.path.join(rgb_base, d))])
        
        for scene_name in scene_dirs:
            scene_rgb_path = os.path.join(rgb_base, scene_name)
            scene_depth_path = os.path.join(depth_base, scene_name)
            
            if not os.path.isdir(scene_rgb_path) or not os.path.isdir(scene_depth_path):
                continue
            
            # Find all variant directories (e.g., '15-deg-left', 'rain', etc.)
            variant_dirs = sorted([d for d in os.listdir(scene_rgb_path)
                                  if os.path.isdir(os.path.join(scene_rgb_path, d))])
            
            for variant_name in variant_dirs:
                variant_rgb_path = os.path.join(scene_rgb_path, variant_name, 'frames', 'rgb')
                variant_depth_path = os.path.join(scene_depth_path, variant_name, 'frames', 'depth')
                variant_textgt_path = os.path.join(textgt_base, scene_name, variant_name) if os.path.exists(textgt_base) else None
                
                if not os.path.exists(variant_rgb_path) or not os.path.exists(variant_depth_path):
                    continue
                
                # Find all camera directories
                camera_dirs = sorted([d for d in os.listdir(variant_rgb_path)
                                     if os.path.isdir(os.path.join(variant_rgb_path, d))])
                
                for camera_dir in camera_dirs:
                    camera_rgb_path = os.path.join(variant_rgb_path, camera_dir)
                    camera_depth_path = os.path.join(variant_depth_path, camera_dir)
                    
                    if not os.path.exists(camera_depth_path):
                        continue
                    
                    # Find all RGB images
                    rgb_files = sorted(glob(os.path.join(camera_rgb_path, 'rgb_*.jpg'))) + \
                               sorted(glob(os.path.join(camera_rgb_path, 'rgb_*.png')))
                    
                    for rgb_path in rgb_files:
                        # Extract frame number from filename (e.g., rgb_00000.jpg -> 00000)
                        base_name = os.path.basename(rgb_path)
                        frame_match = re.search(r'rgb_(\d+)\.(jpg|png)', base_name)
                        if not frame_match:
                            continue
                        
                        frame_num = frame_match.group(1)
                        
                        # Find corresponding depth file
                        depth_file = os.path.join(camera_depth_path, f'depth_{frame_num}.png')
                        
                        if not os.path.exists(depth_file):
                            continue
                        
                        # Create item ID: Scene_Variant_Camera_Frame
                        item_id = f"{scene_name}_{variant_name}_{camera_dir}_{frame_num}"
                        
                        # Load intrinsic matrix if available
                        metadata = {
                            'scene': scene_name,
                            'variant': variant_name,
                            'camera': camera_dir
                        }
                        
                        # Try to load intrinsic matrix
                        if variant_textgt_path and os.path.exists(variant_textgt_path):
                            intrinsic_file = os.path.join(variant_textgt_path, 'intrinsic.txt')
                            if os.path.exists(intrinsic_file):
                                try:
                                    intrinsic_matrix = self._load_intrinsic_matrix(intrinsic_file)
                                    if intrinsic_matrix is not None:
                                        metadata['intrinsic'] = intrinsic_matrix
                                except Exception as e:
                                    # If loading fails, continue without intrinsics
                                    pass
                        
                        items.append(DatasetItem(
                            item_id=item_id,
                            image_path=rgb_path,
                            gt_path=depth_file,
                            camera_id=camera_dir,
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
        
        if self.max_items is not None:
            items = items[:self.max_items]
        
        print(f"Found {len(items)} VKITTI image pairs")
        return items
    
    def _load_intrinsic_matrix(self, intrinsic_file: str) -> Optional[np.ndarray]:
        """
        Load intrinsic matrix from VKITTI intrinsic.txt file.
        
        Args:
            intrinsic_file: Path to intrinsic.txt file
            
        Returns:
            3x3 intrinsic matrix or None if loading fails
        """
        try:
            with open(intrinsic_file, 'r') as f:
                lines = f.readlines()
            
            # VKITTI intrinsic.txt format: typically contains camera matrix
            # Format may vary, try to parse common formats
            intrinsic = None
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try to parse matrix format
                if '[' in line or 'K' in line.upper():
                    # Extract numbers from line
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                    if len(numbers) >= 9:
                        # Assume 3x3 matrix (row-major)
                        values = [float(n) for n in numbers[:9]]
                        intrinsic = np.array(values).reshape(3, 3)
                        break
                    elif len(numbers) >= 4:
                        # Try to parse as fx, fy, cx, cy
                        fx, fy, cx, cy = [float(n) for n in numbers[:4]]
                        intrinsic = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])
                        break
            
            return intrinsic
        except Exception:
            return None
    
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """Load VKITTI depth map and convert from cm to meters."""
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Depth file not found: {gt_path}")
        
        # VKITTI depth is stored as uint16 PNG in centimeters
        depth_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            raise ValueError(f"Could not read depth image: {gt_path}")
        
        # Convert to float and divide by 100 to get depth in meters
        depth = depth_img.astype(np.float32) / 100.0
        
        # Zero values indicate invalid pixels
        depth[depth == 0] = np.nan
        
        # Clip to max depth
        depth[depth > self.max_depth] = np.nan
        
        return depth
    
    def get_output_subdir(self) -> str:
        """Get output subdirectory name."""
        return 'vkitti'


