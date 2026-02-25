"""
KITTI dataset implementation for depth estimation evaluation.

KITTI Depth Prediction Benchmark:
- Ground truth depth from LiDAR projected onto images
- Camera intrinsics stored in separate .txt files
- Depth stored as uint16 PNG (depth_in_meters = pixel_value / 256.0)

Dataset structure:
    kitti/
    ├── depth_selection/
    │   ├── test_depth_completion_anonymous/
    │   │   ├── image/
    │   │   ├── intrinsics/
    │   │   └── velodyne_raw/
    │   ├── test_depth_prediction_anonymous/
    │   │   ├── image/
    │   │   └── intrinsics/
    │   └── val_selection_cropped/
    │       ├── groundtruth_depth/
    │       ├── image/
    │       ├── intrinsics/
    │       └── velodyne_raw/
    ├── train/
    │   └── {date}_drive_{seq}_sync/
    │       └── proj_depth/
    │           ├── groundtruth/
    │           │   ├── image_02/
    │           │   └── image_03/
    │           └── velodyne_raw/
    │               ├── image_02/
    │               └── image_03/
    └── val/
        └── {date}_drive_{seq}_sync/
            └── proj_depth/
                ├── groundtruth/
                │   ├── image_02/
                │   └── image_03/
                └── velodyne_raw/
                    ├── image_02/
                    └── image_03/

Camera Intrinsics:
    KITTI uses calibrated cameras. Intrinsics are stored in:
    - For depth_selection splits: intrinsics/*.txt files
    - For train/val splits: Need to be read from raw KITTI calibration files
    
    Intrinsic file format (depth_selection):
        Single line with space-separated values: fx fy cx cy
        
    For train/val splits, the intrinsics depend on the camera (image_02 or image_03).
    Default KITTI stereo camera intrinsics for reference resolution (1242x375):
        fx = fy ≈ 721.5377 (varies slightly per sequence)
        cx ≈ 609.5593
        cy ≈ 172.854
"""

import os
import re
from typing import List, Optional
from glob import glob
import numpy as np
import cv2

from .base import BaseDataset, DatasetItem, DatasetConfig


class KITTIDataset(BaseDataset):
    """
    KITTI Depth Prediction/Completion dataset for depth estimation evaluation.
    
    Supports multiple splits:
    - 'val_selection': Validation set from depth_selection/val_selection_cropped/
    - 'test_depth_prediction': Test set for depth prediction
    - 'test_depth_completion': Test set for depth completion
    - 'train': Training sequences from train/ folder
    - 'val': Validation sequences from val/ folder
    
    Camera Intrinsics:
        KITTI uses calibrated stereo cameras with known intrinsic parameters.
        For the reference resolution (approximately 1242x375):
        
        Default intrinsics (approximate, varies per sequence):
            fx = fy ≈ 721.5377 pixels
            cx ≈ 609.5593 pixels
            cy ≈ 172.854 pixels
        
        The intrinsic matrix K is:
            K = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
    """
    
    # Default KITTI intrinsic parameters (reference resolution ~1242x375)
    # These are approximate values; per-image intrinsics are loaded when available
    DEFAULT_FX = 721.5377
    DEFAULT_FY = 721.5377
    DEFAULT_CX = 609.5593
    DEFAULT_CY = 172.854
    DEFAULT_WIDTH = 1242
    DEFAULT_HEIGHT = 375
    
    # Maximum depth for KITTI outdoor scenes
    MAX_DEPTH = 80.0  # meters
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.max_depth = self.MAX_DEPTH
        # Cache for intrinsics files to avoid re-reading
        self._intrinsics_cache = {}
    
    def get_default_path(self) -> str:
        """Get default KITTI dataset path."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'kitti')
    
    def find_items(self) -> List[DatasetItem]:
        """Find all KITTI image pairs based on the split."""
        items = []
        
        split = self.split.lower()
        
        if split in ['val_selection', 'val_selection_cropped']:
            items = self._find_val_selection_items()
        elif split == 'test_depth_prediction':
            items = self._find_test_depth_prediction_items()
        elif split == 'test_depth_completion':
            items = self._find_test_depth_completion_items()
        elif split in ['train', 'val']:
            items = self._find_sequence_items(split)
        else:
            # Default to val_selection if available, otherwise val sequences
            val_selection_path = os.path.join(self.dataset_path, 'depth_selection', 'val_selection_cropped')
            if os.path.exists(val_selection_path):
                items = self._find_val_selection_items()
            else:
                items = self._find_sequence_items('val')
        
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
        
        print(f"Found {len(items)} KITTI image pairs (split: {split})")
        return items
    
    def _find_val_selection_items(self) -> List[DatasetItem]:
        """Find items from depth_selection/val_selection_cropped/."""
        items = []
        
        base_path = os.path.join(self.dataset_path, 'depth_selection', 'val_selection_cropped')
        image_dir = os.path.join(base_path, 'image')
        depth_dir = os.path.join(base_path, 'groundtruth_depth')
        intrinsics_dir = os.path.join(base_path, 'intrinsics')
        
        if not os.path.exists(image_dir):
            print(f"Warning: KITTI val_selection image directory not found: {image_dir}")
            return items
        
        # Find all image files
        image_files = sorted(glob(os.path.join(image_dir, '*.png')))
        
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Find corresponding depth file
            depth_path = os.path.join(depth_dir, base_name + '.png')
            
            if not os.path.exists(depth_path):
                continue
            
            # Find intrinsics file
            intrinsics_path = os.path.join(intrinsics_dir, base_name + '.txt')
            
            metadata = {
                'split': 'val_selection',
                'has_intrinsics': os.path.exists(intrinsics_path)
            }
            
            if os.path.exists(intrinsics_path):
                metadata['intrinsics_path'] = intrinsics_path
            
            items.append(DatasetItem(
                item_id=base_name,
                image_path=img_path,
                gt_path=depth_path,
                camera_id=None,
                metadata=metadata
            ))
        
        return items
    
    def _find_test_depth_prediction_items(self) -> List[DatasetItem]:
        """Find items from depth_selection/test_depth_prediction_anonymous/."""
        items = []
        
        base_path = os.path.join(self.dataset_path, 'depth_selection', 'test_depth_prediction_anonymous')
        image_dir = os.path.join(base_path, 'image')
        intrinsics_dir = os.path.join(base_path, 'intrinsics')
        
        if not os.path.exists(image_dir):
            print(f"Warning: KITTI test_depth_prediction directory not found: {image_dir}")
            return items
        
        # Find all image files (no ground truth for test set)
        image_files = sorted(glob(os.path.join(image_dir, '*.png')))
        
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # No ground truth for test set
            intrinsics_path = os.path.join(intrinsics_dir, base_name + '.txt')
            
            metadata = {
                'split': 'test_depth_prediction',
                'has_intrinsics': os.path.exists(intrinsics_path),
                'is_test': True  # No ground truth
            }
            
            if os.path.exists(intrinsics_path):
                metadata['intrinsics_path'] = intrinsics_path
            
            items.append(DatasetItem(
                item_id=base_name,
                image_path=img_path,
                gt_path=None,  # No ground truth
                camera_id=None,
                metadata=metadata
            ))
        
        return items
    
    def _find_test_depth_completion_items(self) -> List[DatasetItem]:
        """Find items from depth_selection/test_depth_completion_anonymous/."""
        items = []
        
        base_path = os.path.join(self.dataset_path, 'depth_selection', 'test_depth_completion_anonymous')
        image_dir = os.path.join(base_path, 'image')
        intrinsics_dir = os.path.join(base_path, 'intrinsics')
        velodyne_dir = os.path.join(base_path, 'velodyne_raw')
        
        if not os.path.exists(image_dir):
            print(f"Warning: KITTI test_depth_completion directory not found: {image_dir}")
            return items
        
        # Find all image files
        image_files = sorted(glob(os.path.join(image_dir, '*.png')))
        
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # No ground truth for test set, but velodyne_raw is available
            intrinsics_path = os.path.join(intrinsics_dir, base_name + '.txt')
            velodyne_path = os.path.join(velodyne_dir, base_name + '.png')
            
            metadata = {
                'split': 'test_depth_completion',
                'has_intrinsics': os.path.exists(intrinsics_path),
                'has_velodyne': os.path.exists(velodyne_path),
                'is_test': True  # No ground truth
            }
            
            if os.path.exists(intrinsics_path):
                metadata['intrinsics_path'] = intrinsics_path
            if os.path.exists(velodyne_path):
                metadata['velodyne_path'] = velodyne_path
            
            items.append(DatasetItem(
                item_id=base_name,
                image_path=img_path,
                gt_path=None,  # No ground truth
                camera_id=None,
                metadata=metadata
            ))
        
        return items
    
    def _find_sequence_items(self, split: str) -> List[DatasetItem]:
        """Find items from train/ or val/ sequence folders."""
        items = []
        
        split_path = os.path.join(self.dataset_path, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: KITTI {split} directory not found: {split_path}")
            return items
        
        # Find all sequence directories
        sequence_dirs = sorted([d for d in os.listdir(split_path) 
                               if os.path.isdir(os.path.join(split_path, d))
                               and '_drive_' in d])
        
        for seq_name in sequence_dirs:
            seq_path = os.path.join(split_path, seq_name)
            proj_depth_path = os.path.join(seq_path, 'proj_depth')
            
            if not os.path.exists(proj_depth_path):
                continue
            
            # Look for groundtruth depth in image_02 and image_03
            for camera in ['image_02', 'image_03']:
                gt_dir = os.path.join(proj_depth_path, 'groundtruth', camera)
                
                if not os.path.exists(gt_dir):
                    continue
                
                # Find all depth files
                depth_files = sorted(glob(os.path.join(gt_dir, '*.png')))
                
                for depth_path in depth_files:
                    frame_name = os.path.splitext(os.path.basename(depth_path))[0]
                    
                    # Construct image path
                    # Images are typically in KITTI raw data structure:
                    # data/{date}/{date}_drive_{seq}_sync/image_{02,03}/data/{frame}.png
                    # But for depth benchmark, images might be provided separately
                    
                    # Try to find corresponding image in common locations
                    img_path = self._find_image_for_depth(seq_name, camera, frame_name)
                    
                    if img_path is None or not os.path.exists(img_path):
                        # Skip if no image found
                        continue
                    
                    # Create item ID: sequence_camera_frame
                    item_id = f"{seq_name}_{camera}_{frame_name}"
                    
                    metadata = {
                        'split': split,
                        'sequence': seq_name,
                        'camera': camera,
                        'frame': frame_name
                    }
                    
                    items.append(DatasetItem(
                        item_id=item_id,
                        image_path=img_path,
                        gt_path=depth_path,
                        camera_id=camera,
                        metadata=metadata
                    ))
        
        return items
    
    def _find_image_for_depth(self, seq_name: str, camera: str, frame_name: str) -> Optional[str]:
        """
        Find the corresponding RGB image for a depth map.
        
        KITTI images might be stored in different locations depending on how
        the dataset was organized. This method checks common locations.
        """
        # Common image locations to check
        possible_paths = []
        
        # Extract date from sequence name (e.g., 2011_09_26 from 2011_09_26_drive_0001_sync)
        date_match = re.match(r'(\d{4}_\d{2}_\d{2})_drive_\d+_sync', seq_name)
        if date_match:
            date = date_match.group(1)
            
            # Check in raw KITTI structure
            # kitti_raw/{date}/{seq_name}/{camera}/data/{frame}.png
            raw_path = os.path.join(self.dataset_path, 'kitti_raw', date, seq_name, camera, 'data', f'{frame_name}.png')
            possible_paths.append(raw_path)
            
            # Also check jpg extension
            raw_path_jpg = os.path.join(self.dataset_path, 'kitti_raw', date, seq_name, camera, 'data', f'{frame_name}.jpg')
            possible_paths.append(raw_path_jpg)
        
        # Check in same split directory structure
        split = self.split if self.split in ['train', 'val'] else 'val'
        split_img_path = os.path.join(self.dataset_path, split, seq_name, camera, 'data', f'{frame_name}.png')
        possible_paths.append(split_img_path)
        
        # Check for images folder in the sequence
        seq_img_path = os.path.join(self.dataset_path, split, seq_name, 'image', camera, f'{frame_name}.png')
        possible_paths.append(seq_img_path)
        
        # Check for flat structure
        flat_path = os.path.join(self.dataset_path, split, 'images', f'{seq_name}_{camera}_{frame_name}.png')
        possible_paths.append(flat_path)
        
        # Return first existing path
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_intrinsics_from_file(self, intrinsics_path: str) -> Optional[np.ndarray]:
        """
        Load camera intrinsics from a KITTI intrinsics file.
        
        KITTI depth_selection intrinsics files contain a single line with:
            fx fy cx cy
        
        Args:
            intrinsics_path: Path to the intrinsics .txt file
            
        Returns:
            3x3 intrinsic matrix or None if loading fails
        """
        if intrinsics_path in self._intrinsics_cache:
            return self._intrinsics_cache[intrinsics_path]
        
        try:
            with open(intrinsics_path, 'r') as f:
                line = f.readline().strip()
            
            # Parse the intrinsics values
            values = line.split()
            
            if len(values) >= 4:
                # Format: fx fy cx cy
                fx = float(values[0])
                fy = float(values[1])
                cx = float(values[2])
                cy = float(values[3])
                
                intrinsic = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                
                self._intrinsics_cache[intrinsics_path] = intrinsic
                return intrinsic
            elif len(values) >= 9:
                # Full 3x3 matrix in row-major order
                matrix_values = [float(v) for v in values[:9]]
                intrinsic = np.array(matrix_values).reshape(3, 3).astype(np.float32)
                self._intrinsics_cache[intrinsics_path] = intrinsic
                return intrinsic
            elif len(values) >= 12:
                # 3x4 projection matrix P, extract K
                # P = K[R|t], for rectified images R=I, so first 3 columns = K
                matrix_values = [float(v) for v in values[:12]]
                P = np.array(matrix_values).reshape(3, 4)
                # Extract fx, fy, cx, cy from projection matrix
                fx = P[0, 0]
                fy = P[1, 1]
                cx = P[0, 2]
                cy = P[1, 2]
                
                intrinsic = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                
                self._intrinsics_cache[intrinsics_path] = intrinsic
                return intrinsic
                
        except Exception as e:
            print(f"Warning: Failed to load intrinsics from {intrinsics_path}: {e}")
        
        return None
    
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """
        Load KITTI ground truth depth map.
        
        KITTI depth maps are stored as uint16 PNG images.
        The depth in meters is: depth_meters = pixel_value / 256.0
        Invalid pixels have value 0.
        
        Args:
            gt_path: Path to ground truth depth file
            item: DatasetItem with metadata
            
        Returns:
            Ground truth depth map in meters (invalid pixels as NaN)
        """
        if gt_path is None:
            raise ValueError("No ground truth path available (test set)")
        
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Depth file not found: {gt_path}")
        
        # Load depth as uint16
        depth_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise ValueError(f"Could not read depth image: {gt_path}")
        if len(depth_img.shape) > 2:
            depth_img = depth_img[:, :, 0]
        
        # Convert to float and divide by 256 to get depth in meters
        depth = depth_img.astype(np.float32) / 256.0
        
        # Zero values indicate invalid pixels
        depth[depth_img == 0] = np.nan
        
        # Clip to max depth
        depth[depth > self.max_depth] = np.nan
        
        return depth
    
    def get_output_subdir(self) -> str:
        """Get output subdirectory name."""
        return 'kitti'
    
    def load_intrinsic(self, item: DatasetItem) -> Optional[np.ndarray]:
        """
        Load camera intrinsic matrix for a KITTI image.
        
        KITTI stores intrinsics in .txt files for depth_selection splits.
        For train/val sequences, we use default values or try to find
        calibration files.
        
        Args:
            item: DatasetItem containing the image path and metadata
            
        Returns:
            3x3 numpy array with camera intrinsic matrix K:
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
            Returns None if intrinsics cannot be found.
        """
        # Check if intrinsics path is in metadata
        if item.metadata and 'intrinsics_path' in item.metadata:
            intrinsics_path = item.metadata['intrinsics_path']
            intrinsic = self._load_intrinsics_from_file(intrinsics_path)
            if intrinsic is not None:
                return intrinsic
        
        # Try to find intrinsics file based on item path
        if item.metadata and item.metadata.get('split') in ['val_selection', 'test_depth_prediction', 'test_depth_completion']:
            # For depth_selection splits, intrinsics are in intrinsics/ folder
            base_name = item.item_id
            split = item.metadata['split']
            
            if split == 'val_selection':
                split_folder = 'val_selection_cropped'
            elif split == 'test_depth_prediction':
                split_folder = 'test_depth_prediction_anonymous'
            else:
                split_folder = 'test_depth_completion_anonymous'
            
            intrinsics_path = os.path.join(
                self.dataset_path, 'depth_selection', split_folder, 'intrinsics', f'{base_name}.txt'
            )
            
            if os.path.exists(intrinsics_path):
                intrinsic = self._load_intrinsics_from_file(intrinsics_path)
                if intrinsic is not None:
                    return intrinsic
        
        # For train/val sequences, try to load from calibration file
        if item.metadata and 'sequence' in item.metadata:
            seq_name = item.metadata['sequence']
            camera = item.metadata.get('camera', 'image_02')
            
            # Extract date from sequence name
            date_match = re.match(r'(\d{4}_\d{2}_\d{2})_drive_\d+_sync', seq_name)
            if date_match:
                date = date_match.group(1)
                
                # Look for calib_cam_to_cam.txt in various locations
                possible_calib_paths = [
                    os.path.join(self.dataset_path, 'kitti_raw', date, 'calib_cam_to_cam.txt'),
                    os.path.join(self.dataset_path, 'calib', date, 'calib_cam_to_cam.txt'),
                    os.path.join(self.dataset_path, date, 'calib_cam_to_cam.txt'),
                ]
                
                for calib_path in possible_calib_paths:
                    if os.path.exists(calib_path):
                        intrinsic = self._load_intrinsics_from_calib(calib_path, camera)
                        if intrinsic is not None:
                            return intrinsic
        
        # Fall back to default intrinsics scaled to image size
        return self.get_default_intrinsic(item)
    
    def _load_intrinsics_from_calib(self, calib_path: str, camera: str) -> Optional[np.ndarray]:
        """
        Load intrinsics from KITTI calib_cam_to_cam.txt file.
        
        The file contains projection matrices P_rect_XX for each camera.
        For image_02 and image_03, we need P_rect_02 and P_rect_03.
        
        Args:
            calib_path: Path to calib_cam_to_cam.txt
            camera: Camera identifier ('image_02' or 'image_03')
            
        Returns:
            3x3 intrinsic matrix or None
        """
        cache_key = f"{calib_path}_{camera}"
        if cache_key in self._intrinsics_cache:
            return self._intrinsics_cache[cache_key]
        
        try:
            # Determine which projection matrix to use
            if camera == 'image_02':
                p_key = 'P_rect_02'
            elif camera == 'image_03':
                p_key = 'P_rect_03'
            else:
                p_key = 'P_rect_02'  # Default
            
            with open(calib_path, 'r') as f:
                for line in f:
                    if line.startswith(p_key):
                        # Parse projection matrix
                        values = line.split(':')[1].strip().split()
                        P = np.array([float(v) for v in values]).reshape(3, 4)
                        
                        # Extract intrinsic parameters
                        fx = P[0, 0]
                        fy = P[1, 1]
                        cx = P[0, 2]
                        cy = P[1, 2]
                        
                        intrinsic = np.array([
                            [fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float32)
                        
                        self._intrinsics_cache[cache_key] = intrinsic
                        return intrinsic
                        
        except Exception as e:
            print(f"Warning: Failed to load calibration from {calib_path}: {e}")
        
        return None
    
    def get_default_intrinsic(self, item: Optional[DatasetItem] = None) -> np.ndarray:
        """
        Get default KITTI intrinsic matrix, scaled to image size if available.
        
        Args:
            item: Optional DatasetItem to get image dimensions
            
        Returns:
            3x3 numpy array with camera intrinsic matrix
        """
        # Try to get actual image dimensions
        if item is not None and os.path.exists(item.image_path):
            try:
                image = cv2.imread(item.image_path)
                if image is not None:
                    img_height, img_width = image.shape[:2]
                    
                    # Scale intrinsics from default resolution
                    scale_x = img_width / self.DEFAULT_WIDTH
                    scale_y = img_height / self.DEFAULT_HEIGHT
                    
                    intrinsic = np.array([
                        [self.DEFAULT_FX * scale_x, 0.0, self.DEFAULT_CX * scale_x],
                        [0.0, self.DEFAULT_FY * scale_y, self.DEFAULT_CY * scale_y],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                    
                    return intrinsic
            except Exception:
                pass
        
        # Return default intrinsics
        intrinsic = np.array([
            [self.DEFAULT_FX, 0.0, self.DEFAULT_CX],
            [0.0, self.DEFAULT_FY, self.DEFAULT_CY],
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
        # Scale from default resolution
        scale_x = width / self.DEFAULT_WIDTH
        scale_y = height / self.DEFAULT_HEIGHT
        
        fx = self.DEFAULT_FX * scale_x
        fy = self.DEFAULT_FY * scale_y
        cx = self.DEFAULT_CX * scale_x
        cy = self.DEFAULT_CY * scale_y
        
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return intrinsic
    
    def has_intrinsics(self) -> bool:
        """KITTI provides camera intrinsic parameters."""
        return True
    
    def supports_multiple_cameras(self) -> bool:
        """
        KITTI has multiple cameras (image_02, image_03) for stereo pairs.
        
        For depth_selection splits, items are single camera.
        For train/val sequences, we have both cameras.
        """
        # Return False for now - we treat each camera as separate items
        return False


