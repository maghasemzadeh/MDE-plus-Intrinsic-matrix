"""
PyTorch Dataset classes for training depth estimation models.
These datasets read from split files in datasets/raw_data/ and use transforms from metric_depth.
"""

import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

# Import transforms from metric_depth
import sys
_metric_depth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 
                                   'models', 'raw_models', 'DepthAnythingV2-revised', 'metric_depth')
if _metric_depth_path not in sys.path:
    sys.path.insert(0, _metric_depth_path)

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class VKITTI2TrainingDataset(Dataset):
    """
    VKITTI2 dataset for depth estimation training.
    
    Reads from split files in datasets/raw_data/vkitti/splits/
    File list format (space-separated):
        image_path depth_path [intrinsics_path]
    
    If intrinsics_path is provided, it should be a .npy file containing a 3x3 numpy array.
    """
    
    def __init__(self, filelist_path, mode='train', size=(518, 518)):
        """
        Initialize VKITTI2 training dataset.
        
        Args:
            filelist_path: Path to file list (e.g., datasets/raw_data/vkitti/splits/train.txt)
            mode: 'train' or 'val'
            size: (width, height) for image resizing
        """
        self.mode = mode
        self.size = size
        
        # Get project root
        current_file = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file))
        
        # Read file list
        if not os.path.isabs(filelist_path):
            filelist_path = os.path.join(self.project_root, filelist_path)
        
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"File list not found: {filelist_path}")
        
        with open(filelist_path, 'r') as f:
            raw_filelist = f.read().splitlines()
        
        # Filter out empty lines and invalid entries
        # Don't check file existence during init - handle at runtime for better performance
        self.filelist = []
        for line in raw_filelist:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # Keep the line - we'll resolve and check paths at runtime in __getitem__
            self.filelist.append(line)
        
        if len(self.filelist) == 0:
            raise ValueError(f"No valid entries found in filelist: {filelist_path}. "
                           f"File should contain lines with at least 2 space-separated paths.")
        
        print(f"Loaded {len(self.filelist)} entries from {filelist_path}")
        
        # Setup transforms
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def _resolve_path(self, path):
        """
        Resolve a path that might be absolute from another system or relative.
        Returns the resolved path if it exists, otherwise returns the original path.
        """
        # Normalize the path first (handles things like .., ., etc.)
        path = os.path.normpath(path)
        
        # If path exists as-is, use it
        if os.path.exists(path):
            return path
        
        # If it's already absolute but doesn't exist, try to resolve symlinks
        if os.path.isabs(path):
            try:
                resolved = os.path.realpath(path)
                if os.path.exists(resolved):
                    return resolved
            except Exception:
                pass
        
        # If relative path, try relative to project root
        if not os.path.isabs(path):
            resolved = os.path.join(self.project_root, path)
            resolved = os.path.normpath(resolved)
            if os.path.exists(resolved):
                return resolved
        
        # Return original path (will fail with clear error if file doesn't exist)
        return path
    
    def __getitem__(self, item):
        # Split by space, but handle paths with spaces correctly
        # The format is: img_path depth_path [intrinsics_path]
        # Paths may contain spaces, so we need to reconstruct them
        line_str = self.filelist[item].strip()
        
        # Find paths by looking for patterns: they start with / and end with file extensions
        # We know there are 2-3 paths separated by spaces
        parts = []
        current_path = []
        
        for token in line_str.split():
            if token.startswith('/'):
                # Start of a new path
                if current_path:
                    parts.append(' '.join(current_path))
                current_path = [token]
            else:
                # Continuation of current path
                current_path.append(token)
        
        # Add the last path
        if current_path:
            parts.append(' '.join(current_path))
        
        if len(parts) < 2:
            raise ValueError(f"Invalid file list line: {line_str}. Expected at least 2 paths (image, depth). Got {len(parts)} parts.")
        
        img_path = parts[0]
        depth_path = parts[1]
        intrinsics_path = parts[2] if len(parts) > 2 else None
        
        # Resolve and normalize paths
        img_path = self._resolve_path(img_path)
        depth_path = self._resolve_path(depth_path)
        if intrinsics_path:
            intrinsics_path = self._resolve_path(intrinsics_path)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}. File may not exist or be corrupted.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Load depth (VKITTI depth is in cm, convert to meters)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Could not load depth: {depth_path}")
        depth = depth / 100.0  # cm to m
        
        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})
        
        # Convert to tensors
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        
        # Create valid mask
        sample['valid_mask'] = (sample['depth'] <= 80) & (sample['depth'] > 0)
        
        # Load intrinsics if provided
        if intrinsics_path and os.path.exists(intrinsics_path):
            try:
                if intrinsics_path.endswith('.npy'):
                    intrinsics = np.load(intrinsics_path)
                    if intrinsics is not None and intrinsics.shape == (3, 3):
                        sample['intrinsics'] = torch.from_numpy(intrinsics).float()
                else:
                    # Try to load from text file (intrinsic.txt)
                    with open(intrinsics_path, 'r') as f:
                        lines = f.readlines()
                    intrinsic = None
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        # Extract numbers from line
                        numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                        if len(numbers) >= 9:
                            values = [float(n) for n in numbers[:9]]
                            intrinsic = np.array(values).reshape(3, 3)
                            break
                        elif len(numbers) >= 4:
                            fx, fy, cx, cy = [float(n) for n in numbers[:4]]
                            intrinsic = np.array([
                                [fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]
                            ])
                            break
                    if intrinsic is not None and intrinsic.shape == (3, 3):
                        sample['intrinsics'] = torch.from_numpy(intrinsic).float()
            except Exception:
                # If loading fails, don't include intrinsics (will use None in training)
                pass
        
        sample['image_path'] = img_path
        
        return sample
    
    def __len__(self):
        return len(self.filelist)


class KITTITrainingDataset(Dataset):
    """
    KITTI dataset for depth estimation validation.
    
    Reads from split files in datasets/raw_data/ or metric_depth/dataset/splits/
    File list format (space-separated):
        image_path depth_path
    """
    
    def __init__(self, filelist_path, mode='val', size=(518, 518)):
        """
        Initialize KITTI training dataset.
        
        Args:
            filelist_path: Path to file list
            mode: 'val' (only validation supported)
            size: (width, height) for image resizing
        """
        if mode != 'val':
            raise NotImplementedError(f"KITTI dataset only supports 'val' mode, got '{mode}'")
        
        self.mode = mode
        self.size = size
        
        # Get project root
        current_file = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file))
        
        # Read file list
        if not os.path.isabs(filelist_path):
            # Try relative to project root first
            test_path = os.path.join(self.project_root, filelist_path)
            if not os.path.exists(test_path):
                # Try relative to metric_depth
                metric_depth_path = os.path.join(self.project_root, 'models', 'raw_models', 
                                                 'DepthAnythingV2-revised', 'metric_depth')
                test_path = os.path.join(metric_depth_path, filelist_path)
            filelist_path = test_path
        
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"File list not found: {filelist_path}")
        
        with open(filelist_path, 'r') as f:
            raw_filelist = f.read().splitlines()
        
        # Filter out empty lines and invalid entries
        # Don't check file existence during init - handle at runtime for better performance
        self.filelist = []
        for line in raw_filelist:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # Keep the line - we'll resolve and check paths at runtime in __getitem__
            self.filelist.append(line)
        
        if len(self.filelist) == 0:
            raise ValueError(f"No valid entries found in filelist: {filelist_path}. "
                           f"File should contain lines with at least 2 space-separated paths.")
        
        print(f"Loaded {len(self.filelist)} entries from {filelist_path}")
        
        # Setup transforms
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def _resolve_path(self, path):
        """
        Resolve a path that might be absolute from another system or relative.
        Returns the resolved path if it exists, otherwise returns the original path.
        """
        # Normalize the path first (handles things like .., ., etc.)
        path = os.path.normpath(path)
        
        # If path exists as-is, use it
        if os.path.exists(path):
            return path
        
        # If it's already absolute but doesn't exist, try to resolve symlinks
        if os.path.isabs(path):
            try:
                resolved = os.path.realpath(path)
                if os.path.exists(resolved):
                    return resolved
            except Exception:
                pass
        
        # If relative path, try relative to project root
        if not os.path.isabs(path):
            resolved = os.path.join(self.project_root, path)
            resolved = os.path.normpath(resolved)
            if os.path.exists(resolved):
                return resolved
        
        # Return original path (will fail with clear error if file doesn't exist)
        return path
    
    def __getitem__(self, item):
        parts = self.filelist[item].split()
        img_path = parts[0]
        depth_path = parts[1]
        
        # Resolve and normalize paths
        img_path = self._resolve_path(img_path)
        depth_path = self._resolve_path(depth_path)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}. File may not exist or be corrupted.")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Load depth (KITTI depth is stored as uint16, divide by 256 to get meters)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth: {depth_path}. File may not exist or be corrupted.")
        depth = depth.astype('float32')
        
        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})
        
        # Convert to tensors
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['depth'] = sample['depth'] / 256.0  # convert to meters
        
        # Create valid mask
        sample['valid_mask'] = sample['depth'] > 0
        
        sample['image_path'] = img_path
        
        return sample
    
    def __len__(self):
        return len(self.filelist)

