"""
PyTorch Dataset classes for training depth estimation models.
These datasets read from split files in datasets/raw_data/ and use transforms from metric_depth.
"""

import os
import random
import re
import shlex
import warnings
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


def _read_pfm(filepath):
    """Read a .pfm disparity file."""
    with open(filepath, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        dims = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dims.split())
        scale = float(f.readline().decode('utf-8').rstrip())
        data = np.fromfile(f, '<f' if scale < 0 else '>f')
        data = np.reshape(data, (height, width))
        data = np.flipud(data)
        return data


def _depth_from_disparity(disparity, f, baseline, doffs):
    """Convert disparity map to depth in meters."""
    depth = np.zeros_like(disparity, dtype=np.float32)
    mask = np.isfinite(disparity) & ((disparity + doffs) != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth[mask] = (baseline * f) / (disparity[mask] + doffs)
    depth[~np.isfinite(depth)] = 0.0
    return depth / 1000.0  # mm to meters


def _parse_middlebury_calib(calib_file):
    """Parse Middlebury calib.txt to get f0,f1,cx0,cy0,cx1,cy1,doffs,baseline."""
    calib = {}
    current_key = None
    current_vals = []

    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '=' in line:
                if current_key is not None:
                    calib[current_key] = current_vals
                    current_vals = []
                key, val = line.split('=', 1)
                key = key.strip()
                if '[' in val:
                    current_key = key
                    val_clean = val.split('[', 1)[1].replace(']', ' ').replace(';', ' ')
                    vals = val_clean.split()
                    current_vals.extend([float(v) for v in vals if v])
                    if ']' in val:
                        calib[current_key] = current_vals
                        current_key = None
                        current_vals = []
                else:
                    val_clean = val.replace('[', ' ').replace(']', ' ').replace(';', ' ')
                    vals = list(map(float, val_clean.split()))
                    calib[key] = vals
                    current_key = None
            else:
                if current_key is not None:
                    val_clean = line.replace(']', ' ').replace(';', ' ')
                    vals = val_clean.split()
                    current_vals.extend([float(v) for v in vals if v])
                    if ']' in line:
                        calib[current_key] = current_vals
                        current_key = None
                        current_vals = []
    if current_key is not None:
        calib[current_key] = current_vals

    cam0 = calib.get('cam0')
    cam1 = calib.get('cam1')
    if cam0 is None or cam1 is None or len(cam0) < 6 or len(cam1) < 6:
        raise ValueError(f"Invalid calib file: cam0/cam1 missing or insufficient values")

    f0, cx0, cy0 = cam0[0], cam0[2], cam0[5] if len(cam0) > 5 else cam0[4]
    f1, cx1, cy1 = cam1[0], cam1[2], cam1[5] if len(cam1) > 5 else cam1[4]
    doffs = calib['doffs'][0] if 'doffs' in calib else (cx1 - cx0)
    baseline = calib['baseline'][0] if 'baseline' in calib else 1.0

    return f0, f1, cx0, cy0, cx1, cy1, doffs, baseline


class RobustDataset(Dataset):
    """
    Wrapper that catches load errors in __getitem__ and returns a fallback sample instead of crashing.
    Useful when some files in the dataset are corrupted (e.g., libpng CRC errors on PNG files).
    """

    def __init__(self, dataset, max_retries=10):
        """
        Args:
            dataset: The underlying dataset to wrap.
            max_retries: Maximum number of random fallback attempts when the requested index fails.
        """
        self.dataset = dataset
        self.max_retries = max_retries

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except (ValueError, OSError, RuntimeError) as e:
            warnings.warn(f"Skipping corrupted sample at index {index}: {e}", UserWarning)
            # Try random fallback indices
            for _ in range(self.max_retries - 1):
                fallback_idx = random.randint(0, len(self.dataset) - 1)
                try:
                    return self.dataset[fallback_idx]
                except (ValueError, OSError, RuntimeError):
                    continue
            # Re-raise if all retries failed
            raise


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
        # Parse paths and validate file existence after path resolution
        self.filelist = []
        skipped_count = 0
        
        for line in raw_filelist:
            line = line.strip()
            if not line:
                continue
            
            # Parse paths (handle paths with spaces)
            parts = []
            current_path = []
            
            for token in line.split():
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
                skipped_count += 1
                continue
            
            # Resolve paths and check if they exist
            img_path = self._resolve_path(parts[0])
            depth_path = self._resolve_path(parts[1])
            
            # Only keep entries where both files exist after path resolution
            if os.path.exists(img_path) and os.path.exists(depth_path):
                self.filelist.append(line)
            else:
                skipped_count += 1
        
        if len(self.filelist) == 0:
            raise ValueError(f"No valid entries found in filelist: {filelist_path}. "
                           f"Original filelist had {len(raw_filelist)} entries, but none resolved to existing files. "
                           f"Please check that the paths in the filelist are correct.")
        
        if skipped_count > 0:
            print(f"Warning: Skipped {skipped_count} entries with missing files (after path resolution)")
        print(f"Loaded {len(self.filelist)} valid entries from {filelist_path}")
        
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
        Maps paths from other systems (e.g., /mnt/bn/liheyang/DepthDatasets/vKitti2/...)
        to local project structure (datasets/raw_data/vkitti/...).
        Returns the resolved path if it exists, otherwise returns the original path.
        """
        # Normalize the path first (handles things like .., ., etc.)
        path = os.path.normpath(path)
        
        # If path exists as-is, use it
        if os.path.exists(path):
            return path
        
        # If it's an absolute path from another system, try to map it to local structure
        if os.path.isabs(path):
            path_lower = path.lower()
            
            # Check if this is a VKITTI path from another system
            if 'vkitti' in path_lower or 'vkitti2' in path_lower:
                # Extract the relative part after vKitti2 or vkitti
                parts = [p for p in path.split('/') if p]
                
                # Find the index of vKitti2 or vkitti
                vkitti_idx = None
                for i, part in enumerate(parts):
                    if 'vkitti' in part.lower():
                        vkitti_idx = i
                        break
                
                if vkitti_idx is not None:
                    # Get the relative path after vKitti2/vkitti
                    relative_parts = parts[vkitti_idx + 1:]
                    
                    # Determine the base directory based on path content
                    if 'rgb' in path_lower or path_lower.endswith(('.jpg', '.jpeg', '.png')):
                        base_name = 'vkitti_2.0.3_rgb'
                    elif 'depth' in path_lower:
                        base_name = 'vkitti_2.0.3_depth'
                    elif 'intrinsics' in path_lower or path_lower.endswith('.npy'):
                        # Intrinsics are in splits/intrinsics/
                        if len(relative_parts) > 0:
                            local_path = os.path.join(self.project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'intrinsics', *relative_parts)
                            if os.path.exists(local_path):
                                return local_path
                        filename = os.path.basename(path)
                        local_path = os.path.join(self.project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'intrinsics', filename)
                        if os.path.exists(local_path):
                            return local_path
                        return path
                    else:
                        base_name = 'vkitti_2.0.3_textgt'
                    
                    # Construct local path: datasets/raw_data/vkitti/{base_name}/{relative_parts}
                    if len(relative_parts) > 0:
                        local_path = os.path.join(self.project_root, 'datasets', 'raw_data', 'vkitti', base_name, *relative_parts)
                        local_path = os.path.normpath(local_path)
                        
                        if os.path.exists(local_path):
                            return local_path
                        
                        # If exact path doesn't exist, try to find the file by searching
                        # The variant name might be different (e.g., sunset vs morning)
                        # or the frame number might be different
                        if len(relative_parts) >= 3:
                            scene_name = relative_parts[0]
                            variant_name = relative_parts[1]
                            filename = os.path.basename(path)
                            
                            # Try to find the file in the scene directory
                            scene_dir = os.path.join(self.project_root, 'datasets', 'raw_data', 'vkitti', base_name, scene_name)
                            if os.path.exists(scene_dir):
                                # First, try exact filename in all variants
                                for variant_dir in os.listdir(scene_dir):
                                    variant_full_path = os.path.join(scene_dir, variant_dir)
                                    if not os.path.isdir(variant_full_path):
                                        continue
                                    
                                    # Try to find the file
                                    if 'rgb' in path_lower:
                                        file_path = os.path.join(variant_full_path, 'frames', 'rgb', 'Camera_0', filename)
                                    elif 'depth' in path_lower:
                                        file_path = os.path.join(variant_full_path, 'frames', 'depth', 'Camera_0', filename)
                                    else:
                                        continue
                                    
                                    if os.path.exists(file_path):
                                        return file_path
                                
                                # If not found, try to find any file with similar name pattern
                                # Extract frame number from filename (e.g., rgb_00089.jpg -> 00089)
                                import re
                                frame_match = re.search(r'(\d+)\.(jpg|png|jpeg)', filename, re.IGNORECASE)
                                if frame_match:
                                    frame_num = frame_match.group(1)
                                    # Search for files with the same frame number pattern
                                    for variant_dir in os.listdir(scene_dir):
                                        variant_full_path = os.path.join(scene_dir, variant_dir)
                                        if not os.path.isdir(variant_full_path):
                                            continue
                                        
                                        if 'rgb' in path_lower:
                                            frames_dir = os.path.join(variant_full_path, 'frames', 'rgb', 'Camera_0')
                                        elif 'depth' in path_lower:
                                            frames_dir = os.path.join(variant_full_path, 'frames', 'depth', 'Camera_0')
                                        else:
                                            continue
                                        
                                        if os.path.exists(frames_dir):
                                            # Look for files with similar frame numbers
                                            for f in os.listdir(frames_dir):
                                                if frame_num in f and f.endswith(('.jpg', '.png', '.jpeg')):
                                                    file_path = os.path.join(frames_dir, f)
                                                    if os.path.exists(file_path):
                                                        return file_path
            
            # Try to resolve symlinks for absolute paths
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
            # Provide helpful error message with path resolution info
            error_msg = f"Could not load image: {img_path}\n"
            error_msg += f"  Original line: {self.filelist[item][:200]}\n"
            error_msg += f"  Resolved path: {img_path}\n"
            error_msg += f"  File exists: {os.path.exists(img_path)}\n"
            # Check if directory exists
            img_dir = os.path.dirname(img_path)
            if os.path.exists(img_dir):
                files_in_dir = os.listdir(img_dir)
                error_msg += f"  Directory exists with {len(files_in_dir)} files\n"
                if len(files_in_dir) > 0:
                    error_msg += f"  Sample files: {files_in_dir[:3]}\n"
            else:
                error_msg += f"  Directory does not exist: {img_dir}\n"
            raise ValueError(error_msg)
        # Validate image dimensions
        if len(image.shape) < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(
                f"Invalid image dimensions: shape={image.shape}, path={img_path}. "
                f"This usually indicates a corrupted or empty image file."
            )
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # Record original resolution before any resizing (Bug 1 fix)
        orig_h, orig_w = image.shape[:2]
        
        # Load depth (VKITTI depth is in cm, convert to meters)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            error_msg = f"Could not load depth: {depth_path}\n"
            error_msg += f"  Resolved path: {depth_path}\n"
            error_msg += f"  File exists: {os.path.exists(depth_path)}\n"
            raise ValueError(error_msg)
        depth = depth / 100.0  # cm to m
        
        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})
        
        # Convert to tensors
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        
        # Create valid mask
        sample['valid_mask'] = (sample['depth'] <= 80) & (sample['depth'] > 0)
        
        # Store original image dimensions so intrinsics can be correctly normalised
        # during training (the intrinsics correspond to this resolution, not the resized one)
        sample['original_size'] = torch.tensor([orig_h, orig_w], dtype=torch.long)
        
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


class NYUTrainingDataset(Dataset):
    """
    NYU Depth V2 dataset for depth estimation training.
    
    Loads data directly from the nyu_depth_v2_labeled.mat file which contains
    1,449 densely labeled pairs of aligned RGB and depth images.
    
    The .mat file uses HDF5 format (MATLAB v7.3), so we use h5py to read it.
    Images are 640x480 resolution with known camera intrinsics.
    
    Camera Intrinsics (640×480 resolution):
        fx = 518.8579
        fy = 519.4696
        cx = 325.5824
        cy = 253.7362
    """
    
    # Camera intrinsic parameters (640x480 resolution)
    INTRINSIC_FX = 518.8579
    INTRINSIC_FY = 519.4696
    INTRINSIC_CX = 325.5824
    INTRINSIC_CY = 253.7362
    
    def __init__(self, filelist_path_or_mat_path, mode='train', size=(518, 518)):
        """
        Initialize NYU Depth V2 training dataset.
        
        Args:
            filelist_path_or_mat_path: Path to the nyu_depth_v2_labeled.mat file
                                       or a split file containing indices
            mode: 'train' or 'val' (dataset will be split 80/20 if no split file)
            size: (width, height) for image resizing
        """
        self.mode = mode
        self.size = size
        
        # Get project root
        current_file = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file))
        
        # Try to import h5py
        try:
            import h5py
            self.h5py = h5py
        except ImportError:
            raise ImportError(
                "h5py is required to read NYU Depth V2 .mat files. "
                "Install it with: pip install h5py"
            )
        
        # Determine if input is a .mat file or a split file
        if not os.path.isabs(filelist_path_or_mat_path):
            filelist_path_or_mat_path = os.path.join(self.project_root, filelist_path_or_mat_path)
        
        if filelist_path_or_mat_path.endswith('.mat'):
            self.mat_path = filelist_path_or_mat_path
            self.indices = None  # Will be set after loading
        else:
            # It's a split file with indices
            # Format: one index per line, or path to .mat file + indices
            mat_dir = os.path.dirname(filelist_path_or_mat_path)
            self.mat_path = os.path.join(mat_dir, 'nyu_depth_v2_labeled.mat')
            
            # If mat file not found in same dir, check default location
            if not os.path.exists(self.mat_path):
                self.mat_path = os.path.join(self.project_root, 'datasets', 'raw_data', 'nyu', 'nyu_depth_v2_labeled.mat')
            
            # Read indices from split file
            with open(filelist_path_or_mat_path, 'r') as f:
                lines = f.read().strip().splitlines()
            
            # Parse indices (one per line)
            self.indices = []
            for line in lines:
                line = line.strip()
                if line and line.isdigit():
                    self.indices.append(int(line))
            
            if not self.indices:
                raise ValueError(f"No valid indices found in split file: {filelist_path_or_mat_path}")
        
        if not os.path.exists(self.mat_path):
            raise FileNotFoundError(
                f"NYU Depth V2 .mat file not found: {self.mat_path}\n"
                f"Please download the labeled subset from:\n"
                f"https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html\n"
                f"and place nyu_depth_v2_labeled.mat in: {os.path.dirname(self.mat_path)}"
            )
        
        # Load the .mat file
        print(f"Loading NYU Depth V2 dataset from {self.mat_path}...")
        print("(This may take a moment for the first load)")
        
        with self.h5py.File(self.mat_path, 'r') as f:
            # HDF5/MATLAB v7.3 format stores data transposed
            # When loaded with h5py, we need to handle the reference-based storage
            # Load the actual data (h5py may return references that need to be dereferenced)
            images_ref = f['images']
            depths_ref = f['depths']
            
            # Get the actual shape (may be different due to HDF5 reference storage)
            images_shape = images_ref.shape
            depths_shape = depths_ref.shape
            
            # Load the data
            self._images = np.array(images_ref)
            self._depths = np.array(depths_ref)
        
        # Check actual loaded shapes
        original_img_shape = self._images.shape
        original_depth_shape = self._depths.shape
        
        print(f"Loaded NYU data - Images shape: {original_img_shape}, Depths shape: {original_depth_shape}")
        
        # NYU Depth V2 has 1449 images, typically 480x640 resolution (height x width)
        n_images_expected = 1449
        expected_height = 480
        expected_width = 640
        
        # Transpose images to [n_images, height, width, channels]
        # The .mat file from NYU stores data in MATLAB format, which h5py loads with reversed dimensions
        # We need to identify which dimension is which and construct the correct transpose
        img_dims = list(original_img_shape)
        
        if len(img_dims) != 4:
            raise ValueError(
                f"Expected 4D image array from NYU .mat file, got shape {original_img_shape}. "
                f"Expected dimensions: [n_images={n_images_expected}, height={expected_height}, "
                f"width={expected_width}, channels=3] in some order."
            )
        
        # Find each dimension by its expected size
        # n_images = 1449, channels = 3, height = 480, width = 640
        def find_dim_idx(dims, target_value, exclude_indices=None):
            """Find index of dimension matching target value, excluding already-matched indices."""
            exclude_indices = exclude_indices or []
            for i, d in enumerate(dims):
                if d == target_value and i not in exclude_indices:
                    return i
            return None
        
        # Find n_images (1449) - most unique value
        img_n_idx = find_dim_idx(img_dims, n_images_expected)
        
        # Find channels (3)
        img_c_idx = find_dim_idx(img_dims, 3, exclude_indices=[img_n_idx] if img_n_idx is not None else [])
        
        # Find height (480) and width (640)
        exclude = [i for i in [img_n_idx, img_c_idx] if i is not None]
        img_h_idx = find_dim_idx(img_dims, expected_height, exclude_indices=exclude)
        exclude = [i for i in [img_n_idx, img_c_idx, img_h_idx] if i is not None]
        img_w_idx = find_dim_idx(img_dims, expected_width, exclude_indices=exclude)
        
        # Build transpose order: target is [n_images, height, width, channels]
        if img_n_idx is not None and img_c_idx is not None:
            # We found n_images and channels, figure out height/width from remaining dims
            remaining_indices = [i for i in range(4) if i not in [img_n_idx, img_c_idx]]
            
            if img_h_idx is not None and img_w_idx is not None:
                # All dimensions identified - construct transpose directly
                transpose_order = (img_n_idx, img_h_idx, img_w_idx, img_c_idx)
            elif img_h_idx is not None:
                # Height identified, width is the other remaining dimension
                img_w_idx = [i for i in remaining_indices if i != img_h_idx][0]
                transpose_order = (img_n_idx, img_h_idx, img_w_idx, img_c_idx)
            elif img_w_idx is not None:
                # Width identified, height is the other remaining dimension
                img_h_idx = [i for i in remaining_indices if i != img_w_idx][0]
                transpose_order = (img_n_idx, img_h_idx, img_w_idx, img_c_idx)
            else:
                # Neither height nor width identified - infer from remaining dimensions
                # remaining_indices has the two spatial dimensions
                # Try to figure out which is height (480) and which is width (640)
                dim0_size = img_dims[remaining_indices[0]]
                dim1_size = img_dims[remaining_indices[1]]
                
                # Height should be smaller than width for typical landscape images
                if dim0_size <= dim1_size:
                    img_h_idx, img_w_idx = remaining_indices[0], remaining_indices[1]
                else:
                    img_h_idx, img_w_idx = remaining_indices[1], remaining_indices[0]
                
                transpose_order = (img_n_idx, img_h_idx, img_w_idx, img_c_idx)
            
            print(f"  Identified image dimensions: n_images={img_n_idx}, height={img_h_idx}, width={img_w_idx}, channels={img_c_idx}")
            print(f"  Applying transpose: {transpose_order}")
            
            if transpose_order != (0, 1, 2, 3):
                self._images = np.transpose(self._images, transpose_order)
        else:
            # Could not identify key dimensions - try common fallback patterns
            print(f"  Warning: Could not identify all dimensions. Trying fallback patterns.")
            print(f"    Found: n_images_idx={img_n_idx}, channels_idx={img_c_idx}")
            
            # Common HDF5/MATLAB patterns:
            # Pattern 1: (3, 480, 640, 1449) -> [C, H, W, N] - transpose to [N, H, W, C]
            # Pattern 2: (1449, 3, 640, 480) -> [N, C, W, H] - transpose to [N, H, W, C]
            # Pattern 3: (480, 640, 3, 1449) -> [H, W, C, N] - transpose to [N, H, W, C]
            
            if img_dims[0] == 3 and img_dims[-1] == n_images_expected:
                # [C, H, W, N] -> [N, H, W, C]
                self._images = np.transpose(self._images, (3, 1, 2, 0))
                print(f"    Applied pattern 1: (3, 1, 2, 0)")
            elif img_dims[-1] == 3 and img_dims[0] == n_images_expected:
                # Already [N, ?, ?, C] - may need height/width swap
                if img_dims[1] > img_dims[2]:
                    # [N, W, H, C] -> [N, H, W, C]
                    self._images = np.transpose(self._images, (0, 2, 1, 3))
                    print(f"    Applied height/width swap: (0, 2, 1, 3)")
            elif img_dims[0] == n_images_expected and img_dims[1] == 3:
                # [N, C, ?, ?] -> [N, H, W, C]
                # Determine if dims 2,3 are (W,H) or (H,W)
                if img_dims[2] > img_dims[3]:
                    # [N, C, W, H] -> [N, H, W, C]
                    self._images = np.transpose(self._images, (0, 3, 2, 1))
                    print(f"    Applied pattern: (0, 3, 2, 1)")
                else:
                    # [N, C, H, W] -> [N, H, W, C]
                    self._images = np.transpose(self._images, (0, 2, 3, 1))
                    print(f"    Applied pattern: (0, 2, 3, 1)")
            elif img_dims[-1] == n_images_expected:
                # [?, ?, ?, N] -> [N, H, W, C]
                if img_dims[0] == 3:
                    self._images = np.transpose(self._images, (3, 1, 2, 0))
                elif img_dims[2] == 3:
                    self._images = np.transpose(self._images, (3, 0, 1, 2))
                else:
                    # Assume [H, W, C, N]
                    self._images = np.transpose(self._images, (3, 0, 1, 2))
                print(f"    Applied N-last pattern")
            else:
                raise ValueError(
                    f"Cannot determine correct transpose for images. Shape: {original_img_shape}. "
                    f"Expected to find n_images={n_images_expected} and channels=3 in dimensions. "
                    f"Please ensure the .mat file contains valid NYU Depth V2 data."
                )
        
        # Handle depths: need [n_images, height, width]
        depth_dims = list(original_depth_shape)
        
        if len(depth_dims) != 3:
            raise ValueError(
                f"Expected 3D depth array from NYU .mat file, got shape {original_depth_shape}. "
                f"Expected dimensions: [n_images={n_images_expected}, height, width] in some order."
            )
        
        depth_n_idx = find_dim_idx(depth_dims, n_images_expected)
        
        if depth_n_idx is not None:
            remaining = [i for i in range(3) if i != depth_n_idx]
            # Put n_images first, keep spatial dims in order (smaller=height, larger=width)
            if depth_dims[remaining[0]] <= depth_dims[remaining[1]]:
                depth_h_idx, depth_w_idx = remaining[0], remaining[1]
            else:
                depth_h_idx, depth_w_idx = remaining[1], remaining[0]
            
            transpose_order = (depth_n_idx, depth_h_idx, depth_w_idx)
            print(f"  Identified depth dimensions: n_images={depth_n_idx}, height={depth_h_idx}, width={depth_w_idx}")
            
            if transpose_order != (0, 1, 2):
                self._depths = np.transpose(self._depths, transpose_order)
                print(f"  Applied depth transpose: {transpose_order}")
        else:
            # Fallback: assume last dimension is n_images
            if depth_dims[-1] == n_images_expected or depth_dims[2] > depth_dims[0]:
                # [H, W, N] -> [N, H, W]
                self._depths = np.transpose(self._depths, (2, 0, 1))
                print(f"  Applied depth fallback transpose: (2, 0, 1)")
        
        # Validate final shapes
        final_img_shape = self._images.shape
        final_depth_shape = self._depths.shape
        
        print(f"After transpose - Images shape: {final_img_shape}, Depths shape: {final_depth_shape}")
        
        # Validate that we have correct format: [n_images, H, W, 3] for images
        if len(final_img_shape) != 4:
            raise ValueError(f"Image array should be 4D, got shape {final_img_shape}")
        
        if final_img_shape[-1] != 3:
            raise ValueError(
                f"After transpose, image shape is {final_img_shape}, but last dimension should be 3 (channels). "
                f"Original shape was {original_img_shape}. The transpose logic may need adjustment."
            )
        
        if final_img_shape[0] != n_images_expected:
            raise ValueError(
                f"After transpose, image shape is {final_img_shape}, but first dimension should be {n_images_expected} (n_images). "
                f"Original shape was {original_img_shape}. The transpose logic may need adjustment."
            )
        
        # Validate depth shape: [n_images, H, W]
        if len(final_depth_shape) != 3:
            raise ValueError(f"Depth array should be 3D, got shape {final_depth_shape}")
        
        if final_depth_shape[0] != n_images_expected:
            raise ValueError(
                f"After transpose, depth shape is {final_depth_shape}, but first dimension should be {n_images_expected} (n_images). "
                f"Original shape was {original_depth_shape}. The transpose logic may need adjustment."
            )
        
        print(f"After transpose - Images shape: {final_img_shape}, Depths shape: {final_depth_shape}")
        
        n_total = len(self._images)
        
        # Set up indices based on mode if not explicitly provided
        if self.indices is None:
            # Default 80/20 split
            np.random.seed(42)  # Fixed seed for reproducibility
            all_indices = np.random.permutation(n_total)
            split_idx = int(0.8 * n_total)
            
            if mode == 'train':
                self.indices = all_indices[:split_idx].tolist()
            else:
                self.indices = all_indices[split_idx:].tolist()
        
        print(f"Loaded NYU Depth V2 {mode} set: {len(self.indices)} samples (from {n_total} total)")
        
        # Build intrinsic matrix
        self.intrinsic = np.array([
            [self.INTRINSIC_FX, 0.0, self.INTRINSIC_CX],
            [0.0, self.INTRINSIC_FY, self.INTRINSIC_CY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
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
    
    def __getitem__(self, item):
        # Get the actual index from our subset
        idx = self.indices[item]
        
        # Load image and depth from cached arrays
        image = self._images[idx].astype(np.float32) / 255.0  # [H, W, 3] normalized to [0, 1]
        depth = self._depths[idx].astype(np.float32)  # [H, W] in meters
        # Record original resolution before any resizing (Bug 1 fix)
        orig_h, orig_w = image.shape[:2]
        
        # Validate image dimensions and shape
        if len(image.shape) < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(
                f"Invalid image dimensions: shape={image.shape}, index={idx}. "
                f"This usually indicates corrupted data in the .mat file or incorrect transpose."
            )
        
        # Check for expected shape [H, W, 3] or [H, W]
        # Handle case where channels might be in wrong position due to transpose issues
        if len(image.shape) == 3:
            # Check if channels are in the last dimension (expected)
            if image.shape[2] in [1, 3]:
                # Correct format [H, W, C]
                pass
            elif image.shape[0] == 3 or image.shape[1] == 3:
                # Channels are in wrong position - try to fix
                old_shape = image.shape
                if image.shape[0] == 3:
                    # [C, H, W] -> [H, W, C]
                    image = np.transpose(image, (1, 2, 0))
                    print(f"Warning: Fixed image shape for index {idx} from {old_shape} to {image.shape}")
                elif image.shape[1] == 3:
                    # [H, C, W] -> [H, W, C]  
                    image = np.transpose(image, (0, 2, 1))
                    print(f"Warning: Fixed image shape for index {idx} from {old_shape} to {image.shape}")
            else:
                # Unexpected shape
                raise ValueError(
                    f"Unexpected image shape: {image.shape}, index={idx}. "
                    f"Expected [H, W, 3] or [H, W, 1], but channels dimension (size 3 or 1) not found. "
                    f"This indicates the .mat file transpose may be incorrect. "
                    f"Full array shape: {self._images.shape}, indexed shape: {image.shape}."
                )
        
        # Check for extremely unusual aspect ratios (likely data corruption)
        img_height, img_width = image.shape[0], image.shape[1]
        aspect_ratio = img_width / img_height if img_height > 0 else 0
        
        if aspect_ratio < 0.01 or aspect_ratio > 100:
            raise ValueError(
                f"Extreme aspect ratio in NYU image: width={img_width}, height={img_height}, "
                f"ratio={aspect_ratio:.4f}, shape={image.shape}, index={idx}. "
                f"This usually indicates corrupted data in the .mat file. "
                f"NYU images should be approximately 640x480."
            )
        
        # Check for extremely small dimensions
        if img_width < 10 or img_height < 10:
            raise ValueError(
                f"Image has extremely small dimensions: width={img_width}, height={img_height}, "
                f"shape={image.shape}, index={idx}. Minimum expected dimension is 10 pixels. "
                f"This usually indicates corrupted data in the .mat file."
            )
        
        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})
        
        # Convert to tensors
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        
        # Create valid mask (NYU max depth is ~10m for indoor scenes)
        sample['valid_mask'] = (sample['depth'] <= 10.0) & (sample['depth'] > 0)
        
        # Store original image dimensions for correct intrinsics normalisation (Bug 1 fix)
        sample['original_size'] = torch.tensor([orig_h, orig_w], dtype=torch.long)
        
        # Intrinsics are for the original (640x480) resolution
        sample['intrinsics'] = torch.from_numpy(self.intrinsic).float()
        
        # Add image identifier
        sample['image_path'] = f"nyu_depth_v2_labeled.mat::{idx}"
        
        return sample
    
    def __len__(self):
        return len(self.indices)


class DIODETrainingDataset(Dataset):
    """
    DIODE dataset for depth estimation training.
    
    Reads from the DIODE dataset folder structure:
        diode/val/{indoors,outdoor}/scene_XXXXX/scan_XXXXX/
            - {name}.png (RGB image)
            - {name}_depth.npy (depth in meters)
            - {name}_depth_mask.npy (valid mask)
    
    Camera Intrinsics (1024×768 resolution):
        fx = 886.81
        fy = 927.06
        cx = 512
        cy = 384
    """
    
    # Camera intrinsic parameters (1024x768 resolution)
    INTRINSIC_FX = 886.81
    INTRINSIC_FY = 927.06
    INTRINSIC_CX = 512.0
    INTRINSIC_CY = 384.0
    
    def __init__(self, dataset_path, mode='train', size=(518, 518)):
        """
        Initialize DIODE training dataset.
        
        Args:
            dataset_path: Path to the DIODE dataset root (e.g., datasets/raw_data/diode)
            mode: 'train' or 'val'
            size: (width, height) for image resizing
        """
        self.mode = mode
        self.size = size
        
        # Get project root
        current_file = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file))
        
        # Resolve dataset path
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(self.project_root, dataset_path)
        
        self.dataset_path = dataset_path
        
        # Find all image/depth pairs
        self.samples = self._find_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in DIODE dataset at: {dataset_path}")
        
        # Split into train/val (80/20)
        np.random.seed(42)  # Fixed seed for reproducibility
        all_indices = np.random.permutation(len(self.samples))
        split_idx = int(0.8 * len(self.samples))
        
        if mode == 'train':
            indices = all_indices[:split_idx]
        else:
            indices = all_indices[split_idx:]
        
        self.samples = [self.samples[i] for i in indices]
        
        print(f"Loaded DIODE {mode} set: {len(self.samples)} samples")
        
        # Build intrinsic matrix
        self.intrinsic = np.array([
            [self.INTRINSIC_FX, 0.0, self.INTRINSIC_CX],
            [0.0, self.INTRINSIC_FY, self.INTRINSIC_CY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
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
    
    def _find_samples(self):
        """Find all valid image/depth pairs in the dataset."""
        from glob import glob
        
        samples = []
        
        # Look in 'val' split (or 'train' if it exists)
        for split_name in ['val', 'train']:
            split_dir = os.path.join(self.dataset_path, split_name)
            if not os.path.exists(split_dir):
                continue
            
            # Find scene type directories (indoors, outdoor)
            for scene_type in os.listdir(split_dir):
                scene_type_path = os.path.join(split_dir, scene_type)
                if not os.path.isdir(scene_type_path):
                    continue
                
                # Find scene directories
                for scene_name in os.listdir(scene_type_path):
                    if not scene_name.startswith('scene_'):
                        continue
                    scene_path = os.path.join(scene_type_path, scene_name)
                    if not os.path.isdir(scene_path):
                        continue
                    
                    # Find scan directories
                    for scan_name in os.listdir(scene_path):
                        if not scan_name.startswith('scan_'):
                            continue
                        scan_path = os.path.join(scene_path, scan_name)
                        if not os.path.isdir(scan_path):
                            continue
                        
                        # Find all PNG files (excluding depth-related files)
                        png_files = glob(os.path.join(scan_path, '*.png'))
                        
                        for rgb_path in png_files:
                            base_name = os.path.basename(rgb_path)
                            
                            # Skip depth-related files
                            if '_depth' in base_name:
                                continue
                            
                            # Get base name without extension
                            name_without_ext = os.path.splitext(base_name)[0]
                            
                            # Find corresponding depth file
                            depth_file = os.path.join(scan_path, f'{name_without_ext}_depth.npy')
                            
                            if not os.path.exists(depth_file):
                                # Try with wildcards
                                depth_files = glob(os.path.join(scan_path, f'{name_without_ext}_depth.npy*'))
                                if len(depth_files) > 0:
                                    depth_file = depth_files[0]
                                else:
                                    continue
                            
                            # Find mask file (optional)
                            mask_file = os.path.join(scan_path, f'{name_without_ext}_depth_mask.npy')
                            mask_files = glob(os.path.join(scan_path, f'{name_without_ext}_depth_mask.npy*'))
                            if len(mask_files) > 0:
                                mask_file = mask_files[0]
                            elif not os.path.exists(mask_file):
                                mask_file = None
                            
                            samples.append({
                                'rgb_path': rgb_path,
                                'depth_path': depth_file,
                                'mask_path': mask_file,
                                'scene_type': scene_type,
                                'scene_name': scene_name,
                                'scan_name': scan_name,
                            })
        
        return samples
    
    def __getitem__(self, item):
        sample_info = self.samples[item]
        
        # Load image
        image = cv2.imread(sample_info['rgb_path'])
        if image is None:
            raise ValueError(f"Could not load image: {sample_info['rgb_path']}")
        
        # Validate image dimensions
        if len(image.shape) < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(
                f"Invalid image dimensions: shape={image.shape}, path={sample_info['rgb_path']}. "
                f"This usually indicates a corrupted or empty image file."
            )
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # Record original resolution before any resizing (Bug 1 fix)
        orig_h, orig_w = image.shape[:2]
        
        # Load depth (stored in meters)
        depth = np.load(sample_info['depth_path']).astype(np.float32)
        
        # Load mask if available
        if sample_info['mask_path'] and os.path.exists(sample_info['mask_path']):
            mask = np.load(sample_info['mask_path'])
            # Apply mask - set invalid pixels to 0
            if mask.dtype == bool:
                depth[~mask] = 0
            else:
                depth[mask == 0] = 0
        
        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})
        
        # Convert to tensors
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        
        # Create valid mask (DIODE max depth ~10m for indoor scenes)
        sample['valid_mask'] = (sample['depth'] <= 10.0) & (sample['depth'] > 0)
        
        # Store original image dimensions for correct intrinsics normalisation (Bug 1 fix)
        sample['original_size'] = torch.tensor([orig_h, orig_w], dtype=torch.long)
        
        # Intrinsics are for the original (1024x768) resolution
        sample['intrinsics'] = torch.from_numpy(self.intrinsic).float()
        
        # Add image identifier
        sample['image_path'] = sample_info['rgb_path']
        
        return sample
    
    def __len__(self):
        return len(self.samples)


class MiddleburyTrainingDataset(Dataset):
    """
    Middlebury stereo dataset for depth estimation training.

    Loads from the Middlebury Stereo Evaluation structure:
        middlebury/{scene_name}/
            - calib.txt    # Camera intrinsics (cam0, cam1, doffs, baseline)
            - im0.png      # Left camera image
            - im1.png      # Right camera image
            - disp0.pfm    # Left disparity (convertible to depth)
            - disp1.pfm    # Right disparity

    Intrinsics are loaded from calib.txt per scene/camera.
    Depth is computed from disparity: depth = (baseline * f) / (disparity + doffs)
    """

    def __init__(self, dataset_path, mode='train', size=(518, 518), train_ratio=0.8):
        """
        Initialize Middlebury training dataset.

        Args:
            dataset_path: Path to the Middlebury dataset root (e.g., datasets/raw_data/middlebury)
            mode: 'train' or 'val'
            size: (width, height) for image resizing
            train_ratio: Fraction of scenes for training (remainder for validation)
        """
        self.mode = mode
        self.size = size

        current_file = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file))

        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(self.project_root, dataset_path)

        self.dataset_path = dataset_path

        # Find all valid scene samples (each scene yields 2 samples: cam0 and cam1)
        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid Middlebury samples found at: {dataset_path}\n"
                f"Each scene needs: calib.txt, im0.png, im1.png, disp0.pfm, disp1.pfm"
            )

        # Split by scene (avoid same scene in train and val)
        scene_to_samples = {}
        for s in self.samples:
            scene = s['scene_name']
            if scene not in scene_to_samples:
                scene_to_samples[scene] = []
            scene_to_samples[scene].append(s)

        scenes = sorted(scene_to_samples.keys())
        np.random.seed(42)
        np.random.shuffle(scenes)
        split_idx = int(len(scenes) * train_ratio)
        train_scenes = set(scenes[:split_idx])
        val_scenes = set(scenes[split_idx:])

        if mode == 'train':
            self.samples = [s for s in self.samples if s['scene_name'] in train_scenes]
        else:
            self.samples = [s for s in self.samples if s['scene_name'] in val_scenes]

        print(f"Loaded Middlebury {mode} set: {len(self.samples)} samples from {len(scene_to_samples)} scenes")

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

    def _find_samples(self):
        """Find all valid image/depth pairs with intrinsics."""
        samples = []
        if not os.path.isdir(self.dataset_path):
            return samples

        for scene_name in sorted(os.listdir(self.dataset_path)):
            scene_path = os.path.join(self.dataset_path, scene_name)
            if not os.path.isdir(scene_path):
                continue

            calib_file = os.path.join(scene_path, 'calib.txt')
            if not os.path.exists(calib_file):
                continue

            try:
                f0, f1, cx0, cy0, cx1, cy1, doffs, baseline = _parse_middlebury_calib(calib_file)
            except Exception:
                continue

            for cam_id in [0, 1]:
                im_path = os.path.join(scene_path, f'im{cam_id}.png')
                disp_path = os.path.join(scene_path, f'disp{cam_id}.pfm')
                if not (os.path.exists(im_path) and os.path.exists(disp_path)):
                    continue

                f_val = f0 if cam_id == 0 else f1
                cx_val = cx0 if cam_id == 0 else cx1
                cy_val = cy0 if cam_id == 0 else cy1

                intrinsic = np.array([
                    [f_val, 0.0, cx_val],
                    [0.0, f_val, cy_val],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)

                samples.append({
                    'image_path': im_path,
                    'disp_path': disp_path,
                    'scene_name': scene_name,
                    'cam_id': cam_id,
                    'f': f_val,
                    'baseline': baseline,
                    'doffs': doffs,
                    'intrinsic': intrinsic,
                })

        return samples

    def __getitem__(self, item):
        s = self.samples[item]

        image = cv2.imread(s['image_path'])
        if image is None:
            raise ValueError(f"Could not load image: {s['image_path']}")

        if len(image.shape) < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"Invalid image dimensions: {image.shape}, path={s['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        orig_h, orig_w = image.shape[:2]

        disparity = _read_pfm(s['disp_path'])
        depth = _depth_from_disparity(
            disparity,
            s['f'],
            s['baseline'],
            s['doffs'],
        )

        # Ensure depth shape matches image (PFM can have different conventions)
        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()

        # Indoor Middlebury: max depth ~20m
        sample['valid_mask'] = (sample['depth'] > 0) & (sample['depth'] <= 20.0)

        sample['original_size'] = torch.tensor([orig_h, orig_w], dtype=torch.long)
        sample['intrinsics'] = torch.from_numpy(s['intrinsic']).float()
        sample['image_path'] = s['image_path']

        return sample

    def __len__(self):
        return len(self.samples)


class CityscapesTrainingDataset(Dataset):
    """
    Cityscapes dataset for depth estimation training.

    Uses left images and disparity maps. Disparity is converted to depth via:
        depth = (baseline * focal_length) / disparity

    Cityscapes camera (fixed): baseline=0.22 m, fx=fy=2262.52 px, standard resolution 2048x1024.
    Intrinsics are scaled when image resolution differs.
    """

    STANDARD_WIDTH = 2048
    STANDARD_HEIGHT = 1024
    BASELINE = 0.22
    FOCAL_LENGTH = 2262.52
    MIN_DISPARITY = 0.1
    MAX_DEPTH = 200.0

    def __init__(self, dataset_path, mode='train', size=(518, 518)):
        """
        Initialize Cityscapes training dataset.

        Args:
            dataset_path: Path to Cityscapes root (e.g., datasets/raw_data/cityscapes or .../CityScapes)
            mode: 'train' or 'val'
            size: (width, height) for image resizing
        """
        self.mode = mode
        self.size = size

        current_file = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file))

        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(self.project_root, dataset_path)
        if not os.path.isdir(dataset_path):
            for name in ('cityscapes', 'CityScapes'):
                candidate = os.path.join(self.project_root, 'datasets', 'raw_data', name)
                if os.path.isdir(candidate):
                    dataset_path = candidate
                    break
        self.dataset_path = dataset_path
        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"Cityscapes: no image/disparity pairs found at {dataset_path}\n"
                f"Expected: leftImg8bit/{{train,val}}/ and disparity/{{train,val}}/"
            )

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

    def _find_samples(self):
        """Find (image_path, disparity_path) for the current split."""
        split = self.mode  # 'train' or 'val'
        gt_dir = os.path.join(self.dataset_path, 'disparity', split)
        if not os.path.exists(gt_dir):
            gt_dir = os.path.join(self.dataset_path, 'disparity', split.lower())
        if not os.path.exists(gt_dir):
            return []

        possible_left_dirs = [
            os.path.join(self.dataset_path, 'leftImg8bit', split),
            os.path.join(self.dataset_path, 'leftImg8bit', split.lower()),
            os.path.join(self.dataset_path, 'images', split),
        ]
        left_img_dir = None
        for d in possible_left_dirs:
            if os.path.exists(d):
                left_img_dir = d
                break
        if left_img_dir is None:
            return []

        samples = []
        for city_dir in sorted(os.listdir(gt_dir)):
            if city_dir.startswith('._'):
                continue
            city_gt_path = os.path.join(gt_dir, city_dir)
            if not os.path.isdir(city_gt_path):
                continue
            for fname in sorted(os.listdir(city_gt_path)):
                if fname.startswith('._') or not fname.endswith('_disparity.png'):
                    continue
                base_name = fname.replace('_disparity.png', '')
                disp_path = os.path.join(city_gt_path, fname)
                city_left = os.path.join(left_img_dir, city_dir)
                if not os.path.isdir(city_left):
                    city_left = left_img_dir
                for img_name in (f'{base_name}_leftImg8bit.png', f'{base_name}_left.png', f'{base_name}.png'):
                    img_path = os.path.join(city_left, img_name)
                    if os.path.exists(img_path) and not os.path.basename(img_path).startswith('._'):
                        samples.append({'image_path': img_path, 'disp_path': disp_path})
                        break
        return samples

    def _disparity_to_depth(self, disparity):
        """Convert Cityscapes disparity (float pixels) to depth in meters."""
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = np.isfinite(disparity) & (disparity >= self.MIN_DISPARITY)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth[valid] = (self.BASELINE * self.FOCAL_LENGTH) / disparity[valid]
            depth[valid] = np.clip(depth[valid], 0.0, self.MAX_DEPTH)
        depth[~valid] = 0.0
        return depth

    def _intrinsic_for_size(self, width, height):
        scale_x = width / self.STANDARD_WIDTH
        scale_y = height / self.STANDARD_HEIGHT
        fx = self.FOCAL_LENGTH * scale_x
        fy = self.FOCAL_LENGTH * scale_y
        cx = (self.STANDARD_WIDTH / 2.0) * scale_x
        cy = (self.STANDARD_HEIGHT / 2.0) * scale_y
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def __getitem__(self, item):
        s = self.samples[item]
        image = cv2.imread(s['image_path'])
        if image is None:
            raise ValueError(f"Could not load image: {s['image_path']}")
        if len(image.shape) < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"Invalid image dimensions: {image.shape}, path={s['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        orig_h, orig_w = image.shape[:2]

        disp_img = cv2.imread(s['disp_path'], cv2.IMREAD_UNCHANGED)
        if disp_img is None:
            raise ValueError(f"Could not load disparity: {s['disp_path']}")
        if len(disp_img.shape) > 2:
            disp_img = disp_img[:, :, 0]
        disparity = disp_img.astype(np.float32) / 256.0
        disparity[disparity == 0] = np.nan
        depth = self._disparity_to_depth(disparity)
        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        sample = self.transform({'image': image, 'depth': depth})
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        sample['valid_mask'] = (sample['depth'] > 0) & (sample['depth'] <= 80.0)
        sample['original_size'] = torch.tensor([orig_h, orig_w], dtype=torch.long)
        sample['intrinsics'] = torch.from_numpy(self._intrinsic_for_size(orig_w, orig_h)).float()
        sample['image_path'] = s['image_path']
        return sample

    def __len__(self):
        return len(self.samples)


class KITTITrainingDataset(Dataset):
    """
    KITTI dataset for depth estimation training/validation.
    
    Supports two modes:
    1. File list mode: Reads from split files (e.g., metric_depth/dataset/splits/kitti/val.txt)
    2. Directory mode: Reads directly from KITTI depth benchmark structure
    
    KITTI dataset structure:
        kitti/
        ├── depth_selection/
        │   └── val_selection_cropped/
        │       ├── groundtruth_depth/
        │       ├── image/
        │       └── intrinsics/
        ├── train/
        │   └── {date}_drive_{seq}_sync/
        │       └── proj_depth/groundtruth/image_{02,03}/
        └── val/
            └── {date}_drive_{seq}_sync/
                └── proj_depth/groundtruth/image_{02,03}/
    
    Camera Intrinsics:
        KITTI intrinsics are loaded from .txt files in the intrinsics/ folder.
        Default values (reference resolution ~1242x375):
        - fx = fy ≈ 721.5377 pixels
        - cx ≈ 609.5593 pixels
        - cy ≈ 172.854 pixels
    """
    
    # Default KITTI intrinsic parameters (reference resolution ~1242x375)
    DEFAULT_FX = 721.5377
    DEFAULT_FY = 721.5377
    DEFAULT_CX = 609.5593
    DEFAULT_CY = 172.854
    DEFAULT_WIDTH = 1242
    DEFAULT_HEIGHT = 375
    
    def __init__(self, filelist_or_path, mode='val', size=(518, 518)):
        """
        Initialize KITTI training dataset.
        
        Args:
            filelist_or_path: Path to file list OR path to KITTI dataset root
            mode: 'train' or 'val'
            size: (width, height) for image resizing
        """
        self.mode = mode
        self.size = size
        
        # Get project root
        current_file = os.path.abspath(__file__)
        self.project_root = os.path.dirname(os.path.dirname(current_file))
        
        # Determine if we have a filelist or a directory path
        if not os.path.isabs(filelist_or_path):
            # Try relative to project root first
            test_path = os.path.join(self.project_root, filelist_or_path)
            if not os.path.exists(test_path):
                # Try relative to metric_depth
                metric_depth_path = os.path.join(self.project_root, 'models', 'raw_models', 
                                                 'DepthAnythingV2-revised', 'metric_depth')
                test_path = os.path.join(metric_depth_path, filelist_or_path)
            filelist_or_path = test_path
        
        # First, check if a filelist exists (preferred method, like VKITTI)
        if os.path.isdir(filelist_or_path):
            # Check for splits directory with filelist
            splits_dir = os.path.join(filelist_or_path, 'splits')
            filelist_path = os.path.join(splits_dir, f'{self.mode}.txt')
            if os.path.exists(filelist_path):
                print(f"Found KITTI filelist at: {filelist_path}")
                self.dataset_path = filelist_or_path
                self.filelist = self._load_filelist(filelist_path)
            else:
                # Directory mode - scan for image/depth pairs
                self.dataset_path = filelist_or_path
                self.filelist = self._scan_directory()
        elif os.path.isfile(filelist_or_path) and filelist_or_path.endswith('.txt'):
            # Filelist mode
            self.dataset_path = os.path.dirname(filelist_or_path)
            self.filelist = self._load_filelist(filelist_or_path)
        else:
            raise FileNotFoundError(f"KITTI dataset path not found: {filelist_or_path}")
        
        if len(self.filelist) == 0:
            # Provide more helpful error message
            error_msg = f"No valid entries found in KITTI dataset at: {filelist_or_path}\n"
            if os.path.isdir(filelist_or_path):
                error_msg += f"\nDirectory structure found:\n"
                # List what directories exist
                if os.path.exists(filelist_or_path):
                    subdirs = [d for d in os.listdir(filelist_or_path) 
                              if os.path.isdir(os.path.join(filelist_or_path, d))]
                    if subdirs:
                        error_msg += f"  Subdirectories: {', '.join(subdirs[:10])}\n"
                    else:
                        error_msg += f"  (directory is empty or contains only files)\n"
                
                # Check for expected structure
                train_path = os.path.join(filelist_or_path, 'train')
                val_path = os.path.join(filelist_or_path, 'val')
                depth_selection_path = os.path.join(filelist_or_path, 'depth_selection')
                
                if not os.path.exists(train_path) and not os.path.exists(val_path) and not os.path.exists(depth_selection_path):
                    error_msg += f"\nExpected one of:\n"
                    error_msg += f"  - {train_path}/ (with sequence directories)\n"
                    error_msg += f"  - {val_path}/ (with sequence directories)\n"
                    error_msg += f"  - {depth_selection_path}/val_selection_cropped/\n"
                    error_msg += f"  - {os.path.join(filelist_or_path, 'splits', f'{self.mode}.txt')} (filelist)\n"
            raise ValueError(error_msg)
        
        print(f"Loaded KITTI {self.mode} set: {len(self.filelist)} samples")
        
        # Build default intrinsic matrix
        self.default_intrinsic = np.array([
            [self.DEFAULT_FX, 0.0, self.DEFAULT_CX],
            [0.0, self.DEFAULT_FY, self.DEFAULT_CY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
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
    
    def _scan_directory(self) -> list:
        """Scan KITTI directory structure for image/depth pairs."""
        from glob import glob
        samples = []
        debug_info = []
        
        # Check for depth_selection/val_selection_cropped (mainly for validation)
        val_selection_path = os.path.join(self.dataset_path, 'depth_selection', 'val_selection_cropped')
        if os.path.exists(val_selection_path) and self.mode == 'val':
            image_dir = os.path.join(val_selection_path, 'image')
            # Try both possible depth directory names
            depth_dir = os.path.join(val_selection_path, 'groundtruth_depth')
            if not os.path.exists(depth_dir):
                depth_dir = os.path.join(val_selection_path, 'groundtruth')
            intrinsics_dir = os.path.join(val_selection_path, 'intrinsics')
            
            if os.path.exists(image_dir) and os.path.exists(depth_dir):
                img_files = sorted(glob(os.path.join(image_dir, '*.png'))) + sorted(glob(os.path.join(image_dir, '*.jpg')))
                for img_file in img_files:
                    base_name = os.path.splitext(os.path.basename(img_file))[0]
                    depth_file = os.path.join(depth_dir, f'{base_name}.png')
                    intrinsics_file = os.path.join(intrinsics_dir, f'{base_name}.txt')
                    
                    if os.path.exists(depth_file):
                        samples.append({
                            'image_path': img_file,
                            'depth_path': depth_file,
                            'intrinsics_path': intrinsics_file if os.path.exists(intrinsics_file) else None
                        })
                if len(img_files) > 0:
                    debug_info.append(f"Found {len(samples)} samples in depth_selection/val_selection_cropped")
        
        # Check for train/val sequence folders
        for split in ['train', 'val']:
            if self.mode == 'train' and split != 'train':
                continue
            if self.mode == 'val' and split != 'val':
                continue
            
            split_path = os.path.join(self.dataset_path, split)
            if not os.path.exists(split_path):
                debug_info.append(f"Split directory not found: {split_path}")
                continue
            
            # Find all sequence directories
            try:
                seq_dirs = [d for d in os.listdir(split_path) 
                           if os.path.isdir(os.path.join(split_path, d)) and '_drive_' in d]
            except PermissionError:
                debug_info.append(f"Permission denied accessing: {split_path}")
                continue
            
            if len(seq_dirs) == 0:
                debug_info.append(f"No sequence directories found in {split_path}")
                continue
            
            debug_info.append(f"Found {len(seq_dirs)} sequence directories in {split}")
            
            for seq_name in sorted(seq_dirs):
                for camera in ['image_02', 'image_03']:
                    gt_dir = os.path.join(split_path, seq_name, 'proj_depth', 'groundtruth', camera)
                    
                    if not os.path.exists(gt_dir):
                        continue
                    
                    depth_files = sorted(glob(os.path.join(gt_dir, '*.png')))
                    if len(depth_files) == 0:
                        continue
                    
                    found_images = 0
                    missing_images_sample = None
                    for depth_file in depth_files[:10]:  # Check first 10 to diagnose
                        frame_name = os.path.splitext(os.path.basename(depth_file))[0]
                        
                        # Try to find corresponding image
                        img_path = self._find_image_for_depth(seq_name, camera, frame_name, split)
                        
                        if img_path and os.path.exists(img_path):
                            samples.append({
                                'image_path': img_path,
                                'depth_path': depth_file,
                                'intrinsics_path': None,
                                'sequence': seq_name,
                                'camera': camera
                            })
                            found_images += 1
                        elif missing_images_sample is None:
                            missing_images_sample = frame_name
                    
                    # Process remaining depth files if we found at least one image
                    if found_images > 0:
                        for depth_file in depth_files[10:]:
                            frame_name = os.path.splitext(os.path.basename(depth_file))[0]
                            img_path = self._find_image_for_depth(seq_name, camera, frame_name, split)
                            if img_path and os.path.exists(img_path):
                                samples.append({
                                    'image_path': img_path,
                                    'depth_path': depth_file,
                                    'intrinsics_path': None,
                                    'sequence': seq_name,
                                    'camera': camera
                                })
                                found_images += 1
                    
                    if found_images == 0 and len(depth_files) > 0:
                        # Provide more detailed diagnostic
                        sample_frame = missing_images_sample or os.path.splitext(os.path.basename(depth_files[0]))[0]
                        debug_info.append(
                            f"Warning: Found {len(depth_files)} depth files in {gt_dir} but no matching images.\n"
                            f"  Example: Looking for image matching frame '{sample_frame}' for camera '{camera}' in sequence '{seq_name}'.\n"
                            f"  Checked locations include:\n"
                            f"    - {os.path.join(self.dataset_path, 'kitti_raw', '*', seq_name, camera, 'data', f'{sample_frame}.png')}\n"
                            f"    - {os.path.join(self.dataset_path, split, seq_name, camera, 'data', f'{sample_frame}.png')}\n"
                            f"  Note: KITTI depth benchmark may require separate download of RGB images from KITTI Raw dataset."
                        )
        
        # Print debug information if no samples found
        if len(samples) == 0 and len(debug_info) > 0:
            print("KITTI directory scan diagnostics:")
            for info in debug_info:
                print(f"  {info}")
            
            # Check if kitti_raw exists
            kitti_raw_path = os.path.join(self.dataset_path, 'kitti_raw')
            if not os.path.exists(kitti_raw_path):
                print(f"\n  Note: KITTI Raw dataset not found at: {kitti_raw_path}")
                print(f"  KITTI Depth Prediction Benchmark provides depth maps, but RGB images")
                print(f"  need to be downloaded separately from KITTI Raw dataset.")
                print(f"  Expected structure: {kitti_raw_path}/{{date}}/{{sequence}}/{{camera}}/data/{{frame}}.png")
                print(f"\n  Alternative: Create a filelist at:")
                print(f"    {os.path.join(self.dataset_path, 'splits', f'{self.mode}.txt')}")
                print(f"  Format: image_path depth_path [intrinsics_path]")
        
        return samples
    
    def _find_image_for_depth(self, seq_name: str, camera: str, frame_name: str, split: str) -> str:
        """Find corresponding RGB image for a depth map."""
        from glob import glob
        
        # Extract date from sequence name
        date_match = re.match(r'(\d{4}_\d{2}_\d{2})_drive_\d+_sync', seq_name)
        date = None
        if date_match:
            date = date_match.group(1)
        
        # Common image locations to check
        possible_paths = []
        
        # If we have a date, try kitti_raw structure (common KITTI organization)
        if date:
            possible_paths.extend([
                os.path.join(self.dataset_path, 'kitti_raw', date, seq_name, camera, 'data', f'{frame_name}.png'),
                os.path.join(self.dataset_path, 'kitti_raw', date, seq_name, camera, 'data', f'{frame_name}.jpg'),
                # Also try without date in path
                os.path.join(self.dataset_path, 'kitti_raw', seq_name, camera, 'data', f'{frame_name}.png'),
                os.path.join(self.dataset_path, 'kitti_raw', seq_name, camera, 'data', f'{frame_name}.jpg'),
            ])
        
        # Try various locations relative to split directory
        possible_paths.extend([
            os.path.join(self.dataset_path, split, seq_name, camera, 'data', f'{frame_name}.png'),
            os.path.join(self.dataset_path, split, seq_name, camera, 'data', f'{frame_name}.jpg'),
            os.path.join(self.dataset_path, split, seq_name, 'image', camera, f'{frame_name}.png'),
            os.path.join(self.dataset_path, split, seq_name, 'image', camera, f'{frame_name}.jpg'),
            os.path.join(self.dataset_path, split, seq_name, camera, f'{frame_name}.png'),
            os.path.join(self.dataset_path, split, seq_name, camera, f'{frame_name}.jpg'),
            # Check if images are in the same directory structure as depth (sibling to proj_depth)
            os.path.join(self.dataset_path, split, seq_name, camera, 'data', f'{frame_name}.png'),
            os.path.join(self.dataset_path, split, seq_name, camera, 'data', f'{frame_name}.jpg'),
        ])
        
        # Check standard paths first
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If not found in standard locations, try recursive search in sequence directory
        seq_dir = os.path.join(self.dataset_path, split, seq_name)
        if os.path.exists(seq_dir):
            # Search for any image file with matching frame name in the sequence directory
            search_patterns = [
                os.path.join(seq_dir, '**', f'{frame_name}.png'),
                os.path.join(seq_dir, '**', f'{frame_name}.jpg'),
            ]
            
            for pattern in search_patterns:
                matches = glob(pattern, recursive=True)
                # Filter out depth files (they're in proj_depth/groundtruth)
                for match in matches:
                    if 'proj_depth' not in match and 'groundtruth' not in match:
                        return match
        
        # Also try searching in parent kitti_raw if it exists
        if date:
            kitti_raw_base = os.path.join(self.dataset_path, 'kitti_raw')
            if os.path.exists(kitti_raw_base):
                # Search recursively for the frame
                search_patterns = [
                    os.path.join(kitti_raw_base, '**', f'{frame_name}.png'),
                    os.path.join(kitti_raw_base, '**', f'{frame_name}.jpg'),
                ]
                for pattern in search_patterns:
                    matches = glob(pattern, recursive=True)
                    # Prefer matches in the same camera directory
                    for match in matches:
                        if camera in match:
                            return match
                    # If no camera match, return first match
                    if matches:
                        return matches[0]
        
        return None
    
    def _load_filelist(self, filelist_path: str) -> list:
        """Load samples from a file list."""
        with open(filelist_path, 'r') as f:
            raw_filelist = f.read().splitlines()
        
        samples = []
        skipped_count = 0
        
        for line in raw_filelist:
            line = line.strip()
            if not line:
                continue
            
            # Use shlex.split to handle spaces in paths
            parts = shlex.split(line)
            if len(parts) < 2:
                continue
            
            img_path_orig = parts[0]
            depth_path_orig = parts[1]
            
            # Resolve paths
            img_path_resolved = self._resolve_path(img_path_orig)
            depth_path_resolved = self._resolve_path(depth_path_orig)
            
            # Only keep entries where both files exist
            if os.path.exists(img_path_resolved) and os.path.exists(depth_path_resolved):
                samples.append({
                    'image_path': img_path_resolved,
                    'depth_path': depth_path_resolved,
                    'intrinsics_path': None
                })
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Warning: Skipped {skipped_count} entries with missing files")
        
        return samples
    
    def _resolve_path(self, path):
        """
        Resolve a path that might be absolute from another system or relative.
        Maps paths from other systems (e.g., /mnt/bn/liheyang/Kitti/...)
        to local project structure (datasets/raw_data/kitti/...).
        """
        # Normalize the path first
        path = os.path.normpath(path)
        
        # If path exists as-is, use it
        if os.path.exists(path):
            return path
        
        # If it's an absolute path from another system, try to map it to local structure
        if os.path.isabs(path):
            path_lower = path.lower()
            
            if 'kitti' in path_lower and ('raw_data' in path_lower or 'data_depth_annotated' in path_lower):
                parts = [p for p in path.split('/') if p]
                
                kitti_idx = None
                for i, part in enumerate(parts):
                    if 'kitti' in part.lower():
                        kitti_idx = i
                        break
                
                if kitti_idx is not None:
                    relative_parts = parts[kitti_idx + 1:]
                    
                    if len(relative_parts) > 0:
                        local_path = os.path.join(self.project_root, 'datasets', 'raw_data', 'kitti', *relative_parts)
                        local_path = os.path.normpath(local_path)
                        
                        if os.path.exists(local_path):
                            return local_path
            
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
        
        return path
    
    def _load_intrinsics(self, intrinsics_path: str) -> np.ndarray:
        """Load intrinsics from a .txt file."""
        try:
            with open(intrinsics_path, 'r') as f:
                line = f.readline().strip()
            
            values = line.split()
            
            if len(values) >= 4:
                fx = float(values[0])
                fy = float(values[1])
                cx = float(values[2])
                cy = float(values[3])
                
                return np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                
        except Exception:
            pass
        
        return self.default_intrinsic
    
    def __getitem__(self, item):
        sample_info = self.filelist[item]
        
        img_path = sample_info['image_path']
        depth_path = sample_info['depth_path']
        intrinsics_path = sample_info.get('intrinsics_path')
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Validate image dimensions
        if len(image.shape) < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(
                f"Invalid image dimensions: shape={image.shape}, path={img_path}. "
                f"This usually indicates a corrupted or empty image file."
            )
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # Record original resolution before any resizing (Bug 1 fix)
        orig_h, orig_w = image.shape[:2]
        
        # Load depth (KITTI depth is stored as uint16, divide by 256 to get meters)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth: {depth_path}")
        depth = depth.astype('float32') / 256.0  # Convert to meters
        
        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})
        
        # Convert to tensors
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        
        # Create valid mask (KITTI max depth ~80m for outdoor)
        sample['valid_mask'] = (sample['depth'] > 0) & (sample['depth'] <= 80.0)
        
        # Store original image dimensions for correct intrinsics normalisation (Bug 1 fix)
        sample['original_size'] = torch.tensor([orig_h, orig_w], dtype=torch.long)
        
        # Load intrinsics (correspond to the original resolution stored above)
        if intrinsics_path and os.path.exists(intrinsics_path):
            intrinsic = self._load_intrinsics(intrinsics_path)
        else:
            intrinsic = self.default_intrinsic
        sample['intrinsics'] = torch.from_numpy(intrinsic).float()
        
        sample['image_path'] = img_path
        
        return sample
    
    def __len__(self):
        return len(self.filelist)

