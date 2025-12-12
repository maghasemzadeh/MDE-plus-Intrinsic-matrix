"""
Generic dataset class with camera intrinsics support for depth estimation training.

File list format (space-separated):
    image_path depth_path [intrinsics_path]

If intrinsics_path is provided, it should be:
  - A .npy file containing a 3x3 numpy array
  - A .txt file containing the intrinsic matrix (3x3 or fx,fy,cx,cy format)

The dataset supports flexible path resolution for cross-machine compatibility.
"""

import cv2
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


def resolve_path(path, project_root=None):
    """
    Resolve a path that might be from another system to the local dataset location.

    Args:
        path: Original path (may be absolute from another system)
        project_root: Root directory of the project (for finding datasets)

    Returns:
        Resolved local path if found, otherwise original path
    """
    # If path exists, use it
    if os.path.exists(path):
        return path

    # If it's not an absolute path, return as-is
    if not os.path.isabs(path):
        return path

    # Try to extract dataset name from path and remap
    path_lower = path.lower()

    # Common dataset patterns to match
    dataset_patterns = {
        'vkitti': ['vkitti', 'vkitti2', 'virtual_kitti'],
        'kitti': ['kitti', 'kitti_depth'],
        'nyu': ['nyu', 'nyuv2', 'nyu_depth'],
        'hypersim': ['hypersim'],
        'cityscapes': ['cityscapes'],
        'diode': ['diode'],
    }

    for dataset_name, patterns in dataset_patterns.items():
        for pattern in patterns:
            if pattern in path_lower:
                # Try to resolve to local dataset path
                parts = path.split(os.sep)

                # Find the dataset directory in the path
                dataset_idx = None
                for i, part in enumerate(parts):
                    if pattern in part.lower():
                        dataset_idx = i
                        break

                if dataset_idx is not None:
                    # Get relative path after the dataset directory
                    relative_parts = parts[dataset_idx + 1:]

                    if project_root is None:
                        # Infer project root from current file location
                        current_file = os.path.abspath(__file__)
                        # Go up: generic_with_intrinsics.py -> dataset -> metric_depth -> DepthAnythingV2-revised -> raw_models -> models -> project_root
                        project_root = os.path.abspath(os.path.join(current_file, '..', '..', '..', '..', '..', '..'))

                    possible_bases = [
                        os.path.join(project_root, 'datasets', 'raw_data', dataset_name),
                        os.path.expanduser(f'~/datasets/{dataset_name}'),
                        os.path.expanduser(f'~/data/{dataset_name}'),
                        f'/data/{dataset_name}',
                    ]

                    for base in possible_bases:
                        if os.path.exists(base):
                            # Try direct path
                            local_path = os.path.join(base, *relative_parts)
                            if os.path.exists(local_path):
                                return local_path

                            # Try with the original dataset folder included
                            local_path = os.path.join(base, parts[dataset_idx], *relative_parts)
                            if os.path.exists(local_path):
                                return local_path

    # If we can't resolve it, return original (will fail with clear error)
    return path


def load_intrinsics(intrinsics_path):
    """
    Load camera intrinsics from file.

    Supports:
    - .npy files containing 3x3 numpy array
    - .txt files with 3x3 matrix or fx,fy,cx,cy format

    Returns:
        3x3 numpy array or None if loading fails
    """
    if not intrinsics_path or not os.path.exists(intrinsics_path):
        return None

    try:
        if intrinsics_path.endswith('.npy'):
            intrinsics = np.load(intrinsics_path)
            if intrinsics.shape == (3, 3):
                return intrinsics.astype(np.float32)
        else:
            # Try to load from text file
            with open(intrinsics_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Extract numbers from line
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)

                if len(numbers) >= 9:
                    # Full 3x3 matrix
                    values = [float(n) for n in numbers[:9]]
                    return np.array(values, dtype=np.float32).reshape(3, 3)
                elif len(numbers) >= 4:
                    # fx, fy, cx, cy format
                    fx, fy, cx, cy = [float(n) for n in numbers[:4]]
                    return np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
    except Exception:
        pass

    return None


class GenericDatasetWithIntrinsics(Dataset):
    """
    Generic dataset for depth estimation with optional camera intrinsics support.

    File list format (space-separated):
        image_path depth_path [intrinsics_path]

    Supported depth formats:
        - PNG (16-bit or 8-bit)
        - EXR (requires cv2 compiled with EXR support)
        - NPY (numpy array)

    Args:
        filelist_path: Path to file containing list of samples
        mode: 'train' or 'val'
        size: Output image size (width, height)
        depth_scale: Factor to convert depth values to meters (default: 1.0)
        max_depth: Maximum valid depth value in meters (default: 80.0)
        min_depth: Minimum valid depth value in meters (default: 0.001)
    """

    def __init__(self, filelist_path, mode, size=(518, 518),
                 depth_scale=1.0, max_depth=80.0, min_depth=0.001):
        self.mode = mode
        self.size = size
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.min_depth = min_depth

        # Infer project root for path resolution
        current_file = os.path.abspath(__file__)
        self.project_root = os.path.abspath(os.path.join(current_file, '..', '..', '..', '..', '..', '..'))

        # Load file list
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()

        # Filter out empty lines and comments
        self.filelist = [line.strip() for line in self.filelist
                        if line.strip() and not line.strip().startswith('#')]

        if len(self.filelist) == 0:
            raise ValueError(f"No valid entries found in filelist: {filelist_path}")

        # Build transform pipeline
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

    def _load_depth(self, depth_path):
        """Load depth from various formats."""
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        elif depth_path.endswith('.exr'):
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is not None and len(depth.shape) == 3:
                depth = depth[:, :, 0]  # Take first channel
        else:
            # PNG or other image formats
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is not None:
                depth = depth.astype(np.float32)

        return depth

    def __getitem__(self, item):
        line = self.filelist[item].split()

        if len(line) < 2:
            raise ValueError(f"Invalid file list line: {self.filelist[item]}. "
                           f"Expected at least 2 paths (image, depth)")

        # Resolve paths (handles cross-machine path differences)
        img_path = resolve_path(line[0], self.project_root)
        depth_path = resolve_path(line[1], self.project_root)
        intrinsics_path = resolve_path(line[2], self.project_root) if len(line) > 2 else None

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}\n"
                           f"  Original path: {line[0]}\n"
                           f"  Resolved path: {img_path}\n"
                           f"  File exists: {os.path.exists(img_path)}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # Load depth
        depth = self._load_depth(depth_path)
        if depth is None:
            raise ValueError(f"Could not load depth: {depth_path}\n"
                           f"  Original path: {line[1]}\n"
                           f"  Resolved path: {depth_path}\n"
                           f"  File exists: {os.path.exists(depth_path)}")

        # Apply depth scale to convert to meters
        depth = depth * self.depth_scale

        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})

        # Convert to tensors
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()

        # Create valid mask
        sample['valid_mask'] = (sample['depth'] >= self.min_depth) & (sample['depth'] <= self.max_depth)

        # Load intrinsics if provided
        if intrinsics_path:
            intrinsics = load_intrinsics(intrinsics_path)
            if intrinsics is not None:
                sample['intrinsics'] = torch.from_numpy(intrinsics).float()

        sample['image_path'] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)
