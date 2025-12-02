import cv2
import numpy as np
import torch
import os
import re
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
    
    # Try to extract the relative structure and map to local dataset
    # Original: /mnt/bn/liheyang/DepthDatasets/vKitti2/SceneXX/variant/frames/rgb/Camera_X/rgb_XXXXX.jpg
    # Target: datasets/raw_data/vkitti/vkitti_2.0.3_rgb/SceneXX/variant/frames/rgb/Camera_X/rgb_XXXXX.jpg
    
    # Find vKitti2 or vkitti in the path
    path_lower = path.lower()
    if 'vkitti' in path_lower or 'vkitti2' in path_lower:
        # Extract the part after vKitti2 or vkitti
        parts = [p for p in path.split('/') if p]  # Remove empty strings from split
        try:
            # Find the index of vKitti2 or vkitti
            vkitti_idx = None
            for i, part in enumerate(parts):
                if 'vkitti' in part.lower():
                    vkitti_idx = i
                    break
            
            if vkitti_idx is not None:
                # Get the relative path after vKitti2
                relative_parts = parts[vkitti_idx + 1:]
                
                # Determine if it's RGB or depth based on directory structure
                # Path structure: SceneXX/variant/frames/rgb/... or SceneXX/variant/frames/depth/...
                if len(relative_parts) >= 3 and 'frames' in relative_parts:
                    frames_idx = relative_parts.index('frames')
                    if frames_idx + 1 < len(relative_parts):
                        if relative_parts[frames_idx + 1].lower() == 'rgb':
                            base_name = 'vkitti_2.0.3_rgb'
                        elif relative_parts[frames_idx + 1].lower() == 'depth':
                            base_name = 'vkitti_2.0.3_depth'
                        else:
                            # Fallback: check path string
                            if 'rgb' in path_lower:
                                base_name = 'vkitti_2.0.3_rgb'
                            elif 'depth' in path_lower:
                                base_name = 'vkitti_2.0.3_depth'
                            else:
                                base_name = 'vkitti_2.0.3_textgt'
                    else:
                        # Fallback: check path string
                        if 'rgb' in path_lower:
                            base_name = 'vkitti_2.0.3_rgb'
                        elif 'depth' in path_lower:
                            base_name = 'vkitti_2.0.3_depth'
                        else:
                            base_name = 'vkitti_2.0.3_textgt'
                else:
                    # Fallback: check path string
                    if 'rgb' in path_lower:
                        base_name = 'vkitti_2.0.3_rgb'
                    elif 'depth' in path_lower:
                        base_name = 'vkitti_2.0.3_depth'
                    else:
                        base_name = 'vkitti_2.0.3_textgt'
                
                # Try multiple possible base paths
                if project_root is None:
                    # Try to infer project root from current file location
                    current_file = os.path.abspath(__file__)
                    # Go up: vkitti2.py -> dataset -> metric_depth -> DepthAnythingV2-revised -> raw_models -> models -> project_root
                    # That's 6 levels up
                    project_root = os.path.abspath(os.path.join(current_file, '..', '..', '..', '..', '..', '..'))
                
                possible_bases = [
                    os.path.join(project_root, 'datasets', 'raw_data', 'vkitti'),
                    os.path.expanduser('~/datasets/vkitti'),
                    os.path.expanduser('~/data/vkitti'),
                ]
                
                # Filter out None values and check if base directories exist
                valid_bases = []
                for base in possible_bases:
                    if base and os.path.exists(base):
                        valid_bases.append(base)
                
                for base in valid_bases:
                    local_path = os.path.join(base, base_name, *relative_parts)
                    if os.path.exists(local_path):
                        return local_path
                
                # If still not found, try to construct path and return it anyway (will show better error)
                # Use the first valid base if available
                if valid_bases:
                    return os.path.join(valid_bases[0], base_name, *relative_parts)
        except Exception:
            pass
    
    # If we can't resolve it, return original (will fail with clear error)
    return path


class VKITTI2(Dataset):
    """
    VKITTI2 dataset for depth estimation training.
    
    File list format (space-separated):
        image_path depth_path [intrinsics_path]
    
    If intrinsics_path is provided, it should be a .npy file containing a 3x3 numpy array.
    """
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        
        # Try to infer project root for path resolution
        current_file = os.path.abspath(__file__)
        # Go up: vkitti2.py -> dataset -> metric_depth -> DepthAnythingV2-revised -> raw_models -> models -> project_root
        # That's 6 levels up
        self.project_root = os.path.abspath(os.path.join(current_file, '..', '..', '..', '..', '..', '..'))
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        # Filter out empty lines
        self.filelist = [line.strip() for line in self.filelist if line.strip()]
        
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
        line = self.filelist[item].split()
        
        if len(line) < 2:
            raise ValueError(f"Invalid file list line: {self.filelist[item]}. Expected at least 2 paths (image, depth)")
        
        img_path = resolve_path(line[0], self.project_root)
        depth_path = resolve_path(line[1], self.project_root)
        intrinsics_path = resolve_path(line[2], self.project_root) if len(line) > 2 else None
        
        image = cv2.imread(img_path)
        if image is None:
            # Try to find the actual dataset location
            possible_locations = [
                os.path.join(self.project_root, 'datasets', 'raw_data', 'vkitti'),
                os.path.expanduser('~/datasets/vkitti'),
                os.path.expanduser('~/data/vkitti'),
            ]
            existing_locations = [loc for loc in possible_locations if os.path.exists(loc)]
            
            error_msg = f"Could not load image: {img_path}\n"
            error_msg += f"  Original path: {line[0]}\n"
            error_msg += f"  Resolved path: {img_path}\n"
            error_msg += f"  Project root: {self.project_root}\n"
            error_msg += f"  File exists: {os.path.exists(img_path)}\n"
            if existing_locations:
                error_msg += f"  Found dataset at: {existing_locations[0]}\n"
                # Show what the path should be
                if 'vkitti' in line[0].lower():
                    parts = line[0].split('/')
                    vkitti_idx = None
                    for i, part in enumerate(parts):
                        if 'vkitti' in part.lower():
                            vkitti_idx = i
                            break
                    if vkitti_idx is not None:
                        relative_parts = parts[vkitti_idx + 1:]
                        if 'rgb' in line[0].lower() or line[0].endswith(('.jpg', '.png')):
                            expected_path = os.path.join(existing_locations[0], 'vkitti_2.0.3_rgb', *relative_parts)
                            error_msg += f"  Expected path: {expected_path}\n"
                            error_msg += f"  Expected path exists: {os.path.exists(expected_path)}\n"
            else:
                error_msg += f"  No dataset found in expected locations: {possible_locations}\n"
            raise ValueError(error_msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Could not load depth: {depth_path}\n  Original path: {line[1]}\n  Resolved path: {depth_path}\n  Project root: {self.project_root}\n  File exists: {os.path.exists(depth_path)}")
        depth = depth / 100.0  # cm to m
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        
        sample['valid_mask'] = (sample['depth'] <= 80) & (sample['depth'] > 0)
        
        # Load intrinsics if provided
        # Note: We don't include 'intrinsics' key if None to avoid DataLoader collation issues
        # The training code uses sample.get('intrinsics', None) which handles missing keys
        if intrinsics_path and os.path.exists(intrinsics_path):
            try:
                if intrinsics_path.endswith('.npy'):
                    intrinsics = np.load(intrinsics_path)
                    if intrinsics is not None and intrinsics.shape == (3, 3):
                        sample['intrinsics'] = torch.from_numpy(intrinsics).float()
                else:
                    # Try to load from text file (intrinsic.txt) - simplified version
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
            except Exception as e:
                # If loading fails, don't include intrinsics (will use None in training)
                pass
        
        sample['image_path'] = img_path
        
        return sample

    def __len__(self):
        return len(self.filelist)
