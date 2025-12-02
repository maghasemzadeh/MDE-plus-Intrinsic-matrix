"""
Generic dataset class that supports camera intrinsics.
Use this for training with your own data that includes camera intrinsics.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class GenericDatasetWithIntrinsics(Dataset):
    """
    Generic dataset for depth estimation with camera intrinsics support.
    
    File list format (space-separated):
        image_path depth_path [intrinsics_path]
    
    If intrinsics_path is provided, it should be a .npy file containing a 3x3 numpy array.
    If not provided, intrinsics will be None (model will work without them if use_camera_intrinsics=False).
    
    Example file list:
        /path/to/image1.jpg /path/to/depth1.npy /path/to/intrinsics1.npy
        /path/to/image2.jpg /path/to/depth2.npy /path/to/intrinsics2.npy
        /path/to/image3.jpg /path/to/depth3.npy
    """
    
    def __init__(self, filelist_path, mode='train', size=(518, 518), depth_scale=1.0):
        """
        Initialize dataset.
        
        Args:
            filelist_path: Path to text file with image/depth/intrinsics paths
            mode: 'train' or 'val'
            size: Target image size (width, height)
            depth_scale: Scale factor to convert depth to meters (e.g., 0.001 for mm, 1.0 for meters)
        """
        self.mode = mode
        self.size = size
        self.depth_scale = depth_scale
        
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
        """
        Get a sample from the dataset.
        
        Returns:
            dict with keys:
                - 'image': torch.Tensor (3, H, W) - normalized image
                - 'depth': torch.Tensor (H, W) - depth in meters
                - 'valid_mask': torch.Tensor (H, W) - boolean mask for valid depth
                - 'intrinsics': torch.Tensor (3, 3) or None - camera intrinsics matrix
                - 'image_path': str - path to image file
        """
        line = self.filelist[item].split()
        
        if len(line) < 2:
            raise ValueError(f"Invalid file list line: {self.filelist[item]}. Expected at least 2 paths (image, depth)")
        
        img_path = line[0]
        depth_path = line[1]
        intrinsics_path = line[2] if len(line) > 2 else None
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Load depth
        depth = self._load_depth(depth_path)
        
        # Apply transforms
        sample = self.transform({'image': image, 'depth': depth})
        
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float() * self.depth_scale
        
        # Create valid mask
        sample['valid_mask'] = (torch.isnan(sample['depth']) == 0) & (sample['depth'] > 0)
        sample['depth'][sample['valid_mask'] == 0] = 0
        
        # Load intrinsics if provided
        if intrinsics_path:
            intrinsics = np.load(intrinsics_path)
            if intrinsics.shape != (3, 3):
                raise ValueError(f"Intrinsics must be 3x3 matrix, got shape {intrinsics.shape} from {intrinsics_path}")
            sample['intrinsics'] = torch.from_numpy(intrinsics).float()
        else:
            sample['intrinsics'] = None
        
        sample['image_path'] = img_path
        
        return sample
    
    def _load_depth(self, depth_path):
        """
        Load depth from file. Supports multiple formats:
        - .npy files (numpy array)
        - .png/.jpg files (16-bit depth images)
        - .h5 files (h5py format)
        """
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        elif depth_path.endswith('.h5') or depth_path.endswith('.hdf5'):
            import h5py
            with h5py.File(depth_path, 'r') as f:
                # Try common keys
                if 'depth' in f:
                    depth = np.array(f['depth'])
                elif 'dataset' in f:
                    depth = np.array(f['dataset'])
                else:
                    # Use first dataset
                    depth = np.array(f[list(f.keys())[0]])
        elif depth_path.endswith('.png') or depth_path.endswith('.jpg'):
            # 16-bit depth image
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # If it's a 16-bit image, scale appropriately
            if depth.max() > 1000:
                depth = depth / 256.0  # Common scaling for 16-bit depth
        else:
            raise ValueError(f"Unsupported depth file format: {depth_path}")
        
        return depth
    
    def __len__(self):
        return len(self.filelist)


