"""
NYU Depth V2 dataset implementation.

This dataset contains indoor scenes with RGB images and depth maps
from a Microsoft Kinect sensor. The labeled subset contains 1,449
densely labeled pairs of aligned RGB and depth images.

The data is stored in a MATLAB .mat file (nyu_depth_v2_labeled.mat) containing:
    - images: RGB images (shape: [3, height, width, n_images] in HDF5 format)
    - depths: Depth maps (shape: [height, width, n_images] in HDF5 format)
    
Note: The .mat file uses HDF5 format (MATLAB v7.3), so data is transposed.
    - images: actual shape [n_images, width, height, 3] when loaded with h5py
    - depths: actual shape [n_images, width, height] when loaded with h5py

Camera Intrinsics (RGB camera, 640×480 resolution):
    fx = 518.8579
    fy = 519.4696
    cx = 325.5824
    cy = 253.7362

Reference:
    Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012).
    Indoor Segmentation and Support Inference from RGBD Images.
    In ECCV 2012.
    https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""

import os
import re
from typing import List, Optional
import numpy as np

from .base import BaseDataset, DatasetItem, DatasetConfig


# NYU Depth V2 camera intrinsic parameters (640x480 resolution)
NYU_INTRINSIC_FX = 518.8579
NYU_INTRINSIC_FY = 519.4696
NYU_INTRINSIC_CX = 325.5824
NYU_INTRINSIC_CY = 253.7362
NYU_BASE_WIDTH = 640
NYU_BASE_HEIGHT = 480


class NYUDataset(BaseDataset):
    """
    NYU Depth V2 dataset for depth estimation evaluation.
    
    This dataset provides RGB images with corresponding depth maps from the
    NYU Depth V2 labeled subset. The depth values are in meters, captured
    using a Microsoft Kinect sensor.
    
    The dataset is stored as a single .mat file (nyu_depth_v2_labeled.mat)
    containing all 1,449 labeled image pairs.
    
    Attributes:
        max_depth: Maximum valid depth value in meters (default: 10.0 for indoor scenes)
        _images: Cached RGB images from the .mat file
        _depths: Cached depth maps from the .mat file
        _mat_loaded: Flag indicating if the .mat file has been loaded
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize NYU Depth V2 dataset.
        
        Args:
            config: Dataset configuration
        """
        super().__init__(config)
        self.max_depth = 10.0  # meters (NYU indoor typical max ~10m)
        self._images = None
        self._depths = None
        self._mat_loaded = False
        self._mat_file_path = None
    
    def get_default_path(self) -> str:
        """Get default NYU Depth V2 dataset path."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'nyu')
    
    def _load_mat_file(self) -> None:
        """
        Load the NYU Depth V2 .mat file.
        
        Uses h5py to read the HDF5-format .mat file (MATLAB v7.3).
        Caches the images and depths arrays for efficient access.
        """
        if self._mat_loaded:
            return
        
        # Try to import h5py
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required to read NYU Depth V2 .mat files. "
                "Install it with: pip install h5py"
            )
        
        # Find the .mat file
        mat_file = os.path.join(self.dataset_path, 'nyu_depth_v2_labeled.mat')
        
        if not os.path.exists(mat_file):
            raise FileNotFoundError(
                f"NYU Depth V2 .mat file not found: {mat_file}\n"
                f"Please download the labeled subset from:\n"
                f"https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html\n"
                f"and place nyu_depth_v2_labeled.mat in: {self.dataset_path}"
            )
        
        self._mat_file_path = mat_file
        
        print(f"Loading NYU Depth V2 dataset from {mat_file}...")
        print("(This may take a moment for the first load)")
        
        # Load the .mat file using h5py
        with h5py.File(mat_file, 'r') as f:
            # HDF5/MATLAB v7.3 format stores data transposed
            # images: shape [3, height, width, n_images] in MATLAB
            # becomes [n_images, width, height, 3] when loaded with h5py
            # We need to transpose to get [n_images, height, width, 3]
            self._images = np.array(f['images'])
            self._depths = np.array(f['depths'])
        
        # Check the original shape
        original_img_shape = self._images.shape
        original_depth_shape = self._depths.shape
        print(f"NYU .mat file - Original image shape: {original_img_shape}, depth shape: {original_depth_shape}")
        
        # Transpose from HDF5 format to standard format
        # The exact transpose depends on how h5py loads the data
        # Expected final shape: [n_images, height, width, 3] for images
        # Expected final shape: [n_images, height, width] for depths
        
        # Try to determine the correct transpose based on shape
        if len(original_img_shape) == 4:
            # Find dimensions by size: n_images=1449, height=480, width=640, channels=3
            n_imgs_idx = None
            h_idx = None
            w_idx = None
            c_idx = None
            
            for i, size in enumerate(original_img_shape):
                if size == 1449:
                    n_imgs_idx = i
                elif size == 480:
                    h_idx = i
                elif size == 640:
                    w_idx = i
                elif size == 3:
                    c_idx = i
            
            if n_imgs_idx is not None and h_idx is not None and w_idx is not None and c_idx is not None:
                # Create transpose to get [n_images, height, width, channels]
                transpose_order = (n_imgs_idx, h_idx, w_idx, c_idx)
                self._images = np.transpose(self._images, transpose_order)
                print(f"Transposed images using order {transpose_order}")
            else:
                # Fallback to original transpose (may need adjustment)
                self._images = np.transpose(self._images, (3, 2, 1, 0))
                print(f"Using fallback transpose (3, 2, 1, 0)")
        else:
            raise ValueError(f"Unexpected image shape from NYU .mat file: {original_img_shape}")
        
        # Transpose depths similarly
        if len(original_depth_shape) == 3:
            # Find dimensions: n_images=1449, height=480, width=640
            n_imgs_idx = None
            h_idx = None
            w_idx = None
            
            for i, size in enumerate(original_depth_shape):
                if size == 1449:
                    n_imgs_idx = i
                elif size == 480:
                    h_idx = i
                elif size == 640:
                    w_idx = i
            
            if n_imgs_idx is not None and h_idx is not None and w_idx is not None:
                transpose_order = (n_imgs_idx, h_idx, w_idx)
                self._depths = np.transpose(self._depths, transpose_order)
                print(f"Transposed depths using order {transpose_order}")
            else:
                # Fallback
                self._depths = np.transpose(self._depths, (2, 1, 0))
                print(f"Using fallback transpose for depths (2, 1, 0)")
        else:
            raise ValueError(f"Unexpected depth shape from NYU .mat file: {original_depth_shape}")
        
        # Validate final shapes
        if len(self._images.shape) != 4 or self._images.shape[3] != 3:
            raise ValueError(
                f"Invalid final image shape: {self._images.shape}. "
                f"Expected [n_images, height, width, 3]"
            )
        if len(self._depths.shape) != 3:
            raise ValueError(
                f"Invalid final depth shape: {self._depths.shape}. "
                f"Expected [n_images, height, width]"
            )
        
        # Validate dimensions are reasonable
        n_images, height, width, channels = self._images.shape
        if height < 100 or height > 2000 or width < 100 or width > 2000:
            raise ValueError(
                f"Unreasonable image dimensions: {height}x{width}. "
                f"NYU images should be approximately 480x640. "
                f"This suggests incorrect transpose."
            )
        
        self._mat_loaded = True
        print(f"Loaded {len(self._images)} image pairs from NYU Depth V2 dataset")
        print(f"Final image shape: {self._images.shape}, depth shape: {self._depths.shape}")
    
    # Standard 654-image test split (0-based indices into the 1449-image .mat
    # file). Derived from the official NYU Depth V2 splits.mat (testNdxs - 1,
    # http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat).
    # This is the Eigen et al. test split used by most depth estimation papers
    # including DepthAnything V2. Using the full 1449 images gives different
    # (and not comparable) scores. Set split='test' or split='eigen_test' to
    # restrict to these indices; split='all' uses all 1449 images.
    EIGEN_TEST_INDICES = [
        0, 1, 8, 13, 14, 15, 16, 17, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 45, 46, 55, 56, 58, 59, 60, 61, 62, 75,
        76, 77, 78, 83, 84, 85, 86, 87, 88, 89, 90, 116, 117, 118, 124, 125,
        126, 127, 128, 130, 131, 132, 133, 136, 152, 153, 154, 166, 167,
        168, 170, 171, 172, 173, 174, 175, 179, 180, 181, 182, 183, 184,
        185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
        198, 199, 200, 201, 206, 207, 208, 209, 210, 211, 219, 220, 221,
        249, 263, 270, 271, 272, 278, 279, 280, 281, 282, 283, 284, 295,
        296, 297, 298, 299, 300, 301, 309, 310, 311, 314, 315, 316, 324,
        325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 350, 351, 354,
        355, 356, 357, 358, 359, 360, 361, 362, 363, 383, 384, 385, 386,
        387, 388, 389, 394, 395, 396, 410, 411, 412, 413, 429, 430, 431,
        432, 433, 434, 440, 441, 442, 443, 444, 445, 446, 447, 461, 462,
        463, 464, 465, 468, 469, 470, 471, 472, 473, 474, 475, 476, 507,
        508, 509, 510, 511, 512, 514, 515, 516, 517, 518, 519, 520, 521,
        522, 523, 524, 525, 530, 531, 532, 536, 537, 538, 548, 549, 550,
        554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566,
        567, 568, 569, 570, 578, 579, 580, 581, 582, 590, 591, 592, 593,
        602, 603, 604, 605, 606, 611, 612, 616, 617, 618, 619, 620, 632,
        633, 634, 635, 636, 637, 643, 644, 649, 650, 655, 656, 657, 662,
        663, 667, 668, 669, 670, 671, 672, 675, 676, 677, 678, 679, 680,
        685, 686, 687, 688, 689, 692, 693, 696, 697, 698, 705, 706, 707,
        708, 709, 710, 711, 712, 716, 717, 723, 724, 725, 726, 727, 730,
        731, 732, 733, 742, 743, 758, 759, 760, 761, 762, 763, 764, 765,
        766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778,
        779, 780, 781, 782, 783, 784, 785, 786, 799, 800, 801, 802, 803,
        809, 810, 811, 812, 813, 820, 821, 822, 832, 833, 834, 835, 836,
        837, 838, 839, 840, 841, 842, 843, 844, 845, 849, 850, 851, 856,
        857, 858, 859, 860, 861, 868, 869, 870, 905, 906, 907, 916, 917,
        918, 925, 926, 927, 931, 932, 933, 934, 944, 945, 946, 958, 959,
        960, 961, 964, 965, 966, 969, 970, 971, 972, 973, 974, 975, 976,
        990, 991, 992, 993, 994, 1000, 1001, 1002, 1003, 1009, 1010, 1011,
        1020, 1021, 1022, 1031, 1032, 1033, 1037, 1038, 1047, 1048, 1051,
        1052, 1056, 1057, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081,
        1082, 1083, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095,
        1097, 1098, 1099, 1100, 1101, 1102, 1103, 1105, 1106, 1107, 1108,
        1116, 1117, 1118, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129,
        1130, 1134, 1135, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150,
        1151, 1152, 1153, 1154, 1155, 1156, 1157, 1161, 1162, 1163, 1164,
        1165, 1166, 1169, 1170, 1173, 1174, 1175, 1178, 1179, 1180, 1181,
        1182, 1183, 1191, 1192, 1193, 1194, 1195, 1200, 1201, 1202, 1203,
        1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1215, 1216, 1217,
        1218, 1219, 1225, 1226, 1227, 1228, 1229, 1232, 1233, 1234, 1246,
        1247, 1248, 1249, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260,
        1261, 1262, 1263, 1264, 1274, 1275, 1276, 1277, 1278, 1279, 1284,
        1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1296,
        1297, 1298, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1313, 1314,
        1328, 1329, 1330, 1331, 1334, 1335, 1336, 1337, 1338, 1339, 1346,
        1347, 1348, 1352, 1353, 1354, 1355, 1363, 1364, 1367, 1368, 1383,
        1384, 1385, 1386, 1387, 1388, 1389, 1390, 1393, 1394, 1395, 1396,
        1397, 1398, 1399, 1400, 1406, 1407, 1408, 1409, 1410, 1411, 1412,
        1413, 1420, 1421, 1422, 1423, 1429, 1430, 1431, 1432, 1440, 1441,
        1442, 1443, 1444, 1445, 1446, 1447, 1448,
    ]

    def find_items(self) -> List[DatasetItem]:
        """
        Find all NYU Depth V2 image pairs.

        By default uses the standard Eigen 654-image test split (split='test'
        or split='eigen_test'). Use split='all' to iterate all 1449 images.
        Using the full 1449 images is NOT comparable to the DA2 paper results.

        Returns:
            List of DatasetItem objects, each representing an RGB/depth pair
        """
        # Load the .mat file if not already loaded
        self._load_mat_file()

        split = self.split.lower() if self.split else 'test'
        use_all = split in ('all', 'full', 'train')

        if use_all:
            indices = list(range(len(self._images)))
            split_label = 'all-1449'
        else:
            # Default: Eigen 654-image test split (papers-comparable)
            max_idx = len(self._images) - 1
            indices = [i for i in self.EIGEN_TEST_INDICES if i <= max_idx]
            split_label = 'eigen-test-654'

        items = []
        for idx in indices:
            item_id = f"nyu_{idx:05d}"
            image_path = f"__NYU_CACHE__:{idx}:image"
            gt_path = f"__NYU_CACHE__:{idx}:depth"
            metadata = {
                'index': idx,
                'intrinsic': self.get_default_intrinsic()
            }
            items.append(DatasetItem(
                item_id=item_id,
                image_path=image_path,
                gt_path=gt_path,
                camera_id='0',
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

        # Apply max_items limit
        if self.max_items is not None:
            items = items[:self.max_items]

        print(f"Found {len(items)} NYU Depth V2 image pairs (split: {split_label})")
        return items
    
    def load_image(self, item: DatasetItem) -> np.ndarray:
        """
        Load image from the cached data.
        
        Args:
            item: DatasetItem containing the index in metadata
        
        Returns:
            BGR image as numpy array with shape [H, W, 3] and dtype uint8
            (matches cv2.imread format for consistency with other datasets)
        """
        # Ensure mat file is loaded
        self._load_mat_file()
        
        # Get index from metadata
        idx = item.metadata['index']
        
        # Get the cached image (shape: [H, W, 3], RGB format)
        image_rgb = self._images[idx].astype(np.uint8)
        
        # Verify shape is correct
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"Invalid image shape: {image_rgb.shape}. Expected [H, W, 3]. "
                f"This may indicate an issue with the .mat file loading."
            )
        
        # Convert RGB to BGR to match cv2.imread format (which the models expect)
        import cv2
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """
        Load NYU Depth V2 depth map from cached data.
        
        The depth values are in meters. Invalid pixels (depth <= 0 or > max_depth)
        are marked as NaN.
        
        Args:
            gt_path: Path identifier (used to extract index for cached data)
            item: DatasetItem with metadata containing the index
        
        Returns:
            Ground truth depth map in meters (invalid pixels as NaN)
        """
        # Ensure mat file is loaded
        self._load_mat_file()
        
        # Get index from metadata
        idx = item.metadata['index']
        
        # Get depth from cached array
        depth = self._depths[idx].astype(np.float32)
        
        # Mark invalid pixels as NaN
        # Zero or negative depth is invalid
        depth[depth <= 0] = np.nan
        
        # Clip to max depth (values beyond max are unreliable for Kinect)
        depth[depth > self.max_depth] = np.nan
        
        return depth
    
    def get_output_subdir(self) -> str:
        """Get output subdirectory name."""
        return 'nyu'
    
    def load_intrinsic(self, item: DatasetItem) -> Optional[np.ndarray]:
        """
        Load camera intrinsic matrix for a NYU Depth V2 image.
        
        NYU Depth V2 uses fixed camera intrinsics for all images:
            fx = 518.8579
            fy = 519.4696
            cx = 325.5824
            cy = 253.7362
        
        for the default 640x480 resolution.
        
        Args:
            item: DatasetItem containing the image path and metadata
        
        Returns:
            3x3 numpy array with camera intrinsic matrix K:
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
        # Check if intrinsics are in metadata (set during find_items)
        if item.metadata and 'intrinsic' in item.metadata:
            intrinsic = item.metadata['intrinsic']
            if isinstance(intrinsic, np.ndarray) and intrinsic.shape == (3, 3):
                return intrinsic.astype(np.float32)
        
        # Return default NYU intrinsics
        return self.get_default_intrinsic()
    
    def get_default_intrinsic(self) -> np.ndarray:
        """
        Get default NYU Depth V2 intrinsic matrix.
        
        NYU Depth V2 camera parameters (640x480 resolution):
            fx = 518.8579 pixels
            fy = 519.4696 pixels
            cx = 325.5824 pixels
            cy = 253.7362 pixels
        
        Returns:
            3x3 numpy array with default camera intrinsic matrix
        """
        intrinsic = np.array([
            [NYU_INTRINSIC_FX, 0.0, NYU_INTRINSIC_CX],
            [0.0, NYU_INTRINSIC_FY, NYU_INTRINSIC_CY],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return intrinsic
    
    def get_intrinsic_for_size(self, width: int, height: int, 
                                base_intrinsic: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get camera intrinsic matrix scaled to a specific image size.
        
        Scales the intrinsic parameters proportionally based on the ratio
        between the target size and the base NYU resolution (640x480).
        
        Args:
            width: Target image width
            height: Target image height
            base_intrinsic: Optional base intrinsic to scale from 
                          (default: use get_default_intrinsic())
        
        Returns:
            3x3 numpy array with scaled camera intrinsic matrix
        """
        if base_intrinsic is None:
            base_intrinsic = self.get_default_intrinsic()
        
        # Scale from base resolution
        scale_x = width / NYU_BASE_WIDTH
        scale_y = height / NYU_BASE_HEIGHT
        
        scaled_intrinsic = base_intrinsic.copy()
        scaled_intrinsic[0, 0] *= scale_x  # fx
        scaled_intrinsic[1, 1] *= scale_y  # fy
        scaled_intrinsic[0, 2] *= scale_x  # cx
        scaled_intrinsic[1, 2] *= scale_y  # cy
        
        return scaled_intrinsic
    
    def has_intrinsics(self) -> bool:
        """NYU Depth V2 provides camera intrinsic parameters."""
        return True
    
    def is_cached_dataset(self) -> bool:
        """
        Indicates that this dataset uses cached data rather than file paths.
        
        Returns:
            True, indicating images should be loaded via load_image method
        """
        return True

