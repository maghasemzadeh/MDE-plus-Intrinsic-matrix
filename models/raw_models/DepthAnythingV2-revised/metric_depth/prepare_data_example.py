#!/usr/bin/env python3
"""
Example script to prepare your data for training.

This script shows how to organize your data and create file lists.
"""
import os
import re
import numpy as np
import cv2
from glob import glob


def create_intrinsics_from_fov(image_width, image_height, fov_degrees, output_path):
    """
    Create a camera intrinsics matrix from field of view.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_degrees: Field of view in degrees (horizontal)
        output_path: Path to save intrinsics .npy file
    """
    fov_rad = np.radians(fov_degrees)
    fx = fy = (image_width / 2.0) / np.tan(fov_rad / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    intrinsics = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    np.save(output_path, intrinsics)
    print(f"Created intrinsics: {output_path}")
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    return intrinsics


def create_intrinsics_from_focal_length(image_width, image_height, focal_length_pixels, output_path):
    """
    Create a camera intrinsics matrix from focal length.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        focal_length_pixels: Focal length in pixels
        output_path: Path to save intrinsics .npy file
    """
    fx = fy = focal_length_pixels
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    intrinsics = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    np.save(output_path, intrinsics)
    print(f"Created intrinsics: {output_path}")
    return intrinsics


def create_file_list(image_dir, depth_dir, intrinsics_dir=None, output_file='train.txt', 
                     intrinsics_format='from_image_size'):
    """
    Create a file list for training.
    
    Args:
        image_dir: Directory containing images
        depth_dir: Directory containing depth maps (.npy files)
        intrinsics_dir: Directory containing intrinsics (.npy files) or None
        output_file: Output file path for the file list
        intrinsics_format: How to handle missing intrinsics
            - 'from_image_size': Create intrinsics from image dimensions (assumes 60deg FOV)
            - 'skip': Skip samples without intrinsics
            - 'none': Don't include intrinsics path
    """
    # Get all images
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for img_name in image_files:
            base_name = os.path.splitext(img_name)[0]
            
            img_path = os.path.join(image_dir, img_name)
            depth_path = os.path.join(depth_dir, f"{base_name}.npy")
            
            # Check if depth exists
            if not os.path.exists(depth_path):
                print(f"Warning: Depth not found for {img_name}, skipping...")
                continue
            
            line = f"{img_path} {depth_path}"
            
            # Handle intrinsics
            if intrinsics_dir:
                intrinsics_path = os.path.join(intrinsics_dir, f"{base_name}.npy")
                
                if os.path.exists(intrinsics_path):
                    # Intrinsics file exists
                    line += f" {intrinsics_path}"
                elif intrinsics_format == 'from_image_size':
                    # Create intrinsics from image size
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        # Assume 60 degree FOV (common for cameras)
                        create_intrinsics_from_fov(w, h, 60.0, intrinsics_path)
                        line += f" {intrinsics_path}"
                    else:
                        print(f"Warning: Could not read image {img_path}")
                elif intrinsics_format == 'skip':
                    # Skip this sample
                    continue
                # If intrinsics_format == 'none', don't add intrinsics path
            
            f.write(line + '\n')
    
    print(f"Created file list: {output_file} with {len(image_files)} samples")


def load_vkitti_intrinsic_matrix(intrinsic_file: str) -> np.ndarray:
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
    except Exception as e:
        print(f"Warning: Could not load intrinsic from {intrinsic_file}: {e}")
        return None


def _find_vkitti_root(vkitti_root: str) -> str:
    """
    Find VKITTI root directory by trying multiple possible locations.
    
    Args:
        vkitti_root: User-provided path (can be relative or absolute)
        
    Returns:
        Absolute path to VKITTI root directory
        
    Raises:
        ValueError: If VKITTI root cannot be found
    """
    # If already absolute and exists, return it
    if os.path.isabs(vkitti_root):
        abs_path = os.path.abspath(vkitti_root)
        rgb_check = os.path.join(abs_path, 'vkitti_2.0.3_rgb')
        if os.path.exists(rgb_check):
            return abs_path
    
    # Try relative to current working directory
    possible_paths = [
        os.path.join(os.getcwd(), vkitti_root),
    ]
    
    # Try relative to script directory (go up to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # From metric_depth to project root: ../../../../ (4 levels up)
    # metric_depth -> DepthAnythingV2-revised -> raw_models -> models -> MDE plus Intrinsic matrix
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
    possible_paths.extend([
        os.path.join(project_root, vkitti_root),
        os.path.join(project_root, 'datasets', 'raw_data', 'vkitti'),
    ])
    
    # Try common locations
    possible_paths.extend([
        os.path.join(os.path.expanduser('~'), 'datasets', 'vkitti'),
        os.path.join(os.path.expanduser('~'), 'data', 'vkitti'),
    ])
    
    # Check each possible path
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        rgb_check = os.path.join(abs_path, 'vkitti_2.0.3_rgb')
        depth_check = os.path.join(abs_path, 'vkitti_2.0.3_depth')
        if os.path.exists(rgb_check) and os.path.exists(depth_check):
            return abs_path
    
    # If none found, raise error with helpful message
    raise ValueError(
        f"Could not find VKITTI root directory: {vkitti_root}\n"
        f"  Tried paths:\n"
        + "\n".join(f"    - {os.path.abspath(p)}" for p in possible_paths[:5]) +
        f"\n  Current working directory: {os.getcwd()}\n"
        f"  Please provide the correct absolute or relative path to the VKITTI dataset root."
    )


def create_vkitti_file_list(
    vkitti_root: str,
    output_file: str = None,
    include_intrinsics: bool = True,
    intrinsics_output_dir: str = None
):
    """
    Create a file list for VKITTI dataset.
    
    This function handles the VKITTI dataset structure:
    - RGB: vkitti/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg
    - Depth: vkitti/vkitti_2.0.3_depth/Scene01/15-deg-left/frames/depth/Camera_0/depth_00000.png
    - Intrinsics: vkitti/vkitti_2.0.3_textgt/Scene01/15-deg-left/intrinsic.txt
    
    Args:
        vkitti_root: Root directory containing vkitti_2.0.3_rgb, vkitti_2.0.3_depth, vkitti_2.0.3_textgt
                    Can be relative or absolute path. Will try to auto-detect if not found.
        output_file: Output file path for the file list (default: vkitti_root/splits/train.txt)
        include_intrinsics: Whether to include intrinsics paths in file list
        intrinsics_output_dir: Directory to save converted intrinsics as .npy files (default: vkitti_root/splits/intrinsics)
    """
    # Try to find the VKITTI root directory
    vkitti_root = _find_vkitti_root(vkitti_root)
    
    # Set default output paths relative to vkitti_root if not provided
    if output_file is None:
        output_file = os.path.join(vkitti_root, 'splits', 'train.txt')
    elif not os.path.isabs(output_file):
        # If relative path, make it relative to vkitti_root
        output_file = os.path.join(vkitti_root, output_file)
    
    if intrinsics_output_dir is None and include_intrinsics:
        intrinsics_output_dir = os.path.join(vkitti_root, 'splits', 'intrinsics')
    elif intrinsics_output_dir and not os.path.isabs(intrinsics_output_dir):
        # If relative path, make it relative to vkitti_root
        intrinsics_output_dir = os.path.join(vkitti_root, intrinsics_output_dir)
    
    rgb_base = os.path.join(vkitti_root, 'vkitti_2.0.3_rgb')
    depth_base = os.path.join(vkitti_root, 'vkitti_2.0.3_depth')
    textgt_base = os.path.join(vkitti_root, 'vkitti_2.0.3_textgt')
    
    # Check if directories exist with helpful error messages
    if not os.path.exists(vkitti_root):
        raise ValueError(
            f"VKITTI root directory not found: {vkitti_root}\n"
            f"  Current working directory: {os.getcwd()}\n"
            f"  Please provide the correct path to the VKITTI dataset root."
        )
    if not os.path.exists(rgb_base):
        raise ValueError(
            f"VKITTI RGB directory not found: {rgb_base}\n"
            f"  VKITTI root: {vkitti_root}\n"
            f"  Available directories: {', '.join(os.listdir(vkitti_root) if os.path.exists(vkitti_root) else [])}\n"
            f"  Expected structure: vkitti_root/vkitti_2.0.3_rgb/"
        )
    if not os.path.exists(depth_base):
        raise ValueError(
            f"VKITTI depth directory not found: {depth_base}\n"
            f"  VKITTI root: {vkitti_root}\n"
            f"  Available directories: {', '.join(os.listdir(vkitti_root) if os.path.exists(vkitti_root) else [])}\n"
            f"  Expected structure: vkitti_root/vkitti_2.0.3_depth/"
        )
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if intrinsics_output_dir:
        os.makedirs(intrinsics_output_dir, exist_ok=True)
    
    items = []
    
    # Find all scene directories
    scene_dirs = sorted([d for d in os.listdir(rgb_base) 
                       if os.path.isdir(os.path.join(rgb_base, d))])
    
    print(f"Found {len(scene_dirs)} scenes in VKITTI dataset")
    
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
                
                # Get intrinsic matrix for this variant
                intrinsic_matrix = None
                intrinsic_path = None
                
                if include_intrinsics and variant_textgt_path and os.path.exists(variant_textgt_path):
                    intrinsic_txt = os.path.join(variant_textgt_path, 'intrinsic.txt')
                    if os.path.exists(intrinsic_txt):
                        intrinsic_matrix = load_vkitti_intrinsic_matrix(intrinsic_txt)
                        if intrinsic_matrix is not None:
                            if intrinsics_output_dir:
                                # Save as .npy file
                                intrinsic_npy_name = f"{scene_name}_{variant_name}_{camera_dir}.npy"
                                intrinsic_path = os.path.join(intrinsics_output_dir, intrinsic_npy_name)
                                np.save(intrinsic_path, intrinsic_matrix)
                            else:
                                # Use .txt file directly (will need to be converted by dataset class)
                                intrinsic_path = intrinsic_txt
                
                for rgb_path in rgb_files:
                    # Extract frame number from filename
                    base_name = os.path.basename(rgb_path)
                    frame_match = re.search(r'rgb_(\d+)\.(jpg|png)', base_name)
                    if not frame_match:
                        continue
                    
                    frame_num = frame_match.group(1)
                    
                    # Find corresponding depth file
                    depth_file = os.path.join(camera_depth_path, f'depth_{frame_num}.png')
                    
                    if not os.path.exists(depth_file):
                        continue
                    
                    # Create file list entry
                    line = f"{rgb_path} {depth_file}"
                    
                    if include_intrinsics and intrinsic_path:
                        line += f" {intrinsic_path}"
                    
                    items.append(line)
    
    # Write file list
    with open(output_file, 'w') as f:
        for line in items:
            f.write(line + '\n')
    
    print(f"Created VKITTI file list: {output_file}")
    print(f"  Total samples: {len(items)}")
    print(f"  Scenes: {len(scene_dirs)}")
    if include_intrinsics:
        print(f"  Intrinsics: {'Included' if intrinsic_path else 'Not available'}")


def example_usage():
    """Example of how to use the data preparation functions."""
    
    # Example 1: Create file list with existing intrinsics
    create_file_list(
        image_dir='datasets/raw_data/my_dataset/images',
        depth_dir='datasets/raw_data/my_dataset/depths',
        intrinsics_dir='datasets/raw_data/my_dataset/intrinsics',
        output_file='dataset/splits/my_dataset/train.txt',
        intrinsics_format='skip'  # Skip samples without intrinsics
    )
    
    # Example 2: Create file list and generate intrinsics from image size
    create_file_list(
        image_dir='datasets/raw_data/my_dataset/images',
        depth_dir='datasets/raw_data/my_dataset/depths',
        intrinsics_dir='datasets/raw_data/my_dataset/intrinsics',
        output_file='dataset/splits/my_dataset/train.txt',
        intrinsics_format='from_image_size'  # Auto-generate intrinsics
    )
    
    # Example 3: Create file list without intrinsics
    create_file_list(
        image_dir='datasets/raw_data/my_dataset/images',
        depth_dir='datasets/raw_data/my_dataset/depths',
        intrinsics_dir=None,
        output_file='dataset/splits/my_dataset/train.txt',
        intrinsics_format='none'
    )
    
    # Example 4: Create intrinsics manually
    create_intrinsics_from_fov(
        image_width=1920,
        image_height=1080,
        fov_degrees=60.0,
        output_path='datasets/raw_data/my_dataset/intrinsics/camera_001.npy'
    )
    
    # Example 5: Prepare VKITTI dataset
    # This is the main function for VKITTI dataset preparation
    # Output files will be created in vkitti_root/splits/
    create_vkitti_file_list(
        vkitti_root='datasets/raw_data/vkitti',  # Path to VKITTI root directory
        output_file=None,  # None = default to vkitti_root/splits/train.txt
        include_intrinsics=True,  # Set to True if you want to use camera intrinsics
        intrinsics_output_dir=None  # None = default to vkitti_root/splits/intrinsics
    )


if __name__ == '__main__':
    import sys
    
    # For VKITTI dataset preparation, use:
    if len(sys.argv) > 1:
        vkitti_root = sys.argv[1]
        # Default output paths are relative to vkitti_root (will be set in create_vkitti_file_list)
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        intrinsics_dir = sys.argv[3] if len(sys.argv) > 3 else None
        include_intrinsics = sys.argv[4].lower() != 'false' if len(sys.argv) > 4 else True
        
        print(f"Preparing VKITTI dataset...")
        print(f"  VKITTI root: {vkitti_root}")
        print(f"  Output file: {output_file}")
        print(f"  Intrinsics output dir: {intrinsics_dir}")
        print(f"  Include intrinsics: {include_intrinsics}")
        print()
        
        create_vkitti_file_list(
            vkitti_root=vkitti_root,
            output_file=output_file,
            include_intrinsics=include_intrinsics,
            intrinsics_output_dir=intrinsics_dir if include_intrinsics else None
        )
    else:
        print("=" * 80)
        print("VKITTI Dataset Preparation Script")
        print("=" * 80)
        print()
        print("Usage:")
        print("  python prepare_data_example.py <vkitti_root> [output_file] [intrinsics_dir] [include_intrinsics]")
        print()
        print("Arguments:")
        print("  vkitti_root       : Path to VKITTI root directory (containing vkitti_2.0.3_rgb, vkitti_2.0.3_depth, vkitti_2.0.3_textgt)")
        print("                     Can be relative or absolute. Script will auto-detect common locations.")
        print("  output_file       : Output file path (default: <vkitti_root>/splits/train.txt)")
        print("                     If relative, will be created inside vkitti_root directory")
        print("  intrinsics_dir    : Directory to save intrinsics .npy files (default: <vkitti_root>/splits/intrinsics)")
        print("                     If relative, will be created inside vkitti_root directory")
        print("  include_intrinsics: Whether to include intrinsics (default: True, set to 'false' to disable)")
        print()
        print("Examples:")
        print("  # From project root (recommended):")
        print("  cd /path/to/project")
        print("  python models/raw_models/DepthAnythingV2-revised/metric_depth/prepare_data_example.py \\")
        print("      datasets/raw_data/vkitti")
        print()
        print("  # From metric_depth directory:")
        print("  cd models/raw_models/DepthAnythingV2-revised/metric_depth")
        print("  python prepare_data_example.py ../../../../datasets/raw_data/vkitti")
        print()
        print("  # With absolute path:")
        print("  python prepare_data_example.py /absolute/path/to/vkitti")
        print()
        print("  # With custom output path:")
        print("  python prepare_data_example.py datasets/raw_data/vkitti splits/train.txt")
        print()
        print("  # Without intrinsics:")
        print("  python prepare_data_example.py datasets/raw_data/vkitti splits/train.txt '' false")
        print()
        print("=" * 80)
        print()
        print("Other examples:")
        example_usage()


