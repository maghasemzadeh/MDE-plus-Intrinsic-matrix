#!/usr/bin/env python3
"""
Script to prepare VKITTI dataset for training.

This script creates file lists for VKITTI dataset following the structure:
- RGB: vkitti/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg
- Depth: vkitti/vkitti_2.0.3_depth/Scene01/15-deg-left/frames/depth/Camera_0/depth_00000.png
- Intrinsics: vkitti/vkitti_2.0.3_textgt/Scene01/15-deg-left/intrinsic.txt
"""

import os
import re
import numpy as np
from glob import glob
from pathlib import Path


def load_intrinsic_matrix(intrinsic_file: str) -> np.ndarray:
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


def create_vkitti_file_list(
    vkitti_root: str,
    output_file: str,
    split: str = 'train',
    include_intrinsics: bool = True,
    intrinsics_output_dir: str = None
):
    """
    Create a file list for VKITTI dataset.
    
    Args:
        vkitti_root: Root directory containing vkitti_2.0.3_rgb, vkitti_2.0.3_depth, vkitti_2.0.3_textgt
        output_file: Output file path for the file list
        split: Dataset split ('train', 'val', 'test') - currently all data is used
        include_intrinsics: Whether to include intrinsics paths in file list
        intrinsics_output_dir: Directory to save converted intrinsics as .npy files (if None, uses intrinsic.txt directly)
    """
    rgb_base = os.path.join(vkitti_root, 'vkitti_2.0.3_rgb')
    depth_base = os.path.join(vkitti_root, 'vkitti_2.0.3_depth')
    textgt_base = os.path.join(vkitti_root, 'vkitti_2.0.3_textgt')
    
    # Check if directories exist
    if not os.path.exists(rgb_base):
        raise ValueError(f"VKITTI RGB directory not found: {rgb_base}")
    if not os.path.exists(depth_base):
        raise ValueError(f"VKITTI depth directory not found: {depth_base}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if intrinsics_output_dir:
        os.makedirs(intrinsics_output_dir, exist_ok=True)
    
    items = []
    
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
                
                # Get intrinsic matrix for this variant
                intrinsic_matrix = None
                intrinsic_path = None
                
                if include_intrinsics and variant_textgt_path and os.path.exists(variant_textgt_path):
                    intrinsic_txt = os.path.join(variant_textgt_path, 'intrinsic.txt')
                    if os.path.exists(intrinsic_txt):
                        intrinsic_matrix = load_intrinsic_matrix(intrinsic_txt)
                        if intrinsic_matrix is not None:
                            if intrinsics_output_dir:
                                # Save as .npy file
                                intrinsic_npy_name = f"{scene_name}_{variant_name}_{camera_dir}.npy"
                                intrinsic_path = os.path.join(intrinsics_output_dir, intrinsic_npy_name)
                                np.save(intrinsic_path, intrinsic_matrix)
                            else:
                                # Use .txt file directly
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


def main():
    """Main function with example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare VKITTI dataset file list for training')
    parser.add_argument('--vkitti-root', type=str, required=True,
                       help='Root directory containing vkitti_2.0.3_rgb, vkitti_2.0.3_depth, vkitti_2.0.3_textgt')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path for the file list (e.g., dataset/splits/vkitti2/train.txt)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split (currently all data is used)')
    parser.add_argument('--no-intrinsics', action='store_true',
                       help='Do not include intrinsics in file list')
    parser.add_argument('--intrinsics-output-dir', type=str, default=None,
                       help='Directory to save converted intrinsics as .npy files (optional)')
    
    args = parser.parse_args()
    
    create_vkitti_file_list(
        vkitti_root=args.vkitti_root,
        output_file=args.output,
        split=args.split,
        include_intrinsics=not args.no_intrinsics,
        intrinsics_output_dir=args.intrinsics_output_dir
    )


if __name__ == '__main__':
    main()

