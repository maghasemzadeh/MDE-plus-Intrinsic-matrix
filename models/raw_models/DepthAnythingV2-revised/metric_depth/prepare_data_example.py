#!/usr/bin/env python3
"""
Example script to prepare your data for training.

This script shows how to organize your data and create file lists.
"""
import os
import numpy as np
import cv2


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


if __name__ == '__main__':
    example_usage()


