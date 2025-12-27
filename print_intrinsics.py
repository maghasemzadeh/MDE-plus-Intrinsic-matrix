#!/usr/bin/env python3
"""
Script to analyze camera intrinsics from the VKITTI intrinsics folder.

This script loads all .npy files containing 3x3 camera intrinsic matrices,
identifies unique intrinsic matrices, and displays them along with which files use each one.
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict


def arrays_equal(arr1: np.ndarray, arr2: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if two numpy arrays are equal within tolerance.
    
    Args:
        arr1: First array
        arr2: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if arrays are equal within tolerance
    """
    return np.allclose(arr1, arr2, rtol=rtol, atol=atol)


def find_unique_intrinsics(intrinsics_dir: str):
    """
    Load all camera intrinsics and identify unique intrinsic matrices.
    
    Args:
        intrinsics_dir: Path to the directory containing .npy intrinsic files
    """
    intrinsics_path = Path(intrinsics_dir)
    
    if not intrinsics_path.exists():
        print(f"Error: Directory does not exist: {intrinsics_dir}")
        return
    
    # Get all .npy files in the directory
    npy_files = sorted(intrinsics_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {intrinsics_dir}")
        return
    
    print(f"Found {len(npy_files)} intrinsic files in {intrinsics_dir}\n")
    print("=" * 80)
    
    # Load all intrinsics
    loaded_intrinsics = []
    file_to_intrinsic = {}
    errors = []
    
    for npy_file in npy_files:
        try:
            # Load the intrinsic matrix
            intrinsics = np.load(npy_file)
            
            # Validate shape
            if intrinsics.shape != (3, 3):
                errors.append(f"⚠️  {npy_file.name}: shape {intrinsics.shape}, expected (3, 3)")
                continue
            
            # Convert to float32 for consistent comparison
            intrinsics = intrinsics.astype(np.float32)
            loaded_intrinsics.append(intrinsics)
            file_to_intrinsic[npy_file.name] = intrinsics
            
        except Exception as e:
            errors.append(f"❌ {npy_file.name}: {e}")
    
    if errors:
        print("Errors encountered while loading files:")
        for error in errors:
            print(f"  {error}")
        print()
    
    if not loaded_intrinsics:
        print("No valid intrinsic matrices found.")
        return
    
    # Find unique intrinsics
    unique_intrinsics = []
    intrinsic_to_files = defaultdict(list)
    
    for npy_file in npy_files:
        if npy_file.name not in file_to_intrinsic:
            continue
        
        current_intrinsic = file_to_intrinsic[npy_file.name]
        
        # Check if this intrinsic matches any existing unique one
        found_match = False
        for unique_idx, unique_intrinsic in enumerate(unique_intrinsics):
            if arrays_equal(current_intrinsic, unique_intrinsic):
                intrinsic_to_files[unique_idx].append(npy_file.name)
                found_match = True
                break
        
        # If no match found, add as new unique intrinsic
        if not found_match:
            unique_idx = len(unique_intrinsics)
            unique_intrinsics.append(current_intrinsic)
            intrinsic_to_files[unique_idx].append(npy_file.name)
    
    # Print summary
    print(f"Total files processed: {len(npy_files)}")
    print(f"Valid intrinsic matrices: {len(loaded_intrinsics)}")
    print(f"Number of unique intrinsic matrices: {len(unique_intrinsics)}\n")
    print("=" * 80)
    
    # Print each unique intrinsic matrix
    for unique_idx, unique_intrinsic in enumerate(unique_intrinsics):
        files_using_this = intrinsic_to_files[unique_idx]
        
        print(f"\n📷 Unique Intrinsic Matrix #{unique_idx + 1}")
        print("-" * 80)
        print(f"Number of files using this matrix: {len(files_using_this)}")
        print("\nIntrinsic Matrix (3x3):")
        print(unique_intrinsic)
        
        # Extract and display individual parameters
        fx = unique_intrinsic[0, 0]
        fy = unique_intrinsic[1, 1]
        cx = unique_intrinsic[0, 2]
        cy = unique_intrinsic[1, 2]
        
        print(f"\nExtracted Parameters:")
        print(f"  fx (focal length x): {fx:.6f}")
        print(f"  fy (focal length y): {fy:.6f}")
        print(f"  cx (principal point x): {cx:.6f}")
        print(f"  cy (principal point y): {cy:.6f}")
        
        print(f"\nFiles using this intrinsic matrix ({len(files_using_this)}):")
        # Print files in columns for better readability
        for i, filename in enumerate(sorted(files_using_this)):
            if i > 0 and i % 3 == 0:
                print()
            print(f"  {filename:<50}", end="")
        print()
    
    print("\n" + "=" * 80)
    print(f"\nSummary:")
    print(f"  Total files: {len(npy_files)}")
    print(f"  Valid intrinsics: {len(loaded_intrinsics)}")
    print(f"  Unique intrinsics: {len(unique_intrinsics)}")


if __name__ == "__main__":
    # Path to the intrinsics directory
    intrinsics_dir = "datasets/raw_data/vkitti/splits/intrinsics"
    
    # Convert to absolute path if needed
    if not os.path.isabs(intrinsics_dir):
        script_dir = Path(__file__).parent
        intrinsics_dir = script_dir / intrinsics_dir
    
    find_unique_intrinsics(intrinsics_dir)

