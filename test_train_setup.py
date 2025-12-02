#!/usr/bin/env python3
"""
Test script to verify training setup without actually training.
This checks imports, paths, and basic functionality.
"""

import os
import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        print("✓ torch imported")
        
        import numpy as np
        print("✓ numpy imported")
        
        import cv2
        print("✓ cv2 imported")
        
        # Test metric_depth imports
        _metric_depth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         'models', 'raw_models', 'DepthAnythingV2-revised', 'metric_depth')
        if _metric_depth_path not in sys.path:
            sys.path.insert(0, _metric_depth_path)
        
        from dataset.vkitti2 import VKITTI2
        print("✓ VKITTI2 imported")
        
        from depth_anything_v2.dpt import DepthAnythingV2
        print("✓ DepthAnythingV2 imported")
        
        from util.loss import SiLogLoss
        print("✓ SiLogLoss imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_paths():
    """Test that required paths exist."""
    print("\nTesting paths...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Check VKITTI train file
    vkitti_train1 = os.path.join(project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'train.txt')
    vkitti_train2 = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised', 
                                 'metric_depth', 'dataset', 'splits', 'vkitti2', 'train.txt')
    
    if os.path.exists(vkitti_train1):
        print(f"✓ VKITTI train file found: {vkitti_train1}")
        return True
    elif os.path.exists(vkitti_train2):
        print(f"✓ VKITTI train file found: {vkitti_train2}")
        return True
    else:
        print(f"✗ VKITTI train file not found in either location:")
        print(f"  - {vkitti_train1}")
        print(f"  - {vkitti_train2}")
        return False

def test_model_creation():
    """Test that model can be created."""
    print("\nTesting model creation...")
    try:
        _metric_depth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         'models', 'raw_models', 'DepthAnythingV2-revised', 'metric_depth')
        if _metric_depth_path not in sys.path:
            sys.path.insert(0, _metric_depth_path)
        
        from depth_anything_v2.dpt import DepthAnythingV2
        
        model = DepthAnythingV2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024],
            max_depth=80.0,
            use_camera_intrinsics=True
        )
        print("✓ Model created successfully")
        print(f"  - Model has cam_encoder: {hasattr(model, 'cam_encoder')}")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Training Setup Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Paths", test_paths()))
    results.append(("Model Creation", test_model_creation()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed! Training should work.")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

