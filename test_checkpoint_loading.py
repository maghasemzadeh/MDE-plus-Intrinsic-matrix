"""
Test script to verify checkpoint loading and inference works correctly.

This script tests that checkpoints saved from train.py can be loaded
and used for inference in the model wrappers.
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models import create_model_wrapper, DepthAnythingV2RevisedWrapper


def test_checkpoint_loading(checkpoint_path: str, test_image_path: str = None):
    """
    Test loading a checkpoint and running inference.
    
    Args:
        checkpoint_path: Path to checkpoint file to test
        test_image_path: Optional path to test image (if None, creates dummy image)
    """
    print(f"Testing checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    # Try to load checkpoint to verify it's valid
    print("\n1. Testing checkpoint file loading...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("   ✅ Checkpoint file loaded successfully")
        
        # Check checkpoint format
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                print(f"   ✅ Checkpoint format: Training checkpoint (has 'model' key)")
                state_dict = checkpoint['model']
                print(f"   ✅ Model state dict contains {len(state_dict)} parameters")
                if 'epoch' in checkpoint:
                    print(f"   ✅ Epoch: {checkpoint['epoch']}")
                if 'previous_best' in checkpoint:
                    print(f"   ✅ Previous best metrics available")
            else:
                print(f"   ✅ Checkpoint format: State dict (direct)")
                state_dict = checkpoint
                print(f"   ✅ State dict contains {len(state_dict)} parameters")
        else:
            print(f"   ⚠️  Warning: Unexpected checkpoint format")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: Failed to load checkpoint file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try to identify model configuration from checkpoint
    print("\n2. Identifying model configuration...")
    try:
        from models.utils import identify_model_from_checkpoint
        config = identify_model_from_checkpoint(checkpoint_path)

        encoder = config.get('encoder', 'vitl')
        model_type = config.get('model_type', 'metric')
        max_depth = config.get('max_depth', 80.0)
        has_cam_encoder = config.get('use_camera_intrinsics', False)
        cam_token_inject_layer = config.get('cam_token_inject_layer', None)

        # Also check state dict directly for backward compatibility
        has_depth_head = any('depth_head' in k for k in state_dict.keys())
        has_cam_encoder_in_state = any('cam_encoder' in k for k in state_dict.keys())

        print(f"   ✅ Encoder: {encoder}")
        print(f"   ✅ Model type: {model_type}")
        print(f"   ✅ Max depth: {max_depth}")
        print(f"   ✅ Has depth_head: {has_depth_head}")
        print(f"   ✅ Use camera intrinsics: {has_cam_encoder}")
        print(f"   ✅ Has cam_encoder in state: {has_cam_encoder_in_state}")
        if cam_token_inject_layer is not None:
            print(f"   ✅ Cam token inject layer: {cam_token_inject_layer}")

    except Exception as e:
        print(f"   ⚠️  Warning: Could not identify model config: {e}")
        encoder = 'vitl'
        model_type = 'metric'
        max_depth = 80.0
        has_cam_encoder = any('cam_encoder' in k for k in state_dict.keys())
        cam_token_inject_layer = None

    # Try to create model wrapper and load checkpoint
    print("\n3. Testing model wrapper checkpoint loading...")
    try:
        model_config = {
            'model_type': model_type,
            'encoder': encoder,
            'checkpoint_path': checkpoint_path,
            'max_depth': max_depth,
            'use_camera_intrinsics': has_cam_encoder,
            'cam_token_inject_layer': cam_token_inject_layer,
        }

        model = create_model_wrapper('da2-revised', model_config)
        print(f"   ✅ Model wrapper created: {model.get_model_name()}")
        print(f"   ✅ Checkpoint path: {model.get_checkpoint_path()}")
        print(f"   ✅ Use camera intrinsics: {model.use_camera_intrinsics}")
        
    except Exception as e:
        print(f"   ❌ ERROR: Failed to create model wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try to run inference
    print("\n4. Testing inference...")
    try:
        # Create or load test image
        if test_image_path and os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            print(f"   ✅ Loaded test image: {test_image_path} ({image.shape})")
        else:
            # Create dummy test image
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            print(f"   ✅ Created dummy test image: {image.shape}")
        
        # Run inference
        print("   Running inference...")
        depth_map = model.infer_image(image, input_size=518)
        print(f"   ✅ Inference successful!")
        print(f"   ✅ Depth map shape: {depth_map.shape}")
        print(f"   ✅ Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        print(f"   ✅ Depth mean: {depth_map.mean():.3f}")
        
    except Exception as e:
        print(f"   ❌ ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED! Checkpoint can be loaded and used for inference.")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(description='Test checkpoint loading and inference')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file to test')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Optional path to test image (if not provided, uses dummy image)')
    
    args = parser.parse_args()
    
    success = test_checkpoint_loading(args.checkpoint, args.test_image)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

