"""
Script to load and use trained checkpoint for inference.

This script demonstrates how to:
1. Load a trained checkpoint (best.pth or latest.pth)
2. Run inference on images
3. Save depth predictions
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Add metric_depth directory to path for imports
_metric_depth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'models', 'raw_models', 'DepthAnythingV2-revised', 'metric_depth')
if _metric_depth_path not in sys.path:
    sys.path.insert(0, _metric_depth_path)

from depth_anything_v2.dpt import DepthAnythingV2


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_model_from_checkpoint(
    checkpoint_path: str,
    encoder: str = 'vitl',
    max_depth: float = 20.0,
    use_camera_intrinsics: bool = False,
    cam_token_inject_layer: int = None,
    device: torch.device = None
):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (best.pth or latest.pth)
        encoder: Model encoder type ('vits', 'vitb', 'vitl', 'vitg')
        max_depth: Maximum depth value
        use_camera_intrinsics: Whether to use camera intrinsics
        cam_token_inject_layer: Layer index to inject camera token
        device: Device to load model on (auto-detected if None)
    
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = get_device()
    
    # Model configuration
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Create model
    model_kwargs = {
        **model_configs[encoder],
        'use_camera_intrinsics': use_camera_intrinsics,
        'cam_token_inject_layer': cam_token_inject_layer,
        'max_depth': max_depth
    }
    model = DepthAnythingV2(**model_kwargs)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Full checkpoint with 'model' key
        state_dict = checkpoint['model']
        if 'previous_best' in checkpoint:
            print(f"Checkpoint info:")
            print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best metrics: {checkpoint.get('previous_best', {})}")
    elif isinstance(checkpoint, dict) and 'pretrained' in list(checkpoint.keys())[0]:
        # Full model state dict
        state_dict = checkpoint
    else:
        # Assume it's a state dict
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Encoder: {encoder}, Max depth: {max_depth}")
    print(f"Camera intrinsics: {use_camera_intrinsics}")
    
    return model


def run_inference_on_image(
    model: DepthAnythingV2,
    image_path: str,
    output_path: str = None,
    input_size: int = 518,
    intrinsics: np.ndarray = None,
    device: torch.device = None
):
    """
    Run inference on a single image.
    
    Args:
        model: Loaded model
        image_path: Path to input image
        output_path: Path to save depth map (optional)
        input_size: Input size for model
        intrinsics: Optional camera intrinsics (3, 3) numpy array
        device: Device to run inference on
    
    Returns:
        Depth map as numpy array
    """
    if device is None:
        device = get_device()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing image: {image_path}")
    print(f"  Image shape: {image.shape}")
    
    # Run inference
    with torch.no_grad():
        depth = model.infer_image(image, input_size=input_size, intrinsics=intrinsics)
    
    print(f"  Depth map shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}] meters")
    
    # Save depth map if output path is provided
    if output_path is not None:
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Save both raw depth (as numpy) and colormap visualization
        np.save(output_path.replace('.png', '.npy'), depth)
        cv2.imwrite(output_path, depth_colormap)
        print(f"  Saved depth map to: {output_path}")
        print(f"  Saved raw depth to: {output_path.replace('.png', '.npy')}")
    
    return depth


def main():
    parser = argparse.ArgumentParser(description='Load checkpoint and run inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (best.pth or latest.pth)')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder type')
    parser.add_argument('--max-depth', type=float, default=20.0,
                        help='Maximum depth value')
    parser.add_argument('--use-camera-intrinsics', action='store_true',
                        help='Enable camera intrinsics support')
    parser.add_argument('--cam-token-inject-layer', type=int, default=None,
                        help='Layer index to inject camera token')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input image size for model')
    
    # Inference arguments
    parser.add_argument('--image', type=str,
                        help='Path to input image for inference')
    parser.add_argument('--image-dir', type=str,
                        help='Directory containing images for batch inference')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--intrinsics', type=str,
                        help='Path to camera intrinsics file (3x3 numpy array .npy file)')
    
    args = parser.parse_args()
    
    # Resolve checkpoint path
    if not os.path.isabs(args.checkpoint):
        project_root = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(project_root, args.checkpoint)
    else:
        checkpoint_path = args.checkpoint
    
    # Load model
    device = get_device()
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        encoder=args.encoder,
        max_depth=args.max_depth,
        use_camera_intrinsics=args.use_camera_intrinsics,
        cam_token_inject_layer=args.cam_token_inject_layer,
        device=device
    )
    
    # Load intrinsics if provided
    intrinsics = None
    if args.intrinsics:
        intrinsics = np.load(args.intrinsics)
        print(f"Loaded camera intrinsics from {args.intrinsics}")
        print(f"  Intrinsics shape: {intrinsics.shape}")
    
    # Run inference
    if args.image:
        # Single image inference
        output_path = os.path.join(args.output_dir, os.path.basename(args.image).replace('.jpg', '_depth.png').replace('.png', '_depth.png'))
        os.makedirs(args.output_dir, exist_ok=True)
        run_inference_on_image(
            model=model,
            image_path=args.image,
            output_path=output_path,
            input_size=args.input_size,
            intrinsics=intrinsics,
            device=device
        )
    
    elif args.image_dir:
        # Batch inference
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(args.image_dir) if f.lower().endswith(ext)])
        
        print(f"\nFound {len(image_files)} images in {args.image_dir}")
        
        for image_file in image_files:
            image_path = os.path.join(args.image_dir, image_file)
            output_path = os.path.join(args.output_dir, image_file.replace('.jpg', '_depth.png').replace('.png', '_depth.png').replace('.jpeg', '_depth.png'))
            
            try:
                run_inference_on_image(
                    model=model,
                    image_path=image_path,
                    output_path=output_path,
                    input_size=args.input_size,
                    intrinsics=intrinsics,
                    device=device
                )
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        print(f"\nBatch inference complete! Results saved to {args.output_dir}")
    
    else:
        print("\nNo image or image directory specified. Use --image or --image-dir to run inference.")
        print("\nExample usage:")
        print("  python load_checkpoint.py --checkpoint checkpoints/vkitti_training/best.pth \\")
        print("                             --image path/to/image.jpg --output-dir ./results")
        print("\n  python load_checkpoint.py --checkpoint checkpoints/vkitti_training/best.pth \\")
        print("                             --image-dir path/to/images --output-dir ./results")


if __name__ == '__main__':
    main()

