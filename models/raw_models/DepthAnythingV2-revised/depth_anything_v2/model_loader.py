"""
Model loading utilities for Depth Anything V2.
Provides a unified interface for loading different depth models using the model registry.
"""
import os
from typing import Optional, Dict, Any, Union

from depth_anything_v2.config import get_device
from depth_anything_v2.models import get_model, BaseDepthModel, list_models


def load_model(
    model_name: str = 'basic',
    encoder: str = 'vitl',
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    max_depth: float = 20.0,
    checkpoints_dir: Optional[str] = None,
    use_camera_intrinsics: bool = False,
    cam_token_inject_layer: Optional[int] = None,
    **kwargs
) -> BaseDepthModel:
    """
    Unified interface for loading depth estimation models.
    Uses the model registry to load any registered model type.
    
    Args:
        model_name: Name of the model ('basic', 'metric', 'depth_anything_v2_basic', etc.)
        encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
        checkpoint_path: Path to checkpoint file. If None, auto-detects.
        device: Device to load model on. If None, auto-detects.
        max_depth: Maximum depth in meters for metric models (20 for indoor, 80 for outdoor)
        checkpoints_dir: Directory containing checkpoints. If None, uses 'checkpoints' folder.
        **kwargs: Additional model-specific parameters
    
    Returns:
        Loaded model instance (BaseDepthModel)
    
    Raises:
        ValueError: If model_name is not registered
        FileNotFoundError: If checkpoint file is not found
    
    Examples:
        >>> # Load basic model
        >>> model = load_model('basic', encoder='vitl')
        >>> 
        >>> # Load metric model
        >>> model = load_model('metric', encoder='vitl', max_depth=20.0)
        >>> 
        >>> # Load with custom checkpoint
        >>> model = load_model('basic', checkpoint_path='path/to/checkpoint.pth')
    """
    # Get model class from registry
    model_class = get_model(model_name)
    
    if model_class is None:
        available_models = list_models()
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {available_models}"
        )
    
    # Auto-detect checkpoint path if not provided
    if checkpoint_path is None:
        if checkpoints_dir is None:
            checkpoints_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'checkpoints'
            )
        
        if model_name in ['basic', 'depth_anything_v2_basic']:
            checkpoint_name = f'depth_anything_v2_{encoder}.pth'
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        elif model_name in ['metric', 'depth_anything_v2_metric']:
            # Try hypersim checkpoint first
            checkpoint_name = f'depth_anything_v2_metric_hypersim_{encoder}.pth'
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
            
            if not os.path.exists(checkpoint_path):
                # Try vkitti checkpoint
                checkpoint_name = f'depth_anything_v2_metric_vkitti_{encoder}.pth'
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                if os.path.exists(checkpoint_path) and max_depth == 20.0:
                    # Auto-adjust max_depth for outdoor model
                    max_depth = 80.0
    
    # Auto-detect device if not provided
    if device is None:
        device = get_device()
    
    # Create model instance
    if model_name in ['metric', 'depth_anything_v2_metric']:
        model = model_class(
            encoder=encoder,
            max_depth=max_depth,
            checkpoint_path=checkpoint_path,
            device=device,
            use_camera_intrinsics=use_camera_intrinsics,
            cam_token_inject_layer=cam_token_inject_layer,
            **kwargs
        )
    else:
        model = model_class(
            encoder=encoder,
            checkpoint_path=checkpoint_path,
            device=device,
            use_camera_intrinsics=use_camera_intrinsics,
            cam_token_inject_layer=cam_token_inject_layer,
            **kwargs
        )
    
    return model


# Backward compatibility functions
def load_basic_model(
    encoder: str = 'vitl',
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    checkpoints_dir: Optional[str] = None
) -> BaseDepthModel:
    """
    Load basic Depth Anything V2 model (with scale ambiguity).
    Backward compatibility wrapper around load_model().
    
    Args:
        encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
        checkpoint_path: Path to checkpoint file. If None, auto-detects.
        device: Device to load model on. If None, auto-detects.
        checkpoints_dir: Directory containing checkpoints. If None, uses 'checkpoints' folder.
    
    Returns:
        Loaded model instance
    """
    return load_model(
        model_name='basic',
        encoder=encoder,
        checkpoint_path=checkpoint_path,
        device=device,
        checkpoints_dir=checkpoints_dir
    )


def load_metric_model(
    encoder: str = 'vitl',
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    max_depth: float = 20.0,
    checkpoints_dir: Optional[str] = None
) -> BaseDepthModel:
    """
    Load metric Depth Anything V2 model (returns depth in meters, no scale ambiguity).
    Backward compatibility wrapper around load_model().
    
    Args:
        encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
        checkpoint_path: Path to checkpoint file. If None, auto-detects.
        device: Device to load model on. If None, auto-detects.
        max_depth: Maximum depth in meters (20 for indoor, 80 for outdoor)
        checkpoints_dir: Directory containing checkpoints. If None, uses 'checkpoints' folder.
    
    Returns:
        Loaded model instance
    """
    return load_model(
        model_name='metric',
        encoder=encoder,
        checkpoint_path=checkpoint_path,
        device=device,
        max_depth=max_depth,
        checkpoints_dir=checkpoints_dir
    )

