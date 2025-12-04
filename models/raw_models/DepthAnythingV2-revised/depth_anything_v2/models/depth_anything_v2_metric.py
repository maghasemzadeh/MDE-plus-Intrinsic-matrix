"""
Depth Anything V2 Metric Model (metric depth in meters).
Wrapper around the metric Depth Anything V2 implementation.
"""
import os
import torch
import numpy as np
from typing import Optional

from .base import BaseDepthModel
from ..config import get_model_config


class DepthAnythingV2MetricModel(BaseDepthModel):
    """
    Metric Depth Anything V2 model wrapper.
    This model outputs metric depth in meters (no scale ambiguity).
    """
    
    def __init__(
        self,
        encoder: str = 'vitl',
        max_depth: float = 20.0,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        use_camera_intrinsics: bool = False,
        cam_token_inject_layer: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize metric Depth Anything V2 model.
        
        Args:
            encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
            max_depth: Maximum depth in meters (20 for indoor, 80 for outdoor)
            checkpoint_path: Path to checkpoint file (if None, auto-detects)
            device: Device to load model on (if None, auto-detects)
            use_camera_intrinsics: Enable camera intrinsics support
            cam_token_inject_layer: Layer index to inject camera token (None = first layer)
            **kwargs: Additional configuration
        """
        super().__init__(
            model_name='depth_anything_v2_metric',
            encoder=encoder,
            max_depth=max_depth,
            is_metric=True,
            **kwargs
        )
        
        self.encoder = encoder
        self.max_depth = max_depth
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_camera_intrinsics = use_camera_intrinsics
        self.cam_token_inject_layer = cam_token_inject_layer
        
        # Build model
        model_config = get_model_config(encoder)
        self.model = self._build_model(
            max_depth=max_depth,
            use_camera_intrinsics=use_camera_intrinsics,
            cam_token_inject_layer=cam_token_inject_layer,
            **model_config
        )
        
        # Initialize model state
        self._is_loaded = False
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_from_checkpoint(checkpoint_path, device)
        elif device is not None:
            self.to(device)
            self.eval()
            self._is_loaded = True
    
    def _build_model(self, **kwargs) -> torch.nn.Module:
        """Build the metric Depth Anything V2 model."""
        # Import metric model (lazy import to avoid conflicts)
        try:
            from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ImportError(
                "Metric depth model not available. "
                "Make sure metric_depth module is accessible."
            )
        
        return DepthAnythingV2(**kwargs)
    
    def _load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """Load checkpoint weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Please download the metric depth checkpoint."
            )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Trained checkpoint format (from train.py)
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'pretrained' in list(checkpoint.keys())[0]:
            # Full model state dict
            state_dict = checkpoint
        else:
            # Assume it's a state dict
            state_dict = checkpoint
        
        # Filter out cam_encoder keys if they don't exist in model (for backward compatibility)
        model_state_dict = self.model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_dict[k] = v
        
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_dict, strict=False)
        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys missing from checkpoint (will use random init)")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def infer_image(
        self,
        image: np.ndarray,
        input_size: int = 518,
        intrinsics: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Run inference on a single image.
        
        Args:
            image: Input image as numpy array (BGR format, uint8)
            input_size: Input size for model
            intrinsics: Optional camera intrinsics (3, 3) numpy array
            **kwargs: Additional inference parameters
        
        Returns:
            Depth map as numpy array (metric depth in meters)
        """
        if not self._is_loaded:
            # Auto-load checkpoint if not already loaded
            if self.checkpoint_path is None:
                checkpoints_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'checkpoints'
                )
                
                # Try hypersim checkpoint first
                checkpoint_name = f'depth_anything_v2_metric_hypersim_{self.encoder}.pth'
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                
                if not os.path.exists(checkpoint_path):
                    # Try vkitti checkpoint
                    checkpoint_name = f'depth_anything_v2_metric_vkitti_{self.encoder}.pth'
                    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                    if os.path.exists(checkpoint_path) and self.max_depth == 20.0:
                        # Auto-adjust max_depth for outdoor model
                        self.max_depth = 80.0
                        self.config['max_depth'] = 80.0
                
                self.checkpoint_path = checkpoint_path
            
            if os.path.exists(self.checkpoint_path):
                self.load_from_checkpoint(self.checkpoint_path, self.device)
            else:
                raise RuntimeError(
                    f"Model not loaded and checkpoint not found at {self.checkpoint_path}. "
                    f"Please load the model first or provide a valid checkpoint path."
                )
        
        return self.model.infer_image(image, input_size, intrinsics=intrinsics)

