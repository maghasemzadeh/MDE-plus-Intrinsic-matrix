"""
Depth Anything V2 Basic Model (with scale ambiguity).
Wrapper around the basic Depth Anything V2 implementation.
"""
import os
import torch
import numpy as np
from typing import Optional

from .base import BaseDepthModel
from ..dpt import DepthAnythingV2
from ..config import get_model_config


class DepthAnythingV2BasicModel(BaseDepthModel):
    """
    Basic Depth Anything V2 model wrapper.
    This model has scale ambiguity and outputs relative depth.
    """
    
    def __init__(
        self,
        encoder: str = 'vitl',
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        use_camera_intrinsics: bool = False,
        cam_token_inject_layer: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize basic Depth Anything V2 model.
        
        Args:
            encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: Path to checkpoint file (if None, auto-detects)
            device: Device to load model on (if None, auto-detects)
            use_camera_intrinsics: Enable camera intrinsics support
            cam_token_inject_layer: Layer index to inject camera token (None = first layer)
            **kwargs: Additional configuration
        """
        super().__init__(
            model_name='depth_anything_v2_basic',
            encoder=encoder,
            is_metric=False,
            **kwargs
        )
        
        self.encoder = encoder
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_camera_intrinsics = use_camera_intrinsics
        self.cam_token_inject_layer = cam_token_inject_layer
        
        # Build model
        model_config = get_model_config(encoder)
        self.model = self._build_model(
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
        """Build the basic Depth Anything V2 model."""
        return DepthAnythingV2(**kwargs)
    
    def _load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """Load checkpoint weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Please download the basic Depth Anything V2 checkpoint."
            )
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
    
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
            Depth map as numpy array (relative depth, unitless)
        """
        if not self._is_loaded:
            # Auto-load checkpoint if not already loaded
            if self.checkpoint_path is None:
                checkpoint_name = f'depth_anything_v2_{self.encoder}.pth'
                checkpoints_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'checkpoints'
                )
                self.checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
            
            if os.path.exists(self.checkpoint_path):
                self.load_from_checkpoint(self.checkpoint_path, self.device)
            else:
                raise RuntimeError(
                    f"Model not loaded and checkpoint not found at {self.checkpoint_path}. "
                    f"Please load the model first or provide a valid checkpoint path."
                )
        
        return self.model.infer_image(image, input_size, intrinsics=intrinsics)

