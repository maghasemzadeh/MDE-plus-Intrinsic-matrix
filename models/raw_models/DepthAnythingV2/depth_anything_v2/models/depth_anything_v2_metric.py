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
        **kwargs
    ):
        """
        Initialize metric Depth Anything V2 model.
        
        Args:
            encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
            max_depth: Maximum depth in meters (20 for indoor, 80 for outdoor)
            checkpoint_path: Path to checkpoint file (if None, auto-detects)
            device: Device to load model on (if None, auto-detects)
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
        
        # Build model
        model_config = get_model_config(encoder)
        self.model = self._build_model(**model_config, max_depth=max_depth)
        
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
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def infer_image(
        self,
        image: np.ndarray,
        input_size: int = 518,
        **kwargs
    ) -> np.ndarray:
        """
        Run inference on a single image.
        
        Args:
            image: Input image as numpy array (BGR format, uint8)
            input_size: Input size for model
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
        
        return self.model.infer_image(image, input_size)

