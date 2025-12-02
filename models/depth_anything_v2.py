"""
Depth Anything V2 model wrapper.
"""

import os
import sys
from typing import Dict, Any, Optional
import numpy as np

from .base import BaseDepthModelWrapper

# Add raw_models to path to import depth_anything_v2
# models/depth_anything_v2.py -> models/ -> models/raw_models/DepthAnythingV2
_raw_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'raw_models', 'DepthAnythingV2')
if _raw_models_path not in sys.path:
    sys.path.insert(0, _raw_models_path)

from depth_anything_v2.model_loader import load_model
from depth_anything_v2.config import get_device


class DepthAnythingV2Wrapper(BaseDepthModelWrapper):
    """
    Wrapper for Depth Anything V2 models.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize Depth Anything V2 wrapper.
        
        Args:
            model_config: Configuration dict with keys:
                - model_type: 'metric' or 'basic'
                - encoder: 'vits', 'vitb', 'vitl', 'vitg'
                - checkpoint_path: Optional path to checkpoint
                - max_depth: Maximum depth for metric models (default: 20.0)
                - device: Device to use (None for auto-detect)
        """
        super().__init__(model_config)
        self.model_type = model_config.get('model_type', 'metric')
        self.encoder = model_config.get('encoder', 'vitl')
        self.checkpoint_path = model_config.get('checkpoint_path', None)
        self.max_depth = model_config.get('max_depth', 20.0)
        self.device = model_config.get('device', None)
        self._is_metric = self.model_type.lower() == 'metric'
    
    def load_model(self) -> None:
        """Load the Depth Anything V2 model."""
        if self._is_loaded:
            return
        
        device = get_device(self.device)
        
        self._model = load_model(
            model_name=self.model_type,
            encoder=self.encoder,
            checkpoint_path=self.checkpoint_path,
            device=device,
            max_depth=self.max_depth
        )
        
        # Store model metadata
        self._model._is_metric = self._is_metric
        self._is_loaded = True
    
    def infer_image(
        self,
        image: np.ndarray,
        input_size: int = 518,
        **kwargs
    ) -> np.ndarray:
        """Run inference on a single image."""
        if not self._is_loaded:
            self.load_model()
        
        import torch
        with torch.no_grad():
            return self._model.infer_image(image, input_size=input_size)
    
    def is_metric(self) -> bool:
        """Whether this model outputs metric depth."""
        return self._is_metric
    
    def get_model_name(self) -> str:
        """Get model name."""
        return f"DepthAnythingV2-{self.model_type}-{self.encoder}"

