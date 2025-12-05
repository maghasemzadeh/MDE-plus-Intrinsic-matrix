"""
Depth Anything V2 (original) model wrapper.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .base import BaseDepthModelWrapper
from .utils import identify_model_from_checkpoint

# Add raw_models to path to import depth_anything_v2
_raw_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'raw_models', 'DepthAnythingV2')

if _raw_models_path not in sys.path:
    sys.path.insert(0, _raw_models_path)

from depth_anything_v2.model_loader import load_model
from depth_anything_v2.config import get_device


def find_checkpoint(
    model_type: str = 'metric',
    encoder: str = 'vitl',
    explicit_checkpoint: Optional[str] = None,
    max_depth: Optional[float] = None
) -> Tuple[Optional[str], Dict]:
    """
    Find checkpoint for Depth Anything V2 (original) model.
    
    Args:
        model_type: Model type ('metric' or 'basic')
        encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
        explicit_checkpoint: Optional explicit checkpoint path override
        max_depth: Optional max depth hint
    
    Returns:
        Tuple of (checkpoint_path, model_config_dict)
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # If explicit checkpoint provided, use it
    if explicit_checkpoint:
        if os.path.isabs(explicit_checkpoint):
            checkpoint_path = explicit_checkpoint
        else:
            checkpoint_path = os.path.join(project_root, explicit_checkpoint)
        
        if os.path.exists(checkpoint_path):
            config = identify_model_from_checkpoint(checkpoint_path)
            return checkpoint_path, config
        else:
            raise FileNotFoundError(f"Explicit checkpoint not found: {explicit_checkpoint}")
    
    # Check in original v2 checkpoints
    checkpoints_dir = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2', 'checkpoints')
    
    # Try to find checkpoint based on model_type and encoder
    if model_type == 'metric':
        # Try metric checkpoints first
        for ckpt_name in [
            f'depth_anything_v2_metric_hypersim_{encoder}.pth',
            f'depth_anything_v2_metric_vkitti_{encoder}.pth',
        ]:
            checkpoint_path = os.path.join(checkpoints_dir, ckpt_name)
            if os.path.exists(checkpoint_path):
                config = identify_model_from_checkpoint(checkpoint_path)
                return checkpoint_path, config
        
        # Try any encoder
        for enc in ['vitl', 'vitb', 'vits', 'vitg']:
            for ckpt_name in [
                f'depth_anything_v2_metric_hypersim_{enc}.pth',
                f'depth_anything_v2_metric_vkitti_{enc}.pth',
            ]:
                checkpoint_path = os.path.join(checkpoints_dir, ckpt_name)
                if os.path.exists(checkpoint_path):
                    config = identify_model_from_checkpoint(checkpoint_path)
                    return checkpoint_path, config
    
    # Try basic checkpoint
    ckpt_name = f'depth_anything_v2_{encoder}.pth'
    checkpoint_path = os.path.join(checkpoints_dir, ckpt_name)
    if os.path.exists(checkpoint_path):
        config = identify_model_from_checkpoint(checkpoint_path)
        return checkpoint_path, config
    
    # Try any encoder for basic
    for enc in ['vitl', 'vitb', 'vits', 'vitg']:
        ckpt_name = f'depth_anything_v2_{enc}.pth'
        checkpoint_path = os.path.join(checkpoints_dir, ckpt_name)
        if os.path.exists(checkpoint_path):
            config = identify_model_from_checkpoint(checkpoint_path)
            return checkpoint_path, config
    
    # If we get here, no checkpoint was found
    raise FileNotFoundError(
        f"Could not find checkpoint for DepthAnythingV2 (model_type={model_type}, encoder={encoder}).\n"
        f"Please ensure checkpoints are available in: {checkpoints_dir}"
    )


class DepthAnythingV2Wrapper(BaseDepthModelWrapper):
    """
    Wrapper for Depth Anything V2 (original) models.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize Depth Anything V2 wrapper.
        
        Args:
            model_config: Configuration dict with keys:
                - model_type: 'metric' or 'basic'
                - encoder: 'vits', 'vitb', 'vitl', 'vitg'
                - checkpoint_path: Optional path to checkpoint (if None, auto-finds)
                - max_depth: Maximum depth for metric models (default: 20.0)
                - device: Device to use (None for auto-detect)
        """
        super().__init__(model_config)
        self.model_type = model_config.get('model_type', 'metric')
        self.encoder = model_config.get('encoder', 'vitl')
        self.max_depth = model_config.get('max_depth', 20.0)
        self.device = model_config.get('device', None)
        self._is_metric = self.model_type.lower() == 'metric'
        
        # Find checkpoint if not provided
        explicit_checkpoint = model_config.get('checkpoint_path', None)
        self.checkpoint_path, checkpoint_config = find_checkpoint(
            model_type=self.model_type,
            encoder=self.encoder,
            explicit_checkpoint=explicit_checkpoint,
            max_depth=self.max_depth
        )
        
        # Update config from checkpoint if needed
        if not explicit_checkpoint:
            self.model_type = checkpoint_config.get('model_type', self.model_type)
            self.encoder = checkpoint_config.get('encoder', self.encoder)
            if checkpoint_config.get('max_depth'):
                self.max_depth = checkpoint_config['max_depth']
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
    
    def get_checkpoint_path(self) -> Optional[str]:
        """Get the checkpoint path used by this model."""
        return self.checkpoint_path
