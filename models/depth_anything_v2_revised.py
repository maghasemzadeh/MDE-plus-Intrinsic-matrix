"""
Depth Anything V2 (revised) model wrapper.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .base import BaseDepthModelWrapper
from .utils import identify_model_from_checkpoint

# Add raw_models to path to import depth_anything_v2
_raw_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'raw_models', 'DepthAnythingV2-revised')

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
    Find checkpoint for Depth Anything V2 (revised) model.
    
    Priority order:
    1. models/raw_models/DepthAnythingV2-revised/checkpoints/revised/ (trained models)
    2. models/raw_models/DepthAnythingV2-revised/checkpoints/ (pretrained)
    3. Project checkpoints directory (trained models)
    
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
    
    # Priority 1: Check in revised/checkpoints/revised/ folder (trained models)
    revised_checkpoints_dir = os.path.join(
        project_root, 
        'models', 
        'raw_models', 
        'DepthAnythingV2-revised', 
        'checkpoints', 
        'revised'
    )
    
    if os.path.isdir(revised_checkpoints_dir):
        # Check for best.pth or latest.pth first (trained models)
        best_path = os.path.join(revised_checkpoints_dir, 'best.pth')
        latest_path = os.path.join(revised_checkpoints_dir, 'latest.pth')
        
        if os.path.exists(best_path):
            config = identify_model_from_checkpoint(best_path)
            return best_path, config
        elif os.path.exists(latest_path):
            print(f"⚠️  Warning: best.pth not found in revised checkpoints, using latest.pth instead")
            config = identify_model_from_checkpoint(latest_path)
            return latest_path, config
        
        # Check for any .pth files in revised directory
        for file in os.listdir(revised_checkpoints_dir):
            if file.endswith('.pth'):
                checkpoint_path = os.path.join(revised_checkpoints_dir, file)
                config = identify_model_from_checkpoint(checkpoint_path)
                return checkpoint_path, config
    
    # Priority 2: Check in checkpoints/ root (pretrained models)
    checkpoints_dir = os.path.join(
        project_root, 
        'models', 
        'raw_models', 
        'DepthAnythingV2-revised', 
        'checkpoints'
    )
    
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
    
    # Priority 3: Check project checkpoints directory (trained models)
    project_checkpoints_dir = os.path.join(project_root, 'checkpoints')
    if os.path.isdir(project_checkpoints_dir):
        for subdir in os.listdir(project_checkpoints_dir):
            subdir_path = os.path.join(project_checkpoints_dir, subdir)
            if os.path.isdir(subdir_path):
                best_path = os.path.join(subdir_path, 'best.pth')
                latest_path = os.path.join(subdir_path, 'latest.pth')
                if os.path.exists(best_path):
                    config = identify_model_from_checkpoint(best_path)
                    print(f"Found trained da2-revised model in {subdir}")
                    return best_path, config
                elif os.path.exists(latest_path):
                    print(f"⚠️  Warning: best.pth not found for da2-revised (found in {subdir}), using latest.pth instead")
                    config = identify_model_from_checkpoint(latest_path)
                    return latest_path, config
    
    # If we get here, no checkpoint was found
    raise FileNotFoundError(
        f"Could not find checkpoint for DepthAnythingV2-revised (model_type={model_type}, encoder={encoder}).\n"
        f"Please ensure checkpoints are available in:\n"
        f"  - {revised_checkpoints_dir} (priority)\n"
        f"  - {checkpoints_dir}\n"
        f"  - {project_checkpoints_dir}"
    )


class DepthAnythingV2RevisedWrapper(BaseDepthModelWrapper):
    """
    Wrapper for Depth Anything V2 (revised) models.
    Supports camera intrinsics and trained checkpoints.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize Depth Anything V2 (revised) wrapper.
        
        Args:
            model_config: Configuration dict with keys:
                - model_type: 'metric' or 'basic'
                - encoder: 'vits', 'vitb', 'vitl', 'vitg'
                - checkpoint_path: Optional path to checkpoint (if None, auto-finds)
                - max_depth: Maximum depth for metric models (default: 20.0)
                - device: Device to use (None for auto-detect)
                - use_camera_intrinsics: Whether to use camera intrinsics (default: False)
                - cam_token_inject_layer: Layer to inject camera token (default: None)
        """
        super().__init__(model_config)
        self.model_type = model_config.get('model_type', 'metric')
        self.encoder = model_config.get('encoder', 'vitl')
        self.max_depth = model_config.get('max_depth', 20.0)
        self.device = model_config.get('device', None)
        self.use_camera_intrinsics = model_config.get('use_camera_intrinsics', False)
        self.cam_token_inject_layer = model_config.get('cam_token_inject_layer', None)
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
        """Load the Depth Anything V2 (revised) model."""
        if self._is_loaded:
            return
        
        device = get_device(self.device)
        
        self._model = load_model(
            model_name=self.model_type,
            encoder=self.encoder,
            checkpoint_path=self.checkpoint_path,
            device=device,
            max_depth=self.max_depth,
            use_camera_intrinsics=self.use_camera_intrinsics,
            cam_token_inject_layer=self.cam_token_inject_layer
        )
        
        # Store model metadata
        self._model._is_metric = self._is_metric
        self._is_loaded = True
    
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
            Depth map as numpy array
        """
        if not self._is_loaded:
            self.load_model()
        
        import torch
        with torch.no_grad():
            if self.use_camera_intrinsics and intrinsics is not None:
                return self._model.infer_image(image, input_size=input_size, intrinsics=intrinsics)
            else:
                return self._model.infer_image(image, input_size=input_size)
    
    def is_metric(self) -> bool:
        """Whether this model outputs metric depth."""
        return self._is_metric
    
    def get_model_name(self) -> str:
        """Get model name."""
        return f"DepthAnythingV2-revised-{self.model_type}-{self.encoder}"
    
    def get_checkpoint_path(self) -> Optional[str]:
        """Get the checkpoint path used by this model."""
        return self.checkpoint_path

