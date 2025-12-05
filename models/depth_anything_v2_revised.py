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

# Remove any conflicting paths (DepthAnythingV2 original) to ensure we use revised
_original_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'raw_models', 'DepthAnythingV2')
if _original_path in sys.path:
    sys.path.remove(_original_path)

# Use importlib to explicitly load from file paths, bypassing Python's import resolution
# This ensures we always get the modules from DepthAnythingV2-revised
import importlib.util

# Clear ALL cached depth_anything_v2 modules to ensure clean import
_modules_to_clear = [mod_name for mod_name in list(sys.modules.keys()) 
                     if mod_name.startswith('depth_anything_v2')]
for mod_name in _modules_to_clear:
    del sys.modules[mod_name]

# Ensure revised path is first and original is removed
if _original_path in sys.path:
    sys.path.remove(_original_path)
if _raw_models_path in sys.path:
    sys.path.remove(_raw_models_path)
sys.path.insert(0, _raw_models_path)

# CRITICAL: Pre-load modules in dependency order so sub-imports resolve correctly
# 1. Load config first (needed by models)
_config_path = os.path.join(_raw_models_path, 'depth_anything_v2', 'config.py')
_config_spec = importlib.util.spec_from_file_location("depth_anything_v2.config", _config_path)
_config_module = importlib.util.module_from_spec(_config_spec)
sys.modules['depth_anything_v2.config'] = _config_module
_config_spec.loader.exec_module(_config_module)
get_device = _config_module.get_device

# 2. Pre-load the actual model classes before models/__init__.py registers them
# This ensures they're from the revised path
_models_dir = os.path.join(_raw_models_path, 'depth_anything_v2', 'models')
_base_model_path = os.path.join(_models_dir, 'base.py')
_basic_model_path = os.path.join(_models_dir, 'depth_anything_v2_basic.py')
_metric_model_path = os.path.join(_models_dir, 'depth_anything_v2_metric.py')
_registry_path = os.path.join(_models_dir, 'registry.py')

# Load base and registry first (dependencies)
_base_spec = importlib.util.spec_from_file_location("depth_anything_v2.models.base", _base_model_path)
_base_module = importlib.util.module_from_spec(_base_spec)
sys.modules['depth_anything_v2.models.base'] = _base_module
_base_spec.loader.exec_module(_base_module)

_registry_spec = importlib.util.spec_from_file_location("depth_anything_v2.models.registry", _registry_path)
_registry_module = importlib.util.module_from_spec(_registry_spec)
sys.modules['depth_anything_v2.models.registry'] = _registry_module
_registry_spec.loader.exec_module(_registry_module)

# Load model implementations
_basic_spec = importlib.util.spec_from_file_location("depth_anything_v2.models.depth_anything_v2_basic", _basic_model_path)
_basic_module = importlib.util.module_from_spec(_basic_spec)
sys.modules['depth_anything_v2.models.depth_anything_v2_basic'] = _basic_module
_basic_spec.loader.exec_module(_basic_module)

_metric_spec = importlib.util.spec_from_file_location("depth_anything_v2.models.depth_anything_v2_metric", _metric_model_path)
_metric_module = importlib.util.module_from_spec(_metric_spec)
sys.modules['depth_anything_v2.models.depth_anything_v2_metric'] = _metric_module
_metric_spec.loader.exec_module(_metric_module)

# 3. Now load models/__init__.py (it will import the already-loaded model classes)
_models_init_path = os.path.join(_models_dir, '__init__.py')
_models_init_spec = importlib.util.spec_from_file_location("depth_anything_v2.models", _models_init_path)
_models_init_module = importlib.util.module_from_spec(_models_init_spec)
sys.modules['depth_anything_v2.models'] = _models_init_module
_models_init_spec.loader.exec_module(_models_init_module)

# 4. Now load model_loader (it will import models, which is already loaded from revised path)
_model_loader_path = os.path.join(_raw_models_path, 'depth_anything_v2', 'model_loader.py')
_loader_spec = importlib.util.spec_from_file_location("depth_anything_v2.model_loader", _model_loader_path)
_model_loader_module = importlib.util.module_from_spec(_loader_spec)
sys.modules['depth_anything_v2.model_loader'] = _model_loader_module
_loader_spec.loader.exec_module(_model_loader_module)
load_model = _model_loader_module.load_model

# Verify all critical modules loaded from correct path
_ml_file = getattr(_model_loader_module, '__file__', '')
_models_file = getattr(_models_init_module, '__file__', '')
_metric_file = getattr(_metric_module, '__file__', '')
_config_file = getattr(_config_module, '__file__', '')

if not _ml_file or 'DepthAnythingV2-revised' not in _ml_file:
    raise ImportError(f"model_loader loaded from wrong path! Got: {_ml_file}")
if not _models_file or 'DepthAnythingV2-revised' not in _models_file:
    raise ImportError(f"models module loaded from wrong path! Got: {_models_file}")
if not _metric_file or 'DepthAnythingV2-revised' not in _metric_file:
    raise ImportError(f"metric model loaded from wrong path! Got: {_metric_file}")
if not _config_file or 'DepthAnythingV2-revised' not in _config_file:
    raise ImportError(f"config module loaded from wrong path! Got: {_config_file}")


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
        
        # Before loading, verify models module is correct
        import depth_anything_v2.models as _models_check
        _models_file = getattr(_models_check, '__file__', '')
        if _models_file and 'DepthAnythingV2-revised' not in _models_file:
            raise ImportError(
                f"Cannot load model: models module is from wrong path! "
                f"Expected DepthAnythingV2-revised, but got: {_models_file}"
            )
        
        self._model = load_model(
            model_name=self.model_type,
            encoder=self.encoder,
            checkpoint_path=self.checkpoint_path,
            device=device,
            max_depth=self.max_depth,
            use_camera_intrinsics=self.use_camera_intrinsics,
            cam_token_inject_layer=self.cam_token_inject_layer
        )
        
        # Verify the loaded model class is from the correct path
        model_class_file = getattr(self._model.__class__, '__module__', '')
        if hasattr(self._model.__class__, '__module__'):
            # Check if the module file is from the correct path
            import importlib
            try:
                model_module = importlib.import_module(self._model.__class__.__module__)
                model_module_file = getattr(model_module, '__file__', '')
                if model_module_file and 'DepthAnythingV2-revised' not in model_module_file:
                    raise ImportError(
                        f"Loaded model class is from wrong path! "
                        f"Expected DepthAnythingV2-revised, but got: {model_module_file}"
                    )
            except Exception:
                pass  # If we can't verify, continue anyway
        
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

