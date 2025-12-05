"""
Depth Anything 3 model wrapper.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

from .base import BaseDepthModelWrapper
from .utils import identify_model_from_checkpoint

# Add raw_models to path to import depth_anything_3
_raw_models_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'raw_models', 
    'Depth-Anything-3',
    'src'
)

if _raw_models_path not in sys.path:
    sys.path.insert(0, _raw_models_path)

try:
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False
    DepthAnything3 = None


def find_checkpoint(
    model_name: str = 'da3-large',
    explicit_checkpoint: Optional[str] = None
) -> Tuple[Optional[str], Dict]:
    """
    Find checkpoint for Depth Anything 3 model.
    
    DA3 uses model directories (not single checkpoint files).
    The model directory should contain model.safetensors and config.json.
    
    Args:
        model_name: Model preset name ('da3-large', 'da3-giant', etc.)
        explicit_checkpoint: Optional explicit model directory path override
    
    Returns:
        Tuple of (model_dir_path, model_config_dict)
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # If explicit checkpoint/directory provided, use it
    if explicit_checkpoint:
        if os.path.isabs(explicit_checkpoint):
            model_dir = explicit_checkpoint
        else:
            model_dir = os.path.join(project_root, explicit_checkpoint)
        
        if os.path.isdir(model_dir):
            # Check if it's a valid DA3 model directory
            safetensors_file = os.path.join(model_dir, 'model.safetensors')
            config_file = os.path.join(model_dir, 'config.json')
            if os.path.exists(safetensors_file) or os.path.exists(config_file):
                return model_dir, {'model_name': model_name, 'model_type': 'metric'}
        else:
            raise FileNotFoundError(f"Explicit model directory not found: {explicit_checkpoint}")
    
    # Check in DA3 checkpoints directory
    checkpoints_dir = os.path.join(
        project_root, 
        'models', 
        'raw_models', 
        'Depth-Anything-3', 
        'checkpoints'
    )
    
    if os.path.isdir(checkpoints_dir):
        # Look for model directories
        for item in os.listdir(checkpoints_dir):
            item_path = os.path.join(checkpoints_dir, item)
            if os.path.isdir(item_path):
                safetensors_file = os.path.join(item_path, 'model.safetensors')
                config_file = os.path.join(item_path, 'config.json')
                if os.path.exists(safetensors_file) or os.path.exists(config_file):
                    return item_path, {'model_name': model_name, 'model_type': 'metric'}
    
    # If we get here, no checkpoint was found
    raise FileNotFoundError(
        f"Could not find model directory for DepthAnything3 (model_name={model_name}).\n"
        f"Please ensure model directory is available in: {checkpoints_dir}\n"
        f"Or use HuggingFace Hub: DepthAnything3.from_pretrained('huggingface/model-name')"
    )


class DepthAnything3Wrapper(BaseDepthModelWrapper):
    """
    Wrapper for Depth Anything 3 models.
    
    Note: DA3 uses a different API than DA2. This wrapper provides
    a compatible interface but may have limitations.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize Depth Anything 3 wrapper.
        
        Args:
            model_config: Configuration dict with keys:
                - model_name: Model preset name (default: 'da3-large')
                - checkpoint_path: Optional path to model directory (if None, auto-finds)
                - device: Device to use (None for auto-detect)
        """
        super().__init__(model_config)
        
        if not DA3_AVAILABLE:
            raise ImportError(
                "Depth Anything 3 is not available. "
                "Please ensure the DA3 package is properly installed."
            )
        
        self.model_name = model_config.get('model_name', 'da3-large')
        self.device = model_config.get('device', None)
        self._is_metric = True  # DA3 models are metric
        
        # Find checkpoint/model directory if not provided
        explicit_checkpoint = model_config.get('checkpoint_path', None)
        self.checkpoint_path, checkpoint_config = find_checkpoint(
            model_name=self.model_name,
            explicit_checkpoint=explicit_checkpoint
        )
        
        # Determine device
        if self.device is None:
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
    
    def load_model(self) -> None:
        """Load the Depth Anything 3 model."""
        if self._is_loaded:
            return
        
        # Try to load from local directory first
        if os.path.isdir(self.checkpoint_path):
            try:
                self._model = DepthAnything3.from_pretrained(self.checkpoint_path)
            except Exception as e:
                # If local loading fails, try creating with model_name
                print(f"Warning: Could not load from {self.checkpoint_path}, using model_name instead: {e}")
                self._model = DepthAnything3(model_name=self.model_name)
        else:
            # Create model with preset name
            self._model = DepthAnything3(model_name=self.model_name)
        
        self._model = self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True
    
    def infer_image(
        self,
        image: np.ndarray,
        input_size: int = 518,
        **kwargs
    ) -> np.ndarray:
        """
        Run inference on a single image.
        
        Note: DA3's API is designed for batch processing and export.
        This wrapper extracts depth from a single image.
        
        Args:
            image: Input image as numpy array (BGR format, uint8)
            input_size: Input size for model (not directly used by DA3)
            **kwargs: Additional inference parameters
        
        Returns:
            Depth map as numpy array (in meters)
        """
        if not self._is_loaded:
            self.load_model()
        
        import torch
        import tempfile
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save image temporarily (DA3 expects file paths)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            import PIL.Image
            pil_image = PIL.Image.fromarray(image_rgb)
            pil_image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Run inference (DA3 expects list of image paths)
            with torch.no_grad():
                # Use DA3's inference method
                # Note: This is a simplified version - DA3's full API is more complex
                prediction = self._model.inference(
                    image=[tmp_path],
                    export_dir=None,
                    export_format=None
                )
                
                # Extract depth from prediction
                # DA3 returns a Prediction object with depth information
                if hasattr(prediction, 'depth'):
                    depth = prediction.depth
                elif isinstance(prediction, dict) and 'depth' in prediction:
                    depth = prediction['depth']
                elif isinstance(prediction, list) and len(prediction) > 0:
                    # If prediction is a list, get first item
                    if hasattr(prediction[0], 'depth'):
                        depth = prediction[0].depth
                    elif isinstance(prediction[0], dict) and 'depth' in prediction[0]:
                        depth = prediction[0]['depth']
                    else:
                        raise ValueError("Could not extract depth from DA3 prediction")
                else:
                    raise ValueError("Could not extract depth from DA3 prediction")
                
                # Convert to numpy if needed
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                
                # Handle batch dimension if present
                if len(depth.shape) == 4:  # (B, C, H, W) or (B, H, W)
                    depth = depth[0]
                if len(depth.shape) == 3:  # (C, H, W)
                    depth = depth[0] if depth.shape[0] == 1 else depth
                
                return depth.astype(np.float32)
        
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def is_metric(self) -> bool:
        """Whether this model outputs metric depth."""
        return self._is_metric
    
    def get_model_name(self) -> str:
        """Get model name."""
        return f"DepthAnything3-{self.model_name}"
    
    def get_checkpoint_path(self) -> Optional[str]:
        """Get the checkpoint/model directory path used by this model."""
        return self.checkpoint_path

