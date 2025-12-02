"""
Base interface for depth estimation models.
All depth models should inherit from this class to ensure consistent interface.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class BaseDepthModel(nn.Module, ABC):
    """
    Base class for all depth estimation models.
    Provides a common interface for model loading, inference, and configuration.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize base depth model.
        
        Args:
            model_name: Name identifier for this model
            **kwargs: Model-specific configuration parameters
        """
        super().__init__()
        self.model_name = model_name
        self.config = kwargs
        self._is_loaded = False
    
    @abstractmethod
    def _build_model(self, **kwargs) -> nn.Module:
        """
        Build the underlying model architecture.
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Model configuration parameters
        
        Returns:
            The model architecture (nn.Module)
        """
        pass
    
    @abstractmethod
    def _load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """
        Load model weights from checkpoint.
        Must be implemented by subclasses.
        
        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional loading parameters
        """
        pass
    
    def load_from_checkpoint(
        self, 
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> 'BaseDepthModel':
        """
        Load model from checkpoint and move to device.
        
        Args:
            checkpoint_path: Path to checkpoint file (if None, uses default)
            device: Device to load model on (if None, auto-detects)
            **kwargs: Additional loading parameters
        
        Returns:
            Self for method chaining
        """
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, **kwargs)
        
        if device is not None:
            self.to(device)
        else:
            # Auto-detect device
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            self.to(device)
        
        self.eval()
        self._is_loaded = True
        return self
    
    def to(self, device):
        """Move model to device and update underlying model."""
        super().to(device)
        if hasattr(self, 'model'):
            self.model = self.model.to(device)
        return self
    
    @abstractmethod
    def infer_image(
        self, 
        image: np.ndarray, 
        input_size: int = 518,
        **kwargs
    ) -> np.ndarray:
        """
        Run inference on a single image.
        Must be implemented by subclasses.
        
        Args:
            image: Input image as numpy array (BGR format, uint8)
            input_size: Input size for model
            **kwargs: Additional inference parameters
        
        Returns:
            Depth map as numpy array
        """
        pass
    
    def infer_batch(
        self,
        images: list,
        input_size: int = 518,
        **kwargs
    ) -> list:
        """
        Run inference on a batch of images.
        Default implementation processes images sequentially.
        Can be overridden for batch processing optimization.
        
        Args:
            images: List of input images (numpy arrays, BGR format, uint8)
            input_size: Input size for model
            **kwargs: Additional inference parameters
        
        Returns:
            List of depth maps (numpy arrays)
        """
        return [self.infer_image(img, input_size, **kwargs) for img in images]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'config': self.config,
            'is_loaded': self._is_loaded,
            'device': next(self.parameters()).device if self._is_loaded else None,
        }
    
    def is_metric(self) -> bool:
        """
        Check if model outputs metric depth (in meters) or relative depth.
        
        Returns:
            True if model outputs metric depth, False if relative depth
        """
        return self.config.get('is_metric', False)
    
    def get_max_depth(self) -> Optional[float]:
        """
        Get maximum depth value for metric models.
        
        Returns:
            Maximum depth in meters, or None if not applicable
        """
        return self.config.get('max_depth', None)

