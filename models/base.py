"""
Base model wrapper interface for depth estimation models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseDepthModelWrapper(ABC):
    """
    Abstract wrapper for depth estimation models.
    
    This provides a unified interface for different depth estimation models,
    abstracting away model-specific details.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize model wrapper.
        
        Args:
            model_config: Model configuration dictionary
        """
        self.model_config = model_config
        self._model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model (lazy loading).
        """
        pass
    
    @abstractmethod
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
            Depth map as numpy array (in meters for metric models, unitless for basic)
        """
        pass
    
    @abstractmethod
    def is_metric(self) -> bool:
        """
        Whether this model outputs metric depth (True) or relative depth (False).
        
        Returns:
            True if metric depth, False if relative depth
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get model name/identifier.
        
        Returns:
            Model name string
        """
        pass
    
    def requires_scale_factor(self) -> bool:
        """
        Whether this model requires a scale factor (for non-metric models).
        
        Returns:
            True if scale factor is needed, False otherwise
        """
        return not self.is_metric()
    
    def calculate_scale_factor(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        user_scale_factor: Optional[float] = None
    ) -> float:
        """
        Calculate scale factor for non-metric models.
        
        Args:
            pred_depth: Predicted depth (unitless)
            gt_depth: Ground truth depth (in meters)
            user_scale_factor: User-provided scale factor (if None, auto-calculate)
        
        Returns:
            Scale factor to apply to predictions
        """
        if user_scale_factor is not None:
            return user_scale_factor
        
        # Auto-scale: match median depth
        valid_gt = gt_depth[np.isfinite(gt_depth) & (gt_depth > 0)]
        valid_pred = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
        
        if valid_gt.size > 0 and valid_pred.size > 0:
            median_gt = np.median(valid_gt)
            median_pred = np.median(valid_pred)
            return (median_gt / median_pred) if median_pred > 0 else 1.0
        
        return 1.0

