"""
Model registry and factory for Depth Anything V2 models.
Provides a unified interface for loading and using different model types.
"""
from .registry import ModelRegistry, get_model, list_models, register_model, has_model
from .base import BaseDepthModel

# Lazy import model implementations to avoid circular imports
# Models will be registered when first accessed

__all__ = [
    'BaseDepthModel',
    'ModelRegistry',
    'get_model',
    'list_models',
    'register_model',
    'has_model',
    '_register_builtin_models',
]


def _register_builtin_models():
    """Register built-in models. Called automatically on first import."""
    from .depth_anything_v2_basic import DepthAnythingV2BasicModel
    from .depth_anything_v2_metric import DepthAnythingV2MetricModel
    
    # Register models with their full names
    register_model('depth_anything_v2_basic', DepthAnythingV2BasicModel)
    register_model('depth_anything_v2_metric', DepthAnythingV2MetricModel)
    
    # Register aliases
    register_model('basic', DepthAnythingV2BasicModel)
    register_model('metric', DepthAnythingV2MetricModel)


# Auto-register built-in models on import
_register_builtin_models()

