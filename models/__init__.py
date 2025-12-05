"""
Model interfaces and implementations for depth estimation.
"""

from typing import Dict
from .base import BaseDepthModelWrapper
from .depth_anything_v2 import DepthAnythingV2Wrapper, find_checkpoint as find_da2_checkpoint
from .depth_anything_v2_revised import DepthAnythingV2RevisedWrapper, find_checkpoint as find_da2_revised_checkpoint
from .depth_anything_3 import DepthAnything3Wrapper, find_checkpoint as find_da3_checkpoint
from .utils import identify_model_from_checkpoint

# Factory function to create model wrapper by name
def create_model_wrapper(model_name: str, model_config: Dict) -> BaseDepthModelWrapper:
    """
    Create a model wrapper by name.
    
    Args:
        model_name: Model name ('da2', 'da2-revised', 'da3')
        model_config: Model configuration dictionary
    
    Returns:
        Model wrapper instance
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'da2':
        return DepthAnythingV2Wrapper(model_config)
    elif model_name_lower == 'da2-revised':
        return DepthAnythingV2RevisedWrapper(model_config)
    elif model_name_lower == 'da3':
        return DepthAnything3Wrapper(model_config)
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Supported models: 'da2', 'da2-revised', 'da3'"
        )


__all__ = [
    'BaseDepthModelWrapper',
    'DepthAnythingV2Wrapper',
    'DepthAnythingV2RevisedWrapper',
    'DepthAnything3Wrapper',
    'find_da2_checkpoint',
    'find_da2_revised_checkpoint',
    'find_da3_checkpoint',
    'identify_model_from_checkpoint',
    'create_model_wrapper',
]
