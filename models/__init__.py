"""
Model interfaces and implementations for depth estimation.
"""

from .base import BaseDepthModelWrapper
from .depth_anything_v2 import DepthAnythingV2Wrapper

__all__ = [
    'BaseDepthModelWrapper',
    'DepthAnythingV2Wrapper',
]

