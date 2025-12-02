"""
Common configuration and utility functions for Depth Anything V2.
This module provides shared constants and utilities to avoid code duplication.
"""
import torch
from typing import Dict, Optional


# Model configurations for different encoder types
MODEL_CONFIGS: Dict[str, Dict[str, any]] = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available device for PyTorch operations.
    
    Args:
        device: Optional device string ('cuda', 'mps', 'cpu'). If None, auto-detects.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if device is not None:
        return device
    
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_model_config(encoder: str) -> Dict[str, any]:
    """
    Get model configuration for a given encoder type.
    
    Args:
        encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
    
    Returns:
        Dictionary with model configuration
    
    Raises:
        ValueError: If encoder type is not supported
    """
    if encoder not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported encoder: {encoder}. "
            f"Supported encoders: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[encoder].copy()

