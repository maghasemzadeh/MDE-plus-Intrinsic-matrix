"""
Utility functions for model checkpoint identification and selection.
"""

import os
import torch
from typing import Dict, Optional, Tuple


def identify_model_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Identify model configuration from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dictionary with model configuration (model_type, encoder, max_depth)
    """
    # Try to load checkpoint to get info
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Try to infer encoder from state dict
        encoder = 'vitl'  # default
        if 'pretrained.patch_embed.proj.weight' in state_dict:
            weight_shape = state_dict['pretrained.patch_embed.proj.weight'].shape
            if weight_shape[0] == 384:  # vitg
                encoder = 'vitg'
            elif weight_shape[0] == 256:  # vitl
                encoder = 'vitl'
            elif weight_shape[0] == 192:
                # Could be vitb or vits, check more
                if 'pretrained.blocks.11' in state_dict:  # vitb has 12 blocks
                    encoder = 'vitb'
                else:  # vits has fewer blocks
                    encoder = 'vits'
        
        # Check if it's a metric model (has depth_head)
        model_type = 'metric'  # default
        if 'depth_head' in str(state_dict.keys()):
            model_type = 'metric'
        else:
            model_type = 'basic'
        
        # Infer max_depth from checkpoint metadata or filename
        max_depth = 80.0  # default outdoor
        if isinstance(checkpoint, dict) and 'previous_best' in checkpoint:
            # This is a trained checkpoint, likely metric
            model_type = 'metric'
        
    except Exception as e:
        # If we can't load, infer from filename
        model_type = 'metric'
        encoder = 'vitl'
        max_depth = 80.0
    
    # Infer from filename as fallback
    filename = os.path.basename(checkpoint_path).lower()
    dirname = os.path.basename(os.path.dirname(checkpoint_path)).lower()
    
    # Check for encoder in filename or directory
    for enc in ['vitg', 'vitl', 'vitb', 'vits']:
        if enc in filename or enc in dirname:
            encoder = enc
            break
    
    # Check for model type
    if 'basic' in filename or 'basic' in dirname:
        model_type = 'basic'
        max_depth = 20.0
    elif 'metric' in filename or 'metric' in dirname:
        model_type = 'metric'
        # Check for indoor/outdoor
        if 'hypersim' in filename or 'hypersim' in dirname:
            max_depth = 20.0
        elif 'vkitti' in filename or 'vkitti' in dirname or 'cityscapes' in dirname or 'drivingstereo' in dirname:
            max_depth = 80.0
        else:
            max_depth = 80.0  # default outdoor
    
    return {
        'model_type': model_type,
        'encoder': encoder,
        'max_depth': max_depth
    }

