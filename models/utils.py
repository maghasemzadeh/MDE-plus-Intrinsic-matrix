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
        Dictionary with model configuration (model_type, encoder, max_depth, use_camera_intrinsics, etc.)
    """
    # Default values - basic by default, only metric if explicitly detected
    encoder = 'vitl'
    model_type = 'basic'  # Default to basic, only set to metric if explicitly detected
    max_depth = 80.0
    use_camera_intrinsics = False
    cam_token_inject_layer = None

    # Try to load checkpoint to get info
    # #region agent log
    _logp = None
    try:
        _d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            _d = os.path.dirname(_d)
            if os.path.exists(os.path.join(_d, 'train.py')) or os.path.exists(os.path.join(_d, 'compare_models.py')):
                _logp = os.path.join(_d, '.cursor', 'debug.log')
                break
        if _logp:
            with open(_logp, 'a') as _lf:
                import json
                _lf.write(json.dumps({"hypothesisId": "E", "location": "identify_model_from_checkpoint:before_load", "message": "about to torch.load in identify", "data": {"checkpoint_path": checkpoint_path, "is_file": os.path.isfile(checkpoint_path), "is_dir": os.path.isdir(checkpoint_path)}, "timestamp": __import__('time').time()}) + '\n')
    except Exception:
        pass
    # #endregion
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Check if checkpoint has saved config (from new train.py format)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            encoder = config.get('encoder', encoder)
            # Use config's model_type if provided, otherwise default to basic
            model_type = config.get('model_type', 'basic')
            max_depth = config.get('max_depth', max_depth)
            use_camera_intrinsics = config.get('use_camera_intrinsics', use_camera_intrinsics)
            cam_token_inject_layer = config.get('cam_token_inject_layer', cam_token_inject_layer)
            # Return early if config was found
            return {
                'model_type': model_type,
                'encoder': encoder,
                'max_depth': max_depth,
                'use_camera_intrinsics': use_camera_intrinsics,
                'cam_token_inject_layer': cam_token_inject_layer,
                'basic_target_space': config.get('basic_target_space'),
                # train.py checkpoints use the unified ReLU-head dpt class for
                # BOTH model types; the released metric checkpoints instead use
                # the sigmoid*max_depth head. Loaders must build the right one.
                'from_train_py': True,
            }

        # Extract state dict for older checkpoint format
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Try to infer encoder from state dict
        if 'pretrained.patch_embed.proj.weight' in state_dict:
            weight_shape = state_dict['pretrained.patch_embed.proj.weight'].shape
            embed_dim = weight_shape[0]
            # Map embed_dim to encoder type
            if embed_dim == 384:
                encoder = 'vits'
            elif embed_dim == 768:
                encoder = 'vitb'
            elif embed_dim == 1024:
                encoder = 'vitl'
            elif embed_dim == 1536:
                encoder = 'vitg'
        elif 'patch_embed.proj.weight' in state_dict:
            weight_shape = state_dict['patch_embed.proj.weight'].shape
            embed_dim = weight_shape[0]
            if embed_dim == 384:
                encoder = 'vits'
            elif embed_dim == 768:
                encoder = 'vitb'
            elif embed_dim == 1024:
                encoder = 'vitl'
            elif embed_dim == 1536:
                encoder = 'vitg'
        elif 'pretrained.cls_token' in state_dict:
            embed_dim = state_dict['pretrained.cls_token'].shape[-1]
            if embed_dim == 384:
                encoder = 'vits'
            elif embed_dim == 768:
                encoder = 'vitb'
            elif embed_dim == 1024:
                encoder = 'vitl'
            elif embed_dim == 1536:
                encoder = 'vitg'

        # NOTE: both metric and basic models have 'depth_head' keys — we cannot
        # distinguish them from state-dict keys alone.  The filename/dirname
        # fallback below is authoritative for official DA2 checkpoints.
        # For trained checkpoints the 'config' block above already returned early.
        # Leave model_type at its default ('basic') here; the filename check below
        # will set it to 'metric' when appropriate.

        # Check if model has camera intrinsics support (has cam_encoder keys)
        use_camera_intrinsics = any('cam_encoder' in k for k in state_dict.keys())

        # Infer max_depth from checkpoint metadata or filename
        # Note: previous_best doesn't necessarily mean metric - check state dict first
        # (already done above with depth_head check)

    except Exception as e:
        # #region agent log
        if _logp:
            try:
                with open(_logp, 'a') as _lf:
                    import json
                    _lf.write(json.dumps({"hypothesisId": "A,E", "location": "identify_model_from_checkpoint:catch", "message": "torch.load failed, using filename fallback", "data": {"checkpoint_path": checkpoint_path, "error_type": type(e).__name__, "error_msg": str(e)[:200]}, "timestamp": __import__('time').time()}) + '\n')
            except Exception:
                pass
        # #endregion
        # If we can't load, infer from filename
        pass
    
    # Infer from filename as fallback
    filename = os.path.basename(checkpoint_path).lower()
    dirname = os.path.basename(os.path.dirname(checkpoint_path)).lower()
    
    # Check for encoder in filename or directory
    for enc in ['vitg', 'vitl', 'vitb', 'vits']:
        if enc in filename or enc in dirname:
            encoder = enc
            break
    
    # Check for model type from filename
    # Default is basic, only set to metric if "metric" is explicitly in the name
    if 'metric' in filename or 'metric' in dirname:
        model_type = 'metric'
        # Check for indoor/outdoor
        if 'hypersim' in filename or 'hypersim' in dirname:
            max_depth = 20.0
        elif 'vkitti' in filename or 'vkitti' in dirname or 'cityscapes' in dirname or 'drivingstereo' in dirname:
            max_depth = 80.0
        else:
            max_depth = 80.0  # default outdoor
    elif 'basic' in filename or 'basic' in dirname:
        model_type = 'basic'
        max_depth = 20.0
    elif 'depth_anything_v2' in filename:
        # DepthAnythingV2 naming convention:
        # - depth_anything_v2_{encoder}.pth = basic (non-metric) - default
        # - depth_anything_v2_metric_{dataset}_{encoder}.pth = metric
        # If "metric" is not in filename, it's basic (already the default)
        if 'metric' not in filename:
            model_type = 'basic'  # Explicitly set, though already default
            max_depth = 80.0  # default outdoor for basic models
    
    return {
        'model_type': model_type,
        'encoder': encoder,
        'max_depth': max_depth,
        'use_camera_intrinsics': use_camera_intrinsics,
        'cam_token_inject_layer': cam_token_inject_layer,
    }

