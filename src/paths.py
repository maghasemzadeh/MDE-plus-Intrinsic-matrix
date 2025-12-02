"""
Path utilities for organizing output files.
"""

import os
from typing import Dict, Optional


def get_output_paths(
    scene_output_dir: str,
    model_label: str,
    cam: Optional[str] = None
) -> Dict[str, str]:
    """
    Get output paths for a scene/model/camera combination.
    
    Args:
        scene_output_dir: Base output directory for the scene/item
        model_label: 'metric' or 'basic'
        cam: Camera ID ('0', '1', etc.) or None for single camera
    
    Returns:
        Dictionary with paths:
            - model_dir: Model-specific directory
            - compare_dir: Comparison directory
            - disp_dir: Disparity/visualization directory
            - numpy_dir: Numpy data directory
    """
    paths = {}
    
    # Model-specific directory
    paths['model_dir'] = os.path.join(scene_output_dir, model_label)
    
    # Comparison directory
    paths['compare_dir'] = os.path.join(paths['model_dir'], 'compare')
    
    # Camera-specific directories
    if cam is not None:
        cam_dir = f'disp{cam}'
        paths['disp_dir'] = os.path.join(paths['model_dir'], cam_dir)
        paths['numpy_dir'] = os.path.join(paths['disp_dir'], 'numpy_matrix')
    else:
        # Single camera (use disp0)
        paths['disp_dir'] = os.path.join(paths['model_dir'], 'disp0')
        paths['numpy_dir'] = os.path.join(paths['disp_dir'], 'numpy_matrix')
    
    return paths

