"""
Source package for depth estimation evaluation.
Contains core processing, utilities, and tools.
"""

from .metrics import compute_depth_metrics
from .visualization import (
    depth_to_color,
    create_depth_color_scale_overlay,
    create_error_scale_overlay,
    concat_with_guide,
    generate_visualization_images,
    generate_visualizations_for_dataset
)
from .paths import get_output_paths
from .pipeline import ProcessingPipeline

__all__ = [
    'compute_depth_metrics',
    'depth_to_color',
    'create_depth_color_scale_overlay',
    'create_error_scale_overlay',
    'concat_with_guide',
    'generate_visualization_images',
    'generate_visualizations_for_dataset',
    'get_output_paths',
    'ProcessingPipeline',
]
