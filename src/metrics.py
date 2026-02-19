"""
Metrics computation for depth estimation evaluation.
"""

import numpy as np
from typing import Dict


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    is_metric_model: bool = True
) -> Dict[str, float]:
    """
    Compute depth estimation metrics for a single image.

    Always returns all four metrics regardless of model type. For non-metric
    (basic) models, median scale alignment is applied before computing metrics.

    Args:
        pred_depth: Predicted depth map (in meters)
        gt_depth: Ground truth depth map (in meters)
        is_metric_model: If False, apply median scale alignment before computing
                         metrics.

    Returns:
        Dictionary with metrics: abs_rel, rmse, rmse_log, silog, n_valid
    """
    # Get valid mask (finite, positive values)
    valid_mask = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (gt_depth > 0) & (pred_depth > 0)

    if np.sum(valid_mask) < 10:
        return {
            'abs_rel': np.nan,
            'rmse': np.nan,
            'rmse_log': np.nan,
            'silog': np.nan,
            'n_valid': 0
        }

    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]

    if not is_metric_model:
        # Apply median scale alignment for relative/basic models
        scale = np.median(gt_valid) / np.median(pred_valid)
        pred_valid = pred_valid * scale

    diff = pred_valid - gt_valid
    diff_log = np.log(pred_valid) - np.log(gt_valid)

    abs_rel = np.mean(np.abs(diff) / gt_valid)
    rmse = np.sqrt(np.mean(diff ** 2))
    rmse_log = np.sqrt(np.mean(diff_log ** 2))
    silog = np.sqrt(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2)

    return {
        'abs_rel': float(abs_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'silog': float(silog),
        'n_valid': int(np.sum(valid_mask))
    }

