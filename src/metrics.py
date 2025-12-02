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
    
    Args:
        pred_depth: Predicted depth map (in meters)
        gt_depth: Ground truth depth map (in meters)
        is_metric_model: If True, compute AbsRel and RMSE. If False, compute SILog.
    
    Returns:
        Dictionary with metrics: abs_rel, rmse (for metric) or silog (for non-metric)
    """
    # Get valid mask (finite, positive values)
    valid_mask = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (gt_depth > 0) & (pred_depth > 0)
    
    if np.sum(valid_mask) < 10:
        # Not enough valid pixels
        if is_metric_model:
            return {
                'abs_rel': np.nan,
                'rmse': np.nan,
                'n_valid': 0
            }
        else:
            return {
                'silog': np.nan,
                'n_valid': 0
            }
    
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    
    if is_metric_model:
        # For metric models: use AbsRel and RMSE
        # 1. Absolute Relative Error
        abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
        
        # 2. RMSE
        rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
        
        return {
            'abs_rel': float(abs_rel),
            'rmse': float(rmse),
            'n_valid': int(np.sum(valid_mask))
        }
    else:
        # For non-metric models: use SILog (with scale alignment)
        # Apply median scale alignment
        scale = np.median(gt_valid) / np.median(pred_valid)
        pred_valid_scaled = pred_valid * scale
        
        # SILog (Scale-Invariant Log RMSE)
        diff_log = np.log(pred_valid_scaled) - np.log(gt_valid)
        silog = np.sqrt(np.mean(diff_log ** 2) - 0.5 * (np.mean(diff_log) ** 2))
        
        return {
            'silog': float(silog),
            'n_valid': int(np.sum(valid_mask))
        }

