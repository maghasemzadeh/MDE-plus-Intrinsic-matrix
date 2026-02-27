"""
Metrics computation for depth estimation evaluation.
"""

import numpy as np
from typing import Dict


def align_non_metric_predictions(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Align non-metric (relative/basic) model predictions to ground-truth depth.

    The Depth Anything V2 basic model outputs **affine-invariant inverse depth**
    (per the paper, Section 5.2): values are proportional to 1/z with unknown
    scale and shift, so larger values mean *closer* to the camera.

    The paper achieves abs_rel ~0.078 on KITTI by using affine scale+shift
    alignment.  Simple median-scale alignment assumes the output is already in
    the depth domain (larger = farther), which is wrong for inverse-depth
    models and can inflate abs_rel to 80–90%.

    Algorithm
    ---------
    1. Detect whether the model output is inverse-depth (Pearson r < 0 with GT).
    2a. If inverse-depth: fit  s·pred + t ≈ 1/gt  (alignment in inverse-depth
        space), then convert: depth_pred = 1 / max(s·pred + t, ε).
    2b. If direct-depth: fit  s·pred + t ≈ gt  (affine alignment in depth
        space), then clip negatives to ε.
    Fallback in either branch: median-scale alignment.

    This matches the evaluation protocol used in the Depth Anything V2 paper
    for zero-shot relative depth estimation (Table 2).

    Args:
        pred: Valid predicted values (1-D, all finite and > 0)
        gt:   Corresponding GT depth values (1-D, all finite and > 0)

    Returns:
        Aligned prediction array (same shape as pred, all > 0)
    """
    n = len(pred)
    if n < 2:
        return pred.copy()

    correlation = float(np.corrcoef(pred, gt)[0, 1])

    if correlation < 0:
        # Inverse-depth model (e.g. original DA2 pretrained basic weights).
        # Align in inverse-depth space: solve  s·pred + t ≈ 1/gt
        gt_inv = 1.0 / np.maximum(gt, 1e-8)
        A = np.stack([pred, np.ones(n, dtype=pred.dtype)], axis=1)
        try:
            result = np.linalg.lstsq(A, gt_inv, rcond=None)
            s, t = float(result[0][0]), float(result[0][1])
            if s > 0:
                pred_inv_aligned = s * pred + t
                pred_inv_aligned = np.maximum(pred_inv_aligned, 1e-8)
                return 1.0 / pred_inv_aligned
        except Exception:
            pass
        # Fallback: invert then median scale
        pred_inv = 1.0 / np.maximum(pred, 1e-8)
        scale = np.median(gt) / np.median(pred_inv)
        return pred_inv * max(scale, 1e-8)
    else:
        # Direct-depth model (e.g. fine-tuned / metric-conditioned models).
        # Affine scale+shift alignment in depth space: solve  s·pred + t ≈ gt
        A = np.stack([pred, np.ones(n, dtype=pred.dtype)], axis=1)
        try:
            result = np.linalg.lstsq(A, gt, rcond=None)
            s, t = float(result[0][0]), float(result[0][1])
            if s > 0:
                aligned = s * pred + t
                return np.maximum(aligned, 1e-8)
        except Exception:
            pass
        # Fallback: median scale
        scale = np.median(gt) / np.median(pred)
        return pred * max(scale, 1e-8)


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    is_metric_model: bool = True
) -> Dict[str, float]:
    """
    Compute depth estimation metrics for a single image.

    Always returns all metrics regardless of model type.  For non-metric
    (basic) models, affine scale+shift alignment is applied via
    ``align_non_metric_predictions``, which automatically handles both
    direct-depth and inverse-depth model outputs.

    The Depth Anything V2 *basic* pretrained model produces affine-invariant
    *inverse* depth (per the paper).  Evaluating it with simple median-scale
    alignment gives abs_rel ~0.85 even for a high-quality model; the correct
    affine-inverse-depth alignment recovers abs_rel ~0.08–0.15, consistent
    with the paper's reported numbers.

    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map (in metres)
        is_metric_model: If False, apply affine alignment before computing metrics.

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
        pred_valid = align_non_metric_predictions(pred_valid, gt_valid)

    pred_valid = np.maximum(pred_valid, 1e-8)
    gt_valid = np.maximum(gt_valid, 1e-8)

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

