"""
Metrics computation for depth estimation evaluation.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def align_non_metric_predictions(
    pred: np.ndarray,
    gt: np.ndarray,
    outputs_inverse_depth: bool = False,
) -> np.ndarray:
    """
    Align non-metric (relative/basic) model predictions to ground-truth depth.

    Inverse depth (outputs_inverse_depth=True) means the model outputs values
    proportional to 1/z: larger = closer to the camera. The Depth Anything V2
    *pretrained* basic model does this (paper Section 5.2: "affine-invariant
    inverse depth"). Direct depth (outputs_inverse_depth=False) means larger =
    farther; e.g. fine-tuned basic models trained with depth targets.

    Alignment uses affine scale+shift (least squares). For inverse-depth models
    we fit s·pred + t ≈ 1/gt then convert to depth as 1/(s·pred+t). For
    direct-depth we fit s·pred + t ≈ gt. Callers should pass
    outputs_inverse_depth from model metadata or script argument, not derived
    from correlation.

    Args:
        pred: Valid predicted values (1-D, all finite and > 0)
        gt:   Corresponding GT depth values (1-D, all finite and > 0)
        outputs_inverse_depth: True if model outputs inverse depth (1/z).

    Returns:
        Aligned prediction array (same shape as pred, all > 0)
    """
    n = len(pred)
    if n < 2:
        return pred.copy()

    if outputs_inverse_depth:
        # Align in inverse-depth space: s·pred + t ≈ 1/gt, then depth = 1/(s·pred+t)
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
        pred_inv = 1.0 / np.maximum(pred, 1e-8)
        scale = np.median(gt) / np.median(pred_inv)
        return pred_inv * max(scale, 1e-8)
    else:
        # Direct-depth: affine alignment s·pred + t ≈ gt
        A = np.stack([pred, np.ones(n, dtype=pred.dtype)], axis=1)
        try:
            result = np.linalg.lstsq(A, gt, rcond=None)
            s, t = float(result[0][0]), float(result[0][1])
            if s > 0:
                aligned = s * pred + t
                return np.maximum(aligned, 1e-8)
        except Exception:
            pass
        scale = np.median(gt) / np.median(pred)
        return pred * max(scale, 1e-8)


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    is_metric_model: bool = True,
    outputs_inverse_depth: bool = False,
    pred_clip: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Compute depth estimation metrics for a single image.

    For non-metric (basic) models, alignment is applied using
    ``align_non_metric_predictions``. Pass ``outputs_inverse_depth`` from the
    model (e.g. ``model.outputs_inverse_depth()``) or from script args; do not
    infer it from data.

    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map (in metres)
        is_metric_model: If False, apply affine alignment before computing metrics.
        outputs_inverse_depth: If True (and not metric), treat predictions as
            inverse depth (1/z). Only used when is_metric_model is False.
        pred_clip: Optional (min, max) depth range to clip predictions to
            after alignment. Standard eigen/MiDaS-protocol evaluation clips
            predictions to the evaluation depth range; without this, aligned
            inverse-depth predictions near zero disparity explode to huge
            depths and dominate per-image AbsRel.

    Returns:
        Dictionary with metrics: d1, d2, d3, abs_rel, sq_rel, rmse, rmse_log,
        log10, silog, n_valid (same definitions as the official DA2
        util/metric.py eval_depth)
    """
    # Get valid mask (finite, positive values)
    valid_mask = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (gt_depth > 0) & (pred_depth > 0)

    if np.sum(valid_mask) < 10:
        return {
            'd1': np.nan,
            'd2': np.nan,
            'd3': np.nan,
            'abs_rel': np.nan,
            'sq_rel': np.nan,
            'rmse': np.nan,
            'rmse_log': np.nan,
            'log10': np.nan,
            'silog': np.nan,
            'n_valid': 0
        }

    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]

    if not is_metric_model:
        pred_valid = align_non_metric_predictions(
            pred_valid, gt_valid, outputs_inverse_depth=outputs_inverse_depth
        )

    if pred_clip is not None:
        pred_valid = np.clip(pred_valid, pred_clip[0], pred_clip[1])

    pred_valid = np.maximum(pred_valid, 1e-8)
    gt_valid = np.maximum(gt_valid, 1e-8)

    diff = pred_valid - gt_valid
    diff_log = np.log(pred_valid) - np.log(gt_valid)

    # Threshold accuracy (delta) metrics, as in DA2 util/metric.py
    thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
    d1 = np.mean(thresh < 1.25)
    d2 = np.mean(thresh < 1.25 ** 2)
    d3 = np.mean(thresh < 1.25 ** 3)

    abs_rel = np.mean(np.abs(diff) / gt_valid)
    sq_rel = np.mean(diff ** 2 / gt_valid)
    rmse = np.sqrt(np.mean(diff ** 2))
    rmse_log = np.sqrt(np.mean(diff_log ** 2))
    log10 = np.mean(np.abs(np.log10(pred_valid) - np.log10(gt_valid)))
    silog = np.sqrt(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2)

    return {
        'd1': float(d1),
        'd2': float(d2),
        'd3': float(d3),
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'log10': float(log10),
        'silog': float(silog),
        'n_valid': int(np.sum(valid_mask))
    }

