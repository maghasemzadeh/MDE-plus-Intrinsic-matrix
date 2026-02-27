"""
Show how abs_rel (Absolute Relative Error) is calculated for a sample image.

This script uses the same arguments as compare_models.py. It runs one image
through both models and prints:
  - Sample pixels: predicted depth, ground truth depth, and per-pixel abs_rel = |pred - gt| / gt
  - Image-level abs_rel = mean of per-pixel abs_rel over all valid pixels

It also saves (to --output-dir):
  - depth_ground_truth.png: ground truth depth colormap
  - depth_<model1>.png, depth_<model2>.png: each model's predicted depth colormap
  - input_image_with_sample_pixels.png: input image with numbered circles at sample pixels
    (circle index 1 = first row in the table, 2 = second row, etc.)

Usage:
  python show_absrel_calculation.py --dataset CityScapes --model1 da2-revised --model2 da2-revised \\
    --model1-checkpoint path/to/checkpoint1/best.pth --model2-checkpoint path/to/checkpoint2/best.pth \\
    --num-sample-pixels 10 --output-dir absrel_sample_output

  By default, a random image from the dataset is selected each run. Use --item-index N for a specific image.
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2

from datasets import (
    DatasetConfig,
    CityscapesDataset,
    DrivingStereoDataset,
    MiddleburyDataset,
    VKITTIDataset,
    DIODEDataset,
    NYUDataset,
    KITTIDataset,
)
from src import compute_depth_metrics
from src.visualization import depth_to_color

# Import model loading from compare_models to keep same behavior
from compare_models import find_model_by_name, checkpoint_output_slug, display_name_for_comparison


def create_dataset(args):
    """Create dataset instance from args (same logic as compare_models.py)."""
    dataset_name = args.dataset.lower()
    dataset_config = DatasetConfig(
        dataset_path=args.dataset_path,
        split=args.split,
        max_items=args.max_items,
        regex_filter=args.filter_regex,
    )
    if dataset_name == "cityscapes":
        return CityscapesDataset(dataset_config)
    if dataset_name == "drivingstereo":
        return DrivingStereoDataset(dataset_config)
    if dataset_name == "middlebury":
        return MiddleburyDataset(dataset_config)
    if dataset_name == "vkitti":
        return VKITTIDataset(dataset_config)
    if dataset_name == "diode":
        return DIODEDataset(dataset_config)
    if dataset_name == "nyu":
        return NYUDataset(dataset_config)
    if dataset_name == "kitti":
        return KITTIDataset(dataset_config)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def run_one_image_and_show_absrel(
    model,
    model_display_name: str,
    item,
    dataset,
    input_size: int,
    is_metric: bool,
    num_sample_pixels: int,
):
    """
    Load image and GT, run model inference, then show sample pixels and image-level abs_rel.
    """
    # Load image
    if dataset.is_cached_dataset():
        image = dataset.load_image(item)
    else:
        image = cv2.imread(item.image_path)
    if image is None:
        return None, f"Could not read image: {item.image_path}"

    # Load ground truth depth
    try:
        gt_depth = dataset.load_gt_depth(item.gt_path, item)
    except Exception as e:
        return None, f"Could not load GT: {e}"
    # Ensure 2D (some OpenCV/loaders return BGR for grayscale PNG)
    if len(gt_depth.shape) == 3:
        gt_depth = gt_depth[:, :, 0]
    if len(gt_depth.shape) > 2:
        gt_depth = np.squeeze(gt_depth)
    if len(gt_depth.shape) != 2:
        return None, f"Expected 2D gt_depth, got shape {gt_depth.shape}"

    # Load intrinsics if available
    intrinsics = None
    if hasattr(dataset, "load_intrinsic") and hasattr(dataset, "has_intrinsics") and dataset.has_intrinsics():
        try:
            intrinsics = dataset.load_intrinsic(item)
        except Exception:
            intrinsics = None

    # Run inference
    import torch
    with torch.no_grad():
        pred_depth_raw = model.infer_image(image, input_size=input_size, intrinsics=intrinsics)
    # If model outputs all zeros and we passed intrinsics, retry without intrinsics
    # (some checkpoints trained without intrinsics may fail when intrinsics are passed)
    if pred_depth_raw is not None and intrinsics is not None:
        n_positive = np.sum(np.isfinite(pred_depth_raw) & (pred_depth_raw > 0))
        if n_positive == 0:
            with torch.no_grad():
                pred_depth_raw = model.infer_image(image, input_size=input_size, intrinsics=None)
    if pred_depth_raw is None:
        return None, "Model inference returned None"
    pred_depth = pred_depth_raw.astype(np.float32)
    if len(pred_depth.shape) > 2:
        pred_depth = np.squeeze(pred_depth)
    if len(pred_depth.shape) == 3:
        pred_depth = pred_depth[:, :, 0]  # ensure 2D if model returned 3 channels
    if len(pred_depth.shape) != 2:
        return None, f"Expected 2D pred_depth, got shape {pred_depth.shape}"

    # Resize pred to GT shape if needed
    if pred_depth.shape != gt_depth.shape:
        pred_depth = cv2.resize(
            pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Same valid mask and optional scale as in compute_depth_metrics
    valid_mask = (
        np.isfinite(pred_depth) & np.isfinite(gt_depth)
        & (gt_depth > 0) & (pred_depth > 0)
    )
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]

    if not is_metric:
        scale = np.median(gt_valid) / np.median(pred_valid)
        pred_valid = pred_valid * scale
        pred_depth_aligned = pred_depth.copy()
        pred_depth_aligned[valid_mask] = pred_valid
    else:
        pred_depth_aligned = pred_depth.copy()

    n_valid = int(np.sum(valid_mask))
    if n_valid < 10:
        # Diagnostic info to help debug "0 valid pixels" across all images
        n_gt_finite = int(np.sum(np.isfinite(gt_depth)))
        n_gt_positive = int(np.sum(np.isfinite(gt_depth) & (gt_depth > 0)))
        n_pred_finite = int(np.sum(np.isfinite(pred_depth)))
        n_pred_positive = int(np.sum(np.isfinite(pred_depth) & (pred_depth > 0)))
        gt_valid_vals = gt_depth[np.isfinite(gt_depth) & (gt_depth > 0)]
        pred_valid_vals = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
        diag = (
            f"Insufficient valid pixels: {n_valid}. "
            f"Diagnostics: gt_depth shape={gt_depth.shape} (n_finite={n_gt_finite}, n_>0={n_gt_positive}), "
            f"pred_depth shape={pred_depth.shape} (n_finite={n_pred_finite}, n_>0={n_pred_positive}). "
        )
        if n_gt_positive > 0:
            diag += f"GT valid range: [{float(np.min(gt_valid_vals)):.4f}, {float(np.max(gt_valid_vals)):.4f}]m. "
        if n_pred_positive > 0:
            diag += f"Pred valid range: [{float(np.min(pred_valid_vals)):.4f}, {float(np.max(pred_valid_vals)):.4f}]."
        else:
            diag += "Pred has no positive finite values (model may output zeros or invalid)."
        return None, diag

    # Per-pixel abs_rel contribution: |pred - gt| / gt (for valid pixels only)
    diff = pred_valid - gt_valid
    abs_rel_per_pixel = np.abs(diff) / gt_valid
    abs_rel_image = float(np.mean(abs_rel_per_pixel))

    # Pick sample pixel indices (from flattened valid indices)
    valid_flat_idx = np.where(valid_mask.ravel())[0]
    n_show = min(num_sample_pixels, len(valid_flat_idx))
    rng = np.random.default_rng(42)
    sample_flat_indices = rng.choice(valid_flat_idx, size=n_show, replace=False)
    sample_flat_indices = np.sort(sample_flat_indices)

    # Map back to (row, col) and get values
    h, w = gt_depth.shape
    rows = sample_flat_indices // w
    cols = sample_flat_indices % w

    sample_rows = rows.tolist()
    sample_cols = cols.tolist()
    sample_pred = pred_depth_aligned[rows, cols].tolist()
    sample_gt = gt_depth[rows, cols].tolist()
    sample_abs_rel = (np.abs(pred_depth_aligned[rows, cols] - gt_depth[rows, cols]) / gt_depth[rows, cols]).tolist()

    # Verify with official metric
    metrics = compute_depth_metrics(pred_depth, gt_depth, is_metric_model=is_metric)
    abs_rel_from_api = metrics["abs_rel"]

    return {
        "model_name": model_display_name,
        "item_id": item.item_id,
        "n_valid": n_valid,
        "abs_rel_image": abs_rel_image,
        "abs_rel_from_api": abs_rel_from_api,
        "sample_pixels": [
            {
                "row": int(r),
                "col": int(c),
                "pred_m": float(p),
                "gt_m": float(g),
                "abs_rel_pixel": float(a),
            }
            for r, c, p, g, a in zip(sample_rows, sample_cols, sample_pred, sample_gt, sample_abs_rel)
        ],
        "is_metric": is_metric,
        "image": image.copy(),
        "gt_depth": gt_depth.copy(),
        "pred_depth": pred_depth_aligned.copy(),
    }, None


def print_report(data: dict) -> None:
    """Print a readable report for one model."""
    print(f"\n{'─' * 80}")
    print(f"  Model: {data['model_name']}")
    print(f"  Image: {data['item_id']}")
    print(f"  Valid pixels: {data['n_valid']}  (metric type: {'metric' if data['is_metric'] else 'basic'})")
    print(f"{'─' * 80}")
    print("\n  Formula: abs_rel (per pixel) = |pred - gt| / gt   (all in meters)")
    print("           abs_rel (image)      = mean( abs_rel (per pixel) ) over all valid pixels")
    print("  (Index = circle number in saved image input_image_with_sample_pixels.png)\n")
    print("  Sample pixels:")
    print(f"  {'idx':>4} {'row':>6} {'col':>6} {'pred (m)':>12} {'gt (m)':>12} {'|pred-gt|/gt':>14}")
    print("  " + "-" * 58)
    for idx, s in enumerate(data["sample_pixels"], start=1):
        print(f"  {idx:>4} {s['row']:>6} {s['col']:>6} {s['pred_m']:>12.6f} {s['gt_m']:>12.6f} {s['abs_rel_pixel']:>14.6f}")
    print("  " + "-" * 54)
    print(f"\n  Image-level abs_rel (from this script): {data['abs_rel_image']:.6f}")
    print(f"  Image-level abs_rel (from compute_depth_metrics): {data['abs_rel_from_api']:.6f}")
    print(f"  (Fraction; as percent: {data['abs_rel_image'] * 100:.4f}%)")
    print()


def print_comparison_table(data1: dict, data2: dict) -> None:
    """Print a single table comparing both models on the same pixels.

    Assumes data1["sample_pixels"] and data2["sample_pixels"] share (row, col).
    """
    samples1 = data1["sample_pixels"]
    samples2 = data2["sample_pixels"]
    if len(samples1) != len(samples2):
        print("\n⚠️ Cannot build comparison table: different number of sample pixels.")
        return

    print(f"\n{'─' * 80}")
    print("  Per-pixel comparison (same pixels for both models)")
    print(f"{'─' * 80}")
    print("  Columns:")
    print(f"    - better: which model is closer to GT at that pixel (1, 2, or = for tie)")
    print()
    print(
        "  "
        f"{'idx':>4} {'row':>6} {'col':>6} "
        f"{'gt (m)':>12} "
        f"{'pred1 (m)':>14} {'|p1-gt|/gt':>12} "
        f"{'pred2 (m)':>14} {'|p2-gt|/gt':>12} "
        f"{'better':>8}"
    )
    print("  " + "-" * 96)

    for idx, (s1, s2) in enumerate(zip(samples1, samples2), start=1):
        # Sanity: they should share coordinates and GT
        row = s1["row"]
        col = s1["col"]
        gt = s1["gt_m"]
        pred1 = s1["pred_m"]
        pred2 = s2["pred_m"]
        absrel1 = s1["abs_rel_pixel"]
        absrel2 = s2["abs_rel_pixel"]

        # Determine which model is closer to GT in absolute depth error
        abs_err1 = abs(pred1 - gt)
        abs_err2 = abs(pred2 - gt)
        if abs_err1 < abs_err2:
            better = "1"
        elif abs_err2 < abs_err1:
            better = "2"
        else:
            better = "="

        # Star markers: append '*' directly after the better prediction value
        pred1_str = f"{pred1:.6f}" + ("*" if better in ("1", "=") else "")
        pred2_str = f"{pred2:.6f}" + ("*" if better in ("2", "=") else "")

        print(
            f"  {idx:>4} {row:>6} {col:>6} "
            f"{gt:>12.6f} "
            f"{pred1_str:>14} {absrel1:>12.6f} "
            f"{pred2_str:>14} {absrel2:>12.6f} "
            f"{better:>8}"
        )

    print("  " + "-" * 96)
    print()

def _add_scale_overlay(vis: np.ndarray, max_depth: float, label: str, depth_arr: np.ndarray) -> np.ndarray:
    """Add a text overlay showing scale (0 - max_depth m) and this image's actual range."""
    out = vis.copy()
    valid = depth_arr[np.isfinite(depth_arr) & (depth_arr > 0)]
    d_min = float(np.min(valid)) if len(valid) > 0 else 0.0
    d_max = float(np.max(valid)) if len(valid) > 0 else 0.0
    w = out.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(1.2, w / 800))
    thick = max(1, int(scale * 2))
    y = int(28 * scale)
    cv2.putText(out, f"Scale: 0 - {max_depth:.1f} m (same for all)", (8, y), font, scale, (255, 255, 255), thick + 2)
    cv2.putText(out, f"Scale: 0 - {max_depth:.1f} m (same for all)", (10, y), font, scale, (0, 0, 0), thick)
    y += int(26 * scale)
    cv2.putText(out, f"{label} range: {d_min:.2f} - {d_max:.2f} m", (8, y), font, scale, (255, 255, 255), thick + 2)
    cv2.putText(out, f"{label} range: {d_min:.2f} - {d_max:.2f} m", (10, y), font, scale, (0, 0, 0), thick)
    return out


def save_visualizations(
    output_dir: str,
    image: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth1: np.ndarray,
    pred_depth2: np.ndarray,
    data1: dict,
    data2: dict,
    display_name1: str,
    display_name2: str,
) -> None:
    """
    Save depth visualizations and the input image with sample pixels marked by numbered circles.
    All depth images use the same color scale (0 to max_depth m): blue = near, red = far.
    Table row index 1 = circle label 1, etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Use a common max_depth for consistent color scale across depth maps (from valid GT and preds)
    valid_gt = gt_depth[np.isfinite(gt_depth) & (gt_depth > 0)]
    valid_p1 = pred_depth1[np.isfinite(pred_depth1) & (pred_depth1 > 0)]
    valid_p2 = pred_depth2[np.isfinite(pred_depth2) & (pred_depth2 > 0)]
    all_valid = np.concatenate([
        valid_gt.ravel() if len(valid_gt) else [0],
        valid_p1.ravel() if len(valid_p1) else [0],
        valid_p2.ravel() if len(valid_p2) else [0],
    ])
    max_depth = float(np.percentile(all_valid, 99)) if len(all_valid) > 0 else 100.0
    max_depth = max(max_depth, 1.0)

    # Depth colormaps (same scale for all: JET, 0 = blue, max_depth = red)
    gt_vis = depth_to_color(gt_depth, max_depth=max_depth)
    p1_vis = depth_to_color(pred_depth1, max_depth=max_depth)
    p2_vis = depth_to_color(pred_depth2, max_depth=max_depth)

    # Safe filenames for model names
    def safe_name(s: str, max_len: int = 40) -> str:
        out = "".join(c if c.isalnum() or c in "._-" else "_" for c in s)[:max_len]
        return out or "model"

    name1 = safe_name(display_name1)
    name2 = safe_name(display_name2)

    # Add scale and per-image range overlay so you can see shared scale and compare ranges
    gt_vis = _add_scale_overlay(gt_vis, max_depth, "GT", gt_depth)
    p1_vis = _add_scale_overlay(p1_vis, max_depth, "Pred", pred_depth1)
    p2_vis = _add_scale_overlay(p2_vis, max_depth, "Pred", pred_depth2)

    cv2.imwrite(os.path.join(output_dir, "depth_ground_truth.png"), gt_vis)
    cv2.imwrite(os.path.join(output_dir, f"depth_{name1}.png"), p1_vis)
    cv2.imwrite(os.path.join(output_dir, f"depth_{name2}.png"), p2_vis)

    # Input image with numbered circles at sample pixels (table row 1 = circle 1)
    img_marked = image.copy()
    sample_pixels = data1["sample_pixels"]  # same (row,col) order as in both tables
    radius = max(8, min(image.shape[0], image.shape[1]) // 80)
    thickness = max(2, radius // 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, radius / 20.0)
    font_thick = max(1, thickness)

    for idx, s in enumerate(sample_pixels):
        # Table row is 1-based: first row in table = index 1
        label = str(idx + 1)
        col, row = int(s["col"]), int(s["row"])
        # Draw circle (OpenCV uses (x,y) = (col, row))
        cv2.circle(img_marked, (col, row), radius, (0, 255, 0), thickness)
        # Draw label above the circle so it's visible
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
        tx = col - tw // 2
        ty = row - radius - 4
        if ty < th + 2:
            ty = row + radius + th + 2
        # Background for text readability
        cv2.rectangle(
            img_marked,
            (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + 2),
            (0, 255, 0),
            -1,
        )
        cv2.putText(img_marked, label, (tx, ty), font, font_scale, (0, 0, 0), font_thick)

    cv2.imwrite(os.path.join(output_dir, "input_image_with_sample_pixels.png"), img_marked)

    # Print depth ranges so you can see why colors differ (same scale: blue=low, red=high)
    def _stats(d: np.ndarray) -> tuple:
        v = d[np.isfinite(d) & (d > 0)]
        return (float(np.min(v)), float(np.max(v)), float(np.mean(v))) if len(v) > 0 else (0.0, 0.0, 0.0)
    gt_min, gt_max, gt_mean = _stats(gt_depth)
    p1_min, p1_max, p1_mean = _stats(pred_depth1)
    p2_min, p2_max, p2_mean = _stats(pred_depth2)
    print(f"\n  Depth color scale: same for all images (0 - {max_depth:.1f} m). Blue = near, red = far.")
    print(f"  Depth ranges (if one image is mostly blue and another mostly red, ranges differ):")
    print(f"    GT:     {gt_min:.2f} - {gt_max:.2f} m  (mean {gt_mean:.2f})")
    print(f"    {display_name1}: {p1_min:.2f} - {p1_max:.2f} m  (mean {p1_mean:.2f})")
    print(f"    {display_name2}: {p2_min:.2f} - {p2_max:.2f} m  (mean {p2_mean:.2f})")
    print(f"\n  Saved visualizations to {output_dir}:")
    print(f"    - depth_ground_truth.png")
    print(f"    - depth_{name1}.png")
    print(f"    - depth_{name2}.png")
    print(f"    - input_image_with_sample_pixels.png  (circle index = table row number)")


def main():
    parser = argparse.ArgumentParser(
        description="Show how abs_rel is calculated for a sample image (same args as compare_models.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Same dataset args as compare_models
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (case-insensitive: CityScapes, DrivingStereo, middlebury, vkitti, diode, nyu, kitti)",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional path to dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-items", type=int, default=None, help="Max items to consider")
    parser.add_argument("--filter-regex", type=str, default=None, help="Regex to filter items")
    # Same model args as compare_models
    parser.add_argument("--model1", type=str, required=True, choices=["da2", "da2-revised", "da3"])
    parser.add_argument("--model2", type=str, required=True, choices=["da2", "da2-revised", "da3"])
    parser.add_argument("--model1-checkpoint", type=str, default=None)
    parser.add_argument("--model2-checkpoint", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None, choices=["metric", "basic"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", type=str, default=None)
    # Extra args for this script
    parser.add_argument("--item-index", type=int, default=None,
                        help="Index of the dataset item to use. If omitted, a random image is selected each run.")
    parser.add_argument("--num-sample-pixels", type=int, default=10,
                        help="Number of sample pixels to print (default: 10)")
    parser.add_argument("--output-dir", type=str, default="absrel_sample_output",
                        help="Directory to save depth images and input image with marked sample pixels (default: absrel_sample_output)")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  AbsRel calculation sample (same inputs as compare_models.py)")
    print("=" * 80)

    # Create dataset
    try:
        dataset = create_dataset(args)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    items = dataset.find_items()
    if not items:
        print("\n❌ No items found in dataset.")
        sys.exit(1)

    # For multi-camera datasets, items may be grouped; we take one item (first camera if grouped)
    if args.item_index is not None:
        item_index = min(args.item_index, len(items) - 1)
    else:
        item_index = random.randint(0, len(items) - 1)
    item = items[item_index]
    print(f"\n  Using item index {item_index}: {item.item_id}")

    # Load both models
    print("\n  Loading models...")
    try:
        model1, model1_checkpoint = find_model_by_name(args.model1, args.model1_checkpoint, args.device)
        model2, model2_checkpoint = find_model_by_name(args.model2, args.model2_checkpoint, args.device)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    model1_name = args.model1
    model2_name = args.model2
    slug1 = checkpoint_output_slug(model1_checkpoint)
    slug2 = checkpoint_output_slug(model2_checkpoint)
    display_name1 = display_name_for_comparison(model1_name, model1_checkpoint, slug1, model2_name)
    display_name2 = display_name_for_comparison(model2_name, model2_checkpoint, slug2, model1_name)

    is_metric1 = model1.is_metric()
    is_metric2 = model2.is_metric()
    if args.model_type is not None:
        is_metric1 = args.model_type == "metric"
        is_metric2 = is_metric1

    # Run for model 1
    print(f"\n  Running model 1: {display_name1}")
    data1, err1 = run_one_image_and_show_absrel(
        model1, display_name1, item, dataset, args.input_size, is_metric1, args.num_sample_pixels
    )
    if err1:
        print(f"\n❌ Model 1 failed: {err1}")
        sys.exit(1)

    # Run for model 2
    print(f"\n  Running model 2: {display_name2}")
    data2, err2 = run_one_image_and_show_absrel(
        model2, display_name2, item, dataset, args.input_size, is_metric2, args.num_sample_pixels
    )
    if err2:
        print(f"\n❌ Model 2 failed: {err2}")
        sys.exit(1)

    # Recompute sample pixels so both models use the same (row, col) positions.
    # We sample from the intersection of valid pixels for both models.
    gt_depth = data1["gt_depth"]
    pred1 = data1["pred_depth"]
    pred2 = data2["pred_depth"]

    valid1 = (
        np.isfinite(pred1) & np.isfinite(gt_depth)
        & (gt_depth > 0) & (pred1 > 0)
    )
    valid2 = (
        np.isfinite(pred2) & np.isfinite(gt_depth)
        & (gt_depth > 0) & (pred2 > 0)
    )
    shared_valid = valid1 & valid2
    shared_flat_idx = np.where(shared_valid.ravel())[0]
    if len(shared_flat_idx) == 0:
        print("\n❌ No shared valid pixels between the two models for this image.")
        sys.exit(1)

    n_show = min(args.num_sample_pixels, len(shared_flat_idx))
    rng = np.random.default_rng(42)
    sample_flat_indices = rng.choice(shared_flat_idx, size=n_show, replace=False)
    sample_flat_indices = np.sort(sample_flat_indices)

    h, w = gt_depth.shape
    rows = sample_flat_indices // w
    cols = sample_flat_indices % w

    def _build_sample_pixels(pred_depth_aligned):
        sample_list = []
        for r, c in zip(rows, cols):
            r_int = int(r)
            c_int = int(c)
            pred_val = float(pred_depth_aligned[r_int, c_int])
            gt_val = float(gt_depth[r_int, c_int])
            abs_rel_val = float(abs(pred_val - gt_val) / gt_val)
            sample_list.append(
                {
                    "row": r_int,
                    "col": c_int,
                    "pred_m": pred_val,
                    "gt_m": gt_val,
                    "abs_rel_pixel": abs_rel_val,
                }
            )
        return sample_list

    data1["sample_pixels"] = _build_sample_pixels(pred1)
    data2["sample_pixels"] = _build_sample_pixels(pred2)

    # Now print reports using locked pixel positions
    print_report(data1)
    print_report(data2)

    # And a single comparison table highlighting which model is closer per pixel
    print_comparison_table(data1, data2)

    # Short summary
    print("=" * 80)
    print("  Summary (same image, same formula)")
    print("=" * 80)
    print(f"  Model 1 ({display_name1}): abs_rel = {data1['abs_rel_image']:.6f} ({data1['abs_rel_image'] * 100:.4f}%)")
    print(f"  Model 2 ({display_name2}): abs_rel = {data2['abs_rel_image']:.6f} ({data2['abs_rel_image'] * 100:.4f}%)")
    print("=" * 80)

    # Save depth images and input image with sample pixels marked
    save_visualizations(
        args.output_dir,
        data1["image"],
        data1["gt_depth"],
        data1["pred_depth"],
        data2["pred_depth"],
        data1,
        data2,
        display_name1,
        display_name2,
    )
    print()


if __name__ == "__main__":
    main()
