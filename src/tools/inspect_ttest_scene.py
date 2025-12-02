"""
Inspect intermediate Middlebury evaluation artifacts and mirror the metric logic
from datasets.middlebury.

By default the script auto-selects the first scene directory inside
`results/middlebury`, prints step-by-step samples (depths, errors, masks, Welch
test), then reports AbsRel and RMSE for the scene. Finally, it aggregates *all*
scene folders in the base directory to show how the combined Welch t-test,
AbsRel, and RMSE statistics are derived across the dataset.

Examples:
    # Auto-pick first scene under results/middlebury
    python src/tools/inspect_ttest_scene.py

    # Inspect specific scene and limit aggregation to first 5 folders
    python src/tools/inspect_ttest_scene.py --scene Backpack-perfect --num-scenes 5

    # Provide an explicit folder path
    python src/tools/inspect_ttest_scene.py --folder results/middlebury/Adirondack-perfect
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.stats import ttest_ind, ttest_rel


###############################################################################
# Data containers
###############################################################################


@dataclass
class SceneArrays:
    pred0: np.ndarray
    gt0: np.ndarray
    err0: np.ndarray
    pred1: np.ndarray
    gt1: np.ndarray
    err1: np.ndarray


###############################################################################
# Loading helpers
###############################################################################


def list_scene_folders(base_dir: Path) -> Iterable[Path]:
    return sorted(p for p in base_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def resolve_scene_folder(base_dir: Path, folder_arg: Optional[Path], scene_name: Optional[str]) -> Path:
    if folder_arg is not None:
        return folder_arg
    if scene_name is not None:
        candidate = base_dir / scene_name
        if not candidate.exists():
            raise FileNotFoundError(f"Scene '{scene_name}' not found under {base_dir}")
        return candidate
    folders = list(list_scene_folders(base_dir))
    if not folders:
        raise FileNotFoundError(f"No scene folders found in {base_dir}")
    print(f"No folder provided; automatically selected scene: {folders[0]}")
    return folders[0]


def _load_array_from_path(path: Path, array_name: str) -> Optional[np.ndarray]:
    """Load array from either compressed (.npz) or uncompressed (.npy) format."""
    # Try compressed format first
    if path.is_dir():
        compressed_file = path / "arrays.npz"
        if compressed_file.exists():
            try:
                arrays = np.load(compressed_file)
                name_map = {
                    'pred_depth_meters': 'pred_depth',
                    'gt_depth_meters': 'gt_depth',
                    'error': 'error'
                }
                key = name_map.get(array_name, array_name)
                if key in arrays:
                    return arrays[key]
            except Exception:
                pass
        # Try legacy .npy file
        legacy_file = path / f"{array_name}.npy"
        if legacy_file.exists():
            try:
                return np.load(legacy_file)
            except Exception:
                pass
    else:
        # Direct file path
        if path.exists():
            try:
                if path.suffix == '.npz':
                    arrays = np.load(path)
                    # Try common keys
                    for key in ['pred_depth', 'gt_depth', 'error', array_name]:
                        if key in arrays:
                            return arrays[key]
                else:
                    return np.load(path)
            except Exception:
                pass
    return None


def load_scene(folder: Path) -> SceneArrays:
    """Load scene arrays, checking both root-level and nested metric/ structure."""
    # Try nested structure with compressed format first (new structure: metric/disp*/numpy_matrix/)
    numpy_dirs_nested = {
        "dir0": folder / "metric" / "disp0" / "numpy_matrix",
        "dir1": folder / "metric" / "disp1" / "numpy_matrix",
    }
    
    # Try nested structure with legacy .npy files
    paths_nested = {
        "pred0": folder / "metric" / "disp0" / "numpy_matrix" / "pred_depth_meters.npy",
        "gt0": folder / "metric" / "disp0" / "numpy_matrix" / "gt_depth_meters.npy",
        "err0": folder / "metric" / "disp0" / "numpy_matrix" / "error.npy",
        "pred1": folder / "metric" / "disp1" / "numpy_matrix" / "pred_depth_meters.npy",
        "gt1": folder / "metric" / "disp1" / "numpy_matrix" / "gt_depth_meters.npy",
        "err1": folder / "metric" / "disp1" / "numpy_matrix" / "error.npy",
    }
    
    # Try root-level (old structure)
    paths_root = {
        "pred0": folder / "disp0_pred_depth_meters.npy",
        "gt0": folder / "disp0_gt_depth_meters.npy",
        "err0": folder / "disp0_error.npy",
        "pred1": folder / "disp1_pred_depth_meters.npy",
        "gt1": folder / "disp1_gt_depth_meters.npy",
        "err1": folder / "disp1_error.npy",
    }
    
    # Try compressed format in nested structure first
    if numpy_dirs_nested["dir0"].exists() and numpy_dirs_nested["dir1"].exists():
        pred0 = _load_array_from_path(numpy_dirs_nested["dir0"], "pred_depth_meters")
        gt0 = _load_array_from_path(numpy_dirs_nested["dir0"], "gt_depth_meters")
        err0 = _load_array_from_path(numpy_dirs_nested["dir0"], "error")
        pred1 = _load_array_from_path(numpy_dirs_nested["dir1"], "pred_depth_meters")
        gt1 = _load_array_from_path(numpy_dirs_nested["dir1"], "gt_depth_meters")
        err1 = _load_array_from_path(numpy_dirs_nested["dir1"], "error")
        
        if all(arr is not None for arr in [pred0, gt0, err0, pred1, gt1, err1]):
            return SceneArrays(pred0=pred0, gt0=gt0, err0=err0, pred1=pred1, gt1=gt1, err1=err1)
    
    # Try nested structure with legacy .npy files
    nested_exists = all(p.exists() for p in paths_nested.values())
    if nested_exists:
        return SceneArrays(
            pred0=np.load(paths_nested["pred0"]),
            gt0=np.load(paths_nested["gt0"]),
            err0=np.load(paths_nested["err0"]),
            pred1=np.load(paths_nested["pred1"]),
            gt1=np.load(paths_nested["gt1"]),
            err1=np.load(paths_nested["err1"]),
        )
    
    # Try root-level structure
    root_exists = all(p.exists() for p in paths_root.values())
    if root_exists:
        return SceneArrays(
            pred0=np.load(paths_root["pred0"]),
            gt0=np.load(paths_root["gt0"]),
            err0=np.load(paths_root["err0"]),
            pred1=np.load(paths_root["pred1"]),
            gt1=np.load(paths_root["gt1"]),
            err1=np.load(paths_root["err1"]),
        )
    
    # If we get here, nothing worked
    missing_root = [k for k, p in paths_root.items() if not p.exists()]
    missing_nested = [k for k, p in paths_nested.items() if not p.exists()]
    raise FileNotFoundError(
        f"Missing files in {folder}.\n"
        f"Root structure missing: {missing_root}\n"
        f"Nested structure missing: {missing_nested}"
    )


###############################################################################
# Reporting utilities
###############################################################################


def compute_stats(arr: np.ndarray, label: str) -> None:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"  {label}: no finite values found.")
        return
    print(
        f"  {label}: min={finite.min():.4f}, max={finite.max():.4f}, "
        f"mean={finite.mean():.4f}, std={finite.std(ddof=1):.4f}"
    )


def select_patch(mask: np.ndarray, patch_size: int = 5) -> Tuple[slice, slice]:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return slice(0, patch_size), slice(0, patch_size)
    center_idx = indices[indices.shape[0] // 2]
    half = patch_size // 2
    r, c = center_idx
    r_start = max(r - half, 0)
    c_start = max(c - half, 0)
    r_end = min(r_start + patch_size, mask.shape[0])
    c_end = min(c_start + patch_size, mask.shape[1])
    r_start = max(r_end - patch_size, 0)
    c_start = max(c_end - patch_size, 0)
    return slice(r_start, r_end), slice(c_start, c_end)


def show_patch(scene: SceneArrays, row_slice: slice, col_slice: slice) -> None:
    print("Sample 5x5 window (rows, cols): "
          f"{row_slice.start}:{row_slice.stop}, {col_slice.start}:{col_slice.stop}")

    def fmt(arr: np.ndarray) -> str:
        return np.array2string(arr[row_slice, col_slice], precision=4, suppress_small=False)

    print("  Cam0 predicted depth (m):")
    print(f"    {fmt(scene.pred0)}")
    print("  Cam0 ground-truth depth (m):")
    print(f"    {fmt(scene.gt0)}")
    print("  Cam0 error = |pred - gt| (m):")
    print(f"    {fmt(scene.err0)}")
    print("  Cam0 valid mask (finite errors):")
    print(f"    {fmt(np.isfinite(scene.err0))}")

    print("  Cam1 predicted depth (m):")
    print(f"    {fmt(scene.pred1)}")
    print("  Cam1 ground-truth depth (m):")
    print(f"    {fmt(scene.gt1)}")
    print("  Cam1 error = |pred - gt| (m):")
    print(f"    {fmt(scene.err1)}")
    print("  Cam1 valid mask (finite errors):")
    print(f"    {fmt(np.isfinite(scene.err1))}")


def compute_per_camera_metrics(scene: SceneArrays) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for cam, pred, gt in [
        ("0", scene.pred0, scene.gt0),
        ("1", scene.pred1, scene.gt1),
    ]:
        valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0)
        if np.sum(valid) == 0:
            metrics[cam] = {"abs_rel": float("nan"), "rmse": float("nan"), "n_valid": 0}
            continue
        diff = pred[valid] - gt[valid]
        abs_rel = np.mean(np.abs(diff) / gt[valid])
        rmse = np.sqrt(np.mean(diff ** 2))
        metrics[cam] = {
            "abs_rel": float(abs_rel),
            "rmse": float(rmse),
            "n_valid": int(np.sum(valid)),
        }
    return metrics


def welch_t_test(err0: np.ndarray, err1: np.ndarray) -> Tuple[float, float, int, int]:
    mask0 = np.isfinite(err0)
    mask1 = np.isfinite(err1)
    finite0 = err0[mask0]
    finite1 = err1[mask1]
    if finite0.size == 0 or finite1.size == 0:
        raise ValueError("One of the error arrays has no finite values.")
    stat, pvalue = ttest_ind(finite0, finite1, equal_var=False)
    return stat, pvalue, finite0.size, finite1.size


###############################################################################
# Scene-level inspection
###############################################################################


def inspect_scene(folder: Path) -> Dict[str, Dict[str, float]]:
    print(f"Scene folder: {folder}")
    scene = load_scene(folder)

    print("\nStep 1. Metric depth prediction shapes and dtype:")
    print(f"  Cam0 pred shape={scene.pred0.shape}, dtype={scene.pred0.dtype}")
    print(f"  Cam1 pred shape={scene.pred1.shape}, dtype={scene.pred1.dtype}")

    print("\nStep 2. Ground-truth depth statistics:")
    compute_stats(scene.gt0, "Cam0 GT depth")
    compute_stats(scene.gt1, "Cam1 GT depth")

    print("\nStep 3. Error matrices (|pred - gt|) statistics:")
    compute_stats(scene.err0, "Cam0 error")
    compute_stats(scene.err1, "Cam1 error")

    print("\nStep 4. Valid pixel masks:")
    mask0 = np.isfinite(scene.err0)
    mask1 = np.isfinite(scene.err1)
    total_pixels = scene.err0.size
    print(
        f"  Cam0 valid pixels: {mask0.sum():,} / {total_pixels:,} ({mask0.mean() * 100:.4f}% valid)"
    )
    print(
        f"  Cam1 valid pixels: {mask1.sum():,} / {total_pixels:,} ({mask1.mean() * 100:.4f}% valid)"
    )
    print("  Valid mask computed via np.isfinite(error_matrix).")

    print("\nStep 5. Sample 5x5 patch (centred on a valid pixel):")
    row_slice, col_slice = select_patch(mask0 & mask1, patch_size=5)
    show_patch(scene, row_slice, col_slice)

    print("\nStep 6. Welch t-test on finite error values:")
    stat, pvalue, n0, n1 = welch_t_test(scene.err0, scene.err1)
    mean0 = float(scene.err0[mask0].mean())
    mean1 = float(scene.err1[mask1].mean())
    print(f"  Cam0 finite count: {n0:,}")
    print(f"  Cam1 finite count: {n1:,}")
    print(f"  Mean error (Cam0): {mean0:.6f} m")
    print(f"  Mean error (Cam1): {mean1:.6f} m")
    print(f"  Welch t-statistic: {stat:.6f}")
    print(f"  p-value: {pvalue:.6g}")
    if pvalue == 0.0:
        print("  Note: A printed p-value of 0.0 indicates underflow (extremely small).")

    print("\nStep 7. Scene-level AbsRel and RMSE (per middlebury.py formulas):")
    metrics = compute_per_camera_metrics(scene)
    for cam in ("0", "1"):
        vals = metrics[cam]
        print(f"  Camera {cam}: n_valid={vals['n_valid']:,}")
        print(f"    AbsRel = mean(|pred - gt| / gt) = {vals['abs_rel']:.6f}")
        print(f"    RMSE   = sqrt(mean((pred - gt)^2)) = {vals['rmse']:.6f} m")

    return {
        "errors": {
            "cam0": scene.err0[mask0],
            "cam1": scene.err1[mask1],
            "stat": stat,
            "pvalue": pvalue,
            "mean0": mean0,
            "mean1": mean1,
            "n0": n0,
            "n1": n1,
        },
        "metrics": metrics,
    }


###############################################################################
# Aggregated metrics across folders
###############################################################################


def aggregate_metrics(base_dir: Path, limit: Optional[int] = None) -> None:
    print("\nStep 8. Aggregated statistics across scene folders:")
    scene_paths = list(list_scene_folders(base_dir))
    if limit is not None:
        scene_paths = scene_paths[:limit]
    if not scene_paths:
        print(f"  No scene folders found in {base_dir}, skipping aggregation.")
        return

    err0_all = []
    err1_all = []
    absrel0 = []
    absrel1 = []
    rmse0 = []
    rmse1 = []
    processed = 0

    for scene_folder in scene_paths:
        try:
            scene = load_scene(scene_folder)
        except FileNotFoundError:
            continue
        metrics = compute_per_camera_metrics(scene)
        mask0 = np.isfinite(scene.err0)
        mask1 = np.isfinite(scene.err1)

        if np.any(mask0):
            err0_all.append(scene.err0[mask0])
        if np.any(mask1):
            err1_all.append(scene.err1[mask1])

        if metrics["0"]["n_valid"] > 0:
            absrel0.append(metrics["0"]["abs_rel"])
            rmse0.append(metrics["0"]["rmse"])
        if metrics["1"]["n_valid"] > 0:
            absrel1.append(metrics["1"]["abs_rel"])
            rmse1.append(metrics["1"]["rmse"])

        processed += 1

    if processed == 0 or not err0_all or not err1_all:
        print("  Insufficient data to compute aggregated statistics.")
        return

    err0_all = np.concatenate(err0_all)
    err1_all = np.concatenate(err1_all)
    stat, pvalue = ttest_ind(err0_all, err1_all, equal_var=False)
    print(f"  Scenes aggregated: {processed}")
    print(f"  Combined valid pixels: Cam0={err0_all.size:,}, Cam1={err1_all.size:,}")
    print(f"  Error means: Cam0={err0_all.mean():.6f} m, Cam1={err1_all.mean():.6f} m")
    print(f"  Welch t-statistic (all scenes): {stat:.6f}")
    print(f"  p-value: {pvalue:.6g}")
    if pvalue == 0.0:
        print("  Note: p-value underflowed to 0.0 (extremely small).")

    if absrel0 and absrel1:
        absrel0_arr = np.array(absrel0)
        absrel1_arr = np.array(absrel1)
        diff = absrel0_arr - absrel1_arr
        if diff.size > 1:
            t_stat, t_p = ttest_rel(absrel0_arr, absrel1_arr)
        else:
            t_stat, t_p = float("nan"), float("nan")
        print("\n  AbsRel (scene means mirroring evaluate_per_image_metrics):")
        print(f"    Camera 0 mean={absrel0_arr.mean():.6f}, std={absrel0_arr.std(ddof=1):.6f}")
        print(f"    Camera 1 mean={absrel1_arr.mean():.6f}, std={absrel1_arr.std(ddof=1):.6f}")
        print(f"    Mean difference (Cam0 - Cam1)={diff.mean():.6f}")
        if np.isfinite(t_stat):
            print(f"    Paired t-test t={t_stat:.6f}, p={t_p:.6g}")

    if rmse0 and rmse1:
        rmse0_arr = np.array(rmse0)
        rmse1_arr = np.array(rmse1)
        diff = rmse0_arr - rmse1_arr
        if diff.size > 1:
            t_stat, t_p = ttest_rel(rmse0_arr, rmse1_arr)
        else:
            t_stat, t_p = float("nan"), float("nan")
        print("\n  RMSE (scene means mirroring evaluate_per_image_metrics):")
        print(f"    Camera 0 mean={rmse0_arr.mean():.6f} m, std={rmse0_arr.std(ddof=1):.6f}")
        print(f"    Camera 1 mean={rmse1_arr.mean():.6f} m, std={rmse1_arr.std(ddof=1):.6f}")
        print(f"    Mean difference (Cam0 - Cam1)={diff.mean():.6f} m")
        if np.isfinite(t_stat):
            print(f"    Paired t-test t={t_stat:.6f}, p={t_p:.6g}")


###############################################################################
# Entrypoint
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Middlebury scene outputs and aggregate statistics."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/middlebury"),
        help="Directory containing scene subfolders.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Explicit path to a scene folder (overrides --scene).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name inside base-dir to inspect.",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=None,
        help="Limit the number of scene folders included in the aggregated summary.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    folder_arg = args.folder.resolve() if args.folder else None
    scene_folder = resolve_scene_folder(base_dir, folder_arg, args.scene)
    inspect_scene(scene_folder.resolve())
    aggregate_metrics(base_dir, limit=args.num_scenes)


if __name__ == "__main__":
    main()

