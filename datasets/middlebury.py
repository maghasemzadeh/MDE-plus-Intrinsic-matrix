"""
Middlebury dataset implementation.
"""

import os
import re
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from scipy.stats import ttest_rel

from .base import BaseDataset, DatasetItem, DatasetConfig
from src.metrics import compute_depth_metrics
from src.paths import get_output_paths


# -------------------- Middlebury-specific utility functions --------------------

def _load_depth_array(numpy_dir: str, array_name: str) -> Optional[np.ndarray]:
    """
    Load depth array from either compressed (.npz) or uncompressed (.npy) format.
    
    Args:
        numpy_dir: Directory containing the numpy files
        array_name: Name of the array ('pred_depth', 'gt_depth', or 'error')
    
    Returns:
        Loaded numpy array or None if not found
    """
    # Try compressed format first (preferred)
    compressed_file = os.path.join(numpy_dir, "arrays.npz")
    if os.path.exists(compressed_file):
        try:
            arrays = np.load(compressed_file)
            if array_name in arrays:
                return arrays[array_name]
            # Map legacy names to compressed keys
            name_map = {
                'pred_depth_meters': 'pred_depth',
                'gt_depth_meters': 'gt_depth',
                'error': 'error'
            }
            mapped_name = name_map.get(array_name, array_name)
            if mapped_name in arrays:
                return arrays[mapped_name]
        except Exception:
            pass
    
    # Fallback to legacy .npy files
    legacy_file = os.path.join(numpy_dir, f"{array_name}.npy")
    if os.path.exists(legacy_file):
        try:
            return np.load(legacy_file)
        except Exception:
            pass
    
    return None


def read_pfm(file):
    """Read a .pfm disparity file."""
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        dims = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dims.split())
        scale = float(f.readline().decode('utf-8').rstrip())
        data = np.fromfile(f, '<f' if scale < 0 else '>f')
        data = np.reshape(data, (height, width))
        data = np.flipud(data)
        return data


def depth_from_disparity(disparity, f, baseline, doffs):
    """Convert disparity map to depth in mm."""
    depth = np.zeros_like(disparity, dtype=np.float32)
    mask = np.isfinite(disparity) & ((disparity + doffs) != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth[mask] = (baseline * f) / (disparity[mask] + doffs)
    depth[~np.isfinite(depth)] = np.nan
    return depth


def parse_calib(calib_file):
    """Parse calibration txt file to get f0,f1,cx0,cy0,cx1,cy1,doffs,baseline.
    Handles both single-line and multi-line matrix definitions.
    """
    calib = {}
    current_key = None
    current_vals = []
    
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new key-value pair
            if '=' in line:
                # If we were collecting values for a previous key, save them
                if current_key is not None:
                    calib[current_key] = current_vals
                    current_vals = []
                
                # Parse new key-value pair
                key, val = line.split('=', 1)
                key = key.strip()
                
                # Check if value contains a matrix (starts with [)
                if '[' in val:
                    # Start collecting matrix values
                    current_key = key
                    # Remove opening bracket and collect values from this line
                    val_clean = val.split('[', 1)[1]  # Get everything after '['
                    # Replace closing bracket and semicolons with spaces
                    val_clean = val_clean.replace(']', ' ').replace(';', ' ')
                    vals = val_clean.split()
                    current_vals.extend([float(v) for v in vals if v])
                    
                    # Check if matrix is complete on this line (has closing bracket)
                    if ']' in val:
                        # Matrix is complete, save it
                        calib[current_key] = current_vals
                        current_key = None
                        current_vals = []
                else:
                    # Simple single-line value
                    val_clean = val.replace('[', ' ').replace(']', ' ').replace(';', ' ')
                    vals = list(map(float, val_clean.split()))
                    calib[key] = vals
                    current_key = None
            else:
                # Continuation line (for multi-line matrices)
                if current_key is not None:
                    # This is a continuation of a matrix
                    val_clean = line.replace(']', ' ').replace(';', ' ')
                    vals = val_clean.split()
                    current_vals.extend([float(v) for v in vals if v])
                    
                    # Check if this line closes the matrix
                    if ']' in line:
                        # Matrix is complete, save it
                        calib[current_key] = current_vals
                        current_key = None
                        current_vals = []
    
    # Don't forget the last key if we were collecting values
    if current_key is not None:
        calib[current_key] = current_vals

    cam0 = calib.get('cam0')
    cam1 = calib.get('cam1')
    if cam0 is None or cam1 is None:
        raise ValueError(f"cam0 or cam1 missing in {calib_file}")

    # Check if we have enough values (should be 9 for a 3x3 matrix)
    if len(cam0) < 6:
        raise ValueError(f"cam0 in {calib_file} has insufficient values: {len(cam0)} (expected at least 6). Values: {cam0}")
    if len(cam1) < 6:
        raise ValueError(f"cam1 in {calib_file} has insufficient values: {len(cam1)} (expected at least 6). Values: {cam1}")

    # Extract values from 3x3 camera matrix [f 0 cx; 0 f cy; 0 0 1]
    # Matrix is stored row-major: [f, 0, cx, 0, f, cy, 0, 0, 1]
    # For a 3x3 matrix flattened row-major: indices are [0,1,2,3,4,5,6,7,8]
    # f = index 0 (or 4), cx = index 2, cy = index 5
    f0 = cam0[0]
    cx0 = cam0[2]
    cy0 = cam0[5] if len(cam0) > 5 else cam0[4]  # Handle different formats
    f1 = cam1[0]
    cx1 = cam1[2]
    cy1 = cam1[5] if len(cam1) > 5 else cam1[4]
    
    doffs = calib['doffs'][0] if 'doffs' in calib else (cx1 - cx0)
    baseline = calib['baseline'][0] if 'baseline' in calib else 1.0

    return f0, f1, cx0, cy0, cx1, cy1, doffs, baseline


def bootstrap_confidence_interval(diff_values: np.ndarray, n_bootstrap: int = 10000, 
                                   confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean difference.
    
    Args:
        diff_values: Array of differences (camera0 - camera1)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(diff_values) == 0 or np.all(np.isnan(diff_values)):
        return np.nan, np.nan, np.nan
    
    valid_diffs = diff_values[np.isfinite(diff_values)]
    if len(valid_diffs) == 0:
        return np.nan, np.nan, np.nan
    
    n = len(valid_diffs)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(valid_diffs, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    mean_diff = np.mean(valid_diffs)
    
    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(mean_diff), float(lower), float(upper)


def evaluate_per_image_metrics(output_path: str, max_scenes: Optional[int] = None,
                               is_metric_model: bool = True, regex_pattern: Optional[str] = None) -> Dict:
    """
    Evaluate depth metrics per image and perform statistical comparison for Middlebury dataset.
    
    Args:
        output_path: Path to output directory containing scene folders
        max_scenes: Maximum number of scenes to process (None for all)
        is_metric_model: If True, use AbsRel and RMSE. If False, use SILog.
        regex_pattern: Optional regex pattern to filter scene names
    
    Returns:
        Dictionary with evaluation results
    """
    scene_dirs = sorted([d for d in os.listdir(output_path) 
                        if os.path.isdir(os.path.join(output_path, d))])
    
    # Apply regex filter if provided
    if regex_pattern is not None:
        try:
            pattern = re.compile(regex_pattern)
            scene_dirs = [d for d in scene_dirs if pattern.search(d)]
            if len(scene_dirs) > 0:
                print(f"Regex filter '{regex_pattern}': {len(scene_dirs)} scenes match")
                print(f"  Matching scenes: {', '.join(scene_dirs[:5])}{'...' if len(scene_dirs) > 5 else ''}")
            else:
                print(f"Warning: Regex filter '{regex_pattern}' matched 0 scenes")
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{regex_pattern}': {e}")
            print("Proceeding without regex filter...")
    
    if max_scenes is not None:
        scene_dirs = scene_dirs[:max_scenes]
    
    print(f"\n{'='*80}")
    print(f"Per-Image Depth Evaluation")
    print(f"{'='*80}")
    print(f"Processing {len(scene_dirs)} scenes...")
    print(f"Model type: {'Metric' if is_metric_model else 'Non-metric (Basic)'}")
    print(f"Metrics: {'AbsRel, RMSE' if is_metric_model else 'SILog'}")
    print(f"{'='*80}\n")
    
    # Store per-image metrics
    if is_metric_model:
        all_metrics_cam0 = {'abs_rel': [], 'rmse': []}
        all_metrics_cam1 = {'abs_rel': [], 'rmse': []}
        metric_names = ['abs_rel', 'rmse']
    else:
        all_metrics_cam0 = {'silog': []}
        all_metrics_cam1 = {'silog': []}
        metric_names = ['silog']
    
    scene_names = []
    
    for scene_name in scene_dirs:
        scene_output_dir = os.path.join(output_path, scene_name)
        
        # Determine model type from folder structure
        # Try metric first, then basic
        model_type = None
        if os.path.exists(os.path.join(scene_output_dir, 'metric')):
            model_type = 'metric'
        elif os.path.exists(os.path.join(scene_output_dir, 'basic')):
            model_type = 'basic'
        else:
            # Try to find any model folder
            for possible_type in ['metric', 'basic']:
                if os.path.exists(os.path.join(scene_output_dir, possible_type)):
                    model_type = possible_type
                    break
        
        if model_type is None:
            print(f"  Skipping {scene_name}: No model folder found (metric/basic)")
            continue
        
        # Load depth maps from new folder structure
        paths0 = get_output_paths(scene_output_dir, model_type, cam='0')
        paths1 = get_output_paths(scene_output_dir, model_type, cam='1')
        
        # Load arrays using helper function (handles both compressed and uncompressed)
        pred0 = _load_depth_array(paths0['numpy_dir'], 'pred_depth_meters')
        gt0 = _load_depth_array(paths0['numpy_dir'], 'gt_depth_meters')
        pred1 = _load_depth_array(paths1['numpy_dir'], 'pred_depth_meters')
        gt1 = _load_depth_array(paths1['numpy_dir'], 'gt_depth_meters')
        
        if any(arr is None for arr in [pred0, gt0, pred1, gt1]):
            print(f"  Skipping {scene_name}: Missing depth files")
            continue
        
        # Compute metrics for each camera
        metrics0 = compute_depth_metrics(pred0, gt0, is_metric_model)
        metrics1 = compute_depth_metrics(pred1, gt1, is_metric_model)
        
        # Check if metrics are valid
        if metrics0['n_valid'] < 10 or metrics1['n_valid'] < 10:
            print(f"  Skipping {scene_name}: Insufficient valid pixels")
            continue
        
        # Store metrics
        for key in all_metrics_cam0.keys():
            if not np.isnan(metrics0[key]):
                all_metrics_cam0[key].append(metrics0[key])
            if not np.isnan(metrics1[key]):
                all_metrics_cam1[key].append(metrics1[key])
        
        scene_names.append(scene_name)
        print(f"  Processed {scene_name}: {metrics0['n_valid']} (cam0) + {metrics1['n_valid']} (cam1) valid pixels")
    
    # Aggregate results
    results = {}
    
    for metric in metric_names:
        cam0_vals = np.array(all_metrics_cam0[metric])
        cam1_vals = np.array(all_metrics_cam1[metric])
        
        # Only use scenes where both cameras have valid metrics
        valid_mask = np.isfinite(cam0_vals) & np.isfinite(cam1_vals)
        if np.sum(valid_mask) == 0:
            continue
        
        cam0_vals = cam0_vals[valid_mask]
        cam1_vals = cam1_vals[valid_mask]
        
        # Aggregate statistics
        mean0 = np.mean(cam0_vals)
        std0 = np.std(cam0_vals, ddof=1)
        mean1 = np.mean(cam1_vals)
        std1 = np.std(cam1_vals, ddof=1)
        
        # Difference
        diff = cam0_vals - cam1_vals
        mean_diff = np.mean(diff)
        
        # Paired t-test
        try:
            if len(diff) > 1:
                t_stat, t_pvalue = ttest_rel(cam0_vals, cam1_vals)
            else:
                t_stat, t_pvalue = np.nan, np.nan
        except:
            t_stat, t_pvalue = np.nan, np.nan
        
        # Bootstrap confidence interval
        mean_diff_ci, lower_ci, upper_ci = bootstrap_confidence_interval(diff)
        
        # Determine if difference is meaningful (p < 0.05)
        is_significant = not np.isnan(t_pvalue) and t_pvalue < 0.05
        
        results[metric] = {
            'cam0_mean': float(mean0),
            'cam0_std': float(std0),
            'cam1_mean': float(mean1),
            'cam1_std': float(std1),
            'mean_diff': float(mean_diff),
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            't_pvalue': float(t_pvalue) if not np.isnan(t_pvalue) else None,
            'bootstrap_mean_diff': float(mean_diff_ci) if not np.isnan(mean_diff_ci) else None,
            'bootstrap_ci_lower': float(lower_ci) if not np.isnan(lower_ci) else None,
            'bootstrap_ci_upper': float(upper_ci) if not np.isnan(upper_ci) else None,
            'is_significant': bool(is_significant),
            'n_images': int(len(cam0_vals))
        }
    
    results['num_scenes'] = len(scene_names)
    results['scenes'] = scene_names
    results['is_metric_model'] = is_metric_model
    
    return results


def print_evaluation_results(results: Dict):
    """Print evaluation results in a clear format."""
    print(f"\n{'='*80}")
    print("DEPTH EVALUATION RESULTS: Camera 0 vs Camera 1")
    print(f"{'='*80}\n")
    
    is_metric = results.get('is_metric_model', True)
    
    if is_metric:
        metric_names = ['abs_rel', 'rmse']
        metric_labels = {
            'abs_rel': 'AbsRel',
            'rmse': 'RMSE (m)'
        }
    else:
        metric_names = ['silog']
        metric_labels = {
            'silog': 'SILog'
        }
    
    all_significant = []
    
    for metric in metric_names:
        if metric not in results:
            continue
        
        r = results[metric]
        label = metric_labels[metric]
        
        print(f"\n{'─'*80}")
        print(f"📊 {label}")
        print(f"{'─'*80}")
        print(f"  Camera 0 (Left):  {r['cam0_mean']:>10.6f} ± {r['cam0_std']:>10.6f}")
        print(f"  Camera 1 (Right): {r['cam1_mean']:>10.6f} ± {r['cam1_std']:>10.6f}")
        print(f"\n  Difference (Camera 0 - Camera 1): {r['mean_diff']:>10.6f}")
        
        if r['t_pvalue'] is not None:
            print(f"  t-test p-value: {r['t_pvalue']:>10.6f}")
            if r['bootstrap_ci_lower'] is not None and r['bootstrap_ci_upper'] is not None:
                print(f"  95% CI:         [{r['bootstrap_ci_lower']:>8.6f}, {r['bootstrap_ci_upper']:>8.6f}]")
            
            if r['is_significant']:
                print(f"\n  {'✓'*3} SIGNIFICANT DIFFERENCE (p < 0.05) {'✓'*3}")
                all_significant.append(True)
            else:
                print(f"\n  {'○'*3} NO SIGNIFICANT DIFFERENCE (p >= 0.05) {'○'*3}")
                all_significant.append(False)
        
        print(f"  Scenes analyzed: {r['n_images']}")
        print()
    
    # Final summary
    print(f"{'═'*80}")
    print("📋 CONCLUSION")
    print(f"{'═'*80}")
    if len(all_significant) > 0:
        if all(all_significant):
            print("  ✓ There IS a statistically significant difference between Camera 0 and Camera 1")
            print("    → Camera intrinsics DO affect depth estimation performance")
        else:
            print("  ○ There is NO statistically significant difference between Camera 0 and Camera 1")
            print("    → Camera intrinsics do NOT significantly affect depth estimation performance")
    else:
        print("  ⚠ Unable to determine (insufficient data)")
    print(f"{'═'*80}\n")
    
    print(f"Total scenes evaluated: {results['num_scenes']}")
    print()


class MiddleburyDataset(BaseDataset):
    """
    Middlebury dataset for depth estimation evaluation.
    
    Middlebury provides stereo pairs with calibration data.
    Each scene has two cameras (left and right) with different intrinsics.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize Middlebury dataset.
        
        Args:
            config: Dataset configuration (includes regex_filter for filtering scenes)
        """
        super().__init__(config)
    
    def get_default_path(self) -> str:
        """Get default Middlebury dataset path."""
        import os
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'middlebury')
    
    def find_items(self) -> List[DatasetItem]:
        """Find all Middlebury scenes."""
        items = []
        
        scene_list = sorted([d for d in os.listdir(self.dataset_path) 
                           if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        # Apply regex filter if provided
        if self.regex_filter is not None:
            try:
                pattern = re.compile(self.regex_filter)
                scene_list = [d for d in scene_list if pattern.search(d)]
                if len(scene_list) > 0:
                    print(f"Regex filter '{self.regex_filter}': {len(scene_list)} scenes match")
                else:
                    print(f"Warning: Regex filter '{self.regex_filter}' matched 0 scenes")
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{self.regex_filter}': {e}")
                print("Proceeding without regex filter...")
        
        for scene_name in scene_list:
            scene_path = os.path.join(self.dataset_path, scene_name)
            if not os.path.isdir(scene_path):
                continue
            
            calib_file = os.path.join(scene_path, 'calib.txt')
            if not os.path.exists(calib_file):
                continue
            
            # Parse calibration to get camera parameters
            try:
                f0, f1, cx0, cy0, cx1, cy1, doffs, baseline = parse_calib(calib_file)
                
                # Check if both images exist
                im0_path = os.path.join(scene_path, 'im0.png')
                im1_path = os.path.join(scene_path, 'im1.png')
                disp0_path = os.path.join(scene_path, 'disp0.pfm')
                disp1_path = os.path.join(scene_path, 'disp1.pfm')
                
                if not (os.path.exists(im0_path) and os.path.exists(im1_path) and
                       os.path.exists(disp0_path) and os.path.exists(disp1_path)):
                    continue
                
                # Create items for each camera
                # Camera 0 (left)
                items.append(DatasetItem(
                    item_id=scene_name,
                    image_path=im0_path,
                    gt_path=disp0_path,
                    camera_id='0',
                    metadata={
                        'f': f0,
                        'cx': cx0,
                        'cy': cy0,
                        'baseline': baseline,
                        'doffs': doffs,
                        'scene_path': scene_path
                    }
                ))
                
                # Camera 1 (right)
                items.append(DatasetItem(
                    item_id=scene_name,
                    image_path=im1_path,
                    gt_path=disp1_path,
                    camera_id='1',
                    metadata={
                        'f': f1,
                        'cx': cx1,
                        'cy': cy1,
                        'baseline': baseline,
                        'doffs': doffs,
                        'scene_path': scene_path
                    }
                ))
            except Exception as e:
                print(f"Warning: Could not parse calibration for {scene_name}: {e}")
                continue
        
        if self.max_items is not None:
            items = items[:self.max_items]
        
        print(f"Found {len(items)} Middlebury camera items ({len(items)//2} scenes)")
        return items
    
    def load_gt_depth(self, gt_path: str, item: DatasetItem) -> np.ndarray:
        """Load Middlebury disparity and convert to depth in meters."""
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Disparity file not found: {gt_path}")
        
        # Read PFM disparity
        disparity = read_pfm(gt_path)
        
        # Get calibration parameters from metadata
        f = item.metadata['f']
        baseline = item.metadata['baseline']
        doffs = item.metadata['doffs']
        
        # Convert disparity to depth (in mm)
        depth_mm = depth_from_disparity(disparity, f, baseline, doffs)
        
        # Convert to meters
        depth_m = (depth_mm / 1000.0).astype(np.float32)
        
        return depth_m
    
    def get_output_subdir(self) -> str:
        """Get output subdirectory name."""
        return 'middlebury'
    
    def supports_multiple_cameras(self) -> bool:
        """Middlebury supports multiple cameras per scene."""
        return True
    
    def get_camera_ids(self, item: DatasetItem) -> List[str]:
        """Get camera IDs for a scene."""
        # For Middlebury, we need to find all cameras for the same scene
        scene_name = item.item_id
        scene_path = os.path.join(self.dataset_path, scene_name)
        
        cameras = []
        for cam_id in ['0', '1']:
            im_path = os.path.join(scene_path, f'im{cam_id}.png')
            if os.path.exists(im_path):
                cameras.append(cam_id)
        
        return cameras if cameras else ['0']

