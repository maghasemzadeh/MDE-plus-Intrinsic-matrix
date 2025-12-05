"""
Compare two depth estimation models on a dataset.

This script evaluates two models on the same dataset and compares their
error and precision metrics using statistical tests.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from scipy.stats import ttest_ind
from tqdm import tqdm

from datasets import (
    BaseDataset, DatasetConfig,
    CityscapesDataset,
    DrivingStereoDataset,
    MiddleburyDataset,
    VKITTIDataset
)
from models import BaseDepthModelWrapper, DepthAnythingV2Wrapper
from src import ProcessingPipeline, compute_depth_metrics


def identify_model_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Identify model configuration from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dictionary with model configuration (model_type, encoder, max_depth)
    """
    # Try to load checkpoint to get info
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Try to infer encoder from state dict
        encoder = 'vitl'  # default
        if 'pretrained.patch_embed.proj.weight' in state_dict:
            weight_shape = state_dict['pretrained.patch_embed.proj.weight'].shape
            if weight_shape[0] == 384:  # vitg
                encoder = 'vitg'
            elif weight_shape[0] == 256:  # vitl
                encoder = 'vitl'
            elif weight_shape[0] == 192:
                # Could be vitb or vits, check more
                if 'pretrained.blocks.11' in state_dict:  # vitb has 12 blocks
                    encoder = 'vitb'
                else:  # vits has fewer blocks
                    encoder = 'vits'
        
        # Check if it's a metric model (has depth_head)
        model_type = 'metric'  # default
        if 'depth_head' in str(state_dict.keys()):
            model_type = 'metric'
        else:
            model_type = 'basic'
        
        # Infer max_depth from checkpoint metadata or filename
        max_depth = 80.0  # default outdoor
        if isinstance(checkpoint, dict) and 'previous_best' in checkpoint:
            # This is a trained checkpoint, likely metric
            model_type = 'metric'
        
    except Exception as e:
        # If we can't load, infer from filename
        model_type = 'metric'
        encoder = 'vitl'
        max_depth = 80.0
    
    # Infer from filename as fallback
    filename = os.path.basename(checkpoint_path).lower()
    dirname = os.path.basename(os.path.dirname(checkpoint_path)).lower()
    
    # Check for encoder in filename or directory
    for enc in ['vitg', 'vitl', 'vitb', 'vits']:
        if enc in filename or enc in dirname:
            encoder = enc
            break
    
    # Check for model type
    if 'basic' in filename or 'basic' in dirname:
        model_type = 'basic'
        max_depth = 20.0
    elif 'metric' in filename or 'metric' in dirname:
        model_type = 'metric'
        # Check for indoor/outdoor
        if 'hypersim' in filename or 'hypersim' in dirname:
            max_depth = 20.0
        elif 'vkitti' in filename or 'vkitti' in dirname or 'cityscapes' in dirname or 'drivingstereo' in dirname:
            max_depth = 80.0
        else:
            max_depth = 80.0  # default outdoor
    
    return {
        'model_type': model_type,
        'encoder': encoder,
        'max_depth': max_depth
    }


def find_model_by_name(model_name: str, explicit_checkpoint: Optional[str] = None) -> Tuple[str, Dict]:
    """
    Find checkpoint for a standard model by name.
    
    Supported model names:
    - da2: DepthAnythingV2 (original)
    - da2-revised: DepthAnythingV2-revised
    - da3: Depth-Anything-3
    
    Args:
        model_name: Model name ('da2', 'da3', 'da2-revised')
        explicit_checkpoint: Optional explicit checkpoint path override
    
    Returns:
        Tuple of (checkpoint_path, model_config_dict)
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # If explicit checkpoint provided, use it
    if explicit_checkpoint:
        if os.path.isabs(explicit_checkpoint):
            checkpoint_path = explicit_checkpoint
        else:
            checkpoint_path = os.path.join(project_root, explicit_checkpoint)
        
        if os.path.exists(checkpoint_path):
            config = identify_model_from_checkpoint(checkpoint_path)
            return checkpoint_path, config
        else:
            raise FileNotFoundError(f"Explicit checkpoint not found: {explicit_checkpoint}")
    
    # Map model names to their checkpoint directories
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'da2':
        # DepthAnythingV2 - check in original v2 checkpoints
        checkpoints_dir = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2', 'checkpoints')
        # Try to find any available checkpoint (prefer metric, then basic)
        for encoder in ['vitl', 'vitb', 'vits', 'vitg']:
            for ckpt_name in [
                f'depth_anything_v2_metric_hypersim_{encoder}.pth',
                f'depth_anything_v2_metric_vkitti_{encoder}.pth',
                f'depth_anything_v2_{encoder}.pth'
            ]:
                checkpoint_path = os.path.join(checkpoints_dir, ckpt_name)
                if os.path.exists(checkpoint_path):
                    config = identify_model_from_checkpoint(checkpoint_path)
                    return checkpoint_path, config
    
    elif model_name_lower == 'da2-revised':
        # DepthAnythingV2-revised - check in revised v2 checkpoints
        checkpoints_dir = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised', 'checkpoints')
        # Try to find any available checkpoint (prefer metric, then basic)
        for encoder in ['vitl', 'vitb', 'vits', 'vitg']:
            for ckpt_name in [
                f'depth_anything_v2_metric_hypersim_{encoder}.pth',
                f'depth_anything_v2_metric_vkitti_{encoder}.pth',
                f'depth_anything_v2_{encoder}.pth'
            ]:
                checkpoint_path = os.path.join(checkpoints_dir, ckpt_name)
                if os.path.exists(checkpoint_path):
                    config = identify_model_from_checkpoint(checkpoint_path)
                    return checkpoint_path, config
        
        # Also check in project checkpoints for trained models
        project_checkpoints_dir = os.path.join(project_root, 'checkpoints')
        if os.path.isdir(project_checkpoints_dir):
            for subdir in os.listdir(project_checkpoints_dir):
                subdir_path = os.path.join(project_checkpoints_dir, subdir)
                if os.path.isdir(subdir_path):
                    best_path = os.path.join(subdir_path, 'best.pth')
                    latest_path = os.path.join(subdir_path, 'latest.pth')
                    if os.path.exists(best_path):
                        config = identify_model_from_checkpoint(best_path)
                        print(f"Found trained da2-revised model in {subdir}")
                        return best_path, config
                    elif os.path.exists(latest_path):
                        print(f"⚠️  Warning: best.pth not found for da2-revised (found in {subdir}), using latest.pth instead")
                        config = identify_model_from_checkpoint(latest_path)
                        return latest_path, config
    
    elif model_name_lower == 'da3':
        # Depth-Anything-3 - check in DA3 checkpoints
        checkpoints_dir = os.path.join(project_root, 'models', 'raw_models', 'Depth-Anything-3', 'checkpoints')
        # DA3 might have different checkpoint naming, search for any .pth files
        if os.path.isdir(checkpoints_dir):
            for file in os.listdir(checkpoints_dir):
                if file.endswith('.pth'):
                    checkpoint_path = os.path.join(checkpoints_dir, file)
                    config = identify_model_from_checkpoint(checkpoint_path)
                    return checkpoint_path, config
    
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Supported models: 'da2', 'da2-revised', 'da3'"
        )
    
    # If we get here, no checkpoint was found
    raise FileNotFoundError(
        f"Could not find checkpoint for model '{model_name}'.\n"
        f"Please ensure the model checkpoints are available in the standard locations:\n"
        f"  - da2: models/raw_models/DepthAnythingV2/checkpoints/\n"
        f"  - da2-revised: models/raw_models/DepthAnythingV2-revised/checkpoints/ or checkpoints/\n"
        f"  - da3: models/raw_models/Depth-Anything-3/checkpoints/\n"
        f"\nOr specify --model1-checkpoint/--model2-checkpoint to provide explicit paths."
    )


def compare_model_metrics(
    metrics1: List[Dict[str, float]],
    metrics2: List[Dict[str, float]],
    model1_name: str,
    model2_name: str,
    dataset_name: str,
    output_dir: str
) -> Dict:
    """
    Compare metrics from two models using statistical tests.
    
    Args:
        metrics1: List of metric dictionaries from first model
        metrics2: List of metric dictionaries from second model
        model1_name: Name of first model
        model2_name: Name of second model
        dataset_name: Name of dataset
        output_dir: Output directory for saving results
    
    Returns:
        Dictionary with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON: {model1_name} vs {model2_name}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Determine which metrics to compare
    sample_metrics1 = metrics1[0] if metrics1 else {}
    sample_metrics2 = metrics2[0] if metrics2 else {}
    
    has_abs_rel = 'abs_rel' in sample_metrics1 or 'abs_rel' in sample_metrics2
    has_silog = 'silog' in sample_metrics1 or 'silog' in sample_metrics2
    
    if has_abs_rel:
        metric_names = ['abs_rel', 'rmse']
        metric_labels = {
            'abs_rel': 'Absolute Relative Error',
            'rmse': 'RMSE (meters)'
        }
    elif has_silog:
        metric_names = ['silog']
        metric_labels = {
            'silog': 'SILog (Scale-Invariant Log RMSE)'
        }
    else:
        metric_names = ['abs_rel', 'rmse']
        metric_labels = {
            'abs_rel': 'Absolute Relative Error',
            'rmse': 'RMSE (meters)'
        }
    
    results = {}
    
    for metric in metric_names:
        if metric not in metric_labels:
            continue
        
        label = metric_labels[metric]
        
        # Extract metric values
        vals1 = np.array([m[metric] for m in metrics1 if metric in m and not np.isnan(m[metric])])
        vals2 = np.array([m[metric] for m in metrics2 if metric in m and not np.isnan(m[metric])])
        
        if len(vals1) == 0 or len(vals2) == 0:
            print(f"⚠️  Warning: Insufficient data for {label}")
            continue
        
        n1, n2 = len(vals1), len(vals2)
        
        # Robustness checks
        if n1 < 30 or n2 < 30:
            print(f"⚠️  Warning: Small sample sizes for {label} ({model1_name}: n={n1}, {model2_name}: n={n2}). "
                  f"t-test results may be less reliable with n < 30.")
        
        size_ratio = max(n1, n2) / min(n1, n2) if min(n1, n2) > 0 else float('inf')
        if size_ratio > 5:
            print(f"⚠️  Warning: Large sample size difference for {label} (ratio: {size_ratio:.1f}x). "
                  f"Welch's t-test handles this, but results should be interpreted with caution.")
        
        # Compute statistics
        mean1 = np.mean(vals1)
        std1 = np.std(vals1, ddof=1)
        median1 = np.median(vals1)
        mean2 = np.mean(vals2)
        std2 = np.std(vals2, ddof=1)
        median2 = np.median(vals2)
        
        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = ttest_ind(vals1, vals2, equal_var=False)
        
        # Bootstrap confidence interval for mean difference
        mean_diff = mean1 - mean2
        bootstrap_diffs = []
        bootstrap_iterations = 10000
        
        with tqdm(total=bootstrap_iterations, desc=f"Bootstrap sampling ({label})", 
                  unit="sample", leave=False, ncols=80) as pbar:
            for _ in range(bootstrap_iterations):
                sample1 = np.random.choice(vals1, size=n1, replace=True)
                sample2 = np.random.choice(vals2, size=n2, replace=True)
                bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
                pbar.update(1)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Determine significance
        is_significant = p_value < 0.05
        
        # Determine which model is better (lower is better for all metrics)
        better_model = model1_name if mean1 < mean2 else model2_name
        improvement = abs(mean_diff) / max(mean1, mean2) * 100 if max(mean1, mean2) > 0 else 0
        
        results[metric] = {
            f'{model1_name.lower().replace(" ", "_")}_mean': float(mean1),
            f'{model1_name.lower().replace(" ", "_")}_std': float(std1),
            f'{model1_name.lower().replace(" ", "_")}_median': float(median1),
            f'{model1_name.lower().replace(" ", "_")}_n': int(n1),
            f'{model2_name.lower().replace(" ", "_")}_mean': float(mean2),
            f'{model2_name.lower().replace(" ", "_")}_std': float(std2),
            f'{model2_name.lower().replace(" ", "_")}_median': float(median2),
            f'{model2_name.lower().replace(" ", "_")}_n': int(n2),
            'mean_diff': float(mean_diff),
            'median_diff': float(median1 - median2),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'bootstrap_ci_lower': float(ci_lower),
            'bootstrap_ci_upper': float(ci_upper),
            'is_significant': bool(is_significant),
            'better_model': better_model,
            'improvement_percent': float(improvement)
        }
        
        # Print results with nice formatting
        print(f"\n{'─'*80}")
        print(f"📊 {label}")
        print(f"{'─'*80}")
        print(f"  {model1_name:30s}: {mean1:>10.6f} ± {std1:>10.6f}  (median: {median1:>10.6f}, n={n1:>5})")
        print(f"  {model2_name:30s}: {mean2:>10.6f} ± {std2:>10.6f}  (median: {median2:>10.6f}, n={n2:>5})")
        print(f"\n  Difference ({model1_name} - {model2_name}): {mean_diff:>10.6f}")
        print(f"  Median difference: {median1 - median2:>10.6f}")
        print(f"  t-statistic: {t_stat:>10.6f}")
        print(f"  p-value:     {p_value:>10.6f}")
        print(f"  95% CI:      [{ci_lower:>8.6f}, {ci_upper:>8.6f}]")
        
        if is_significant:
            print(f"\n  {'✓'*3} SIGNIFICANT DIFFERENCE (p < 0.05) {'✓'*3}")
            print(f"  → {better_model} performs BETTER (lower {label.lower()})")
            print(f"  → Improvement: {improvement:.2f}%")
        else:
            print(f"\n  {'○'*3} NO SIGNIFICANT DIFFERENCE (p >= 0.05) {'○'*3}")
        print()
    
    # Save results
    results['dataset_name'] = dataset_name
    results['model1_name'] = model1_name
    results['model2_name'] = model2_name
    results['num_model1_images'] = len(metrics1)
    results['num_model2_images'] = len(metrics2)
    
    # Add metadata
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results['metadata'] = {
        'datetime': datetime_str,
        'dataset_name': dataset_name,
        'model1_name': model1_name,
        'model2_name': model2_name
    }
    
    # Generate filename
    filename = f"{model1_name.lower().replace(' ', '_')}_vs_{model2_name.lower().replace(' ', '_')}_{dataset_name.lower()}_{datetime_str}.json"
    results_file = os.path.join(output_dir, filename)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'═'*80}")
    print("📋 SUMMARY")
    print(f"{'═'*80}")
    significant_count = sum(1 for metric in metric_names 
                          if metric in results and results[metric].get('is_significant', False))
    print(f"  Metrics analyzed: {len([m for m in metric_names if m in results])}")
    print(f"  Significant differences found: {significant_count}/{len([m for m in metric_names if m in results])}")
    print(f"  {model1_name} images: {len(metrics1)}")
    print(f"  {model2_name} images: {len(metrics2)}")
    print(f"\n  Results saved to: {results_file}")
    print(f"{'═'*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare two depth estimation models on a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare da2-revised vs da2 on CityScapes
  python compare_models.py \\
    --dataset CityScapes \\
    --model1 da2-revised \\
    --model2 da2
  
  # Compare da2-revised vs da3 on DrivingStereo
  python compare_models.py \\
    --dataset DrivingStereo \\
    --model1 da2-revised \\
    --model2 da3
  
  # Compare with explicit checkpoint override
  python compare_models.py \\
    --dataset middlebury \\
    --model1 da2-revised \\
    --model2 da2 \\
    --model2-checkpoint checkpoints/custom/path/best.pth
  
  # Compare with custom dataset path
  python compare_models.py \\
    --dataset vkitti \\
    --dataset-path /path/to/vkitti \\
    --model1 da2-revised \\
    --model2 da2 \\
    --max-items 50
        """
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['CityScapes', 'DrivingStereo', 'middlebury', 'vkitti'],
                       help='Dataset name: CityScapes, DrivingStereo, middlebury, or vkitti')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Optional path to dataset. If not provided, uses dataset default path.')
    parser.add_argument('--output-path', type=str, default='results',
                       help='Path to save output results (default: results)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split (train, val, test)')
    parser.add_argument('--max-items', type=int, default=None,
                       help='Maximum number of items to process')
    parser.add_argument('--filter-regex', type=str, default=None,
                       help='Regex pattern to filter items by name')
    
    # Model arguments - standardized model names
    parser.add_argument('--model1', type=str, required=True,
                       choices=['da2', 'da2-revised', 'da3'],
                       help='First model name: da2, da2-revised, or da3')
    parser.add_argument('--model2', type=str, required=True,
                       choices=['da2', 'da2-revised', 'da3'],
                       help='Second model name: da2, da2-revised, or da3')
    parser.add_argument('--model1-checkpoint', type=str, default=None,
                       help='Optional: Explicit checkpoint path for model1 (overrides auto-detection)')
    parser.add_argument('--model2-checkpoint', type=str, default=None,
                       help='Optional: Explicit checkpoint path for model2 (overrides auto-detection)')
    parser.add_argument('--input-size', type=int, default=518,
                       help='Input image size for models')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, mps, or cpu). Auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Find models and their configurations
    print(f"\n🔍 Finding models and checkpoints...")
    try:
        model1_checkpoint, model1_config = find_model_by_name(args.model1, args.model1_checkpoint)
        model2_checkpoint, model2_config = find_model_by_name(args.model2, args.model2_checkpoint)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Check if we're using latest.pth instead of best.pth and warn
    if 'latest.pth' in model1_checkpoint and 'best.pth' not in model1_checkpoint:
        print(f"⚠️  Warning: Using latest.pth for model1 instead of best.pth")
        print(f"   This may not represent the best performing checkpoint.")
    
    if 'latest.pth' in model2_checkpoint and 'best.pth' not in model2_checkpoint:
        print(f"⚠️  Warning: Using latest.pth for model2 instead of best.pth")
        print(f"   This may not represent the best performing checkpoint.")
    
    print(f"\n✅ Model 1: {args.model1}")
    print(f"   Checkpoint: {model1_checkpoint}")
    print(f"   Type: {model1_config['model_type']}, Encoder: {model1_config['encoder']}, Max depth: {model1_config['max_depth']}")
    print(f"\n✅ Model 2: {args.model2}")
    print(f"   Checkpoint: {model2_checkpoint}")
    print(f"   Type: {model2_config['model_type']}, Encoder: {model2_config['encoder']}, Max depth: {model2_config['max_depth']}")
    
    # Create models
    model1_name = f"{args.model1}"
    model2_name = f"{args.model2}"
    
    print(f"\nLoading models...")
    model1 = DepthAnythingV2Wrapper({
        'model_type': model1_config['model_type'],
        'encoder': model1_config['encoder'],
        'checkpoint_path': model1_checkpoint,
        'max_depth': model1_config['max_depth'],
        'device': args.device
    })
    
    model2 = DepthAnythingV2Wrapper({
        'model_type': model2_config['model_type'],
        'encoder': model2_config['encoder'],
        'checkpoint_path': model2_checkpoint,
        'max_depth': model2_config['max_depth'],
        'device': args.device
    })
    
    # Create dataset
    dataset_config = DatasetConfig(
        dataset_path=args.dataset_path,
        split=args.split,
        max_items=args.max_items,
        regex_filter=args.filter_regex
    )
    
    # Map dataset names (case-sensitive matching)
    if args.dataset == 'CityScapes':
        dataset = CityscapesDataset(dataset_config)
    elif args.dataset == 'DrivingStereo':
        dataset = DrivingStereoDataset(dataset_config)
    elif args.dataset == 'middlebury':
        dataset = MiddleburyDataset(dataset_config)
    elif args.dataset == 'vkitti':
        dataset = VKITTIDataset(dataset_config)
    else:
        print(f"Error: Unknown dataset: {args.dataset}")
        print(f"Supported datasets: CityScapes, DrivingStereo, middlebury, vkitti")
        sys.exit(1)
    
    # Create output directories - structure: results/{dataset}/{model}/
    # The ProcessingPipeline uses output_base_dir, and dataset.get_item_output_dir() 
    # creates: {output_base_dir}/{dataset_subdir}/{item_id}/
    # We want: {output_path}/{dataset_subdir}/{model}/{item_id}/
    # So we pass: {output_path}/{dataset_subdir}/{model}/ as output_base_dir
    # But get_item_output_dir will add dataset_subdir again, creating:
    # {output_path}/{dataset_subdir}/{model}/{dataset_subdir}/{item_id}/ (WRONG!)
    # 
    # Solution: Create a wrapper that overrides get_item_output_dir to skip adding dataset_subdir
    dataset_output_subdir = dataset.get_output_subdir()  # e.g., 'cityscapes', 'drivingstereo', etc.
    
    # Create the desired structure: results/{dataset}/{model}/
    output_base_dir1 = os.path.join(args.output_path, dataset_output_subdir, model1_name)
    output_base_dir2 = os.path.join(args.output_path, dataset_output_subdir, model2_name)
    comparison_output_dir = os.path.join(args.output_path, dataset_output_subdir, f"comparison_{model1_name}_vs_{model2_name}")
    
    # Create a wrapper dataset that modifies get_item_output_dir to not add dataset_subdir again
    # since we've already included it in the base_dir
    class DatasetWrapper:
        """Wrapper to fix output directory structure for model-specific folders."""
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __getattr__(self, name):
            return getattr(self.dataset, name)
        
        def get_item_output_dir(self, base_output_dir, item):
            # Override: base_output_dir already includes dataset_subdir/model, so just add item_id
            return os.path.join(base_output_dir, item.item_id)
    
    # Wrap datasets to fix output directory structure
    wrapped_dataset1 = DatasetWrapper(dataset)
    wrapped_dataset2 = DatasetWrapper(dataset)
    
    # Count items for progress bar
    items = dataset.find_items()
    total_items = len(items)
    
    # Group items by scene/item_id for multi-camera datasets
    if dataset.supports_multiple_cameras():
        from collections import defaultdict
        items_by_scene = defaultdict(list)
        for item in items:
            items_by_scene[item.item_id].append(item)
        total_items = len(items_by_scene)
    
    # Process with both models
    print(f"\n{'='*80}")
    print(f"Evaluating {model1_name} on {args.dataset}...")
    print(f"{'='*80}")
    
    # Create unified progress bar for model1
    pbar1 = tqdm(
        total=total_items,
        desc=f"🚀 {model1_name} on {args.dataset}",
        unit="item",
        ncols=120,
        dynamic_ncols=False,
        file=sys.stderr
    )
    
    pipeline1 = ProcessingPipeline(
        dataset=wrapped_dataset1,
        model=model1,
        output_base_dir=output_base_dir1,
        input_size=args.input_size,
        max_depth=model1_config['max_depth']
    )
    
    metrics1 = pipeline1.process_dataset(progress_bar=pbar1)
    pbar1.close()
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model2_name} on {args.dataset}...")
    print(f"{'='*80}")
    
    # Create unified progress bar for model2
    pbar2 = tqdm(
        total=total_items,
        desc=f"🚀 {model2_name} on {args.dataset}",
        unit="item",
        ncols=120,
        dynamic_ncols=False,
        file=sys.stderr
    )
    
    pipeline2 = ProcessingPipeline(
        dataset=wrapped_dataset2,
        model=model2,
        output_base_dir=output_base_dir2,
        input_size=args.input_size,
        max_depth=model2_config['max_depth']
    )
    
    metrics2 = pipeline2.process_dataset(progress_bar=pbar2)
    pbar2.close()
    
    # Compare results
    if len(metrics1) > 0 and len(metrics2) > 0:
        comparison_results = compare_model_metrics(
            metrics1,
            metrics2,
            model1_name,
            model2_name,
            args.dataset,
            comparison_output_dir
        )
    else:
        print("\n" + "="*80)
        print("COMPARISON SKIPPED")
        print("="*80)
        print(f"{model1_name}: {len(metrics1)} images processed")
        print(f"{model2_name}: {len(metrics2)} images processed")
        print("\nCannot compare models - need data from both models.")
        print("="*80)
    
    # Print final summary
    print(f"\n{'═'*80}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'═'*80}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model 1 ({model1_name}): {len(metrics1)} images")
    print(f"  Model 2 ({model2_name}): {len(metrics2)} images")
    print(f"  Output directory: {args.output_path}")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()
