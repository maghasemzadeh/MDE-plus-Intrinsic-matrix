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
from datetime import datetime
from typing import Optional, List, Dict
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


def parse_model_config(model_str: str) -> Dict:
    """
    Parse model configuration string.
    
    Format: model_type:encoder:checkpoint_path:max_depth
    Examples:
        - metric:vitl:checkpoints/best.pth:80.0
        - metric:vitl::80.0  (auto-detect checkpoint)
        - basic:vitl::  (basic model, auto-detect)
    
    Args:
        model_str: Model configuration string
    
    Returns:
        Dictionary with model configuration
    """
    parts = model_str.split(':')
    
    if len(parts) < 2:
        raise ValueError(f"Invalid model format: {model_str}. Expected: model_type:encoder[:checkpoint_path][:max_depth]")
    
    model_type = parts[0].strip()
    encoder = parts[1].strip()
    checkpoint_path = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    max_depth = float(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else None
    
    if model_type not in ['metric', 'basic']:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'metric' or 'basic'")
    
    if encoder not in ['vits', 'vitb', 'vitl', 'vitg']:
        raise ValueError(f"Invalid encoder: {encoder}. Must be 'vits', 'vitb', 'vitl', or 'vitg'")
    
    # Set default max_depth based on model type
    if max_depth is None:
        max_depth = 80.0 if model_type == 'metric' else 20.0
    
    return {
        'model_type': model_type,
        'encoder': encoder,
        'checkpoint_path': checkpoint_path,
        'max_depth': max_depth
    }


def find_checkpoint(checkpoint_path: Optional[str], model_type: str, encoder: str, max_depth: float) -> Optional[str]:
    """
    Find checkpoint path, searching in multiple locations.
    If best.pth is not found, falls back to latest.pth with a warning.
    
    Args:
        checkpoint_path: Explicit checkpoint path (if provided)
        model_type: Model type ('metric' or 'basic')
        encoder: Encoder type
        max_depth: Maximum depth
    
    Returns:
        Checkpoint path or None
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # If explicit path provided, check it
    if checkpoint_path:
        if os.path.isabs(checkpoint_path):
            if os.path.exists(checkpoint_path):
                return checkpoint_path
        else:
            # Try relative to project root
            full_path = os.path.join(project_root, checkpoint_path)
            if os.path.exists(full_path):
                return full_path
            # If explicit path doesn't exist, try to find best/latest in that directory
            if os.path.isdir(full_path):
                best_path = os.path.join(full_path, 'best.pth')
                latest_path = os.path.join(full_path, 'latest.pth')
                if os.path.exists(best_path):
                    return best_path
                elif os.path.exists(latest_path):
                    print(f"⚠️  Warning: best.pth not found in {full_path}, using latest.pth instead")
                    return latest_path
    
    # Search in project checkpoints directory (trained models)
    project_checkpoints_dir = os.path.join(project_root, 'checkpoints')
    if os.path.isdir(project_checkpoints_dir):
        for subdir in os.listdir(project_checkpoints_dir):
            subdir_path = os.path.join(project_checkpoints_dir, subdir)
            if os.path.isdir(subdir_path):
                # Try best.pth first
                best_path = os.path.join(subdir_path, 'best.pth')
                latest_path = os.path.join(subdir_path, 'latest.pth')
                if os.path.exists(best_path):
                    return best_path
                elif os.path.exists(latest_path):
                    print(f"⚠️  Warning: best.pth not found in {subdir_path}, using latest.pth instead")
                    return latest_path
    
    # Search in v2-revised checkpoints
    v2_revised_dir = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised', 'checkpoints')
    if model_type == 'metric':
        for name in [
            f'depth_anything_v2_metric_hypersim_{encoder}.pth',
            f'depth_anything_v2_metric_vkitti_{encoder}.pth'
        ]:
            path = os.path.join(v2_revised_dir, name)
            if os.path.exists(path):
                return path
    else:
        name = f'depth_anything_v2_{encoder}.pth'
        path = os.path.join(v2_revised_dir, name)
        if os.path.exists(path):
            return path
    
    # Search in original v2 checkpoints
    v2_dir = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2', 'checkpoints')
    if model_type == 'metric':
        for name in [
            f'depth_anything_v2_metric_hypersim_{encoder}.pth',
            f'depth_anything_v2_metric_vkitti_{encoder}.pth'
        ]:
            path = os.path.join(v2_dir, name)
            if os.path.exists(path):
                return path
    else:
        name = f'depth_anything_v2_{encoder}.pth'
        path = os.path.join(v2_dir, name)
        if os.path.exists(path):
            return path
    
    return None


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
    # Check if we have metric model metrics (abs_rel, rmse) or basic model metrics (silog)
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
        # Default to metric metrics
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
  # Compare two metric models on Cityscapes
  python compare_models.py \\
    --dataset cityscapes \\
    --model1 "metric:vitl:checkpoints/model1/best.pth:80.0" \\
    --model2 "metric:vitl:checkpoints/model2/best.pth:80.0"
  
  # Compare metric vs basic model (auto-detect checkpoints)
  python compare_models.py \\
    --dataset drivingstereo \\
    --model1 "metric:vitl::80.0" \\
    --model2 "basic:vitl::"
  
  # Compare with custom dataset path
  python compare_models.py \\
    --dataset middlebury \\
    --dataset-path /path/to/middlebury \\
    --model1 "metric:vitl::20.0" \\
    --model2 "metric:vitb::20.0" \\
    --max-items 50
        """
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cityscapes', 'drivingstereo', 'middlebury', 'vkitti'],
                       help='Dataset name')
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
    
    # Model arguments
    parser.add_argument('--model1', type=str, required=True,
                       help='First model: model_type:encoder[:checkpoint_path][:max_depth]')
    parser.add_argument('--model2', type=str, required=True,
                       help='Second model: model_type:encoder[:checkpoint_path][:max_depth]')
    parser.add_argument('--input-size', type=int, default=518,
                       help='Input image size for models')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, mps, or cpu). Auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Parse model configurations
    try:
        model1_config = parse_model_config(args.model1)
        model2_config = parse_model_config(args.model2)
    except ValueError as e:
        print(f"Error parsing model configuration: {e}")
        sys.exit(1)
    
    # Find checkpoints
    model1_checkpoint = find_checkpoint(
        model1_config['checkpoint_path'],
        model1_config['model_type'],
        model1_config['encoder'],
        model1_config['max_depth']
    )
    model2_checkpoint = find_checkpoint(
        model2_config['checkpoint_path'],
        model2_config['model_type'],
        model2_config['encoder'],
        model2_config['max_depth']
    )
    
    if model1_checkpoint is None:
        print(f"❌ Error: Could not find checkpoint for model1: {model1_config}")
        print("Please specify checkpoint path explicitly or ensure checkpoints are in standard locations.")
        sys.exit(1)
    
    if model2_checkpoint is None:
        print(f"❌ Error: Could not find checkpoint for model2: {model2_config}")
        print("Please specify checkpoint path explicitly or ensure checkpoints are in standard locations.")
        sys.exit(1)
    
    # Check if we're using latest.pth instead of best.pth and warn
    if 'latest.pth' in model1_checkpoint and 'best.pth' not in model1_checkpoint:
        print(f"⚠️  Warning: Using latest.pth for model1 instead of best.pth")
        print(f"   This may not represent the best performing checkpoint.")
    
    if 'latest.pth' in model2_checkpoint and 'best.pth' not in model2_checkpoint:
        print(f"⚠️  Warning: Using latest.pth for model2 instead of best.pth")
        print(f"   This may not represent the best performing checkpoint.")
    
    print(f"Model 1: {model1_config['model_type']} ({model1_config['encoder']})")
    print(f"  Checkpoint: {model1_checkpoint}")
    print(f"  Max depth: {model1_config['max_depth']}")
    print(f"\nModel 2: {model2_config['model_type']} ({model2_config['encoder']})")
    print(f"  Checkpoint: {model2_checkpoint}")
    print(f"  Max depth: {model2_config['max_depth']}")
    
    # Create models
    model1_name = f"{model1_config['model_type']}-{model1_config['encoder']}"
    model2_name = f"{model2_config['model_type']}-{model2_config['encoder']}"
    
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
    
    if args.dataset.lower() == 'cityscapes':
        dataset = CityscapesDataset(dataset_config)
    elif args.dataset.lower() == 'drivingstereo':
        dataset = DrivingStereoDataset(dataset_config)
    elif args.dataset.lower() == 'middlebury':
        dataset = MiddleburyDataset(dataset_config)
    elif args.dataset.lower() == 'vkitti':
        dataset = VKITTIDataset(dataset_config)
    else:
        print(f"Error: Unknown dataset: {args.dataset}")
        sys.exit(1)
    
    # Create output directories
    output_dir1 = os.path.join(args.output_path, f"{args.dataset}_{model1_name}")
    output_dir2 = os.path.join(args.output_path, f"{args.dataset}_{model2_name}")
    comparison_output_dir = os.path.join(args.output_path, f"comparison_{args.dataset}")
    
    # Process with both models
    print(f"\n{'='*80}")
    print(f"Evaluating {model1_name} on {args.dataset}...")
    print(f"{'='*80}")
    
    pipeline1 = ProcessingPipeline(
        dataset=dataset,
        model=model1,
        output_base_dir=output_dir1,
        input_size=args.input_size,
        max_depth=model1_config['max_depth']
    )
    
    metrics1 = pipeline1.process_dataset()
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model2_name} on {args.dataset}...")
    print(f"{'='*80}")
    
    pipeline2 = ProcessingPipeline(
        dataset=dataset,
        model=model2,
        output_base_dir=output_dir2,
        input_size=args.input_size,
        max_depth=model2_config['max_depth']
    )
    
    metrics2 = pipeline2.process_dataset()
    
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

