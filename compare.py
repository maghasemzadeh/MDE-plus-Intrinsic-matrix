"""
Unified command-line interface for depth estimation evaluation and comparison.
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
    MiddleburyDataset
)
from collections import defaultdict
from models import BaseDepthModelWrapper, DepthAnythingV2Wrapper
from src import ProcessingPipeline, compute_depth_metrics

# Import Middlebury-specific comparison functions
from datasets.middlebury import (
    evaluate_per_image_metrics,
    print_evaluation_results
)


def _auto_select_checkpoint_and_encoder(
    model_type: str,
    encoder: str,
    max_depth: float
) -> tuple:
    """
    Auto-detect a suitable checkpoint and encoder from the local 'checkpoints' directory.

    Returns: (resolved_model_type, resolved_encoder, checkpoint_path_or_none, resolved_max_depth)
    """
    checkpoints_dir = os.path.join(os.path.dirname(__file__), 'models', 'raw_models', 'DepthAnythingV2', 'checkpoints')
    resolved_model_type = model_type
    resolved_encoder = encoder
    resolved_max_depth = max_depth
    checkpoint_path = None
    
    if not os.path.isdir(checkpoints_dir):
        return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth
    
    def exists(name: str) -> Optional[str]:
        p = os.path.join(checkpoints_dir, name)
        return p if os.path.exists(p) else None
    
    if model_type == 'metric':
        # exact encoder first
        ck_h = exists(f'depth_anything_v2_metric_hypersim_{encoder}.pth')
        ck_v = exists(f'depth_anything_v2_metric_vkitti_{encoder}.pth')
        if ck_h:
            checkpoint_path = ck_h
            resolved_max_depth = 20.0 if max_depth is None else max_depth
            return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth
        if ck_v:
            checkpoint_path = ck_v
            resolved_max_depth = 80.0
            return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth
        
        # try any encoder
        candidates = []
        for enc in ['vits', 'vitb', 'vitl', 'vitg']:
            ck_v_any = exists(f'depth_anything_v2_metric_vkitti_{enc}.pth')
            ck_h_any = exists(f'depth_anything_v2_metric_hypersim_{enc}.pth')
            if ck_v_any:
                candidates.append(('vkitti', enc, ck_v_any))
            if ck_h_any:
                candidates.append(('hypersim', enc, ck_h_any))
        if candidates:
            # prefer vkitti then hypersim
            pref = None
            for c in candidates:
                if c[0] == 'vkitti':
                    pref = c
                    break
            if pref is None:
                pref = candidates[0]
            domain, enc, path = pref
            resolved_encoder = enc
            checkpoint_path = path
            resolved_max_depth = 80.0 if domain == 'vkitti' else 20.0
            return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth
        
        # fallback to basic if no metric checkpoint
        for enc in ['vits', 'vitb', 'vitl', 'vitg']:
            ck_b = exists(f'depth_anything_v2_{enc}.pth')
            if ck_b:
                resolved_model_type = 'basic'
                resolved_encoder = enc
                checkpoint_path = ck_b
                return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth
        return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth
    else:
        # basic
        ck_b = exists(f'depth_anything_v2_{encoder}.pth')
        if ck_b:
            checkpoint_path = ck_b
            return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth
        for enc in ['vits', 'vitb', 'vitl', 'vitg']:
            ck = exists(f'depth_anything_v2_{enc}.pth')
            if ck:
                resolved_encoder = enc
                checkpoint_path = ck
                return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth
        return resolved_model_type, resolved_encoder, checkpoint_path, resolved_max_depth


def compare_datasets(
    metrics1: List[Dict[str, float]],
    metrics2: List[Dict[str, float]],
    dataset1_name: str,
    dataset2_name: str,
    is_metric_model: bool,
    output_dir: str,
    model_name: str,
    model_type: str,
    encoder: str
) -> Dict:
    """
    Compare two datasets using t-test and other statistics.
    
    Args:
        metrics1: List of metric dictionaries from first dataset
        metrics2: List of metric dictionaries from second dataset
        dataset1_name: Name of first dataset
        dataset2_name: Name of second dataset
        is_metric_model: Whether using metric model
        output_dir: Output directory for saving results
        model_name: Name of the model used for evaluation
        model_type: Type of model ('metric' or 'basic')
        encoder: Encoder type used in the model
    
    Returns:
        Dictionary with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DATASET COMPARISON: {dataset1_name} vs {dataset2_name}")
    print(f"{'='*80}\n")
    
    if is_metric_model:
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
    
    results = {}
    
    for metric in metric_names:
        # Get metric label early for use in progress bar
        label = metric_labels[metric]
        
        # Extract metric values
        vals1 = np.array([m[metric] for m in metrics1 if not np.isnan(m[metric])])
        vals2 = np.array([m[metric] for m in metrics2 if not np.isnan(m[metric])])
        
        if len(vals1) == 0 or len(vals2) == 0:
            print(f"Warning: Insufficient data for {metric}")
            continue
        
        n1, n2 = len(vals1), len(vals2)
        
        # Robustness checks
        if n1 < 30 or n2 < 30:
            print(f"Warning: Small sample sizes for {metric} ({dataset1_name}: n={n1}, {dataset2_name}: n={n2}). "
                  f"t-test results may be less reliable with n < 30.")
        
        size_ratio = max(n1, n2) / min(n1, n2) if min(n1, n2) > 0 else float('inf')
        if size_ratio > 5:
            print(f"Warning: Large sample size difference for {metric} (ratio: {size_ratio:.1f}x). "
                  f"Welch's t-test handles this, but results should be interpreted with caution.")
        
        # Compute statistics
        mean1 = np.mean(vals1)
        std1 = np.std(vals1, ddof=1)
        mean2 = np.mean(vals2)
        std2 = np.std(vals2, ddof=1)
        
        # Perform Welch's t-test
        t_stat, p_value = ttest_ind(vals1, vals2, equal_var=False)
        
        # Bootstrap confidence interval
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
        
        results[metric] = {
            f'{dataset1_name.lower()}_mean': float(mean1),
            f'{dataset1_name.lower()}_std': float(std1),
            f'{dataset1_name.lower()}_n': int(len(vals1)),
            f'{dataset2_name.lower()}_mean': float(mean2),
            f'{dataset2_name.lower()}_std': float(std2),
            f'{dataset2_name.lower()}_n': int(len(vals2)),
            'mean_diff': float(mean_diff),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'bootstrap_ci_lower': float(ci_lower),
            'bootstrap_ci_upper': float(ci_upper),
            'is_significant': bool(is_significant)
        }
        
        # Print results with nice formatting
        print(f"\n{'─'*80}")
        print(f"📊 {label}")
        print(f"{'─'*80}")
        print(f"  {dataset1_name:20s}: {mean1:>10.6f} ± {std1:>10.6f}  (n={len(vals1):>5})")
        print(f"  {dataset2_name:20s}: {mean2:>10.6f} ± {std2:>10.6f}  (n={len(vals2):>5})")
        print(f"\n  Difference ({dataset1_name} - {dataset2_name}): {mean_diff:>10.6f}")
        print(f"  t-statistic: {t_stat:>10.6f}")
        print(f"  p-value:     {p_value:>10.6f}")
        print(f"  95% CI:      [{ci_lower:>8.6f}, {ci_upper:>8.6f}]")
        
        if is_significant:
            print(f"\n  {'✓'*3} SIGNIFICANT DIFFERENCE (p < 0.05) {'✓'*3}")
            if mean_diff > 0:
                print(f"  → {dataset1_name} has HIGHER error than {dataset2_name}")
            else:
                print(f"  → {dataset2_name} has HIGHER error than {dataset1_name}")
        else:
            print(f"\n  {'○'*3} NO SIGNIFICANT DIFFERENCE (p >= 0.05) {'○'*3}")
        print()
    
    # Save results
    results['is_metric_model'] = is_metric_model
    results[f'num_{dataset1_name.lower()}_images'] = len(metrics1)
    results[f'num_{dataset2_name.lower()}_images'] = len(metrics2)
    
    # Add metadata
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results['metadata'] = {
        'datetime': datetime_str,
        'dataset1_name': dataset1_name,
        'dataset2_name': dataset2_name,
        'is_metric_model': is_metric_model,
        'model_name': model_name,
        'model_type': model_type,
        'encoder': encoder
    }
    
    # Generate filename with format: {dataset1}_{dataset2}_{datetime}.json
    filename = f"{dataset1_name.lower()}_{dataset2_name.lower()}_{datetime_str}.json"
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
    print(f"  {dataset1_name} images: {len(metrics1)}")
    print(f"  {dataset2_name} images: {len(metrics2)}")
    print(f"\n  Results saved to: {results_file}")
    print(f"{'═'*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Unified depth estimation evaluation and comparison tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single dataset (uses default dataset and output paths)
  python compare.py --dataset middlebury
  
  # Process with custom output path
  python compare.py --dataset middlebury --output-path custom_output
  
  # Process with custom dataset path
  python compare.py --dataset middlebury --dataset-path /custom/path/to/middlebury
  
  # Compare two datasets (uses all default paths)
  python compare.py --dataset cityscapes,drivingstereo
  
  # Compare with custom paths
  python compare.py --dataset cityscapes,drivingstereo \\
    --dataset-path /path/to/cityscapes,/path/to/drivingstereo \\
    --output-path custom_output
  
  # Use metric model with specific encoder
  python compare.py --dataset middlebury --model-type metric --encoder vitl --max-depth 20.0
  
  # Filter items using regex (works for all datasets)
  python compare.py --dataset middlebury --filter-regex ".*-perfect"
  python compare.py --dataset cityscapes --filter-regex ".*aachen.*"
  python compare.py --dataset drivingstereo --filter-regex "2018-07-09.*"
        """
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name(s): middlebury, cityscapes, drivingstereo, or comma-separated for comparison')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Optional path(s) to dataset(s), comma-separated if multiple datasets. If not provided, uses dataset default paths.')
    parser.add_argument('--output-path', type=str, default='results',
                       help='Path to save output results (default: results)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split (train, val, test)')
    parser.add_argument('--max-items', type=int, default=None,
                       help='Maximum number of items to process per dataset')
    parser.add_argument('--filter-regex', type=str, default=None,
                       help='Regex pattern to filter items by name (works for all datasets)')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='metric', choices=['metric', 'basic'],
                       help='Model type: "metric" or "basic"')
    parser.add_argument('--model-checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: auto-detect)')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Model encoder type')
    parser.add_argument('--max-depth', type=float, default=80.0,
                       help='Maximum depth in meters (80 for outdoor, 20 for indoor)')
    parser.add_argument('--input-size', type=int, default=518,
                       help='Input image size for model')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, mps, or cpu). Auto-detect if not specified')
    parser.add_argument('--scale-factor', type=float, default=None,
                       help='Scale factor for basic model (default: auto-calculate from GT depth)')
    
    # Processing arguments
    parser.add_argument('--force-evaluate', action='store_true',
                       help='Force re-evaluation even if output folder exists')
    
    args = parser.parse_args()
    
    # Parse datasets
    dataset_names = [d.strip() for d in args.dataset.split(',')]
    
    if len(dataset_names) > 2:
        print("Error: Maximum of 2 datasets supported for comparison!")
        sys.exit(1)
    
    # Parse dataset paths (optional)
    if args.dataset_path:
        dataset_paths = [p.strip() for p in args.dataset_path.split(',')]
        if len(dataset_names) != len(dataset_paths):
            print("Error: Number of dataset names must match number of dataset paths!")
            sys.exit(1)
    else:
        dataset_paths = [None] * len(dataset_names)  # Will use default paths
    
    # Resolve checkpoint/encoder
    resolved_model_type, resolved_encoder, resolved_checkpoint, resolved_max_depth = \
        _auto_select_checkpoint_and_encoder(
            model_type=args.model_type,
            encoder=args.encoder,
            max_depth=args.max_depth
        )
    
    # If user supplied a checkpoint, prefer it
    if args.model_checkpoint is not None and os.path.exists(args.model_checkpoint):
        resolved_checkpoint = args.model_checkpoint
        resolved_model_type = args.model_type
        resolved_encoder = args.encoder
        resolved_max_depth = args.max_depth
    
    # Create model
    print(f"Loading Depth Anything V2 {resolved_model_type} model (encoder={resolved_encoder})...")
    model_config = {
        'model_type': resolved_model_type,
        'encoder': resolved_encoder,
        'checkpoint_path': resolved_checkpoint,
        'max_depth': resolved_max_depth,
        'device': args.device
    }
    model = DepthAnythingV2Wrapper(model_config)
    
    # First, count all items across all datasets to create unified progress bar
    print("\nCounting items across all datasets...")
    total_items = 0
    dataset_instances = []
    
    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        # Create dataset config
        dataset_config = DatasetConfig(
            dataset_path=dataset_path,  # None will use dataset's default path
            split=args.split,
            max_items=args.max_items,
            force_evaluate=args.force_evaluate,
            regex_filter=args.filter_regex  # General regex filter for all datasets
        )
        
        # Create dataset instance
        if dataset_name.lower() == 'middlebury':
            dataset = MiddleburyDataset(dataset_config)
        elif dataset_name.lower() == 'cityscapes':
            dataset = CityscapesDataset(dataset_config)
        elif dataset_name.lower() == 'drivingstereo':
            dataset = DrivingStereoDataset(dataset_config)
        else:
            print(f"Error: Unknown dataset: {dataset_name}")
            print("Supported datasets: middlebury, cityscapes, drivingstereo")
            sys.exit(1)
        
        items = dataset.find_items()
        
        # Group items by scene/item_id for multi-camera datasets
        if dataset.supports_multiple_cameras():
            items_by_scene = defaultdict(list)
            for item in items:
                items_by_scene[item.item_id].append(item)
            items_to_process = len(items_by_scene)
        else:
            items_to_process = len(items)
        
        total_items += items_to_process
        dataset_instances.append(dataset)
        print(f"  {dataset_name}: {items_to_process} items")
    
    print(f"\nTotal items to process: {total_items}")
    print(f"{'='*80}")
    
    # Create unified progress bar at fixed position (top)
    # Use file=sys.stderr to keep it separate from stdout prints
    # Use miniters=1 and mininterval=0.1 for smooth updates
    unified_pbar = tqdm(
        total=total_items, 
        desc="🚀 Processing datasets", 
        unit="item", 
        ncols=120, 
        dynamic_ncols=False,
        position=0,
        leave=True,
        file=sys.stderr,  # Use stderr so it doesn't interfere with stdout
        miniters=1,
        mininterval=0.1,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    # Process datasets
    all_metrics = []
    
    for idx, (dataset_name, dataset) in enumerate(zip(dataset_names, dataset_instances)):
        # Use tqdm.write to avoid interfering with progress bar (write to stdout, bar is on stderr)
        tqdm.write(f"\n📂 Processing dataset {idx+1}/{len(dataset_names)}: {dataset_name}", file=sys.stdout)
        tqdm.write(f"   Dataset path: {dataset.dataset_path}", file=sys.stdout)
        
        # Create pipeline and process with unified progress bar
        pipeline = ProcessingPipeline(
            dataset=dataset,
            model=model,
            output_base_dir=args.output_path,
            input_size=args.input_size,
            scale_factor=args.scale_factor,
            max_depth=resolved_max_depth
        )
        
        try:
            metrics = pipeline.process_dataset(progress_bar=unified_pbar)
            all_metrics.append(metrics)
            tqdm.write(f"   ✅ Completed {dataset_name}: {len(metrics)} items processed", file=sys.stdout)
        except Exception as e:
            tqdm.write(f"   ❌ Error processing {dataset_name}: {e}", file=sys.stdout)
            import traceback
            tqdm.write(traceback.format_exc(), file=sys.stdout)
            all_metrics.append([])  # Append empty list to maintain alignment
    
    # Close the unified progress bar
    unified_pbar.close()
    
    # Print results summary for single dataset (non-Middlebury)
    if len(dataset_names) == 1 and dataset_names[0].lower() != 'middlebury':
        print(f"\n{'═'*80}")
        print("📊 PROCESSING RESULTS")
        print(f"{'═'*80}")
        dataset_name = dataset_names[0]
        total_processed = len(all_metrics[0]) if len(all_metrics) > 0 else 0
        
        if total_processed > 0:
            # Calculate aggregate statistics
            if model.is_metric():
                metric_key = 'abs_rel'
                metric_label = 'AbsRel'
            else:
                metric_key = 'silog'
                metric_label = 'SILog'
            
            valid_metrics = [m[metric_key] for m in all_metrics[0] 
                           if metric_key in m and not np.isnan(m[metric_key])]
            
            if len(valid_metrics) > 0:
                mean_metric = np.mean(valid_metrics)
                std_metric = np.std(valid_metrics, ddof=1)
                print(f"  Dataset: {dataset_name}")
                print(f"  Images processed: {total_processed}")
                print(f"  Mean {metric_label}: {mean_metric:.6f} ± {std_metric:.6f}")
                print(f"  Range: [{np.min(valid_metrics):.6f}, {np.max(valid_metrics):.6f}]")
            else:
                print(f"  Dataset: {dataset_name}")
                print(f"  Images processed: {total_processed}")
        else:
            print(f"  Dataset: {dataset_name}")
            print(f"  Images processed: 0 (no valid results)")
        print(f"{'═'*80}\n")
    
    # Compare datasets if two were provided
    if len(dataset_names) == 2:
        if len(all_metrics[0]) > 0 and len(all_metrics[1]) > 0:
            comparison_results = compare_datasets(
                all_metrics[0],
                all_metrics[1],
                dataset_names[0],
                dataset_names[1],
                model.is_metric(),
                args.output_path,
                model.get_model_name(),
                resolved_model_type,
                resolved_encoder
            )
        else:
            print("\n" + "="*80)
            print("DATASET COMPARISON SKIPPED")
            print("="*80)
            print(f"{dataset_names[0]}: {len(all_metrics[0])} images processed")
            print(f"{dataset_names[1]}: {len(all_metrics[1])} images processed")
            print("\nCannot compare datasets - need data from both datasets.")
            print("="*80)
    
    # For Middlebury dataset, perform left/right camera comparison
    if len(dataset_names) == 1 and dataset_names[0].lower() == 'middlebury':
        print("\n" + "═"*80)
        print("🔬 PERFORMING MIDDLEBURY LEFT/RIGHT CAMERA COMPARISON")
        print("═"*80)
        
        # Run per-image evaluation to compare cameras
        middlebury_output_dir = os.path.join(
            args.output_path,
            dataset_instances[0].get_output_subdir()
        )
        
        evaluation_results = evaluate_per_image_metrics(
            output_path=middlebury_output_dir,
            max_scenes=None,
            is_metric_model=model.is_metric(),
            regex_pattern=args.filter_regex
        )
        
        # Print results
        print_evaluation_results(evaluation_results)
        
        # Save results
        results_file = os.path.join(middlebury_output_dir, "depth_evaluation_results.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"💾 Evaluation results saved to: {results_file}")
        print("═"*80 + "\n")
    
    # Print final summary
    print(f"\n{'═'*80}")
    print("✅ PROCESSING COMPLETE")
    print(f"{'═'*80}")
    
    if len(dataset_names) == 1:
        dataset_name = dataset_names[0]
        total_processed = len(all_metrics[0]) if len(all_metrics) > 0 else 0
        print(f"  Dataset: {dataset_name}")
        print(f"  Images processed: {total_processed}")
    else:
        print(f"  Datasets compared: {', '.join(dataset_names)}")
        for i, name in enumerate(dataset_names):
            total = len(all_metrics[i]) if i < len(all_metrics) else 0
            print(f"    {name}: {total} images")
    
    print(f"  Output directory: {args.output_path}")
    print(f"  Model: {resolved_model_type} ({resolved_encoder})")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()

