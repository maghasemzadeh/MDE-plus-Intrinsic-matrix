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
    MiddleburyDataset,
    VKITTIDataset
)
from collections import defaultdict
from models import (
    BaseDepthModelWrapper,
    DepthAnythingV2Wrapper,
    DepthAnythingV2RevisedWrapper,
    create_model_wrapper
)
from src import ProcessingPipeline, compute_depth_metrics

# Import Middlebury-specific comparison functions
from datasets.middlebury import (
    evaluate_per_image_metrics,
    print_evaluation_results
)


def compare_datasets(
    metrics1: List[Dict[str, float]],
    metrics2: List[Dict[str, float]],
    dataset1_name: str,
    dataset2_name: str,
    is_metric_model: bool,
    output_dir: str,
    model_name: str,
    model_type: str,
    encoder: str,
    model_checkpoint: Optional[str] = None
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
        'encoder': encoder,
        'model_checkpoint': model_checkpoint
    }
    
    # Generate filename with format: model_dataset1_dataset2_datetime.json
    model_name_safe = model_name.lower().replace(' ', '_').replace('-', '_')
    filename = f"{model_name_safe}_{dataset1_name.lower()}_{dataset2_name.lower()}_{datetime_str}.json"
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
    
    # Determine which model to use (default to da2-revised for metric, da2 for basic)
    model_name = 'da2-revised' if args.model_type == 'metric' else 'da2'
    
    # Create model config
    model_config = {
        'model_type': args.model_type,
        'encoder': args.encoder,
        'checkpoint_path': args.model_checkpoint,
        'max_depth': args.max_depth,
        'device': args.device
    }
    
    # Create model wrapper
    print(f"Loading {model_name} {args.model_type} model (encoder={args.encoder})...")
    try:
        model = create_model_wrapper(model_name, model_config)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Get resolved values from model
    resolved_checkpoint = model.get_checkpoint_path()
    resolved_model_type = args.model_type
    resolved_encoder = args.encoder
    resolved_max_depth = getattr(model, 'max_depth', args.max_depth)
    
    # Determine model name from checkpoint path for folder structure
    # Try to identify if it's da2, da2-revised, or da3
    model_name = "unknown"
    if resolved_checkpoint:
        checkpoint_lower = resolved_checkpoint.lower()
        # Check for DA3 first (most specific)
        if 'depth-anything-3' in checkpoint_lower or '/depth-anything-3/' in checkpoint_lower:
            model_name = 'da3'
        # Check for DA2-revised
        elif 'depthanythingv2-revised' in checkpoint_lower or '/depthanythingv2-revised/' in checkpoint_lower or 'v2-revised' in checkpoint_lower:
            model_name = 'da2-revised'
        # Check for DA2 (original)
        elif 'depthanythingv2' in checkpoint_lower and 'revised' not in checkpoint_lower and '/depthanythingv2/' in checkpoint_lower:
            model_name = 'da2'
        # Check directory structure
        elif '/depth-anything-3/' in resolved_checkpoint:
            model_name = 'da3'
        elif '/depthanythingv2-revised/' in resolved_checkpoint:
            model_name = 'da2-revised'
        elif '/depthanythingv2/' in resolved_checkpoint and '/depthanythingv2-revised/' not in resolved_checkpoint:
            model_name = 'da2'
    
    # If we couldn't determine, use a default based on model type and encoder
    if model_name == "unknown":
        model_name = f"{resolved_model_type}-{resolved_encoder}"
        print(f"⚠️  Warning: Could not determine model name from checkpoint, using: {model_name}")
    
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
        elif dataset_name.lower() == 'vkitti':
            dataset = VKITTIDataset(dataset_config)
        else:
            print(f"Error: Unknown dataset: {dataset_name}")
            print("Supported datasets: middlebury, cityscapes, drivingstereo, vkitti")
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
        
        # Create output directory structure: results/{dataset}/{model}/
        # The ProcessingPipeline uses output_base_dir, and dataset.get_item_output_dir() 
        # creates: {output_base_dir}/{dataset_subdir}/{item_id}/
        # We want: {output_path}/{dataset_subdir}/{model}/{item_id}/
        dataset_output_subdir = dataset.get_output_subdir()
        output_base_dir = os.path.join(args.output_path, dataset_output_subdir, model_name)
        
        # Create a wrapper dataset that modifies get_item_output_dir to not add dataset_subdir again
        class DatasetWrapper:
            """Wrapper to fix output directory structure for model-specific folders."""
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __getattr__(self, name):
                return getattr(self.dataset, name)
            
            def get_item_output_dir(self, base_output_dir, item):
                # Override: base_output_dir already includes dataset_subdir/model, so just add item_id
                return os.path.join(base_output_dir, item.item_id)
        
        wrapped_dataset = DatasetWrapper(dataset)
        
        # Helper function to check if dataset evaluation already exists
        def check_dataset_evaluation_exists(output_base_dir: str, wrapped_dataset: DatasetWrapper) -> bool:
            """Check if dataset evaluation already exists by checking for any item output directories."""
            items = wrapped_dataset.find_items()
            if len(items) == 0:
                return False
            
            # Check if at least one item has been processed
            sample_item = items[0]
            item_output_dir = wrapped_dataset.get_item_output_dir(output_base_dir, sample_item)
            return os.path.exists(item_output_dir) and os.path.isdir(item_output_dir)
        
        # Check if evaluation already exists
        dataset_exists = check_dataset_evaluation_exists(output_base_dir, wrapped_dataset)
        
        if dataset_exists:
            tqdm.write(f"   ⏭️  Skipping {dataset_name} evaluation - results already exist", file=sys.stdout)
            tqdm.write(f"      Output directory: {output_base_dir}", file=sys.stdout)
            # Load existing metrics
            pipeline = ProcessingPipeline(
                dataset=wrapped_dataset,
                model=model,
                output_base_dir=output_base_dir,
                input_size=args.input_size,
                scale_factor=args.scale_factor,
                max_depth=resolved_max_depth
            )
            metrics = pipeline.process_dataset(progress_bar=None)
            all_metrics.append(metrics)
            tqdm.write(f"   ✅ Loaded {dataset_name}: {len(metrics)} items from existing results", file=sys.stdout)
        else:
            # Create pipeline and process with unified progress bar
            pipeline = ProcessingPipeline(
                dataset=wrapped_dataset,
                model=model,
                output_base_dir=output_base_dir,
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
                resolved_encoder,
                model_checkpoint=resolved_checkpoint
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

