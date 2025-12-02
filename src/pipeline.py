"""
Unified processing pipeline for depth estimation evaluation.
"""

import os
import json
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
try:
    from tqdm import tqdm
    import sys
except ImportError:
    tqdm = None
    sys = None

from datasets import BaseDataset, DatasetItem
from models import BaseDepthModelWrapper
from src import (
    compute_depth_metrics,
    generate_visualization_images,
    get_output_paths
)


def warp_image_to_camera(
    image: np.ndarray,
    depth: np.ndarray,
    source_cam_params: Dict[str, float],
    target_cam_params: Dict[str, float],
    baseline: float
) -> np.ndarray:
    """
    Warp an image from source camera to target camera viewpoint using depth.
    
    This function projects pixels from the source camera's image plane to the 
    target camera's image plane using the depth information and camera calibration.
    Uses backward mapping for accurate results without holes.
    
    Args:
        image: Source image (H, W, C) or (H, W) - can be RGB, grayscale, or depth map
        depth: Depth map in meters (H, W) - same size as image (from source camera)
        source_cam_params: Dictionary with 'f', 'cx', 'cy' for source camera
        target_cam_params: Dictionary with 'f', 'cx', 'cy' for target camera
        baseline: Baseline distance between cameras in meters
    
    Returns:
        Warped image in target camera's viewpoint (same shape as input image)
    """
    h, w = image.shape[:2]
    is_color = len(image.shape) == 3
    
    # Get camera parameters
    f_src = source_cam_params['f']
    cx_src = source_cam_params['cx']
    cy_src = source_cam_params['cy']
    
    f_tgt = target_cam_params['f']
    cx_tgt = target_cam_params['cx']
    cy_tgt = target_cam_params['cy']
    
    # Stereo warping: warp from source camera to target camera using depth
    # For rectified stereo cameras:
    # - A 3D point at depth d seen at pixel (x_src, y) in source camera
    #   appears at pixel (x_tgt, y) in target camera where:
    #   x_tgt = x_src - (baseline * f) / d  (for right-to-left)
    #
    # For backward mapping (cv2.remap): for each target pixel, find source pixel
    #   x_src = x_tgt + (baseline * f) / d
    #   But we need depth d at target location
    
    # Create coordinate grids for target image
    y_tgt, x_tgt = np.meshgrid(np.arange(h, dtype=np.float32), 
                               np.arange(w, dtype=np.float32), 
                               indexing='ij')
    
    # Use source focal length (depth is measured from source camera)
    f_used = f_src
    
    # For backward mapping, we need depth at target coordinates
    # Since we only have depth from source, we approximate by using depth at same location
    # This is reasonable for rectified stereo where corresponding pixels are at same y
    depth_at_target = depth.copy()
    valid_mask = np.isfinite(depth_at_target) & (depth_at_target > 0)
    
    # Compute horizontal shift in pixels
    # Formula: x_src = x_tgt + (baseline * f) / depth
    # baseline is in meters, f is in pixels, depth is in meters
    # Result is in pixels
    with np.errstate(divide='ignore', invalid='ignore'):
        x_shift_pixels = np.zeros_like(x_tgt)
        x_shift_pixels[valid_mask] = (baseline * f_used) / depth_at_target[valid_mask]
    
    # Compute source pixel coordinates
    # For rectified stereo, vertical coordinate is the same
    x_src = x_tgt + x_shift_pixels
    y_src = y_tgt.copy()
    
    # Account for principal point offset between cameras
    # The shift is already in pixel coordinates, but we need to account for
    # different principal points
    x_src = x_src - cx_tgt + cx_src
    y_src = y_src - cy_tgt + cy_src
    
    # Create mapping arrays for remap
    map_x = x_src.astype(np.float32)
    map_y = y_src.astype(np.float32)
    
    # Mark invalid pixels (out of bounds or invalid depth)
    valid_coords = (
        valid_mask &
        (map_x >= 0) & (map_x < w) &
        (map_y >= 0) & (map_y < h)
    )
    
    # Set invalid coordinates to a value that will be handled by remap
    # Using -1 will make remap use borderValue
    map_x[~valid_coords] = -1
    map_y[~valid_coords] = -1
    
    # Use OpenCV remap for efficient backward mapping with interpolation
    if is_color:
        warped = cv2.remap(
            image, map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    else:
        # For grayscale/depth, ensure 2D
        if len(image.shape) == 2:
            warped = cv2.remap(
                image, map_x, map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            warped = cv2.remap(
                image, map_x, map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
    
    return warped


class ProcessingPipeline:
    """
    Unified pipeline for processing datasets with depth estimation models.
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseDepthModelWrapper,
        output_base_dir: str,
        input_size: int = 518,
        scale_factor: Optional[float] = None,
        max_depth: Optional[float] = None
    ):
        """
        Initialize processing pipeline.
        
        Args:
            dataset: Dataset instance
            model: Model wrapper instance
            output_base_dir: Base output directory
            input_size: Input image size for model
            scale_factor: Scale factor for non-metric models (None for auto)
            max_depth: Maximum depth for visualization (None for auto)
        """
        self.dataset = dataset
        self.model = model
        self.output_base_dir = output_base_dir
        self.input_size = input_size
        self.scale_factor = scale_factor
        self.max_depth = max_depth
        self.is_metric = model.is_metric()
        self.model_label = 'metric' if self.is_metric else 'basic'
        
        # Create output directory
        os.makedirs(output_base_dir, exist_ok=True)
    
    def process_dataset(self, progress_bar: Optional[tqdm] = None) -> List[Dict[str, float]]:
        """
        Process all items in the dataset.
        
        Args:
            progress_bar: Optional tqdm progress bar to update. If None, no progress bar is shown.
        
        Returns:
            List of metric dictionaries (one per item)
        """
        items = self.dataset.find_items()
        
        if len(items) == 0:
            if progress_bar is not None:
                tqdm.write("No items found in dataset!")
            else:
                print("No items found in dataset!")
            return []
        
        # Only print dataset info if no progress bar (for backward compatibility)
        if progress_bar is None:
            print(f"\n{'='*80}")
            print(f"Processing {self.dataset.get_output_subdir()} dataset")
            print(f"{'='*80}")
            print(f"Total items: {len(items)}")
            print(f"Model type: {'Metric' if self.is_metric else 'Basic'}")
            print(f"Output directory: {self.output_base_dir}")
            print(f"{'='*80}\n")
        
        all_metrics = []
        processed_items = []
        
        # Group items by scene/item_id for multi-camera datasets
        if self.dataset.supports_multiple_cameras():
            items_by_scene = defaultdict(list)
            for item in items:
                items_by_scene[item.item_id].append(item)
            items_to_process = list(items_by_scene.values())
        else:
            items_to_process = [[item] for item in items]
        
        # Process each item/scene
        for idx, item_group in enumerate(items_to_process):
            try:
                # Get the first item as representative
                main_item = item_group[0]
                item_id = main_item.item_id
                
                # Get output directory
                item_output_dir = self.dataset.get_item_output_dir(
                    self.output_base_dir, main_item
                )
                
                # Update progress bar if provided - show folder path at top
                if progress_bar is not None:
                    # Get relative folder path for display
                    rel_path = os.path.relpath(item_output_dir, self.output_base_dir)
                    # Show folder name at top (in description), item name in postfix
                    progress_bar.set_description(f"📁 {rel_path[:70]}")
                    progress_bar.set_postfix_str(f"Item: {item_id[:35]}")
                
                # Check if already processed
                metrics = None
                if not self.dataset.config.force_evaluate:
                    if self._is_already_processed(item_output_dir, item_group):
                        # Load metrics from already-processed item instead of skipping
                        metrics = self._load_existing_metrics(item_output_dir, item_group)
                        if metrics is not None:
                            all_metrics.append(metrics)
                            processed_items.append(item_id)
                            
                            # Update progress bar
                            if progress_bar is not None:
                                if self.is_metric:
                                    metric_val = metrics.get('abs_rel', 0.0)
                                    progress_bar.set_postfix_str(
                                        f"Item: {item_id[:25]} | Status: ⏭️  loaded | AbsRel: {metric_val:.4f}"
                                    )
                                else:
                                    metric_val = metrics.get('silog', 0.0)
                                    progress_bar.set_postfix_str(
                                        f"Item: {item_id[:25]} | Status: ⏭️  loaded | SILog: {metric_val:.4f}"
                                    )
                                progress_bar.update(1)
                            continue
                        # If loading failed, fall through to process the item
                
                # Process item(s) if not already processed or if force_evaluate
                if metrics is None:
                    metrics = self._process_item_group(item_group, item_output_dir, item_id, progress_bar)
                
                if metrics is not None:
                    all_metrics.append(metrics)
                    processed_items.append(item_id)
                    
                    # Update progress bar with success status
                    if progress_bar is not None:
                        if self.is_metric:
                            metric_val = metrics.get('abs_rel', 0.0)
                            progress_bar.set_postfix_str(
                                f"Item: {item_id[:25]} | Status: ✅ done | AbsRel: {metric_val:.4f}"
                            )
                        else:
                            metric_val = metrics.get('silog', 0.0)
                            progress_bar.set_postfix_str(
                                f"Item: {item_id[:25]} | Status: ✅ done | SILog: {metric_val:.4f}"
                            )
                        progress_bar.update(1)
                else:
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(f"Item: {item_id[:30]} | Status: ⚠️  warning")
                        progress_bar.update(1)
            
            except Exception as e:
                if progress_bar is not None:
                    progress_bar.set_postfix_str(f"Item: {item_id[:30]} | Status: ❌ error: {str(e)[:20]}")
                    progress_bar.update(1)
                    # Use tqdm.write to avoid interfering with progress bar
                    tqdm.write(f"Error processing {item_id}: {e}", file=sys.stdout)
                    import traceback
                    tqdm.write(traceback.format_exc(), file=sys.stdout)
                else:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Use tqdm.write for final summary to avoid interfering with progress bar
        if progress_bar is not None:
            skipped_count = len(items_to_process) - len(all_metrics)
            if skipped_count > 0:
                tqdm.write(
                    f"✅ Dataset {self.dataset.get_output_subdir()}: "
                    f"{len(all_metrics)} items processed, {skipped_count} items skipped (already processed)",
                    file=sys.stdout
                )
            else:
                tqdm.write(
                    f"✅ Dataset {self.dataset.get_output_subdir()}: "
                    f"{len(all_metrics)}/{len(items_to_process)} items processed",
                    file=sys.stdout
                )
        else:
            print(f"\nSuccessfully processed {len(all_metrics)}/{len(items_to_process)} items")
        return all_metrics
    
    def _is_already_processed(self, item_output_dir: str, item_group: List[DatasetItem]) -> bool:
        """Check if item is already processed."""
        if not os.path.exists(item_output_dir):
            return False
        
        # Check for all cameras - prefer metrics.json, then compressed arrays, then legacy .npy
        for item in item_group:
            cam_id = item.camera_id or '0'
            paths = get_output_paths(item_output_dir, self.model_label, cam=cam_id)
            # Check for metrics.json (preferred)
            metrics_file = os.path.join(paths['numpy_dir'], "metrics.json")
            if os.path.exists(metrics_file):
                continue
            # Check for compressed arrays
            compressed_file = os.path.join(paths['numpy_dir'], "arrays.npz")
            if os.path.exists(compressed_file):
                continue
            # Legacy check for .npy files
            error_file = os.path.join(paths['numpy_dir'], "error.npy")
            if not os.path.exists(error_file):
                return False
        
        return True
    
    def _load_existing_metrics(self, item_output_dir: str, item_group: List[DatasetItem]) -> Optional[Dict[str, float]]:
        """Load metrics from already-processed item from saved JSON file."""
        try:
            # Get paths for first camera
            first_item = item_group[0]
            cam_id = first_item.camera_id or '0'
            paths = get_output_paths(item_output_dir, self.model_label, cam=cam_id)
            
            # Try to load metrics from JSON file (preferred - no memory overhead)
            metrics_file = os.path.join(paths['numpy_dir'], "metrics.json")
            if os.path.exists(metrics_file):
                import json
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                if metrics.get('n_valid', 0) >= 10:
                    return metrics
                return None
            
            # Fallback: try compressed numpy format
            compressed_file = os.path.join(paths['numpy_dir'], "arrays.npz")
            if os.path.exists(compressed_file):
                arrays = np.load(compressed_file)
                pred_depth = arrays['pred_depth']
                gt_depth = arrays['gt_depth']
                metrics = compute_depth_metrics(pred_depth, gt_depth, self.is_metric)
                del pred_depth, gt_depth, arrays  # Free memory immediately
                if metrics['n_valid'] >= 10:
                    return metrics
                return None
            
            # Legacy fallback: try uncompressed .npy files (for backward compatibility)
            pred_file = os.path.join(paths['numpy_dir'], "pred_depth_meters.npy")
            gt_file = os.path.join(paths['numpy_dir'], "gt_depth_meters.npy")
            
            if os.path.exists(pred_file) and os.path.exists(gt_file):
                pred_depth = np.load(pred_file)
                gt_depth = np.load(gt_file)
                metrics = compute_depth_metrics(pred_depth, gt_depth, self.is_metric)
                del pred_depth, gt_depth  # Free memory immediately
                if metrics['n_valid'] >= 10:
                    return metrics
                return None
            
            return None
        except Exception as e:
            # If loading fails, return None to trigger reprocessing
            return None
    
    def _process_item_group(
        self,
        item_group: List[DatasetItem],
        item_output_dir: str,
        item_id: str,
        progress_bar: Optional[tqdm] = None
    ) -> Optional[Dict[str, float]]:
        """Process a group of items (e.g., multiple cameras for one scene)."""
        # Create directories
        os.makedirs(item_output_dir, exist_ok=True)
        
        pred_depths = {}
        gt_depths = {}
        error_depths = {}
        images = {}  # Store original images for warping
        
        # Check if we have multiple cameras for the same scene (for comparison)
        has_multiple_cameras = len(item_group) > 1
        compare_dir_created = False
        
        # Process each camera/item
        for item in item_group:
            cam_id = item.camera_id or '0'
            paths = get_output_paths(item_output_dir, self.model_label, cam=cam_id)
            
            os.makedirs(paths['model_dir'], exist_ok=True)
            os.makedirs(paths['disp_dir'], exist_ok=True)
            os.makedirs(paths['numpy_dir'], exist_ok=True)
            
            # Only create compare_dir if we have multiple cameras for the same scene
            if has_multiple_cameras and not compare_dir_created:
                os.makedirs(paths['compare_dir'], exist_ok=True)
                compare_dir_created = True
            
            # Load image
            image = cv2.imread(item.image_path)
            if image is None:
                if progress_bar is not None:
                    tqdm.write(f"    ⚠️  Warning: Could not read image: {item.image_path}", file=sys.stdout)
                else:
                    print(f"    Warning: Could not read image: {item.image_path}")
                continue
            
            # Store image for warping
            images[cam_id] = image
            
            # Load ground truth
            try:
                gt_depth = self.dataset.load_gt_depth(item.gt_path, item)
            except Exception as e:
                if progress_bar is not None:
                    tqdm.write(f"    ⚠️  Warning: Error loading GT for {item_id}: {e}", file=sys.stdout)
                else:
                    print(f"    Warning: Error loading GT for {item_id}: {e}")
                continue
            
            # Predict depth
            import torch
            with torch.no_grad():
                pred_depth_raw = self.model.infer_image(image, input_size=self.input_size)
            
            # Convert to meters
            if self.is_metric:
                pred_depth = pred_depth_raw.astype(np.float32)
            else:
                # Calculate scale factor
                scale = self.model.calculate_scale_factor(
                    pred_depth_raw, gt_depth, self.scale_factor
                )
                pred_depth = (pred_depth_raw * scale).astype(np.float32)
            
            # Compute error
            error_depth = np.full_like(gt_depth, np.nan, dtype=np.float32)
            mask_valid = (
                np.isfinite(gt_depth) & (gt_depth > 0) &
                np.isfinite(pred_depth) & (pred_depth > 0)
            )
            error_depth[mask_valid] = np.abs(gt_depth[mask_valid] - pred_depth[mask_valid])
            
            # Save arrays in compressed format (much smaller file size)
            # Use npz format with compression for efficient storage
            compressed_file = os.path.join(paths['numpy_dir'], "arrays.npz")
            np.savez_compressed(
                compressed_file,
                pred_depth=pred_depth.astype(np.float32),
                gt_depth=gt_depth.astype(np.float32),
                error=error_depth.astype(np.float32)
            )
            
            # Store for later visualization (only keep in memory when needed)
            pred_depths[cam_id] = pred_depth
            gt_depths[cam_id] = gt_depth
            error_depths[cam_id] = error_depth
        
        # Compute metrics (use first camera or combine)
        if len(pred_depths) == 0:
            return None
        
        # Use first camera for metrics
        first_cam = list(pred_depths.keys())[0]
        metrics = compute_depth_metrics(
            pred_depths[first_cam],
            gt_depths[first_cam],
            self.is_metric
        )
        
        if metrics['n_valid'] < 10:
            if progress_bar is not None:
                tqdm.write(f"    ⚠️  Warning: Insufficient valid pixels for {item_id}", file=sys.stdout)
            else:
                print(f"    Warning: Insufficient valid pixels for {item_id}")
            # Free memory before returning
            pred_depths.clear()
            gt_depths.clear()
            error_depths.clear()
            images.clear()
            return None
        
        # Save metrics to JSON file (lightweight, no memory overhead)
        first_item = item_group[0]
        first_cam_id = first_item.camera_id or '0'
        first_paths = get_output_paths(item_output_dir, self.model_label, cam=first_cam_id)
        metrics_file = os.path.join(first_paths['numpy_dir'], "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate visualizations
        for cam_id, pred_depth in pred_depths.items():
            paths = get_output_paths(item_output_dir, self.model_label, cam=cam_id)
            generate_visualization_images(
                pred_depth,
                gt_depths[cam_id],
                error_depths[cam_id],
                paths,
                f"{item_id}_cam{cam_id}",
                max_depth=None,
                is_cityscapes=(self.dataset.get_output_subdir() == 'cityscapes')
            )
        
        # Generate comparison outputs if we have multiple cameras
        if has_multiple_cameras and len(error_depths) >= 2:
            self._generate_comparison_outputs(
                error_depths,
                item_output_dir,
                item_id,
                progress_bar,
                pred_depths=pred_depths,
                item_group=item_group,
                images=images
            )
        
        # Free memory after processing (arrays are saved, no need to keep in memory)
        pred_depths.clear()
        gt_depths.clear()
        error_depths.clear()
        images.clear()
        
        return metrics
    
    def _generate_comparison_outputs(
        self,
        error_depths: Dict[str, np.ndarray],
        item_output_dir: str,
        item_id: str,
        progress_bar: Optional[tqdm] = None,
        pred_depths: Optional[Dict[str, np.ndarray]] = None,
        item_group: Optional[List[DatasetItem]] = None,
        images: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Generate comparison outputs when multiple cameras exist for the same scene.
        
        This includes:
        - Warped images (projection of one camera's view to another)
        - Error disparity (difference between error arrays from different cameras, using warped arrays)
        - Comparison visualization images saved in the compare folder
        
        Args:
            error_depths: Dictionary mapping camera_id to error depth arrays
            item_output_dir: Output directory for this item
            item_id: Item identifier
            progress_bar: Optional progress bar for status updates
            pred_depths: Dictionary mapping camera_id to predicted depth arrays (for warping)
            item_group: List of DatasetItem objects (for accessing camera metadata)
        """
        try:
            # Get compare directory path (use first camera's paths)
            first_cam = list(error_depths.keys())[0]
            paths = get_output_paths(item_output_dir, self.model_label, cam=first_cam)
            compare_dir = paths['compare_dir']
            
            # Ensure compare directory exists
            os.makedirs(compare_dir, exist_ok=True)
            
            # Create numpy_matrix folder for storing numpy arrays
            numpy_matrix_dir = os.path.join(compare_dir, 'numpy_matrix')
            os.makedirs(numpy_matrix_dir, exist_ok=True)
            
            # Get sorted camera IDs for consistent ordering
            cam_ids = sorted(error_depths.keys())
            
            if len(cam_ids) < 2:
                return
            
            # Get camera metadata if available
            cam_metadata = {}
            if item_group is not None:
                for item in item_group:
                    cam_id = item.camera_id or '0'
                    if cam_id in cam_ids and item.metadata is not None:
                        cam_metadata[cam_id] = {
                            'f': item.metadata.get('f'),
                            'cx': item.metadata.get('cx'),
                            'cy': item.metadata.get('cy'),
                            'baseline': item.metadata.get('baseline')
                        }
            
            # Check if we have enough metadata for warping
            can_warp = (
                len(cam_metadata) >= 2 and
                all('f' in m and 'cx' in m and 'cy' in m and 'baseline' in m 
                    for m in cam_metadata.values()) and
                pred_depths is not None and len(pred_depths) >= 2
            )
            
            # Compute error disparity between cameras
            # For each pair of cameras, compute the difference
            for i, cam_id1 in enumerate(cam_ids):
                for cam_id2 in cam_ids[i+1:]:
                    error1 = error_depths[cam_id1]
                    error2 = error_depths[cam_id2]
                    
                    # Ensure same shape
                    if error1.shape != error2.shape:
                        if progress_bar is not None:
                            tqdm.write(
                                f"    ⚠️  Warning: Error arrays have different shapes for {item_id} "
                                f"(cam{cam_id1}: {error1.shape}, cam{cam_id2}: {error2.shape})",
                                file=sys.stdout
                            )
                        continue
                    
                    # If we can warp, warp error2 to cam_id1's viewpoint for accurate comparison
                    if can_warp and cam_id1 in cam_metadata and cam_id2 in cam_metadata:
                        try:
                            # Warp error2 and pred_depth2 to cam_id1's viewpoint
                            pred_depth2 = pred_depths.get(cam_id2)
                            if pred_depth2 is not None and pred_depth2.shape == error2.shape:
                                # Warp error2 using depth from cam_id2
                                error2_warped = warp_image_to_camera(
                                    error2,
                                    pred_depth2,  # Use predicted depth for warping
                                    cam_metadata[cam_id2],
                                    cam_metadata[cam_id1],
                                    cam_metadata[cam_id2]['baseline']
                                )
                                
                                # Save warped error array in compressed format
                                error_warped_filename = f"error_warped.npz"
                                error_warped_file = os.path.join(numpy_matrix_dir, error_warped_filename)
                                np.savez_compressed(error_warped_file, error_warped=error2_warped.astype(np.float32))
                                
                                # Also warp the predicted depth for reference
                                pred_depth2_warped = warp_image_to_camera(
                                    pred_depth2,
                                    pred_depth2,
                                    cam_metadata[cam_id2],
                                    cam_metadata[cam_id1],
                                    cam_metadata[cam_id2]['baseline']
                                )
                                # Save warped predicted depth in compressed format
                                pred_depth_warped_filename = f"pred_depth_warped.npz"
                                pred_depth_warped_file = os.path.join(numpy_matrix_dir, pred_depth_warped_filename)
                                np.savez_compressed(pred_depth_warped_file, pred_depth_warped=pred_depth2_warped.astype(np.float32))
                                
                                # Generate and save visualizations for warped error and pred_depth
                                self._save_warped_visualizations(
                                    error2_warped,
                                    pred_depth2_warped,
                                    compare_dir,
                                    item_id
                                )
                                
                                # Warp and save the RGB image if available
                                if images is not None and cam_id2 in images:
                                    image2 = images[cam_id2]
                                    if image2.shape[:2] == pred_depth2.shape[:2]:
                                        image2_warped = warp_image_to_camera(
                                            image2,
                                            pred_depth2,
                                            cam_metadata[cam_id2],
                                            cam_metadata[cam_id1],
                                            cam_metadata[cam_id2]['baseline']
                                        )
                                        # Save warped image
                                        image_warped_filename = f"image_warped.png"
                                        image_warped_file = os.path.join(compare_dir, image_warped_filename)
                                        cv2.imwrite(image_warped_file, image2_warped)
                                
                                # Use warped error for disparity calculation
                                error_disparity = np.full_like(error1, np.nan, dtype=np.float32)
                                mask_valid = (
                                    np.isfinite(error1) & np.isfinite(error2_warped) &
                                    (error1 > 0) & (error2_warped > 0)
                                )
                                error_disparity[mask_valid] = np.abs(error1[mask_valid] - error2_warped[mask_valid])
                            else:
                                # Fallback to non-warped comparison
                                error_disparity = np.full_like(error1, np.nan, dtype=np.float32)
                                mask_valid = (
                                    np.isfinite(error1) & np.isfinite(error2) &
                                    (error1 > 0) & (error2 > 0)
                                )
                                error_disparity[mask_valid] = np.abs(error1[mask_valid] - error2[mask_valid])
                        except Exception as e:
                            if progress_bar is not None:
                                tqdm.write(
                                    f"    ⚠️  Warning: Failed to warp for {item_id} "
                                    f"(cam{cam_id1} vs cam{cam_id2}): {e}",
                                    file=sys.stdout
                                )
                            # Fallback to non-warped comparison
                            error_disparity = np.full_like(error1, np.nan, dtype=np.float32)
                            mask_valid = (
                                np.isfinite(error1) & np.isfinite(error2) &
                                (error1 > 0) & (error2 > 0)
                            )
                            error_disparity[mask_valid] = np.abs(error1[mask_valid] - error2[mask_valid])
                    else:
                        # No warping available - use direct comparison
                        error_disparity = np.full_like(error1, np.nan, dtype=np.float32)
                        mask_valid = (
                            np.isfinite(error1) & np.isfinite(error2) &
                            (error1 > 0) & (error2 > 0)
                        )
                        error_disparity[mask_valid] = np.abs(error1[mask_valid] - error2[mask_valid])
                    
                    # Save error disparity array in compressed format
                    disparity_filename = f"error_disparity.npz"
                    disparity_file = os.path.join(numpy_matrix_dir, disparity_filename)
                    np.savez_compressed(disparity_file, error_disparity=error_disparity.astype(np.float32))
                    
                    # Generate visualization for error disparity
                    self._save_error_disparity_visualization(
                        error_disparity,
                        error1,
                        error2,
                        compare_dir,
                        item_id,
                        cam_id1,
                        cam_id2
                    )
                
        except Exception as e:
            if progress_bar is not None:
                tqdm.write(
                    f"    ⚠️  Warning: Failed to generate comparison outputs for {item_id}: {e}",
                    file=sys.stdout
                )
            else:
                print(f"    Warning: Failed to generate comparison outputs for {item_id}: {e}")
    
    def _save_warped_visualizations(
        self,
        error_warped: np.ndarray,
        pred_depth_warped: np.ndarray,
        compare_dir: str,
        item_id: str
    ) -> None:
        """
        Save visualizations for warped error and predicted depth.
        
        Args:
            error_warped: Warped error array
            pred_depth_warped: Warped predicted depth array
            compare_dir: Comparison directory
            item_id: Item identifier
        """
        try:
            from src.visualization import (
                depth_to_color,
                create_depth_color_scale_overlay,
                create_error_scale_overlay,
                concat_with_guide
            )
            
            # Get image dimensions
            img_height, img_width = error_warped.shape[:2]
            
            # Calculate statistics for error
            valid_error = error_warped[np.isfinite(error_warped) & (error_warped > 0)]
            if len(valid_error) > 0:
                min_error = float(np.min(valid_error))
                max_error = float(np.max(valid_error))
            else:
                min_error = 0.0
                max_error = 1.0
            
            # Create error visualization
            err_vis = np.nan_to_num(error_warped, nan=0.0, posinf=0.0, neginf=0.0)
            if max_error > 0:
                err_norm = np.clip((err_vis / max_error) * 255, 0, 255).astype(np.uint8)
            else:
                err_norm = np.zeros_like(err_vis, dtype=np.uint8)
            err_color = cv2.applyColorMap(err_norm, cv2.COLORMAP_JET)
            error_overlay = create_error_scale_overlay(min_error, max_error, image_width=img_width)
            err_color = concat_with_guide(err_color, error_overlay, position='bottom', gap=12)
            
            # Save error visualization
            error_vis_file = os.path.join(compare_dir, "error_warped.png")
            cv2.imwrite(error_vis_file, err_color)
            
            # Calculate statistics for depth
            valid_depth = pred_depth_warped[np.isfinite(pred_depth_warped) & (pred_depth_warped > 0)]
            if len(valid_depth) > 0:
                min_depth = float(np.min(valid_depth))
                max_depth = float(np.max(valid_depth))
            else:
                min_depth = 0.0
                max_depth = 80.0
            
            # Create depth visualization
            depth_colored = depth_to_color(pred_depth_warped, max_depth=max_depth)
            depth_overlay = create_depth_color_scale_overlay(
                min_depth, max_depth, image_width=img_width, show_meters=True
            )
            depth_colored = concat_with_guide(depth_colored, depth_overlay, position='bottom', gap=12)
            
            # Save depth visualization
            depth_vis_file = os.path.join(compare_dir, "pred_depth_warped.png")
            cv2.imwrite(depth_vis_file, depth_colored)
            
        except Exception as e:
            # Silently fail - visualization is optional
            pass
    
    def _save_error_disparity_visualization(
        self,
        error_disparity: np.ndarray,
        error1: np.ndarray,
        error2: np.ndarray,
        compare_dir: str,
        item_id: str,
        cam_id1: str,
        cam_id2: str
    ) -> None:
        """
        Save visualization of error disparity.
        
        Args:
            error_disparity: Disparity array (difference between error arrays)
            error1: First error array
            error2: Second error array
            compare_dir: Comparison directory
            item_id: Item identifier
            cam_id1: First camera ID
            cam_id2: Second camera ID
        """
        try:
            from src.visualization import (
                depth_to_color,
                create_error_scale_overlay,
                concat_with_guide
            )
            
            # Calculate statistics
            valid_disparity = error_disparity[np.isfinite(error_disparity) & (error_disparity > 0)]
            
            if len(valid_disparity) == 0:
                return
            
            min_disparity = float(np.min(valid_disparity))
            max_disparity = float(np.max(valid_disparity))
            mean_disparity = float(np.mean(valid_disparity))
            median_disparity = float(np.median(valid_disparity))
            
            # Get image dimensions
            img_height, img_width = error_disparity.shape[:2]
            
            # Create error disparity visualization
            err_vis = np.nan_to_num(error_disparity, nan=0.0, posinf=0.0, neginf=0.0)
            if max_disparity > 0:
                err_norm = np.clip((err_vis / max_disparity) * 255, 0, 255).astype(np.uint8)
            else:
                err_norm = np.zeros_like(err_vis, dtype=np.uint8)
            err_color = cv2.applyColorMap(err_norm, cv2.COLORMAP_JET)
            
            # Create error scale overlay
            error_overlay = create_error_scale_overlay(min_disparity, max_disparity, image_width=img_width)
            err_color = concat_with_guide(err_color, error_overlay, position='bottom', gap=12)
            
            # Save visualization
            vis_filename = f"error_disparity.png"
            vis_file = os.path.join(compare_dir, vis_filename)
            cv2.imwrite(vis_file, err_color)
            
            # Save statistics
            stats_filename = f"error_disparity_stats.txt"
            stats_file = os.path.join(compare_dir, stats_filename)
            with open(stats_file, 'w') as f:
                f.write(f"Error Disparity Statistics: {item_id}\n")
                f.write("=" * 60 + "\n\n")
                f.write("Disparity Statistics (absolute difference in error, m):\n")
                f.write(f"  Min: {min_disparity:.6f} m\n")
                f.write(f"  Max: {max_disparity:.6f} m\n")
                f.write(f"  Mean: {mean_disparity:.6f} m\n")
                f.write(f"  Median: {median_disparity:.6f} m\n")
                f.write(f"  Std: {np.std(valid_disparity):.6f} m\n")
                f.write(f"  Valid pixels: {len(valid_disparity)}\n")
            
        except Exception as e:
            # Silently fail - comparison visualization is optional
            pass
    
