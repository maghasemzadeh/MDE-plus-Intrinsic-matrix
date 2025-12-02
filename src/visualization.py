"""
Visualization utilities for depth estimation evaluation.
"""

import os
import shutil
import numpy as np
import cv2
from typing import Dict, Optional
from pathlib import Path


def depth_to_color(depth: np.ndarray, max_depth: float = 10000.0) -> np.ndarray:
    """Convert depth map to color visualization (input and max_depth share same units)."""
    depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_vis = np.clip(depth_vis, 0, max_depth)
    depth_normalized = (depth_vis / max_depth * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colored


def create_depth_color_scale_overlay(
    min_depth_m: float,
    max_depth_m: float,
    image_width: int,
    bar_height: int = 60,
    show_meters: bool = True,
    show_mm: bool = False
) -> np.ndarray:
    """
    Create a modern horizontal color scale overlay that spans the full image width.
    
    Args:
        min_depth_m: Minimum depth in meters
        max_depth_m: Maximum depth in meters
        image_width: Width of the image (overlay will match this width)
        bar_height: Height of the color bar
        show_meters: Whether to show meter values
        show_mm: Whether to show millimeter values
    
    Returns:
        Horizontal overlay guide image as numpy array (BGR format for OpenCV)
    """
    # Create horizontal color gradient
    gradient = np.zeros((bar_height, image_width, 3), dtype=np.uint8)
    for j in range(image_width):
        # Left to right: blue (low) to red (high)
        normalized = j / (image_width - 1) if image_width > 1 else 0
        normalized_uint8 = int(normalized * 255)
        color = cv2.applyColorMap(np.array([[normalized_uint8]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        gradient[:, j] = color
    
    # Calculate text dimensions - scale fonts based on image width for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font size based on image width (larger images get larger fonts)
    # Base scale for ~1000px width, scale proportionally
    base_width = 1000.0
    font_scale = max(1.2, min(2.0, 0.85 * (image_width / base_width)))  # Larger, more visible fonts
    font_thickness = max(2, int(font_scale * 2))  # Thicker for larger fonts
    label_font_scale = max(0.9, min(1.5, 0.7 * (image_width / base_width)))  # Larger labels
    label_font_thickness = max(2, int(label_font_scale * 2))
    
    # Space for labels above and below the bar (increased to accommodate larger text)
    # Scale margins based on font size to prevent text clipping
    margin_scale = max(1.0, font_scale / 0.85)  # Scale relative to original
    top_margin = int(35 * margin_scale)
    bottom_margin = int(50 * margin_scale)
    overlay_height = bar_height + top_margin + bottom_margin
    
    # Create overlay with modern styling
    overlay = np.ones((overlay_height, image_width, 3), dtype=np.uint8) * 255
    
    # Place color gradient
    y_offset = top_margin
    overlay[y_offset:y_offset+bar_height, :] = gradient
    
    # Add border around the color bar for modern look
    border_thickness = 2
    cv2.rectangle(overlay, (0, y_offset), (image_width-1, y_offset+bar_height-1), (0, 0, 0), border_thickness)
    
    # Add scale markers with tick marks
    num_markers = 8
    tick_length = 8
    
    for i in range(num_markers):
        x_pos = int((i / (num_markers - 1)) * (image_width - 1)) if num_markers > 1 else 0
        depth_m = min_depth_m + (i / (num_markers - 1)) * (max_depth_m - min_depth_m) if num_markers > 1 else min_depth_m
        
        # Draw tick marks above and below the bar
        cv2.line(overlay, (x_pos, y_offset - tick_length), (x_pos, y_offset), (0, 0, 0), 2)
        cv2.line(overlay, (x_pos, y_offset + bar_height), (x_pos, y_offset + bar_height + tick_length), (0, 0, 0), 2)
        
        # Add text labels below the bar
        if show_meters:
            text = f"{depth_m:.1f}m"
        elif show_mm:
            text = f"{depth_m*1000:.0f}mm"
        else:
            text = f"{depth_m:.1f}"
        
        # Get text size for centering
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x_pos - text_w // 2
        text_y = y_offset + bar_height + tick_length + text_h + 5
        
        # Ensure text doesn't go out of bounds
        text_x = max(0, min(text_x, image_width - text_w))
        
        # Draw text with shadow for better visibility
        cv2.putText(overlay, text, (text_x + 1, text_y + 1), font, font_scale, (255, 255, 255), font_thickness + 1)
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    
    # Add min/max labels at the ends
    min_text = f"{min_depth_m:.1f}m" if show_meters else f"{min_depth_m*1000:.0f}mm" if show_mm else f"{min_depth_m:.1f}"
    max_text = f"{max_depth_m:.1f}m" if show_meters else f"{max_depth_m*1000:.0f}mm" if show_mm else f"{max_depth_m:.1f}"
    
    # Min label (left)
    (min_w, min_h), _ = cv2.getTextSize(min_text, font, label_font_scale, label_font_thickness)
    cv2.putText(overlay, min_text, (5, y_offset - 10), font, label_font_scale, (100, 100, 100), label_font_thickness)
    
    # Max label (right)
    (max_w, max_h), _ = cv2.getTextSize(max_text, font, label_font_scale, label_font_thickness)
    cv2.putText(overlay, max_text, (image_width - max_w - 5, y_offset - 10), font, label_font_scale, (100, 100, 100), label_font_thickness)
    
    return overlay


def create_error_scale_overlay(
    min_error: float,
    max_error: float,
    image_width: int,
    bar_height: int = 60
) -> np.ndarray:
    """
    Create a modern horizontal error scale overlay that spans the full image width.
    
    Args:
        min_error: Minimum error value
        max_error: Maximum error value
        image_width: Width of the image (overlay will match this width)
        bar_height: Height of the color bar
    
    Returns:
        Horizontal overlay guide image as numpy array (BGR format for OpenCV)
    """
    # Create horizontal color gradient
    gradient = np.zeros((bar_height, image_width, 3), dtype=np.uint8)
    for j in range(image_width):
        # Left to right: blue (low error) to red (high error)
        normalized = j / (image_width - 1) if image_width > 1 else 0
        normalized_uint8 = int(normalized * 255)
        color = cv2.applyColorMap(np.array([[normalized_uint8]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        gradient[:, j] = color
    
    # Calculate text dimensions - scale fonts based on image width for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font size based on image width (larger images get larger fonts)
    # Base scale for ~1000px width, scale proportionally
    base_width = 1000.0
    font_scale = max(1.2, min(2.0, 0.85 * (image_width / base_width)))  # Larger, more visible fonts
    font_thickness = max(2, int(font_scale * 2))  # Thicker for larger fonts
    label_font_scale = max(0.9, min(1.5, 0.7 * (image_width / base_width)))  # Larger labels
    label_font_thickness = max(2, int(label_font_scale * 2))
    
    # Space for labels above and below the bar (increased to accommodate larger text)
    # Scale margins based on font size to prevent text clipping
    margin_scale = max(1.0, font_scale / 0.85)  # Scale relative to original
    top_margin = int(35 * margin_scale)
    bottom_margin = int(50 * margin_scale)
    overlay_height = bar_height + top_margin + bottom_margin
    
    # Create overlay with modern styling
    overlay = np.ones((overlay_height, image_width, 3), dtype=np.uint8) * 255
    
    # Place color gradient
    y_offset = top_margin
    overlay[y_offset:y_offset+bar_height, :] = gradient
    
    # Add border around the color bar for modern look
    border_thickness = 2
    cv2.rectangle(overlay, (0, y_offset), (image_width-1, y_offset+bar_height-1), (0, 0, 0), border_thickness)
    
    # Add scale markers with tick marks
    num_markers = 8
    tick_length = 8
    
    for i in range(num_markers):
        x_pos = int((i / (num_markers - 1)) * (image_width - 1)) if num_markers > 1 else 0
        error_val = min_error + (i / (num_markers - 1)) * (max_error - min_error) if num_markers > 1 else min_error
        
        # Draw tick marks above and below the bar
        cv2.line(overlay, (x_pos, y_offset - tick_length), (x_pos, y_offset), (0, 0, 0), 2)
        cv2.line(overlay, (x_pos, y_offset + bar_height), (x_pos, y_offset + bar_height + tick_length), (0, 0, 0), 2)
        
        # Add text labels below the bar
        text = f"{error_val:.1f}"
        
        # Get text size for centering
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x_pos - text_w // 2
        text_y = y_offset + bar_height + tick_length + text_h + 5
        
        # Ensure text doesn't go out of bounds
        text_x = max(0, min(text_x, image_width - text_w))
        
        # Draw text with shadow for better visibility
        cv2.putText(overlay, text, (text_x + 1, text_y + 1), font, font_scale, (255, 255, 255), font_thickness + 1)
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    
    # Add min/max labels at the ends
    min_text = f"{min_error:.1f}"
    max_text = f"{max_error:.1f}"
    
    # Min label (left)
    (min_w, min_h), _ = cv2.getTextSize(min_text, font, label_font_scale, label_font_thickness)
    cv2.putText(overlay, min_text, (5, y_offset - 10), font, label_font_scale, (100, 100, 100), label_font_thickness)
    
    # Max label (right)
    (max_w, max_h), _ = cv2.getTextSize(max_text, font, label_font_scale, label_font_thickness)
    cv2.putText(overlay, max_text, (image_width - max_w - 5, y_offset - 10), font, label_font_scale, (100, 100, 100), label_font_thickness)
    
    return overlay


def concat_with_guide(
    image: np.ndarray,
    guide: np.ndarray,
    position: str = 'right',
    gap: int = 12
) -> np.ndarray:
    """
    Concatenate image with guide overlay.
    
    Args:
        image: Main image
        guide: Guide overlay image
        position: 'right' or 'bottom'
        gap: Gap between image and guide
    
    Returns:
        Concatenated image
    """
    h_img, w_img = image.shape[:2]
    h_guide, w_guide = guide.shape[:2]
    
    if position == 'right':
        # Concatenate horizontally
        result = np.ones((max(h_img, h_guide), w_img + w_guide + gap, 3), dtype=np.uint8) * 255
        result[:h_img, :w_img] = image
        result[:h_guide, w_img + gap:] = guide
    else:  # bottom
        # Concatenate vertically
        result = np.ones((h_img + h_guide + gap, max(w_img, w_guide), 3), dtype=np.uint8) * 255
        result[:h_img, :w_img] = image
        result[h_img + gap:, :w_guide] = guide
    
    return result


def generate_visualization_images(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    error_depth: np.ndarray,
    paths: Dict[str, str],
    item_id: str,
    max_depth: Optional[float] = None,
    is_cityscapes: bool = True
) -> None:
    """
    Generate visualization images for depth estimation results.
    
    Args:
        pred_depth: Predicted depth map in meters
        gt_depth: Ground truth depth map in meters
        error_depth: Error depth map in meters
        paths: Dictionary with output paths
        item_id: Identifier for this item
        max_depth: Maximum depth for visualization (None to use per-image max)
        is_cityscapes: Whether this is Cityscapes dataset (for default max_depth)
    """
    try:
        disp_dir = paths['disp_dir']
        os.makedirs(disp_dir, exist_ok=True)
        
        # Calculate per-image depth range if max_depth not provided
        valid_depth = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
        valid_gt_depth = gt_depth[np.isfinite(gt_depth) & (gt_depth > 0)]
        valid_error = error_depth[np.isfinite(error_depth) & (error_depth > 0)]
        
        # Collect all valid depths for range calculation
        all_valid = []
        if len(valid_depth) > 0:
            all_valid.extend(valid_depth.flatten().tolist())
        if len(valid_gt_depth) > 0:
            all_valid.extend(valid_gt_depth.flatten().tolist())
        
        if max_depth is None:
            # Use per-image max depth
            if len(all_valid) > 0:
                max_depth_vis = float(np.max(all_valid))
            else:
                max_depth_vis = 80.0 if not is_cityscapes else 20.0
        else:
            max_depth_vis = max_depth
        
        min_depth_vis = 0.0
        if len(all_valid) > 0:
            min_depth_vis = float(np.min(all_valid))
        
        # Get image dimensions for overlay
        img_height, img_width = pred_depth.shape[:2]
        
        # Create depth scale overlay
        depth_overlay = create_depth_color_scale_overlay(
            min_depth_vis, max_depth_vis, image_width=img_width, show_meters=True
        )
        
        # Calculate statistics
        if len(valid_depth) > 0:
            min_depth_m = float(np.min(valid_depth))
            max_depth_m = float(np.max(valid_depth))
            mean_depth = float(np.mean(valid_depth))
            median_depth = float(np.median(valid_depth))
        else:
            min_depth_m = max_depth_m = mean_depth = median_depth = 0.0
        
        if len(valid_error) > 0:
            min_error = float(np.min(valid_error))
            max_error = float(np.max(valid_error))
            mean_error = float(np.mean(valid_error))
            median_error = float(np.median(valid_error))
        else:
            min_error = max_error = mean_error = median_error = 0.0
        
        # Generate predicted depth visualization
        pred_vis_file = os.path.join(disp_dir, "pred_depth_m.png")
        depth_colored = depth_to_color(pred_depth, max_depth=max_depth_vis)
        depth_colored = concat_with_guide(depth_colored, depth_overlay, position='bottom', gap=12)
        cv2.imwrite(pred_vis_file, depth_colored)
        
        # Generate GT depth visualization
        gt_vis_file = os.path.join(disp_dir, "gt.png")
        
        # Check if GT visualization exists from another model (basic/metric)
        image_output_dir = os.path.dirname(os.path.dirname(disp_dir))  # Go up from disp0 to image dir
        # Extract current model label from path
        if os.path.sep + 'metric' + os.path.sep in disp_dir or disp_dir.endswith(os.path.sep + 'metric'):
            other_model_label = 'basic'
        elif os.path.sep + 'basic' + os.path.sep in disp_dir or disp_dir.endswith(os.path.sep + 'basic'):
            other_model_label = 'metric'
        else:
            other_model_label = 'basic'
        other_gt_path = os.path.join(image_output_dir, other_model_label, 'disp0', 'gt.png')
        
        if os.path.exists(other_gt_path) and os.path.getsize(other_gt_path) > 0:
            # Reuse existing GT visualization from other model
            shutil.copy2(other_gt_path, gt_vis_file)
        else:
            # Generate GT visualization using GT's own depth range
            if len(valid_gt_depth) > 0:
                gt_max_depth = float(np.max(valid_gt_depth))
                gt_min_depth = float(np.min(valid_gt_depth))
            else:
                gt_max_depth = max_depth_vis
                gt_min_depth = 0.0
            
            # Create GT-specific depth overlay
            gt_depth_overlay = create_depth_color_scale_overlay(
                gt_min_depth, gt_max_depth, image_width=img_width, show_meters=True
            )
            
            gt_depth_colored = depth_to_color(gt_depth, max_depth=gt_max_depth)
            gt_depth_colored = concat_with_guide(gt_depth_colored, gt_depth_overlay, position='bottom', gap=12)
            cv2.imwrite(gt_vis_file, gt_depth_colored)
        
        # Generate error visualization
        error_vis_file = os.path.join(disp_dir, "depth_error_m.png")
        if len(valid_error) > 0:
            max_error_vis = float(np.max(valid_error))
        else:
            max_error_vis = 1.0
        err_vis = np.nan_to_num(error_depth, nan=0.0, posinf=0.0, neginf=0.0)
        if max_error_vis > 0:
            err_norm = np.clip((err_vis / max_error_vis) * 255, 0, 255).astype(np.uint8)
        else:
            err_norm = np.zeros_like(err_vis, dtype=np.uint8)
        err_color = cv2.applyColorMap(err_norm, cv2.COLORMAP_JET)
        error_overlay = create_error_scale_overlay(min_error, max_error_vis, image_width=img_width)
        err_color = concat_with_guide(err_color, error_overlay, position='bottom', gap=12)
        cv2.imwrite(error_vis_file, err_color)
        
        # Save statistics
        stats_file = os.path.join(disp_dir, "stats.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Image: {item_id} Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write("Depth Statistics (m):\n")
            f.write(f"  Min: {min_depth_m:.4f} m\n")
            f.write(f"  Max: {max_depth_m:.4f} m\n")
            if len(valid_depth) > 0:
                f.write(f"  Mean: {mean_depth:.4f} m\n")
                f.write(f"  Median: {median_depth:.4f} m\n")
                f.write(f"  Std: {np.std(valid_depth):.4f} m\n")
            f.write(f"\nDepth Range (for visualization): {min_depth_vis:.4f}m - {max_depth_vis:.4f}m\n")
            f.write("\nError Statistics (depth error, m):\n")
            f.write(f"  Min: {min_error:.4f}\n")
            f.write(f"  Max: {max_error_vis:.4f}\n")
            if len(valid_error) > 0:
                f.write(f"  Mean: {mean_error:.4f}\n")
                f.write(f"  Median: {median_error:.4f}\n")
                f.write(f"  Std: {np.std(valid_error):.4f}\n")
    
    except Exception as e:
        print(f"    Warning: Failed to generate visualization images for {item_id}: {e}")


def generate_visualizations_for_dataset(dataset_output_dir, model_label='metric', force=False):
    """
    Generate visualization images for all processed images in a dataset.
    
    This function regenerates visualization images from existing numpy depth files.
    Useful if the visualization pass didn't run or failed.
    
    Args:
        dataset_output_dir: Base directory for the dataset (e.g., 'results/cityscapes')
        model_label: 'metric' or 'basic'
        force: If True, regenerate even if images exist
    """
    dataset_output_dir = Path(dataset_output_dir)
    
    if not dataset_output_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_output_dir}")
        return
    
    # Find all image directories
    image_dirs = [d for d in dataset_output_dir.iterdir() if d.is_dir()]
    
    if len(image_dirs) == 0:
        print(f"No image directories found in {dataset_output_dir}")
        return
    
    print(f"Found {len(image_dirs)} image directories")
    
    # Collect all depth maps to calculate global range
    all_depths = []
    pred_depth_maps = {}
    gt_depth_maps = {}
    error_depth_maps = {}
    
    print("\nLoading depth maps...")
    for img_dir in image_dirs:
        numpy_dir = img_dir / model_label / 'disp0' / 'numpy_matrix'
        
        if not numpy_dir.exists():
            continue
        
        try:
            pred_depth = None
            gt_depth = None
            error_depth = None
            
            # Try compressed format first (preferred)
            compressed_file = numpy_dir / 'arrays.npz'
            if compressed_file.exists():
                try:
                    arrays = np.load(compressed_file)
                    pred_depth = arrays['pred_depth']
                    gt_depth = arrays['gt_depth']
                    error_depth = arrays['error']
                except Exception as e:
                    print(f"Warning: Failed to load compressed arrays from {compressed_file}: {e}")
                    continue
            
            # Fallback to legacy .npy files if compressed format didn't work
            if pred_depth is None:
                pred_file = numpy_dir / 'pred_depth_meters.npy'
                gt_file = numpy_dir / 'gt_depth_meters.npy'
                error_file = numpy_dir / 'error.npy'
                
                if not (pred_file.exists() and gt_file.exists() and error_file.exists()):
                    continue
                
                try:
                    pred_depth = np.load(pred_file)
                    gt_depth = np.load(gt_file)
                    error_depth = np.load(error_file)
                except Exception as e:
                    print(f"Warning: Failed to load arrays from {numpy_dir}: {e}")
                    continue
            
            # Process loaded data
            if pred_depth is not None and gt_depth is not None and error_depth is not None:
                img_name = img_dir.name
                pred_depth_maps[img_name] = pred_depth
                gt_depth_maps[img_name] = gt_depth
                error_depth_maps[img_name] = error_depth
                
                # Collect valid depths for global range
                valid_pred = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
                valid_gt = gt_depth[np.isfinite(gt_depth) & (gt_depth > 0)]
                
                if len(valid_pred) > 0:
                    all_depths.extend(valid_pred.flatten().tolist())
                if len(valid_gt) > 0:
                    all_depths.extend(valid_gt.flatten().tolist())
        except Exception as e:
            print(f"  Error loading {img_dir.name}: {e}")
            continue
    
    if len(all_depths) == 0:
        print("Error: No valid depth maps found!")
        return
    
    # Calculate global depth range
    global_min_depth_m = float(np.min(all_depths))
    global_max_depth_m = float(np.max(all_depths))
    
    print(f"\nGlobal depth range: {global_min_depth_m:.4f}m - {global_max_depth_m:.4f}m")
    
    # Generate visualizations
    print(f"\nGenerating visualization images for {len(pred_depth_maps)} images...")
    images_generated = 0
    
    for img_idx, (img_name, pred_depth) in enumerate(pred_depth_maps.items()):
        try:
            gt_depth = gt_depth_maps[img_name]
            error_depth = error_depth_maps[img_name]
            
            img_dir = dataset_output_dir / img_name
            disp_dir = img_dir / model_label / 'disp0'
            
            # Ensure disp_dir exists
            disp_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if images already exist
            pred_vis_file = disp_dir / "pred_depth_m.png"
            gt_vis_file = disp_dir / "gt.png"
            error_vis_file = disp_dir / "depth_error_m.png"
            stats_file = disp_dir / "stats.txt"
            
            if not force and all(f.exists() for f in [pred_vis_file, gt_vis_file, error_vis_file, stats_file]):
                if (img_idx + 1) % 50 == 0:
                    print(f"  Progress: {img_idx + 1}/{len(pred_depth_maps)} (skipping existing)")
                continue
            
            # Get image dimensions for overlay
            img_height, img_width = pred_depth.shape[:2]
            
            # Create global depth scale overlay for this image width
            global_depth_overlay = create_depth_color_scale_overlay(
                global_min_depth_m, global_max_depth_m, image_width=img_width, show_meters=True
            )
            
            # Calculate statistics
            valid_depth = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0)]
            valid_gt_depth = gt_depth[np.isfinite(gt_depth) & (gt_depth > 0)]
            valid_error = error_depth[np.isfinite(error_depth) & (error_depth > 0)]
            
            if len(valid_depth) > 0:
                min_depth_m = float(np.min(valid_depth))
                max_depth_m = float(np.max(valid_depth))
                mean_depth = float(np.mean(valid_depth))
                median_depth = float(np.median(valid_depth))
            else:
                min_depth_m = max_depth_m = mean_depth = median_depth = 0.0
            
            if len(valid_error) > 0:
                min_error = float(np.min(valid_error))
                max_error = float(np.max(valid_error))
                mean_error = float(np.mean(valid_error))
                median_error = float(np.median(valid_error))
            else:
                min_error = max_error = mean_error = median_error = 0.0
            
            # Save predicted depth visualization
            depth_colored = depth_to_color(pred_depth, max_depth=global_max_depth_m)
            depth_colored = concat_with_guide(depth_colored, global_depth_overlay, position='bottom', gap=12)
            success = cv2.imwrite(str(pred_vis_file), depth_colored)
            if not success:
                raise RuntimeError(f"Failed to write {pred_vis_file}")
            
            # GT depth visualization
            gt_depth_colored = depth_to_color(gt_depth, max_depth=global_max_depth_m)
            gt_depth_colored = concat_with_guide(gt_depth_colored, global_depth_overlay, position='bottom', gap=12)
            success = cv2.imwrite(str(gt_vis_file), gt_depth_colored)
            if not success:
                raise RuntimeError(f"Failed to write {gt_vis_file}")
            
            # Depth error visualization
            if len(valid_error) > 0:
                max_error_vis = float(np.max(valid_error))
            else:
                max_error_vis = 1.0
            err_vis = np.nan_to_num(error_depth, nan=0.0, posinf=0.0, neginf=0.0)
            if max_error_vis > 0:
                err_norm = np.clip((err_vis / max_error_vis) * 255, 0, 255).astype(np.uint8)
            else:
                err_norm = np.zeros_like(err_vis, dtype=np.uint8)
            err_color = cv2.applyColorMap(err_norm, cv2.COLORMAP_JET)
            error_overlay = create_error_scale_overlay(min_error, max_error_vis, image_width=img_width)
            err_color = concat_with_guide(err_color, error_overlay, position='bottom', gap=12)
            success = cv2.imwrite(str(error_vis_file), err_color)
            if not success:
                raise RuntimeError(f"Failed to write {error_vis_file}")
            
            # Save statistics
            with open(stats_file, 'w') as f:
                f.write(f"Image: {img_name} Statistics\n")
                f.write("=" * 50 + "\n\n")
                f.write("Depth Statistics (m):\n")
                f.write(f"  Min: {min_depth_m:.4f} m\n")
                f.write(f"  Max: {max_depth_m:.4f} m\n")
                if len(valid_depth) > 0:
                    f.write(f"  Mean: {mean_depth:.4f} m\n")
                    f.write(f"  Median: {median_depth:.4f} m\n")
                    f.write(f"  Std: {np.std(valid_depth):.4f} m\n")
                f.write(f"\nGlobal Depth Range (for visualization): {global_min_depth_m:.4f}m - {global_max_depth_m:.4f}m\n")
                f.write("\nError Statistics (depth error, m):\n")
                f.write(f"  Min: {min_error:.4f}\n")
                f.write(f"  Max: {max_error_vis:.4f}\n")
                if len(valid_error) > 0:
                    f.write(f"  Mean: {mean_error:.4f}\n")
                    f.write(f"  Median: {median_error:.4f}\n")
                    f.write(f"  Std: {np.std(valid_error):.4f}\n")
            
            images_generated += 1
            if (img_idx + 1) % 50 == 0 or (img_idx + 1) == len(pred_depth_maps):
                print(f"  Progress: {img_idx + 1}/{len(pred_depth_maps)} (generated: {images_generated})")
        
        except Exception as e:
            print(f"  Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nSuccessfully generated visualization images for {images_generated}/{len(pred_depth_maps)} images")

