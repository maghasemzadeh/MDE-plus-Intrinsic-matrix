"""
Integration tests for the full pipeline.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import cv2

from datasets.base import DatasetItem, DatasetConfig
from src.metrics import compute_depth_metrics
from datasets.middlebury import (
    MiddleburyDataset,
    read_pfm,
    depth_from_disparity,
    parse_calib,
    evaluate_per_image_metrics
)


class TestFullPipeline:
    """Test the complete pipeline from data loading to metrics."""
    
    def test_pipeline_data_loading_to_metrics(self, temp_dir):
        """Test complete pipeline: load data -> compute depth -> calculate metrics."""
        # Step 1: Create a mock scene
        scene_dir = os.path.join(temp_dir, "test-scene")
        os.makedirs(scene_dir, exist_ok=True)
        
        # Create calibration
        calib_path = os.path.join(scene_dir, "calib.txt")
        with open(calib_path, 'w') as f:
            f.write("cam0=[1000 0 320; 0 1000 240; 0 0 1]\n")
            f.write("cam1=[1050 0 330; 0 1050 245; 0 0 1]\n")
            f.write("doffs=10.0\n")
            f.write("baseline=0.1\n")
        
        # Create images
        img0 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(scene_dir, "im0.png"), img0)
        cv2.imwrite(os.path.join(scene_dir, "im1.png"), img1)
        
        # Create disparity files
        for i, disp_name in enumerate(["disp0.pfm", "disp1.pfm"]):
            disp_path = os.path.join(scene_dir, disp_name)
            disparity = np.random.uniform(10, 200, (100, 100)).astype(np.float32)
            
            with open(disp_path, 'wb') as f:
                f.write(b'PF\n')
                f.write(b'100 100\n')
                f.write(b'1.0\n')
                flipped = np.flipud(disparity)
                flipped.astype('>f').tofile(f)
        
        # Step 2: Load dataset
        config = DatasetConfig(dataset_path=temp_dir)
        dataset = MiddleburyDataset(config)
        items = dataset.find_items()
        
        assert len(items) == 2  # Two cameras
        
        # Step 3: Load ground truth depth for camera 0
        item0 = items[0]
        gt_depth0 = dataset.load_gt_depth(item0.gt_path, item0)
        
        assert gt_depth0.shape == (100, 100)
        assert gt_depth0.dtype == np.float32
        
        # Step 4: Simulate model prediction (create mock prediction)
        pred_depth0 = gt_depth0 + np.random.normal(0, 0.1, gt_depth0.shape).astype(np.float32)
        pred_depth0 = np.maximum(pred_depth0, 0.1)  # Ensure positive
        
        # Step 5: Calculate error
        error0 = np.full_like(gt_depth0, np.nan, dtype=np.float32)
        mask_valid = (
            np.isfinite(gt_depth0) & (gt_depth0 > 0) &
            np.isfinite(pred_depth0) & (pred_depth0 > 0)
        )
        error0[mask_valid] = np.abs(gt_depth0[mask_valid] - pred_depth0[mask_valid])
        
        assert np.sum(np.isfinite(error0)) > 0
        
        # Step 6: Calculate metrics
        metrics0 = compute_depth_metrics(pred_depth0, gt_depth0, is_metric_model=True)
        
        assert 'abs_rel' in metrics0
        assert 'rmse' in metrics0
        assert metrics0['n_valid'] > 0
        assert not np.isnan(metrics0['abs_rel'])
        assert not np.isnan(metrics0['rmse'])
    
    def test_two_camera_comparison_pipeline(self, temp_dir):
        """Test pipeline comparing two cameras."""
        # Create output directory structure
        output_dir = os.path.join(temp_dir, "results", "middlebury", "test-scene")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metric model directory
        metric_dir = os.path.join(output_dir, "metric")
        os.makedirs(metric_dir, exist_ok=True)
        
        # Create camera directories
        for cam_id in ['0', '1']:
            cam_dir = os.path.join(metric_dir, f"disp{cam_id}")
            numpy_dir = os.path.join(cam_dir, "numpy_matrix")
            os.makedirs(numpy_dir, exist_ok=True)
            
            # Create depth maps
            gt_depth = np.random.uniform(1.0, 5.0, (50, 50)).astype(np.float32)
            pred_depth = gt_depth + np.random.normal(0, 0.1, gt_depth.shape).astype(np.float32)
            pred_depth = np.maximum(pred_depth, 0.1)
            
            # Save numpy arrays
            np.save(os.path.join(numpy_dir, "pred_depth_meters.npy"), pred_depth)
            np.save(os.path.join(numpy_dir, "gt_depth_meters.npy"), gt_depth)
        
        # Run evaluation
        results = evaluate_per_image_metrics(
            output_path=os.path.join(temp_dir, "results", "middlebury"),
            max_scenes=1,
            is_metric_model=True,
            regex_pattern=None
        )
        
        # Check results structure
        assert 'abs_rel' in results or 'rmse' in results
        assert 'num_scenes' in results
        assert results['num_scenes'] == 1
    
    def test_error_statistics_consistency(self):
        """Test that error statistics are consistent across calculations."""
        # Create test data
        pred = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
        gt = pred + np.random.normal(0, 0.2, pred.shape).astype(np.float32)
        gt = np.maximum(gt, 0.1)
        
        # Calculate error
        error = np.full_like(gt, np.nan, dtype=np.float32)
        mask_valid = (
            np.isfinite(gt) & (gt > 0) &
            np.isfinite(pred) & (pred > 0)
        )
        error[mask_valid] = np.abs(gt[mask_valid] - pred[mask_valid])
        
        # Calculate metrics
        metrics = compute_depth_metrics(pred, gt, is_metric_model=True)
        
        # Verify consistency: RMSE from metrics should match sqrt(mean(error^2))
        finite_errors = error[np.isfinite(error)]
        if len(finite_errors) > 0:
            expected_rmse = np.sqrt(np.mean(finite_errors ** 2))
            assert abs(metrics['rmse'] - expected_rmse) < 1e-5
    
    def test_metrics_consistency_across_cameras(self):
        """Test that metrics are calculated consistently for both cameras."""
        # Create identical depth maps for both cameras
        gt = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
        pred0 = gt + np.random.normal(0, 0.1, gt.shape).astype(np.float32)
        pred1 = gt + np.random.normal(0, 0.1, gt.shape).astype(np.float32)
        pred0 = np.maximum(pred0, 0.1)
        pred1 = np.maximum(pred1, 0.1)
        
        # Calculate metrics for both
        metrics0 = compute_depth_metrics(pred0, gt, is_metric_model=True)
        metrics1 = compute_depth_metrics(pred1, gt, is_metric_model=True)
        
        # Both should have same number of valid pixels (same GT)
        assert metrics0['n_valid'] == metrics1['n_valid']
        
        # Both should have valid metrics
        assert not np.isnan(metrics0['abs_rel'])
        assert not np.isnan(metrics1['abs_rel'])
        assert not np.isnan(metrics0['rmse'])
        assert not np.isnan(metrics1['rmse'])


class TestDataConsistency:
    """Test data consistency throughout the pipeline."""
    
    def test_depth_map_shape_consistency(self):
        """Test that depth maps maintain shape throughout pipeline."""
        # Create test data
        height, width = 100, 150
        pred = np.random.uniform(1.0, 5.0, (height, width)).astype(np.float32)
        gt = np.random.uniform(1.0, 5.0, (height, width)).astype(np.float32)
        
        # Error should have same shape
        error = np.abs(pred - gt)
        
        assert error.shape == (height, width)
        assert error.shape == pred.shape
        assert error.shape == gt.shape
    
    def test_valid_mask_shape_consistency(self):
        """Test that valid masks have correct shape."""
        pred = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
        gt = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
        
        mask_valid = (
            np.isfinite(gt) & (gt > 0) &
            np.isfinite(pred) & (pred > 0)
        )
        
        assert mask_valid.shape == pred.shape
        assert mask_valid.shape == gt.shape
        assert mask_valid.dtype == bool
    
    def test_metrics_dtype_consistency(self):
        """Test that metrics have correct data types."""
        pred = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
        gt = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
        
        metrics = compute_depth_metrics(pred, gt, is_metric_model=True)
        
        assert isinstance(metrics['abs_rel'], (float, np.floating))
        assert isinstance(metrics['rmse'], (float, np.floating))
        assert isinstance(metrics['n_valid'], (int, np.integer))

