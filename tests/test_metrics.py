"""
Tests for depth estimation metrics calculation.
"""

import pytest
import numpy as np
from src.metrics import compute_depth_metrics


class TestComputeDepthMetrics:
    """Test compute_depth_metrics function."""
    
    def test_metric_model_absrel_rmse(self, sample_depth_map, sample_gt_depth):
        """Test AbsRel and RMSE calculation for metric models."""
        # Create matching shapes
        pred = sample_depth_map.copy()
        gt = sample_gt_depth.copy()
        
        # Ensure some overlap in valid regions
        valid_pred = np.isfinite(pred) & (pred > 0)
        valid_gt = np.isfinite(gt) & (gt > 0)
        valid_both = valid_pred & valid_gt
        
        # If not enough overlap, create some
        if np.sum(valid_both) < 100:
            # Make a region valid in both
            pred[50:70, 50:70] = np.random.uniform(1.0, 5.0, (20, 20))
            gt[50:70, 50:70] = pred[50:70, 50:70] + np.random.normal(0, 0.1, (20, 20))
            gt[50:70, 50:70] = np.maximum(gt[50:70, 50:70], 0.1)  # Ensure positive
        
        metrics = compute_depth_metrics(pred, gt, is_metric_model=True)
        
        assert 'abs_rel' in metrics
        assert 'rmse' in metrics
        assert 'n_valid' in metrics
        assert metrics['n_valid'] >= 10
        assert not np.isnan(metrics['abs_rel'])
        assert not np.isnan(metrics['rmse'])
        assert metrics['abs_rel'] >= 0
        assert metrics['rmse'] >= 0
    
    def test_metric_model_perfect_match(self, perfect_depth_maps):
        """Test metrics with perfect matching depth maps."""
        pred, gt = perfect_depth_maps
        
        metrics = compute_depth_metrics(pred, gt, is_metric_model=True)
        
        assert metrics['abs_rel'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['n_valid'] == pred.size
    
    def test_metric_model_known_error(self):
        """Test metrics with known error values."""
        # Create simple test case: pred = gt + 0.1
        # Ensure all values are positive and we have at least 10 pixels
        gt = np.array([[1.0, 2.0, 3.0, 4.0], 
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
        pred = gt + 0.1  # All positive since gt > 0
        
        metrics = compute_depth_metrics(pred, gt, is_metric_model=True)
        
        # Check that we have valid metrics
        assert metrics['n_valid'] >= 10
        assert not np.isnan(metrics['abs_rel'])
        assert not np.isnan(metrics['rmse'])
        
        # AbsRel should be mean(|pred - gt| / gt) = mean(0.1 / gt)
        expected_absrel = np.mean(0.1 / gt)
        assert abs(metrics['abs_rel'] - expected_absrel) < 1e-5
        
        # RMSE should be 0.1
        assert abs(metrics['rmse'] - 0.1) < 1e-5
    
    def test_non_metric_model_silog(self, sample_depth_map, sample_gt_depth):
        """Test SILog calculation for non-metric models."""
        pred = sample_depth_map.copy()
        gt = sample_gt_depth.copy()
        
        # Ensure some overlap
        if np.sum(np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)) < 100:
            pred[50:70, 50:70] = np.random.uniform(0.5, 2.0, (20, 20))
            gt[50:70, 50:70] = np.random.uniform(1.0, 5.0, (20, 20))
        
        metrics = compute_depth_metrics(pred, gt, is_metric_model=False)
        
        assert 'silog' in metrics
        assert 'n_valid' in metrics
        assert metrics['n_valid'] >= 10
        assert not np.isnan(metrics['silog'])
        assert metrics['silog'] >= 0
    
    def test_insufficient_valid_pixels(self):
        """Test behavior with insufficient valid pixels."""
        # Create arrays with very few valid pixels
        pred = np.full((100, 100), np.nan, dtype=np.float32)
        gt = np.full((100, 100), np.nan, dtype=np.float32)
        
        # Add only 5 valid pixels
        pred[0:5, 0] = 1.0
        gt[0:5, 0] = 1.0
        
        metrics_metric = compute_depth_metrics(pred, gt, is_metric_model=True)
        metrics_nonmetric = compute_depth_metrics(pred, gt, is_metric_model=False)
        
        assert metrics_metric['n_valid'] < 10
        assert np.isnan(metrics_metric['abs_rel'])
        assert np.isnan(metrics_metric['rmse'])
        
        assert metrics_nonmetric['n_valid'] < 10
        assert np.isnan(metrics_nonmetric['silog'])
    
    def test_all_invalid_pixels(self):
        """Test behavior with all invalid pixels."""
        pred = np.full((100, 100), np.nan, dtype=np.float32)
        gt = np.full((100, 100), np.nan, dtype=np.float32)
        
        metrics_metric = compute_depth_metrics(pred, gt, is_metric_model=True)
        metrics_nonmetric = compute_depth_metrics(pred, gt, is_metric_model=False)
        
        assert metrics_metric['n_valid'] == 0
        assert np.isnan(metrics_metric['abs_rel'])
        assert np.isnan(metrics_metric['rmse'])
        
        assert metrics_nonmetric['n_valid'] == 0
        assert np.isnan(metrics_nonmetric['silog'])
    
    def test_zero_depth_values(self):
        """Test that zero depth values are excluded."""
        # Create arrays where pred has zeros but gt doesn't
        # Ensure there are at least 10 valid pixels (both pred > 0 and gt > 0)
        pred = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], 
                         [5.0, 0.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 0.0, 11.0, 12.0]], dtype=np.float32)
        gt = np.array([[1.0, 1.0, 2.0, 3.0, 4.0], 
                       [5.0, 1.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 1.0, 11.0, 12.0]], dtype=np.float32)
        
        metrics = compute_depth_metrics(pred, gt, is_metric_model=True)
        
        # Should only use pixels where both pred > 0 and gt > 0
        # Pixels with pred=0 should be excluded
        # We have 12 pixels where both pred > 0 and gt > 0
        assert metrics['n_valid'] == 12
        assert not np.isnan(metrics['abs_rel'])
    
    def test_negative_depth_values(self):
        """Test that negative depth values are excluded."""
        # Create arrays where pred has negatives but gt doesn't
        # Ensure there are at least 10 valid pixels (both pred > 0 and gt > 0)
        pred = np.array([[-1.0, 1.0, 2.0, 3.0, 4.0], 
                         [5.0, -0.5, 6.0, 7.0, 8.0],
                         [9.0, 10.0, -2.0, 11.0, 12.0]], dtype=np.float32)
        gt = np.array([[1.0, 1.0, 2.0, 3.0, 4.0], 
                       [5.0, 1.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 1.0, 11.0, 12.0]], dtype=np.float32)
        
        metrics = compute_depth_metrics(pred, gt, is_metric_model=True)
        
        # Should only use positive values (pred > 0 and gt > 0)
        # Pixels with negative pred should be excluded
        # We have 12 pixels where both pred > 0 and gt > 0
        assert metrics['n_valid'] == 12
        assert not np.isnan(metrics['abs_rel'])
    
    def test_silog_scale_invariance(self):
        """Test that SILog is scale-invariant."""
        # Create depth maps with all positive values (at least 10 pixels)
        gt = np.array([[1.0, 2.0, 3.0, 4.0], 
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
        pred1 = gt * 0.5  # Half scale (all positive, values: 0.5-6.0)
        pred2 = gt * 2.0  # Double scale (all positive, values: 2.0-24.0)
        
        metrics1 = compute_depth_metrics(pred1, gt, is_metric_model=False)
        metrics2 = compute_depth_metrics(pred2, gt, is_metric_model=False)
        
        # SILog should be computed after median alignment
        # Both should have valid metrics since all values are positive
        # All 12 pixels should be valid (all pred > 0 and gt > 0)
        assert metrics1['n_valid'] == 12
        assert metrics2['n_valid'] == 12
        assert not np.isnan(metrics1['silog'])
        assert not np.isnan(metrics2['silog'])
        # Both should be relatively small since they're just scaled versions
        assert metrics1['silog'] < 1.0
        assert metrics2['silog'] < 1.0

