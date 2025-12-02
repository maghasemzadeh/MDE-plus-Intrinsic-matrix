"""
Tests for scale factor calculation (for non-metric models).
"""

import pytest
import numpy as np
from models.base import BaseDepthModelWrapper


class MockNonMetricModel(BaseDepthModelWrapper):
    """Mock non-metric model for testing."""
    
    def load_model(self):
        pass
    
    def infer_image(self, image, input_size=518, **kwargs):
        # Return unitless depth
        return np.random.uniform(0.1, 1.0, (100, 100)).astype(np.float32)
    
    def is_metric(self):
        return False
    
    def get_model_name(self):
        return "MockNonMetric"


class TestScaleFactorCalculation:
    """Test scale factor calculation for non-metric models."""
    
    def test_auto_scale_factor_median(self):
        """Test automatic scale factor calculation using median."""
        model = MockNonMetricModel({})
        
        # Create test data
        pred_depth = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)  # Unitless
        gt_depth = np.array([[2.0, 4.0, 6.0]], dtype=np.float32)  # Meters
        
        scale = model.calculate_scale_factor(pred_depth, gt_depth, user_scale_factor=None)
        
        # Expected: median(gt) / median(pred) = 4.0 / 1.0 = 4.0
        expected_scale = np.median(gt_depth[gt_depth > 0]) / np.median(pred_depth[pred_depth > 0])
        
        assert abs(scale - expected_scale) < 1e-6
        assert scale > 0
    
    def test_user_provided_scale_factor(self):
        """Test using user-provided scale factor."""
        model = MockNonMetricModel({})
        
        pred_depth = np.array([[0.5, 1.0]], dtype=np.float32)
        gt_depth = np.array([[2.0, 4.0]], dtype=np.float32)
        user_scale = 5.0
        
        scale = model.calculate_scale_factor(pred_depth, gt_depth, user_scale_factor=user_scale)
        
        assert scale == user_scale
    
    def test_scale_factor_with_nan(self):
        """Test scale factor calculation with NaN values."""
        model = MockNonMetricModel({})
        
        pred_depth = np.array([[0.5, np.nan, 1.0, 0.0]], dtype=np.float32)
        gt_depth = np.array([[2.0, 4.0, np.nan, 6.0]], dtype=np.float32)
        
        scale = model.calculate_scale_factor(pred_depth, gt_depth, user_scale_factor=None)
        
        # Should only use valid (finite, positive) values
        assert not np.isnan(scale)
        assert scale > 0
    
    def test_scale_factor_all_invalid(self):
        """Test scale factor with all invalid values."""
        model = MockNonMetricModel({})
        
        pred_depth = np.array([[np.nan, 0.0]], dtype=np.float32)
        gt_depth = np.array([[np.nan, 0.0]], dtype=np.float32)
        
        scale = model.calculate_scale_factor(pred_depth, gt_depth, user_scale_factor=None)
        
        # Should return default scale of 1.0 when no valid values
        assert scale == 1.0
    
    def test_scale_factor_zero_prediction(self):
        """Test scale factor when prediction median is zero."""
        model = MockNonMetricModel({})
        
        pred_depth = np.array([[0.0, 0.0, 0.1]], dtype=np.float32)
        gt_depth = np.array([[2.0, 4.0, 6.0]], dtype=np.float32)
        
        scale = model.calculate_scale_factor(pred_depth, gt_depth, user_scale_factor=None)
        
        # Should handle zero median gracefully
        assert scale >= 0
        # If median_pred is 0, should return 1.0
        if np.median(pred_depth[pred_depth > 0]) == 0:
            assert scale == 1.0
    
    def test_scale_factor_applied_to_prediction(self):
        """Test that scale factor correctly scales predictions."""
        model = MockNonMetricModel({})
        
        pred_depth = np.array([[0.5, 1.0]], dtype=np.float32)
        gt_depth = np.array([[2.0, 4.0]], dtype=np.float32)
        
        scale = model.calculate_scale_factor(pred_depth, gt_depth, user_scale_factor=None)
        scaled_pred = pred_depth * scale
        
        # Scaled prediction should be closer to GT
        # Check that median of scaled pred is close to median of GT
        median_scaled = np.median(scaled_pred[scaled_pred > 0])
        median_gt = np.median(gt_depth[gt_depth > 0])
        
        assert abs(median_scaled - median_gt) < 0.1  # Should be close
    
    def test_scale_factor_large_values(self):
        """Test scale factor with large depth values."""
        model = MockNonMetricModel({})
        
        pred_depth = np.array([[0.1, 0.2]], dtype=np.float32)
        gt_depth = np.array([[50.0, 100.0]], dtype=np.float32)  # Large values
        
        scale = model.calculate_scale_factor(pred_depth, gt_depth, user_scale_factor=None)
        
        assert scale > 0
        assert not np.isnan(scale)
        assert not np.isinf(scale)

