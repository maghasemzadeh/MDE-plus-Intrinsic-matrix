"""
Tests for error calculation and valid mask computation.
"""

import pytest
import numpy as np


class TestErrorCalculation:
    """Test error calculation functions."""
    
    def test_error_calculation_basic(self):
        """Test basic error calculation |pred - gt|."""
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        gt = np.array([[1.1, 2.2, 2.9], [4.0, 5.1, 6.2]], dtype=np.float32)
        
        error = np.abs(pred - gt)
        
        expected = np.array([[0.1, 0.2, 0.1], [0.0, 0.1, 0.2]], dtype=np.float32)
        np.testing.assert_array_almost_equal(error, expected, decimal=6)
    
    def test_error_calculation_with_nan(self):
        """Test error calculation with NaN values."""
        pred = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        gt = np.array([[1.1, 2.0, np.nan]], dtype=np.float32)
        
        error = np.full_like(gt, np.nan, dtype=np.float32)
        mask_valid = (
            np.isfinite(gt) & (gt > 0) &
            np.isfinite(pred) & (pred > 0)
        )
        error[mask_valid] = np.abs(gt[mask_valid] - pred[mask_valid])
        
        # Only first pixel should have valid error
        assert not np.isnan(error[0, 0])
        assert np.isnan(error[0, 1])  # pred is NaN
        assert np.isnan(error[0, 2])  # gt is NaN
    
    def test_error_calculation_zero_values(self):
        """Test that zero values are excluded from error calculation."""
        pred = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        gt = np.array([[1.0, 1.0, 2.0]], dtype=np.float32)
        
        error = np.full_like(gt, np.nan, dtype=np.float32)
        mask_valid = (
            np.isfinite(gt) & (gt > 0) &
            np.isfinite(pred) & (pred > 0)
        )
        error[mask_valid] = np.abs(gt[mask_valid] - pred[mask_valid])
        
        # First pixel should be NaN (pred is 0)
        assert np.isnan(error[0, 0])
        assert not np.isnan(error[0, 1])
        assert not np.isnan(error[0, 2])
    
    def test_error_calculation_perfect_match(self):
        """Test error calculation with perfect match."""
        pred = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        gt = pred.copy()
        
        error = np.abs(pred - gt)
        
        assert np.all(error == 0.0)
    
    def test_valid_mask_computation(self):
        """Test valid mask computation."""
        pred = np.array([[1.0, np.nan, 3.0, 0.0, 5.0]], dtype=np.float32)
        gt = np.array([[1.1, 2.0, np.nan, 4.0, 5.0]], dtype=np.float32)
        
        mask_valid = (
            np.isfinite(gt) & (gt > 0) &
            np.isfinite(pred) & (pred > 0)
        )
        
        # Expected: [True, False, False, False, True]
        expected = np.array([[True, False, False, False, True]])
        np.testing.assert_array_equal(mask_valid, expected)
    
    def test_error_statistics(self, error_depth_map):
        """Test error statistics computation."""
        error = error_depth_map
        
        # Get finite errors
        finite_errors = error[np.isfinite(error)]
        
        assert len(finite_errors) > 0
        assert np.all(finite_errors >= 0)  # Errors should be non-negative
        assert not np.any(np.isnan(finite_errors))
        
        # Compute statistics
        mean_error = np.mean(finite_errors)
        std_error = np.std(finite_errors, ddof=1)
        
        assert not np.isnan(mean_error)
        assert not np.isnan(std_error)
        assert mean_error >= 0
        assert std_error >= 0
    
    def test_error_calculation_large_values(self):
        """Test error calculation with large depth values."""
        pred = np.array([[100.0, 200.0]], dtype=np.float32)
        gt = np.array([[100.5, 199.5]], dtype=np.float32)
        
        error = np.abs(pred - gt)
        
        expected = np.array([[0.5, 0.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(error, expected, decimal=6)
    
    def test_error_calculation_negative_difference(self):
        """Test that error is always positive (absolute value)."""
        pred = np.array([[1.0, 2.0]], dtype=np.float32)
        gt = np.array([[2.0, 1.0]], dtype=np.float32)
        
        error = np.abs(pred - gt)
        
        # Both should be 1.0 (absolute difference)
        assert error[0, 0] == 1.0
        assert error[0, 1] == 1.0
        assert np.all(error >= 0)

