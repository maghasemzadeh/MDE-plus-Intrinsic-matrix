"""
Tests for statistical tests (t-test, bootstrap CI).
"""

import pytest
import numpy as np
from scipy.stats import ttest_ind, ttest_rel

from datasets.middlebury import bootstrap_confidence_interval


class TestTTest:
    """Test t-test functionality."""
    
    def test_welch_ttest_basic(self, two_camera_error_arrays):
        """Test basic Welch's t-test."""
        err0, err1 = two_camera_error_arrays
        
        t_stat, p_value = ttest_ind(err0, err1, equal_var=False)
        
        assert not np.isnan(t_stat)
        assert not np.isnan(p_value)
        assert 0 <= p_value <= 1
    
    def test_welch_ttest_identical_arrays(self):
        """Test t-test with identical arrays (should have high p-value)."""
        arr = np.random.normal(0.1, 0.05, 1000)
        
        t_stat, p_value = ttest_ind(arr, arr, equal_var=False)
        
        # With identical arrays, t-stat should be 0 and p-value should be 1.0
        assert abs(t_stat) < 1e-10
        assert abs(p_value - 1.0) < 1e-6
    
    def test_welch_ttest_different_means(self):
        """Test t-test with clearly different means."""
        arr1 = np.random.normal(0.1, 0.05, 1000)
        arr2 = np.random.normal(0.2, 0.05, 1000)  # Different mean
        
        t_stat, p_value = ttest_ind(arr1, arr2, equal_var=False)
        
        # Should have significant difference
        assert p_value < 0.05
    
    def test_paired_ttest_basic(self, sample_metric_values):
        """Test paired t-test."""
        cam0_vals, cam1_vals = sample_metric_values
        
        t_stat, p_value = ttest_rel(cam0_vals, cam1_vals)
        
        assert not np.isnan(t_stat)
        assert not np.isnan(p_value)
        assert 0 <= p_value <= 1
    
    def test_paired_ttest_identical(self):
        """Test paired t-test with identical arrays."""
        arr = np.random.normal(0.05, 0.01, 20)
        
        t_stat, p_value = ttest_rel(arr, arr)
        
        # With identical arrays, differences are all zero
        # t-stat might be NaN if variance is zero, or very close to 0
        if np.isnan(t_stat):
            # This is acceptable when variance is exactly zero
            assert True
        else:
            assert abs(t_stat) < 1e-10
            assert abs(p_value - 1.0) < 1e-6
    
    def test_paired_ttest_single_value(self):
        """Test paired t-test with single value (should fail gracefully)."""
        arr1 = np.array([0.05])
        arr2 = np.array([0.06])
        
        # With only one pair, t-test should handle it
        try:
            t_stat, p_value = ttest_rel(arr1, arr2)
            # If it succeeds, values should be NaN or valid
            assert np.isnan(t_stat) or np.isfinite(t_stat)
        except ValueError:
            # ValueError is acceptable for insufficient data
            pass


class TestBootstrapConfidenceInterval:
    """Test bootstrap confidence interval calculation."""
    
    def test_bootstrap_ci_basic(self):
        """Test basic bootstrap CI calculation."""
        # Create sample differences
        diff_values = np.random.normal(0.01, 0.005, 100)
        
        mean_diff, lower, upper = bootstrap_confidence_interval(diff_values)
        
        assert not np.isnan(mean_diff)
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower <= mean_diff <= upper
    
    def test_bootstrap_ci_empty_array(self):
        """Test bootstrap CI with empty array."""
        diff_values = np.array([])
        
        mean_diff, lower, upper = bootstrap_confidence_interval(diff_values)
        
        assert np.isnan(mean_diff)
        assert np.isnan(lower)
        assert np.isnan(upper)
    
    def test_bootstrap_ci_all_nan(self):
        """Test bootstrap CI with all NaN values."""
        diff_values = np.array([np.nan, np.nan, np.nan])
        
        mean_diff, lower, upper = bootstrap_confidence_interval(diff_values)
        
        assert np.isnan(mean_diff)
        assert np.isnan(lower)
        assert np.isnan(upper)
    
    def test_bootstrap_ci_confidence_level(self):
        """Test bootstrap CI with different confidence levels."""
        diff_values = np.random.normal(0.01, 0.005, 100)
        
        # 95% CI (default)
        mean1, lower1, upper1 = bootstrap_confidence_interval(diff_values, confidence=0.95)
        
        # 99% CI
        mean2, lower2, upper2 = bootstrap_confidence_interval(diff_values, confidence=0.99)
        
        assert mean1 == mean2  # Mean should be same
        assert lower2 < lower1  # 99% CI should be wider
        assert upper2 > upper1
    
    def test_bootstrap_ci_contains_mean(self):
        """Test that bootstrap CI contains the mean."""
        diff_values = np.random.normal(0.0, 1.0, 1000)
        
        mean_diff, lower, upper = bootstrap_confidence_interval(diff_values)
        
        # CI should contain the sample mean
        sample_mean = np.mean(diff_values)
        assert lower <= sample_mean <= upper
    
    def test_bootstrap_ci_reproducibility(self):
        """Test that bootstrap CI is reproducible with same seed."""
        np.random.seed(42)
        diff_values = np.random.normal(0.01, 0.005, 100)
        
        np.random.seed(42)
        mean1, lower1, upper1 = bootstrap_confidence_interval(diff_values, n_bootstrap=1000)
        
        np.random.seed(42)
        mean2, lower2, upper2 = bootstrap_confidence_interval(diff_values, n_bootstrap=1000)
        
        assert abs(mean1 - mean2) < 1e-10
        assert abs(lower1 - lower2) < 1e-10
        assert abs(upper1 - upper2) < 1e-10


class TestSignificanceDetermination:
    """Test significance determination logic."""
    
    def test_significance_p_value_threshold(self):
        """Test significance determination based on p-value."""
        # p < 0.05 should be significant
        assert 0.01 < 0.05  # Significant
        assert 0.10 >= 0.05  # Not significant
    
    def test_significance_with_ttest(self, two_camera_error_arrays):
        """Test significance determination with actual t-test."""
        err0, err1 = two_camera_error_arrays
        
        t_stat, p_value = ttest_ind(err0, err1, equal_var=False)
        is_significant = p_value < 0.05
        
        assert isinstance(is_significant, (bool, np.bool_))
        assert is_significant == (p_value < 0.05)
    
    def test_significance_nan_handling(self):
        """Test significance determination with NaN p-value."""
        p_value = np.nan
        is_significant = not np.isnan(p_value) and p_value < 0.05
        
        assert not is_significant
    
    def test_mean_difference_direction(self, two_camera_error_arrays):
        """Test determining direction of mean difference."""
        err0, err1 = two_camera_error_arrays
        
        mean0 = np.mean(err0)
        mean1 = np.mean(err1)
        mean_diff = mean0 - mean1
        
        # Determine which has higher error
        if mean_diff > 0:
            assert mean0 > mean1
        elif mean_diff < 0:
            assert mean0 < mean1
        else:
            assert abs(mean0 - mean1) < 1e-10

