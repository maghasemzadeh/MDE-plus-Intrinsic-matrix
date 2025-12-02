"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Create a sample RGB image (H, W, 3) as numpy array."""
    # Create a simple test image: 100x100 RGB
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return img


@pytest.fixture
def sample_depth_map():
    """Create a sample depth map in meters."""
    # Create depth map with values between 0.5 and 10.0 meters
    depth = np.random.uniform(0.5, 10.0, (100, 100)).astype(np.float32)
    # Add some invalid pixels (NaN)
    mask = np.random.random((100, 100)) < 0.1  # 10% invalid
    depth[mask] = np.nan
    return depth


@pytest.fixture
def sample_gt_depth():
    """Create a sample ground truth depth map."""
    # Similar to sample_depth_map but slightly different for testing
    depth = np.random.uniform(0.5, 10.0, (100, 100)).astype(np.float32)
    # Add some invalid pixels
    mask = np.random.random((100, 100)) < 0.1
    depth[mask] = np.nan
    return depth


@pytest.fixture
def perfect_depth_maps():
    """Create perfect matching depth maps for testing."""
    # Create identical depth maps
    depth = np.random.uniform(1.0, 5.0, (50, 50)).astype(np.float32)
    return depth.copy(), depth.copy()


@pytest.fixture
def error_depth_map():
    """Create a sample error map (|pred - gt|)."""
    pred = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
    gt = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
    error = np.abs(pred - gt)
    # Add some NaN where either pred or gt is invalid
    mask = np.random.random((100, 100)) < 0.1
    error[mask] = np.nan
    return error


@pytest.fixture
def sample_calibration_data():
    """Create sample Middlebury calibration data."""
    return {
        'f0': 1000.0,
        'f1': 1050.0,
        'cx0': 320.0,
        'cy0': 240.0,
        'cx1': 330.0,
        'cy1': 245.0,
        'doffs': 10.0,
        'baseline': 0.1  # 10cm baseline
    }


@pytest.fixture
def sample_disparity_map():
    """Create a sample disparity map (for Middlebury)."""
    # Disparity values typically between 0 and 200 pixels
    disparity = np.random.uniform(10, 200, (100, 100)).astype(np.float32)
    # Add some invalid pixels
    mask = np.random.random((100, 100)) < 0.1
    disparity[mask] = np.nan
    return disparity


@pytest.fixture
def mock_model_output():
    """Create mock model output (non-metric, unitless depth)."""
    # Unitless depth values
    depth = np.random.uniform(0.1, 1.0, (100, 100)).astype(np.float32)
    return depth


@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data for testing."""
    return {
        'abs_rel': 0.05,
        'rmse': 0.15,
        'silog': 0.08,
        'n_valid': 9000
    }


@pytest.fixture
def two_camera_error_arrays():
    """Create two error arrays for camera comparison testing."""
    # Create two error arrays with known difference
    n_pixels = 10000
    err0 = np.random.normal(0.1, 0.05, n_pixels).astype(np.float32)
    err1 = np.random.normal(0.12, 0.05, n_pixels).astype(np.float32)  # Slightly higher mean
    # Ensure all positive
    err0 = np.abs(err0)
    err1 = np.abs(err1)
    return err0, err1


@pytest.fixture
def sample_metric_values():
    """Create sample metric values for statistical testing."""
    # Create two sets of metric values (e.g., AbsRel) for different cameras
    n_scenes = 20
    cam0_vals = np.random.normal(0.05, 0.01, n_scenes)
    cam1_vals = np.random.normal(0.06, 0.01, n_scenes)  # Slightly higher
    return cam0_vals, cam1_vals

