"""
Tests for data loading functionality.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path

from datasets.base import DatasetItem, DatasetConfig
from datasets.middlebury import (
    MiddleburyDataset,
    read_pfm,
    depth_from_disparity,
    parse_calib
)


class TestPFMReading:
    """Test PFM file reading."""
    
    def test_read_pfm_basic(self, temp_dir):
        """Test reading a basic PFM file."""
        # Create a simple PFM file
        pfm_path = os.path.join(temp_dir, "test.pfm")
        
        # Create test data
        test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        height, width = test_data.shape
        
        # Write PFM file manually
        with open(pfm_path, 'wb') as f:
            f.write(b'PF\n')
            f.write(f'{width} {height}\n'.encode())
            f.write(b'1.0\n')
            # Write data in big-endian format
            flipped = np.flipud(test_data)
            flipped.astype('>f').tofile(f)
        
        # Read it back
        result = read_pfm(pfm_path)
        
        assert result.shape == (height, width)
        np.testing.assert_array_almost_equal(result, test_data, decimal=5)
    
    def test_read_pfm_negative_scale(self, temp_dir):
        """Test reading PFM with negative scale (little-endian)."""
        pfm_path = os.path.join(temp_dir, "test_neg.pfm")
        
        test_data = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
        height, width = test_data.shape
        
        with open(pfm_path, 'wb') as f:
            f.write(b'PF\n')
            f.write(f'{width} {height}\n'.encode())
            f.write(b'-1.0\n')  # Negative scale
            flipped = np.flipud(test_data)
            flipped.astype('<f').tofile(f)  # Little-endian
        
        result = read_pfm(pfm_path)
        
        assert result.shape == (height, width)
        np.testing.assert_array_almost_equal(result, test_data, decimal=5)


class TestDepthFromDisparity:
    """Test disparity to depth conversion."""
    
    def test_depth_from_disparity_basic(self):
        """Test basic disparity to depth conversion."""
        disparity = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        f = 1000.0
        baseline = 0.1  # 10cm
        doffs = 0.0
        
        depth = depth_from_disparity(disparity, f, baseline, doffs)
        
        # Expected: depth = (baseline * f) / (disparity + doffs)
        expected = (baseline * f) / (disparity + doffs)
        
        np.testing.assert_array_almost_equal(depth, expected, decimal=2)
    
    def test_depth_from_disparity_with_doffs(self):
        """Test disparity conversion with doffs offset."""
        disparity = np.array([[10.0, 20.0]], dtype=np.float32)
        f = 1000.0
        baseline = 0.1
        doffs = 5.0
        
        depth = depth_from_disparity(disparity, f, baseline, doffs)
        
        expected = (baseline * f) / (disparity + doffs)
        np.testing.assert_array_almost_equal(depth, expected, decimal=2)
    
    def test_depth_from_disparity_invalid_pixels(self):
        """Test that invalid disparity pixels become NaN."""
        disparity = np.array([[10.0, np.nan, np.inf]], dtype=np.float32)
        f = 1000.0
        baseline = 0.1
        doffs = 0.0
        
        depth = depth_from_disparity(disparity, f, baseline, doffs)
        
        # NaN input should result in NaN or 0.0 (both are invalid)
        assert np.isnan(depth[0, 1]) or depth[0, 1] == 0.0
        # For inf, the result might be 0.0 or NaN depending on implementation
        # Check that it's either NaN or 0.0 (both are invalid)
        assert np.isnan(depth[0, 2]) or depth[0, 2] == 0.0
        assert not np.isnan(depth[0, 0]) and depth[0, 0] > 0


class TestCalibrationParsing:
    """Test calibration file parsing."""
    
    def test_parse_calib_basic(self, temp_dir):
        """Test parsing a basic calibration file."""
        calib_path = os.path.join(temp_dir, "calib.txt")
        
        # Create a simple calibration file
        with open(calib_path, 'w') as f:
            f.write("cam0=[1000 0 320; 0 1000 240; 0 0 1]\n")
            f.write("cam1=[1050 0 330; 0 1050 245; 0 0 1]\n")
            f.write("doffs=10.0\n")
            f.write("baseline=0.1\n")
        
        f0, f1, cx0, cy0, cx1, cy1, doffs, baseline = parse_calib(calib_path)
        
        assert f0 == 1000.0
        assert f1 == 1050.0
        assert cx0 == 320.0
        assert cy0 == 240.0
        assert cx1 == 330.0
        assert cy1 == 245.0
        assert doffs == 10.0
        assert baseline == 0.1
    
    def test_parse_calib_multiline_matrix(self, temp_dir):
        """Test parsing calibration with multiline matrix."""
        calib_path = os.path.join(temp_dir, "calib.txt")
        
        with open(calib_path, 'w') as f:
            f.write("cam0=[1000 0 320;\n")
            f.write("0 1000 240;\n")
            f.write("0 0 1]\n")
            f.write("cam1=[1050 0 330; 0 1050 245; 0 0 1]\n")
            f.write("doffs=10.0\n")
            f.write("baseline=0.1\n")
        
        f0, f1, cx0, cy0, cx1, cy1, doffs, baseline = parse_calib(calib_path)
        
        assert f0 == 1000.0
        assert cx0 == 320.0
        assert cy0 == 240.0


class TestMiddleburyDataset:
    """Test Middlebury dataset loading."""
    
    def test_find_items_empty_directory(self, temp_dir):
        """Test finding items in empty directory."""
        config = DatasetConfig(dataset_path=temp_dir)
        dataset = MiddleburyDataset(config)
        
        items = dataset.find_items()
        
        assert len(items) == 0
    
    def test_find_items_with_scene(self, temp_dir):
        """Test finding items with a valid scene."""
        # Create a scene directory
        scene_dir = os.path.join(temp_dir, "test-scene")
        os.makedirs(scene_dir, exist_ok=True)
        
        # Create calibration file
        calib_path = os.path.join(scene_dir, "calib.txt")
        with open(calib_path, 'w') as f:
            f.write("cam0=[1000 0 320; 0 1000 240; 0 0 1]\n")
            f.write("cam1=[1050 0 330; 0 1050 245; 0 0 1]\n")
            f.write("doffs=10.0\n")
            f.write("baseline=0.1\n")
        
        # Create dummy image and disparity files
        import cv2
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(scene_dir, "im0.png"), dummy_img)
        cv2.imwrite(os.path.join(scene_dir, "im1.png"), dummy_img)
        
        # Create dummy PFM files (simplified)
        for disp_name in ["disp0.pfm", "disp1.pfm"]:
            disp_path = os.path.join(scene_dir, disp_name)
            with open(disp_path, 'wb') as f:
                f.write(b'PF\n')
                f.write(b'100 100\n')
                f.write(b'1.0\n')
                # Write minimal data
                data = np.zeros((100, 100), dtype='>f')
                data.tofile(f)
        
        config = DatasetConfig(dataset_path=temp_dir)
        dataset = MiddleburyDataset(config)
        
        items = dataset.find_items()
        
        # Should find 2 items (one for each camera)
        assert len(items) == 2
        assert all(item.item_id == "test-scene" for item in items)
        assert set(item.camera_id for item in items) == {'0', '1'}
    
    def test_load_gt_depth(self, temp_dir):
        """Test loading ground truth depth."""
        # Create a scene with disparity
        scene_dir = os.path.join(temp_dir, "test-scene")
        os.makedirs(scene_dir, exist_ok=True)
        
        # Create PFM disparity file
        disp_path = os.path.join(scene_dir, "disp0.pfm")
        test_disparity = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        height, width = test_disparity.shape
        
        with open(disp_path, 'wb') as f:
            f.write(b'PF\n')
            f.write(f'{width} {height}\n'.encode())
            f.write(b'1.0\n')
            flipped = np.flipud(test_disparity)
            flipped.astype('>f').tofile(f)
        
        # Create item with metadata
        item = DatasetItem(
            item_id="test-scene",
            image_path="dummy.png",
            gt_path=disp_path,
            camera_id='0',
            metadata={
                'f': 1000.0,
                'baseline': 0.1,
                'doffs': 0.0
            }
        )
        
        config = DatasetConfig(dataset_path=temp_dir)
        dataset = MiddleburyDataset(config)
        
        depth = dataset.load_gt_depth(disp_path, item)
        
        # Check that depth is in meters and has correct shape
        assert depth.shape == (height, width)
        assert depth.dtype == np.float32
        # Depth should be positive and reasonable (in meters)
        valid_depth = depth[np.isfinite(depth)]
        assert np.all(valid_depth > 0)
        assert np.all(valid_depth < 100)  # Reasonable upper bound

