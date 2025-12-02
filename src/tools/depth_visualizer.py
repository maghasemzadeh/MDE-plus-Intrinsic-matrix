"""
Simple tool to visualize depth maps from .npy files.
"""

import sys
import os
import numpy as np
import cv2

# Add parent directory to path to import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import depth_to_color


if __name__ == "__main__":
    depth_path = "results/NYUv2/00006_00071_indoors_000_000_depth_mask.npy"  # path to your .npy file
    save_path = "results/NYUv2/depth_colored.png"  # output path

    # Load depth map
    depth = np.load(depth_path)

    # Convert to color
    depth_color = depth_to_color(depth, max_depth=10.0)

    # Save as PNG
    cv2.imwrite(save_path, depth_color)

    print(f"Saved colorized depth map to {save_path}")
