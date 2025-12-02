#!/usr/bin/env python3
"""
Script to generate visualization images from existing numpy depth files.
This is useful if the visualization pass didn't run or failed.
"""

import argparse
from src import generate_visualizations_for_dataset


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Generate visualization images from existing numpy depth files'
    )
    parser.add_argument('--dataset-dir', type=str, default='results/cityscapes',
                        help='Dataset output directory (default: results/cityscapes)')
    parser.add_argument('--model-label', type=str, default='metric', choices=['metric', 'basic'],
                        help='Model label (default: metric)')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if images exist')
    
    args = parser.parse_args()
    
    generate_visualizations_for_dataset(args.dataset_dir, args.model_label, args.force)

