# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a monocular depth estimation (MDE) research project investigating the impact of camera intrinsic parameters on depth estimation performance. It integrates DepthAnythingV2 and Depth-Anything-3 models, with extensions to support camera intrinsics input during training.

## Common Commands

### Training
```bash
# Train with camera intrinsics support on VKITTI dataset
python train.py --dataset vkitti --encoder vitl --use-camera-intrinsics \
    --pretrained-from depth_anything_v2_vitl.pth --save-path checkpoints/run1

# Train without intrinsics (baseline)
python train.py --dataset hypersim --encoder vitl \
    --pretrained-from depth_anything_v2_vitl.pth --save-path checkpoints/baseline
```

Key training arguments:
- `--encoder`: Model size (`vits`, `vitb`, `vitl`, `vitg`)
- `--use-camera-intrinsics`: Enable camera intrinsics input
- `--freeze-dinov2` (default): Freeze DINOv2 backbone during training
- `--use-distillation`: Enable teacher-student knowledge distillation

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_metrics.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov=datasets --cov=models
```

### Evaluation
The evaluation pipeline uses `ProcessingPipeline` from `src/pipeline.py` to process datasets and compute metrics. See `datasets/middlebury.py` for statistical comparison functions including paired t-tests and bootstrap confidence intervals.

## Architecture Overview

### Core Packages

**`src/`** - Core processing and utilities
- `pipeline.py`: Main `ProcessingPipeline` class orchestrating dataset processing, inference, and metric computation
- `metrics.py`: Depth metrics (AbsRel, RMSE for metric models; SILog for non-metric)
- `visualization.py`: Depth visualization utilities (colormaps, scale overlays)
- `paths.py`: Output path structure management

**`datasets/`** - Dataset implementations
- `base.py`: Abstract `BaseDataset` and `DatasetItem` dataclasses
- `middlebury.py`: Middlebury stereo dataset with dual-camera support and statistical evaluation
- `cityscapes.py`, `drivingstereo.py`: Other evaluation datasets
- `training_datasets.py`: PyTorch Dataset classes for training (VKITTI2, KITTI)

**`models/`** - Model wrappers
- `base.py`: Abstract `BaseDepthModelWrapper` interface
- `raw_models/DepthAnythingV2-revised/`: Modified DepthAnythingV2 with camera intrinsics support
- `raw_models/Depth-Anything-3/`: Depth-Anything-3 model

### Key Design Patterns

1. **Dataset abstraction**: All datasets inherit from `BaseDataset` and implement `find_items()`, `load_gt_depth()`, and `get_output_subdir()`

2. **Multi-camera support**: Datasets can return multiple `DatasetItem` objects with the same `item_id` but different `camera_id` values. The pipeline groups these for stereo comparison.

3. **Metric vs Basic models**: The pipeline handles both metric models (direct meter output) and basic/relative models (requires scale alignment using median matching)

4. **Camera intrinsics flow**: When `--use-camera-intrinsics` is enabled, intrinsics are loaded from `.npy` files referenced in dataset split files and passed to the model's forward method

### Output Structure
```
output_dir/{dataset}/{item_id}/
├── metric/           # For metric model outputs
│   ├── cam0/         # Per-camera outputs
│   │   ├── numpy_matrix/  # arrays.npz, metrics.json
│   │   ├── disp/     # Depth visualizations
│   │   └── ...
│   └── compare/      # Multi-camera comparison (warped images, error disparity)
└── basic/            # For non-metric model outputs
```

### Model Checkpoints

Checkpoints are stored in `models/raw_models/DepthAnythingV2-revised/checkpoints/`. The training script auto-detects encoder type from checkpoint tensor shapes.

## Dependencies

Core: PyTorch, torchvision, numpy, scipy, opencv-python, h5py (Hypersim), tqdm, matplotlib, tensorboard, pytest

Install: `pip install -r requirements.txt`
