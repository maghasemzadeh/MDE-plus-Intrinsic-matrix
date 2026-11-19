# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a monocular depth estimation (MDE) research project investigating the impact of camera intrinsic parameters on depth estimation performance. It integrates DepthAnythingV2 and Depth-Anything-3 models, with extensions to support camera intrinsics input during training.

## Common Commands

### Training
```bash
# Train with camera intrinsics on VKITTI dataset
python train.py \
    --encoder vitl \
    --datasets vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
    --save-path models/raw_models/DepthAnythingV2-revised/checkpoints/revised \
    --use-camera-intrinsics \
    --freeze-dinov2 \
    --head-lr-multiplier 10.0 \
    --grad-clip 1.0 \
    --accumulate-grad 4

# Train baseline without intrinsics
python train.py \
    --encoder vitl \
    --datasets vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --pretrained-from models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth \
    --save-path models/raw_models/DepthAnythingV2/checkpoints/baseline \
    --freeze-dinov2
```

Key training arguments:
- `--encoder`: Model size (`vits`, `vitb`, `vitl`, `vitg`)
- `--datasets`: Training dataset(s) (`vkitti`, `kitti`, `nyu`, `hypersim`, etc.)
- `--use-camera-intrinsics`: Enable camera intrinsics as model input
- `--pretrained-from`: Path to pretrained checkpoint to fine-tune from
- `--save-path`: Directory for saving model checkpoints
- `--freeze-dinov2` (default): Keep DINOv2 backbone frozen; remove flag to fine-tune it
- `--use-distillation`: Enable teacher-student knowledge distillation
- `--max-depth`: Maximum depth value for the dataset

### Model Comparison & Evaluation
```bash
# Compare two models on a dataset
# --max-depth controls which metric checkpoint is loaded:
#   >20 → vkitti (outdoor, max_depth=80); <=20 → hypersim (indoor, max_depth=20)
python compare_models.py \
    --dataset cityscapes \
    --model1 da2 \
    --model2 da2-revised \
    --encoder vitl \
    --max-depth 80.0 \
    --max-items 2000

# Compare results across multiple datasets
python compare_dataset_results.py \
    --dataset cityscapes,drivingstereo \
    --model-name da2-revised \
    --encoder vitl \
    --model-type metric \
    --max-depth 120 \
    --max-items 2000

# Load and run inference with a trained checkpoint
python load_checkpoint.py \
    --checkpoint checkpoints/vkitti_training/best.pth \
    --encoder vitl \
    --image path/to/image.jpg \
    --output-dir ./inference_results \
    --use-camera-intrinsics
```

### Dataset Preparation
```bash
# Prepare VKITTI2 dataset (extracts intrinsics to .npy files)
python prepare_vkitti.py \
    --vkitti-root datasets/raw_data/vkitti \
    --intrinsics-output-dir datasets/raw_data/vkitti/splits/intrinsics
```

## Architecture Overview

### Paper-Protocol Evaluation
`evaluate_paper.py` reproduces the DA2 paper's zero-shot protocol (Table 2); verified numbers, protocol essentials, and the Table 4 caveat live in the `paper-eval-protocol` skill.

### Dataset-Specific Evaluation Notes

- **NYU**: Defaults to the Eigen 654-image test split (`split='test'` or `split='eigen_test'`). Use `split='all'` for all 1449 images — results on the full set are **not comparable** to DA2 paper numbers.
- **KITTI**: Supports a DA2-paper-style filelist split (`split='filelist:/path/to/val.txt'`) where each line is `image_path depth_path`. The official DA2 val.txt is at `models/raw_models/DepthAnythingV2/metric_depth/dataset/splits/kitti/val.txt` but requires paths updated to your local KITTI location.

### Key Design Patterns

1. **Dataset abstraction**: All datasets inherit from `BaseDataset` and implement:
   - `find_items()`: Locate images and ground truth pairs
   - `load_gt_depth()`: Load reference depth map for an item
   - `get_output_subdir()`: Directory for this dataset's outputs
   - `get_camera_intrinsics()`: Return camera intrinsics if available (optional, for intrinsics-aware models)

2. **Multi-camera stereo support**: Datasets can return multiple `DatasetItem` objects with the same `item_id` but different `camera_id` values. The pipeline groups and processes these together for stereo comparison (used by Middlebury).

3. **Metric vs non-metric models**: The pipeline handles both:
   - **Metric models** (DepthAnythingV2-metric branch): Output absolute depth in meters
   - **Non-metric/relative models**: Output relative depth; pipeline applies median-based scale alignment to depth before computing metrics

4. **Camera intrinsics flow**: When `--use-camera-intrinsics` is enabled:
   - Intrinsics loaded from `.npy` files referenced in dataset split files
   - Passed to the model's forward method via optional `camera_intrinsics` parameter
   - DepthAnythingV2-revised has a camera encoder branch that processes intrinsics

5. **Training vs evaluation paths**: 
   - Training uses `training_datasets.py` PyTorch DataLoader wrappers
   - Evaluation uses evaluation dataset classes that provide ground truth for metric computation

### Model Checkpoints

- Baseline DepthAnythingV2: `models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_*.pth` (pre-trained, available for download)
- DepthAnythingV2-revised: `models/raw_models/DepthAnythingV2-revised/checkpoints/` (includes metric and basic branches)
- Trained models: Saved to `--save-path` during training as `latest.pth` (most recent epoch) and `best.pth` (best validation metrics)

The training script auto-detects encoder type (`vits`/`vitb`/`vitl`/`vitg`) from checkpoint tensor shapes (see `ARCHITECTURE_SIZES.md`).

**Checkpoint auto-selection**: For metric models with both `hypersim` and `vkitti` checkpoints available, `--max-depth` controls which is preferred. `max_depth > 20` (outdoor) → vkitti checkpoint; `max_depth <= 20` (indoor) → hypersim checkpoint. Omitting `--max-depth` when comparing on outdoor datasets (KITTI, Cityscapes, DrivingStereo) risks silently loading the indoor hypersim model.

**Model type detection**: Metric vs basic cannot be determined from state-dict keys alone — the filename/dirname is the authoritative source for official DA2 checkpoints. Trained checkpoints use an embedded `config` block.

## Checkpoint and Model Notes

See `CHECKPOINT_USAGE.md` for detailed checkpoint loading and inference workflows.
See `ARCHITECTURE_SIZES.md` for encoder dimension mappings for all ViT sizes.

