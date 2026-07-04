# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a monocular depth estimation (MDE) research project investigating the impact of camera intrinsic parameters on depth estimation performance. It integrates DepthAnythingV2 and Depth-Anything-3 models, with extensions to support camera intrinsics input during training.

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

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

### Testing
```bash
# Run all tests
pytest -v

# Run tests excluding slow tests
pytest -v -m "not slow"

# Run specific test file
pytest tests/test_metrics.py -v

# Run tests with custom output
python tests/run_tests.py
```

## Architecture Overview

### Core Packages

**`src/`** - Core pipeline and utilities
- `pipeline.py`: `ProcessingPipeline` class orchestrating dataset iteration, model inference, and metric computation
- `metrics.py`: Depth error metrics (AbsRel, RMSE for metric models; SILog for relative depth)
- `visualization.py`: Depth map visualization (colormaps, scale overlays, error heatmaps)
- `paths.py`: Output directory structure management
- `tools/`: Miscellaneous utility modules

**`datasets/`** - Dataset interfaces and implementations
- `base.py`: `BaseDataset` abstract class and `DatasetItem` dataclass
- `cityscapes.py`, `drivingstereo.py`, `middlebury.py`, `vkitti.py`, `diode.py`, `nyu.py`, `kitti.py`: Evaluation datasets with ground truth
- `training_datasets.py`: PyTorch Dataset wrappers for training (VKITTI2, KITTI, NYU, DIODE, Middlebury, Cityscapes, RobustDataset)
- Available training datasets in `train.py`: vkitti, kitti, nyu, diode, middlebury, cityscapes, robust

**`models/`** - Model wrappers and integrations
- `base.py`: `BaseDepthModelWrapper` abstract interface
- `depth_anything_v2.py`: Wrapper for baseline DepthAnythingV2
- `depth_anything_v2_revised.py`: Extended DepthAnythingV2 with camera intrinsics encoder
- `depth_anything_3.py`: Wrapper for Depth-Anything-3
- `raw_models/`: Unmodified vendor model code
  - `DepthAnythingV2/`: Original DepthAnythingV2 implementation
  - `DepthAnythingV2-revised/`: Modified DepthAnythingV2 with intrinsics support and metric training branch
  - `Depth-Anything-3/`: Depth-Anything-3 implementation
- `utils.py`: Model initialization and checkpoint handling utilities

**`tests/`** - Test suite
- `conftest.py`: Pytest fixtures and configuration
- `test_metrics.py`: Metric computation correctness
- `test_integration.py`: End-to-end pipeline tests
- `test_data_loading.py`: Dataset loading and preprocessing
- `test_scale_factor.py`, `test_statistical_tests.py`: Evaluation-specific tests

### Paper-Protocol Evaluation (verified against DA2 paper)

`evaluate_paper.py` reproduces the DA2 paper's zero-shot relative-depth protocol
(Table 2) with the released *basic* checkpoints. Verified results (2026-07, lingotube):

| run | ours (AbsRel/δ1) | paper |
|---|---|---|
| NYU vitl | 0.043 / 0.979 | 0.045 / 0.979 |
| NYU vits | 0.051 / 0.973 | 0.053 / 0.973 |
| KITTI vitl (no crop) | 0.076 / 0.948 | 0.074 / 0.946 |
| KITTI vits (no crop) | 0.083 / 0.934 | 0.078 / 0.936 |

(KITTI with the garg crop instead: vitl 0.070/0.956, vits 0.077/0.944 — the two
crop variants bracket the paper numbers; remaining differences are within normal
reproduction tolerance for this benchmark.)

Protocol essentials (all implemented in `evaluate_paper.py` + `src/metrics.py`):
- NYU: eigen 654 split (canonical indices from splits.mat), **rawDepths** GT
  (not the inpainted `depths` field), eigen crop, caps 0.001–10 m.
- KITTI: DA2's 652-image eigen-val filelist (`datasets/raw_data/kitti/da2_val.txt`
  on the server; images under `kitti_raw/`), **no crop**, caps 0.001–80 m.
- Basic model: per-image scale+shift alignment in inverse-depth space, then
  predictions clipped to the eval depth range (without clipping, near-zero
  aligned disparities explode and wreck the mean).
- Paper Table 4 (metric NYU/KITTI) uses in-domain fine-tuned checkpoints that
  were never released — it is NOT reproducible with the public
  hypersim/vkitti metric checkpoints.

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

### Output Structure
```
output_dir/{dataset}/{item_id}/
├── metric/                    # Metric model outputs (absolute depth)
│   ├── cam0/
│   │   ├── numpy_matrix/      # arrays.npz (depth + metrics), metrics.json
│   │   ├── disp/              # PNG depth visualizations
│   │   └── error/             # Error map visualizations
│   └── compare/               # Multi-camera stereo comparison
│       ├── warped.png         # Reprojection results
│       └── error_disp.png     # Cross-camera error maps
└── basic/                     # Non-metric model outputs (relative depth)
    ├── cam0/
    │   ├── numpy_matrix/
    │   └── disp/
    └── compare/
```

### Model Checkpoints

- Baseline DepthAnythingV2: `models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_*.pth` (pre-trained, available for download)
- DepthAnythingV2-revised: `models/raw_models/DepthAnythingV2-revised/checkpoints/` (includes metric and basic branches)
- Trained models: Saved to `--save-path` during training as `latest.pth` (most recent epoch) and `best.pth` (best validation metrics)

The training script auto-detects encoder type (`vits`/`vitb`/`vitl`/`vitg`) from checkpoint tensor shapes (see `ARCHITECTURE_SIZES.md`).

**Checkpoint auto-selection**: For metric models with both `hypersim` and `vkitti` checkpoints available, `--max-depth` controls which is preferred. `max_depth > 20` (outdoor) → vkitti checkpoint; `max_depth <= 20` (indoor) → hypersim checkpoint. Omitting `--max-depth` when comparing on outdoor datasets (KITTI, Cityscapes, DrivingStereo) risks silently loading the indoor hypersim model.

**Model type detection**: Metric vs basic cannot be determined from state-dict keys alone — the filename/dirname is the authoritative source for official DA2 checkpoints. Trained checkpoints use an embedded `config` block.

### Testing Results

Run `python tests/run_tests.py` or `pytest tests/` to execute the full test suite. Key test categories:
- **Unit tests**: Metric computation, scale factor calculation, statistical tests
- **Integration tests**: Full pipeline on small sample datasets
- **Data loading tests**: Dataset iteration, intrinsics loading, multi-camera grouping

## Dependencies

Core: PyTorch (`torch`, `torchvision`), NumPy, SciPy, OpenCV, matplotlib, tqdm, tensorboard, pytest

Dataset-specific: h5py (Hypersim), gradio (for interactive demos)

Install all: `pip install -r requirements.txt`

## Checkpoint and Model Notes

See `CHECKPOINT_USAGE.md` for detailed checkpoint loading and inference workflows.
See `ARCHITECTURE_SIZES.md` for encoder dimension mappings for all ViT sizes.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
