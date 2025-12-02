# Quick Start: Training with Camera Intrinsics

## 1. Prepare Your Data

### Directory Structure
```
datasets/raw_data/my_dataset/
├── images/          # Your RGB images (.jpg, .png)
├── depths/          # Depth maps in meters (.npy files)
└── intrinsics/      # Camera intrinsics 3x3 matrices (.npy files) - OPTIONAL
```

### Create File Lists

Run the preparation script:
```bash
python prepare_data_example.py
```

Or manually create `dataset/splits/my_dataset/train.txt`:
```
datasets/raw_data/my_dataset/images/img_001.jpg datasets/raw_data/my_dataset/depths/img_001.npy datasets/raw_data/my_dataset/intrinsics/img_001.npy
datasets/raw_data/my_dataset/images/img_002.jpg datasets/raw_data/my_dataset/depths/img_002.npy datasets/raw_data/my_dataset/intrinsics/img_002.npy
```

## 2. Download Pretrained Checkpoint

Download `depth_anything_v2_metric_hypersim_vitl.pth` and place in `checkpoints/` folder.

## 3. Start Training

**With Knowledge Distillation (Recommended):**
```bash
python train.py \
    --encoder vitl \
    --dataset my_dataset \
    --pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --teacher-checkpoint checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --save-path ./checkpoints/my_training_run \
    --use-camera-intrinsics \
    --use-distillation \
    --freeze-dinov2 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005
```

**Without Distillation:**
```bash
python train.py \
    --encoder vitl \
    --dataset my_dataset \
    --pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --save-path ./checkpoints/my_training_run \
    --use-camera-intrinsics \
    --freeze-dinov2 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005
```

## 4. Monitor Training

```bash
tensorboard --logdir ./checkpoints/my_training_run
```

## Key Points

- ✅ **Knowledge Distillation**: Teacher model (frozen) guides student training
- ✅ DINOv2 is **frozen** by default (faster training, less memory)
- ✅ Camera encoder is **randomly initialized** (learns from your data)
- ✅ Depth head is **loaded from checkpoint** (already trained)
- ✅ Intrinsics are **optional** (model works with or without them)
- ✅ Standard SiLogLoss used - DepthAnythingV2 handles distillation internally

For detailed information, see `TRAINING_GUIDE.md`.

