# Training Commands for VKITTI Dataset

## Quick Start

Run the training script from the project root:

```bash
./train_vkitti.sh
```

Or run directly with Python:

## Basic Training (with Camera Intrinsics)

```bash
python train.py \
    --encoder vitl \
    --dataset vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
    --save-path checkpoints/vkitti_training \
    --use-camera-intrinsics \
    --freeze-dinov2
```

## Training with Knowledge Distillation (Recommended)

```bash
python train.py \
    --encoder vitl \
    --dataset vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
    --teacher-checkpoint models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
    --save-path checkpoints/vkitti_training_distillation \
    --use-camera-intrinsics \
    --use-distillation \
    --freeze-dinov2
```

## Training without Camera Intrinsics

```bash
python train.py \
    --encoder vitl \
    --dataset vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
    --save-path checkpoints/vkitti_training_basic \
    --freeze-dinov2
```

## Multi-GPU Training

For distributed training on multiple GPUs:

```bash
torchrun --nproc_per_node=4 train.py \
    --encoder vitl \
    --dataset vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
    --save-path checkpoints/vkitti_training \
    --use-camera-intrinsics \
    --freeze-dinov2
```

## Parameters Explanation

- `--encoder vitl`: Use ViT-Large encoder (options: vits, vitb, vitl, vitg)
- `--dataset vkitti`: Use VKITTI dataset
- `--max-depth 80.0`: Maximum depth in meters (VKITTI max is 80m)
- `--epochs 40`: Number of training epochs
- `--bs 4`: Batch size per GPU
- `--lr 0.000005`: Learning rate
- `--pretrained-from`: Path to pretrained checkpoint
- `--save-path`: Directory to save training checkpoints
- `--use-camera-intrinsics`: Enable camera intrinsics support
- `--freeze-dinov2`: Freeze DINOv2 backbone (recommended)
- `--use-distillation`: Enable knowledge distillation (requires --teacher-checkpoint)
- `--teacher-checkpoint`: Path to teacher model for distillation

## Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir checkpoints/vkitti_training
```

## Checkpoints

Checkpoints are saved in the `--save-path` directory:
- `latest.pth`: Latest checkpoint after each epoch
- TensorBoard logs: Training and validation metrics

