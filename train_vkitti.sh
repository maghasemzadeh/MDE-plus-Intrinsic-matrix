#!/bin/bash
# Training script for VKITTI dataset with Depth-Anything-V2-revised

# Activate virtual environment if needed
# source venv/bin/activate

# Set CUDA devices (adjust as needed)
export CUDA_VISIBLE_DEVICES=0

# Training command for VKITTI with camera intrinsics
# Note: Checkpoints are in models/raw_models/DepthAnythingV2-revised/checkpoints/
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

# Alternative: Training with knowledge distillation (recommended)
# python train.py \
#     --encoder vitl \
#     --dataset vkitti \
#     --max-depth 80.0 \
#     --epochs 40 \
#     --bs 4 \
#     --lr 0.000005 \
#     --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
#     --teacher-checkpoint models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
#     --save-path checkpoints/vkitti_training_distillation \
#     --use-camera-intrinsics \
#     --use-distillation \
#     --freeze-dinov2
