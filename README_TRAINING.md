# VKITTI Training - Quick Start Guide

## ✅ All Issues Fixed

The training script has been updated and tested. All errors have been resolved:

1. ✓ Fixed numpy RankWarning compatibility
2. ✓ Fixed distributed training setup for single-GPU mode
3. ✓ Fixed missing generic_with_intrinsics import
4. ✓ Fixed path resolution for datasets and checkpoints
5. ✓ Fixed model state_dict extraction for DDP

## Training Command

Run from the **project root**:

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

## Or Use the Shell Script

```bash
./train_vkitti.sh
```

## Test Setup First (Optional)

Before training, you can test that everything is set up correctly:

```bash
python test_train_setup.py
```

This will verify:
- All imports work
- Dataset paths are correct
- Model can be created

## Important Notes

1. **Single GPU Mode**: The script now works in single-GPU mode without requiring distributed training setup
2. **Multi-GPU**: For multi-GPU training, use `torchrun` (see TRAINING_COMMANDS.md)
3. **Dataset Path**: The script automatically finds VKITTI dataset in `datasets/raw_data/vkitti/splits/train.txt`
4. **Checkpoints**: Checkpoints are saved in `checkpoints/vkitti_training/`

## Monitor Training

```bash
tensorboard --logdir checkpoints/vkitti_training
```

## Troubleshooting

If you encounter any issues:

1. **Check dataset preparation**: Make sure you've run `prepare_data_example.py` first
2. **Check checkpoint path**: Verify the pretrained checkpoint exists
3. **Check CUDA**: Make sure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Run test script**: `python test_train_setup.py` to diagnose issues

