# Training Guide: DepthAnythingV2 with Camera Intrinsics

This guide explains how to train the DepthAnythingV2 model with camera intrinsics support using teacher-student knowledge distillation.

## Overview

The model has been extended to accept camera intrinsics as input. The architecture includes:
- **DINOv2 Backbone**: Frozen during training (can be unfrozen if needed)
- **Camera Encoder**: New MLP + transformer blocks that process intrinsics into tokens
- **DPT Decoder**: Depth prediction head (trainable)

## Teacher-Student Knowledge Distillation

The training process supports **knowledge distillation** similar to Depth-Anything-2 and Depth-Anything-3:

### Architecture
- **Teacher Model**: Original pretrained model (without camera intrinsics) - **FROZEN**
- **Student Model**: New model with camera intrinsics support - **TRAINABLE**

### Training Process
1. **Teacher Model**: Loads pretrained checkpoint, runs inference (no gradients)
2. **Student Model**: Trains with standard loss function
3. **Knowledge Distillation**: DepthAnythingV2 handles distillation internally when teacher model is provided

### Benefits
- ✅ Better generalization from teacher's knowledge
- ✅ Smoother training (teacher provides guidance)
- ✅ Improved performance on real-world data
- ✅ Faster convergence

## Data Format

### Directory Structure

Organize your data in the following structure:

```
datasets/raw_data/your_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── depths/
│   ├── depth_001.npy
│   ├── depth_002.npy
│   └── ...
├── intrinsics/          # Optional
│   ├── intrinsics_001.npy
│   ├── intrinsics_002.npy
│   └── ...
└── splits/
    ├── train.txt
    └── val.txt
```

### File Formats

1. **Images**: Standard image formats (`.jpg`, `.png`, etc.)
   - RGB images
   - Any resolution (will be resized during training)

2. **Depth Maps**: 
   - **Recommended**: `.npy` files (numpy arrays) with depth in meters
   - Also supported: `.png` (16-bit), `.h5` (h5py)
   - Depth values should be in meters (or use `depth_scale` parameter)

3. **Camera Intrinsics** (Optional):
   - `.npy` files containing 3x3 numpy arrays
   - Format:
     ```
     [[fx,  0, cx],
      [ 0, fy, cy],
      [ 0,  0,  1]]
     ```
   - Where:
     - `fx, fy`: Focal lengths in pixels
     - `cx, cy`: Principal point (image center) in pixels

### Creating File Lists

Create text files (`train.txt`, `val.txt`) with space-separated paths:

**Format 1: Without intrinsics**
```
/path/to/images/image_001.jpg /path/to/depths/depth_001.npy
/path/to/images/image_002.jpg /path/to/depths/depth_002.npy
```

**Format 2: With intrinsics**
```
/path/to/images/image_001.jpg /path/to/depths/depth_001.npy /path/to/intrinsics/intrinsics_001.npy
/path/to/images/image_002.jpg /path/to/depths/depth_002.npy /path/to/intrinsics/intrinsics_002.npy
```

**Example script to create file lists:**

```python
import os

def create_file_list(image_dir, depth_dir, intrinsics_dir=None, output_file='train.txt'):
    """Create a file list for training."""
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    with open(output_file, 'w') as f:
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            
            img_path = os.path.join(image_dir, img_name)
            depth_path = os.path.join(depth_dir, f"{base_name}.npy")
            
            line = f"{img_path} {depth_path}"
            
            if intrinsics_dir:
                intrinsics_path = os.path.join(intrinsics_dir, f"{base_name}.npy")
                if os.path.exists(intrinsics_path):
                    line += f" {intrinsics_path}"
            
            f.write(line + '\n')

# Usage
create_file_list(
    image_dir='datasets/raw_data/your_dataset/images',
    depth_dir='datasets/raw_data/your_dataset/depths',
    intrinsics_dir='datasets/raw_data/your_dataset/intrinsics',
    output_file='dataset/splits/your_dataset/train.txt'
)
```

## Training Process

### Step 1: Prepare Your Data

1. Organize images, depths, and intrinsics in the directory structure above
2. Create `train.txt` and `val.txt` file lists
3. Place file lists in `dataset/splits/your_dataset/`

### Step 2: Update Dataset Class (if needed)

If you're using a custom dataset, you can use the `GenericDatasetWithIntrinsics` class:

```python
from dataset.generic_with_intrinsics import GenericDatasetWithIntrinsics

trainset = GenericDatasetWithIntrinsics(
    filelist_path='dataset/splits/your_dataset/train.txt',
    mode='train',
    size=(518, 518),
    depth_scale=1.0  # Adjust if depth is not in meters
)
```

Or modify `train.py` to use your dataset.

### Step 3: Download Pretrained Checkpoint

Download a pretrained DepthAnythingV2 checkpoint:
- Basic model: `depth_anything_v2_vitl.pth`
- Metric model: `depth_anything_v2_metric_hypersim_vitl.pth`

Place it in the `checkpoints/` directory.

### Step 4: Start Training

**With Knowledge Distillation (Recommended):**
```bash
python train.py \
    --encoder vitl \
    --dataset your_dataset \
    --img-size 518 \
    --max-depth 20.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --teacher-checkpoint checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --save-path ./checkpoints/training_run \
    --use-camera-intrinsics \
    --use-distillation \
    --cam-token-inject-layer 0 \
    --freeze-dinov2
```

**Without Knowledge Distillation (Standard Training):**
```bash
python train.py \
    --encoder vitl \
    --dataset your_dataset \
    --img-size 518 \
    --max-depth 20.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --save-path ./checkpoints/training_run \
    --use-camera-intrinsics \
    --cam-token-inject-layer 0 \
    --freeze-dinov2
```

**Multi-GPU (Distributed):**
```bash
torchrun --nproc_per_node=4 train.py \
    --encoder vitl \
    --dataset your_dataset \
    --img-size 518 \
    --max-depth 20.0 \
    --epochs 40 \
    --bs 2 \
    --lr 0.000005 \
    --pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --save-path ./checkpoints/training_run \
    --use-camera-intrinsics \
    --cam-token-inject-layer 0 \
    --freeze-dinov2
```

### Training Arguments

**Basic Arguments:**
- `--encoder`: Encoder type (`vits`, `vitb`, `vitl`, `vitg`)
- `--dataset`: Dataset name (must match your dataset class)
- `--img-size`: Input image size (default: 518)
- `--max-depth`: Maximum depth in meters (20 for indoor, 80 for outdoor)
- `--epochs`: Number of training epochs
- `--bs`: Batch size per GPU
- `--lr`: Learning rate for DINOv2 (if not frozen) or base LR
- `--pretrained-from`: Path to pretrained checkpoint (for student model initialization)
- `--save-path`: Directory to save checkpoints
- `--use-camera-intrinsics`: Enable camera intrinsics support
- `--cam-token-inject-layer`: Layer to inject camera token (None = first layer, default: 0)
- `--freeze-dinov2`: Freeze DINOv2 backbone (default: True)

**Knowledge Distillation Arguments:**
- `--use-distillation`: Enable teacher-student knowledge distillation
- `--teacher-checkpoint`: Path to teacher model checkpoint (pretrained model without intrinsics)

## Training Details

### What Gets Trained?

**Student Model** (with `--freeze-dinov2`, default):
- ✅ **Camera Encoder**: Trained from scratch (random initialization)
- ✅ **DPT Decoder**: Trained (loaded from checkpoint)
- ❌ **DINOv2 Backbone**: Frozen (not updated)

**Teacher Model** (when using `--use-distillation`):
- ❌ **Completely Frozen**: Only used for inference, no gradients

### Knowledge Distillation

DepthAnythingV2 handles knowledge distillation internally. When `--use-distillation` is enabled:
- Teacher model generates predictions (frozen, no gradients)
- Student model trains with standard SiLogLoss
- The model architecture handles the distillation process automatically

### Learning Rates

- **DINOv2** (if not frozen): `lr` (e.g., 0.000005)
- **Depth Head + Camera Encoder**: `lr * 10.0` (e.g., 0.00005)

### Checkpoint Loading

**Student Model:**
1. Loads the full checkpoint (`--pretrained-from`) - pretrained + depth_head
2. Skips `cam_encoder` weights (randomly initialized)
3. Uses `strict=False` to handle missing keys gracefully

**Teacher Model** (when using `--use-distillation`):
1. Loads teacher checkpoint (`--teacher-checkpoint`)
2. Should be the original pretrained model (without camera intrinsics)
3. Completely frozen - no gradients computed
4. Runs inference only to provide soft targets for student

### Monitoring Training

Checkpoints are saved to `--save-path`:
- `latest.pth`: Latest checkpoint with model, optimizer, epoch, and best metrics

View training progress with TensorBoard:
```bash
tensorboard --logdir ./checkpoints/training_run
```

**TensorBoard Metrics**:
- `train/loss`: Training loss
- `eval/*`: Evaluation metrics (d1, d2, d3, abs_rel, etc.)

**Training Logs** show:
- Loss at each iteration
- Learning rate schedule
- Evaluation metrics per epoch

## Inference After Training

```python
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
import numpy as np
import torch

# Load model
model = DepthAnythingV2(
    encoder='vitl',
    max_depth=20.0,
    use_camera_intrinsics=True,
    cam_token_inject_layer=0
)
model.load_state_dict(torch.load('checkpoints/training_run/latest.pth')['model'])
model.eval()

# Load image and intrinsics
image = cv2.imread('path/to/image.jpg')
intrinsics = np.load('path/to/intrinsics.npy')  # 3x3 matrix

# Run inference
depth = model.infer_image(image, input_size=518, intrinsics=intrinsics)
```

## Tips

1. **Use Knowledge Distillation**: Recommended for better results - teacher provides valuable guidance
2. **Start with frozen DINOv2**: Faster training, less memory, good results
3. **Use intrinsics when available**: Even if some samples don't have intrinsics, the model can handle None
4. **Batch size**: Adjust based on GPU memory (teacher model adds ~2x memory overhead when using distillation)
5. **Learning rate**: Start with default (0.000005), adjust if loss doesn't decrease
6. **Data augmentation**: The dataset applies random horizontal flips during training

### Knowledge Distillation Best Practices

- **Teacher checkpoint**: Use the best pretrained model available (same encoder size as student)
- **Memory**: Teacher model doubles memory usage - reduce batch size if needed
- **Convergence**: Distillation often leads to faster convergence and better final performance

## Troubleshooting

**Issue**: "Missing keys (cam_encoder)"
- **Solution**: This is normal! Camera encoder is randomly initialized.

**Issue**: "CUDA out of memory"
- **Solution**: Reduce batch size (`--bs`) or image size (`--img-size`)

**Issue**: Loss not decreasing
- **Solution**: Check data format, verify intrinsics are correct, try unfreezing DINOv2

**Issue**: CUDA out of memory with distillation
- **Solution**: Reduce batch size (`--bs`), teacher model doubles memory usage

**Issue**: Intrinsics shape error
- **Solution**: Ensure intrinsics are 3x3 numpy arrays saved as `.npy` files

