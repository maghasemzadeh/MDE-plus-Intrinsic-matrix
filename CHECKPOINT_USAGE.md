# Loading and Using Trained Checkpoints

This guide explains how to load and use trained checkpoints for inference after training.

## Checkpoint Files

After training, you'll find checkpoint files in your `--save-path` directory:

- **`latest.pth`**: The most recent checkpoint from the last epoch
- **`best.pth`**: The checkpoint with the best validation metrics (automatically saved when metrics improve)

Both checkpoints contain:
- `model`: Model state dictionary
- `optimizer`: Optimizer state (for resuming training)
- `epoch`: Training epoch number
- `previous_best`: Dictionary with best metrics seen so far

## Loading Checkpoints

### Method 1: Using the `load_checkpoint.py` Script

The easiest way to load and use a checkpoint is with the provided script:

#### Single Image Inference

```bash
python load_checkpoint.py \
    --checkpoint checkpoints/vkitti_training/best.pth \
    --encoder vitl \
    --max-depth 20.0 \
    --image path/to/your/image.jpg \
    --output-dir ./inference_results \
    --use-camera-intrinsics \
    --cam-token-inject-layer 0
```

#### Batch Inference on Directory

```bash
python load_checkpoint.py \
    --checkpoint checkpoints/vkitti_training/best.pth \
    --encoder vitl \
    --image-dir path/to/images/ \
    --output-dir ./inference_results
```

#### With Camera Intrinsics

If your model was trained with camera intrinsics:

```bash
python load_checkpoint.py \
    --checkpoint checkpoints/vkitti_training/best.pth \
    --encoder vitl \
    --use-camera-intrinsics \
    --cam-token-inject-layer 0 \
    --intrinsics path/to/intrinsics.npy \
    --image path/to/image.jpg \
    --output-dir ./inference_results
```

### Method 2: Loading in Python Code

You can also load checkpoints directly in your Python code:

```python
import torch
import numpy as np
import cv2
from depth_anything_v2.dpt import DepthAnythingV2

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration (must match training config)
encoder = 'vitl'  # Must match training encoder
max_depth = 20.0  # Must match training max_depth
use_camera_intrinsics = True  # Must match training setting
cam_token_inject_layer = 0  # Must match training setting

# Create model
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
model_kwargs = {
    **model_configs[encoder],
    'use_camera_intrinsics': use_camera_intrinsics,
    'cam_token_inject_layer': cam_token_inject_layer,
    'max_depth': max_depth
}
model = DepthAnythingV2(**model_kwargs)

# Load checkpoint
checkpoint_path = 'checkpoints/vkitti_training/best.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract model state dict
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best metrics: {checkpoint.get('previous_best', {})}")
else:
    state_dict = checkpoint

# Load weights
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# Run inference
image = cv2.imread('path/to/image.jpg')
depth = model.infer_image(image, input_size=518, intrinsics=None)

# Save result
np.save('depth.npy', depth)
depth_colormap = cv2.applyColorMap(
    ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8),
    cv2.COLORMAP_JET
)
cv2.imwrite('depth_visualization.png', depth_colormap)
```

## Important Notes

1. **Model Configuration Must Match**: When loading a checkpoint, the model configuration (encoder, max_depth, use_camera_intrinsics, etc.) must match the configuration used during training.

2. **Checkpoint Format**: The script handles different checkpoint formats:
   - Full checkpoint with `'model'` key (from training)
   - Direct state dictionary (from pretrained models)

3. **Device**: The model will automatically use CUDA if available, otherwise MPS (Mac), or CPU.

4. **Camera Intrinsics**: If your model was trained with `--use-camera-intrinsics`, you should provide intrinsics during inference for best results. The intrinsics should be a 3x3 numpy array saved as `.npy` file.

## Finding the Best Checkpoint

The training script automatically saves `best.pth` when validation metrics improve. You can also check the metrics in the checkpoint:

```python
import torch

checkpoint = torch.load('checkpoints/vkitti_training/best.pth', map_location='cpu')
print("Best metrics:", checkpoint.get('previous_best', {}))
print("Epoch:", checkpoint.get('epoch', 'unknown'))
```

## Example: Complete Inference Pipeline

```python
import torch
import numpy as np
import cv2
from load_checkpoint import load_model_from_checkpoint, run_inference_on_image

# Load model
model = load_model_from_checkpoint(
    checkpoint_path='checkpoints/vkitti_training/best.pth',
    encoder='vitl',
    max_depth=20.0,
    use_camera_intrinsics=True,
    cam_token_inject_layer=0
)

# Run inference
depth = run_inference_on_image(
    model=model,
    image_path='test_image.jpg',
    output_path='depth_result.png',
    input_size=518,
    intrinsics=None  # Or provide intrinsics if available
)
```

## Troubleshooting

1. **KeyError or missing keys**: Make sure the model configuration matches training. Check encoder type, max_depth, and camera intrinsics settings.

2. **Shape mismatches**: Ensure the input image size matches what the model expects (default 518x518).

3. **Device errors**: If you get CUDA errors, try setting device to CPU explicitly or check your PyTorch installation.

4. **Intrinsics format**: Camera intrinsics should be a 3x3 numpy array. If loading from file, use `np.load('intrinsics.npy')`.

