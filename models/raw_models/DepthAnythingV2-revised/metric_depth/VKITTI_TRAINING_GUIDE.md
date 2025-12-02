# VKITTI Dataset Training Guide

This guide explains how to prepare and train on the VKITTI dataset using Depth-Anything-V2-revised.

## Dataset Structure

The VKITTI dataset should be organized as follows:

```
datasets/raw_data/vkitti/
├── vkitti_2.0.3_rgb/
│   └── Scene01/
│       └── 15-deg-left/
│           └── frames/
│               └── rgb/
│                   └── Camera_0/
│                       └── rgb_00000.jpg
├── vkitti_2.0.3_depth/
│   └── Scene01/
│       └── 15-deg-left/
│           └── frames/
│               └── depth/
│                   └── Camera_0/
│                       └── depth_00000.png
└── vkitti_2.0.3_textgt/
    └── Scene01/
        └── 15-deg-left/
            └── intrinsic.txt
```

## Step 1: Prepare the Dataset

Run the preparation script to create the training file list:

```bash
cd models/raw_models/DepthAnythingV2-revised/metric_depth

python prepare_data_example.py datasets/raw_data/vkitti
```

Or with explicit paths:

```bash
python prepare_data_example.py \
    datasets/raw_data/vkitti \
    splits/train.txt \
    splits/intrinsics
```

This will:
- Scan all scenes, variants, and cameras in the VKITTI dataset
- Create a file list with image and depth paths
- Convert intrinsic.txt files to .npy format (if intrinsics are available)
- Save everything to `datasets/raw_data/vkitti/splits/train.txt` (inside the VKITTI root directory)

### Options:

- **Without intrinsics**: Add `false` as the last argument
  ```bash
  python prepare_data_example.py datasets/raw_data/vkitti splits/train.txt '' false
  ```

- **Custom output path**: Specify your own output file
  ```bash
  python prepare_data_example.py datasets/raw_data/vkitti my_custom_path/train.txt
  ```

## Step 2: Prepare Validation Set (Optional)

If you have a validation split, create `datasets/raw_data/vkitti/splits/val.txt` in the same format:
```
/path/to/image1.jpg /path/to/depth1.png [/path/to/intrinsics1.npy]
/path/to/image2.jpg /path/to/depth2.png [/path/to/intrinsics2.npy]
```

## Step 3: Train the Model

**Note:** The file list is now located at `datasets/raw_data/vkitti/splits/train.txt`. 
The train.py script will automatically find it in this location.

### Basic Training (without camera intrinsics):

```bash
python train.py \
    --encoder vitl \
    --dataset vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --save-path ./checkpoints/vkitti_training \
    --freeze-dinov2
```

**Note:** If train.py can't find the file list, create a symlink:
```bash
mkdir -p dataset/splits/vkitti2
ln -s ../../../../datasets/raw_data/vkitti/splits/train.txt dataset/splits/vkitti2/train.txt
ln -s ../../../../datasets/raw_data/vkitti/splits/intrinsics dataset/splits/vkitti2/intrinsics
```

### Training with Camera Intrinsics:

```bash
python train.py \
    --encoder vitl \
    --dataset vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --save-path ./checkpoints/vkitti_training_with_intrinsics \
    --use-camera-intrinsics \
    --freeze-dinov2
```

### Training with Knowledge Distillation (Recommended):

```bash
python train.py \
    --encoder vitl \
    --dataset vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --teacher-checkpoint checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --save-path ./checkpoints/vkitti_training_distillation \
    --use-camera-intrinsics \
    --use-distillation \
    --freeze-dinov2
```

## Important Notes

1. **Max Depth**: VKITTI depth maps are in centimeters and converted to meters. The max depth is 80 meters, so use `--max-depth 80.0`.

2. **Intrinsics**: The VKITTI2 dataset class now supports optional intrinsics. If intrinsics are included in the file list, they will be automatically loaded and used when `--use-camera-intrinsics` is enabled.

3. **File List Format**: Each line in the file list should be:
   ```
   /path/to/image.jpg /path/to/depth.png [/path/to/intrinsics.npy]
   ```
   The intrinsics path is optional.

4. **Validation**: If you don't have a validation set, the training script will use KITTI validation set by default. You can create your own validation file list if needed.

## Troubleshooting

- **"VKITTI RGB directory not found"**: Make sure the path to your VKITTI dataset is correct and contains `vkitti_2.0.3_rgb`, `vkitti_2.0.3_depth` subdirectories.

- **"Intrinsics must be 3x3 matrix"**: The intrinsic.txt files are automatically converted to .npy format. Make sure the conversion completed successfully.

- **Out of memory**: Reduce batch size with `--bs 2` or use a smaller encoder like `--encoder vitb`.

## File List Example

After running the preparation script, `datasets/raw_data/vkitti/splits/train.txt` will look like:

```
datasets/raw_data/vkitti/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg datasets/raw_data/vkitti/vkitti_2.0.3_depth/Scene01/15-deg-left/frames/depth/Camera_0/depth_00000.png datasets/raw_data/vkitti/splits/intrinsics/Scene01_15-deg-left_Camera_0.npy
datasets/raw_data/vkitti/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg datasets/raw_data/vkitti/vkitti_2.0.3_depth/Scene01/15-deg-left/frames/depth/Camera_0/depth_00001.png datasets/raw_data/vkitti/splits/intrinsics/Scene01_15-deg-left_Camera_0.npy
...
```

