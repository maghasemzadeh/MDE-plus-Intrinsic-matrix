
# compare cityscapes and drivingstereo:
```bash
python compare.py 
    --dataset cityscapes,drivingstereo 
    --encoder vitl 
    --model-type basic 
    --max-depth 120 
    --max-items 2000
```

# prepare vkitti2 dataset
 ```bash 
  python prepare_data_example.py \
    datasets/raw_data/vkitti \
    splits/train.txt \
    splits/intrinsics
```

# train on the vkitti2 dataset
```bash
python train.py \
    --encoder vitl \
    --dataset vkitti \
    --max-depth 80.0 \
    --epochs 40 \
    --bs 4 \
    --lr 0.000005 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_metric_hypersim_vits.pth  \
    --save-path checkpoints/vkitti_training \
    --use-camera-intrinsics \
    --freeze-dinov2
```