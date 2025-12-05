# prepare vkitti dataset
```bash
python prepare_vkitti.py --vkitti-root datasets/raw_data/vkitti
```

# compare cityscapes and drivingstereo:
```bash
python compare_dataset_results.py \
    --dataset cityscapes,drivingstereo \
    --encoder vitl \
    --model-type basic \
    --max-depth 120 \
    --max-items 2000
```


# prepare vkitti2 dataset on revised model
```bash
python prepare_vkitti.py \
    --vkitti-root datasets/raw_data/vkitti \
    --intrinsics-output-dir datasets/raw_data/vkitti/splits/intrinsics
```

# compare two models
 ```bash 
  python compare_models.py \
    --dataset CityScapes \
    --model1 da2 \
    --model2 da2-revised \
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
    --save-path models/raw_models/DepthAnythingV2-revised/checkpoints/revised \
    --use-camera-intrinsics \
    --freeze-dinov2
```