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
  python compare_models.py \                                                                                                                                                                      130 ↵ ──(Fri,Dec05)─┘
    --dataset cityscapes \
    --model1 "metric:vitl:checkpoints/vkitti_training/checkpoints/vkitti_training/events.out.tfevents.1764855234.Ubuntu.388503.0" \
    --model2 "metric:vitl:models/raw_modles/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth" \
    --output-path results/comparison
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