# Metric comparison against Depth Anything V2

## Objective

Produce a defensible metric comparison with four model categories: the published DA2 metric results, a paper-style DA2 reproduction trained by us, a matched no-camera control, and the proposed camera-aware model.

Paper metric values:

| Dataset | AbsRel | RMSE | delta1 |
|---|---:|---:|---:|
| NYU-D | 0.056 | 0.206 | 0.984 |
| KITTI | 0.045 | 1.861 | 0.983 |

These are metric-depth results and must remain separate from the paper's relative-depth Table 2 values (`0.045/0.979` for NYU-D and `0.074/0.946` for KITTI).

## Reproducibility rule

The original metric checkpoints are not released. Report **published paper results**, **our paper-style DA2 reproduction**, **our matched control**, and **our proposed model** as separate categories. The paper-style run is a reimplementation unless every training detail and checkpoint is independently confirmed.

## Server and data

Run all training and inference on `lingotube` using the GPU skill, with `source venv/bin/activate`, dedicated tmux windows, and logs containing `EXIT_CODE=`.

Confirm these paths exist:

```text
datasets/raw_data/nyu/nyu_depth_v2_labeled.mat
datasets/raw_data/nyu/splits.mat
datasets/raw_data/kitti/kitti_raw
datasets/raw_data/kitti/val
datasets/raw_data/kitti/da2_val.txt
datasets/raw_data/vkitti/vkitti_2.0.3_rgb
datasets/raw_data/vkitti/vkitti_2.0.3_depth
datasets/raw_data/vkitti/vkitti_2.0.3_textgt
```

NYU-D in the paper is NYU Depth V2. Use the canonical Eigen test indices from `splits.mat`. Use the official Depth Anything V2 652-image KITTI Eigen filelist after remapping only absolute paths.

## Staged training design

### Stage 1: paper-style DA2 metric reproduction

Recreate the model described in the paper using the released DA2 ViT-L backbone and the official metric-training code/configuration as closely as the repository permits:

- Hypersim metric training for indoor NYU-D evaluation.
- Virtual KITTI 2 metric training for outdoor KITTI evaluation.
- The paper's reported input size, metric head, loss, depth range, augmentation, optimizer, schedule, and validation procedure.

Save this as `paper_style_da2`. Compare it with the paper values, but label it **our reimplementation**. A close result validates the implementation and protocol; a mismatch must be investigated before attributing any gain to camera intrinsics.

### Stage 2: matched control and proposed models

Using the exact Stage-1 data manifests, preprocessing, seed, schedule, and checkpoint-selection rule, train:

1. **Control:** identical metric architecture with camera intrinsics disabled.
2. **Proposed:** identical metric architecture with the normalized 9D intrinsic descriptor and camera token enabled.

The only intended difference between these two arms is access to `K`. Use the same images, depth targets, geometric transforms, and random sequence. When a transform changes image geometry, update `K` exactly for the proposed arm and apply the same transformed image to the control arm.

Use the corrected metric branch. Checkpoint metadata must prove `model_type=metric`, `encoder=vitl`, `outputs_inverse_depth=false`, and the expected `uses_camera_intrinsics` value. The evaluation head must be the same head used during training. Never include test images in training. Any geometric crop or focal augmentation must update `K` exactly for the proposed arm and apply the identical image transform to the control arm.

Save checkpoints under `checkpoints/paper_metric_comparison/` and record the command, git revision, seed, dataset manifest, and metadata.

Use Hypersim for indoor/NYU-D and Virtual KITTI 2 for outdoor/KITTI, as documented by the DA2 metric release. If the project scripts cannot support the paper recipe without a target-space or head change, make a documented configuration change and validate it with a short run first. Record every deviation from the paper.

### Stage 3: dataset-matched secondary experiment

If Stage 1 does not approach the paper numbers, do not silently tune until it does. Preserve the failed reproduction and run a separate controlled experiment in which paper-style DA2, control, and proposed models are all trained on the same dataset and split used by the successful existing project checkpoints. This tests whether the camera-aware improvement is robust to the training domain, while keeping the paper-style result scientifically traceable.

## Evaluation

NYU-D, exactly 654 Eigen images:

```bash
python evaluate_paper.py --dataset nyu --model da2-revised --model-type metric \
  --encoder vitl --checkpoint CHECKPOINT --nyu-depth-field rawDepths \
  --crop eigen --clip-pred false \
  --output-json results/paper_metric_comparison/nyud_MODEL.json
```

KITTI, exactly the official 652-image Eigen filelist:

```bash
python evaluate_paper.py --dataset kitti --model da2-revised --model-type metric \
  --encoder vitl --checkpoint CHECKPOINT \
  --kitti-filelist datasets/raw_data/kitti/da2_val.txt \
  --crop none --clip-pred false \
  --output-json results/paper_metric_comparison/kitti_MODEL.json
```

Metric evaluation uses no per-image scale/shift alignment. Confirm the JSON metadata reports metric depth, the expected crop, sample count, and `outputs_inverse_depth=false`.

## Validation

Require 654 unique NYU IDs, 652 KITTI entries, zero skipped images, no train/test overlap, identical control/proposed test IDs and masks, finite values, and recomputed aggregate means. Report AbsRel, RMSE, delta1, paired tests, and bootstrap confidence intervals for proposed versus control. Preserve raw JSON, logs, metadata, and commands.

## Final tables

### Paper metric comparison

| Dataset | Source/model | AbsRel | RMSE | delta1 | Status |
|---|---|---:|---:|---:|---|
| NYU-D | Depth Anything V2 paper | 0.056 | 0.206 | 0.984 | published |
| NYU-D | Our DA2 metric control | ... | ... | ... | reimplemented |
| NYU-D | Our camera-aware metric model | ... | ... | ... | reimplemented |
| KITTI | Depth Anything V2 paper | 0.045 | 1.861 | 0.983 | published |
| KITTI | Our DA2 metric control | ... | ... | ... | reimplemented |
| KITTI | Our camera-aware metric model | ... | ... | ... | reimplemented |

### Matched intrinsic-ablation comparison

| Dataset | Model | AbsRel | RMSE | delta1 | Status |
|---|---|---:|---:|---:|---|
| NYU-D | Paper-style DA2 reproduction | ... | ... | ... | reimplemented |
| NYU-D | Matched control (no K) | ... | ... | ... | reimplemented |
| NYU-D | Proposed (+K) | ... | ... | ... | reimplemented |
| KITTI | Paper-style DA2 reproduction | ... | ... | ... | reimplemented |
| KITTI | Matched control (no K) | ... | ... | ... | reimplemented |
| KITTI | Proposed (+K) | ... | ... | ... | reimplemented |

Report proposed-minus-control differences, relative changes, confidence intervals, and p-values. Do not calculate improvement against the paper row unless training protocols are confirmed comparable.

## Thesis wording

> Under absolute metric-depth evaluation, the camera-aware model is compared with a matched no-intrinsics control using the same test split and evaluation protocol as the published Depth Anything V2 metric benchmark. The published paper values are external references; because the original metric checkpoints are unavailable, our trained baseline and proposed values are reported as reimplementations.

The causal claim comes from proposed versus matched control. The paper result is a contextual benchmark, while the paper-style reproduction tests whether the implementation can recover the paper's scale. If the proposed model beats the matched control but not the paper-style reproduction, report both facts rather than selecting the more favorable baseline.

## Completion

Finish only when Stage 1 and Stage 2 have valid NYU-D and KITTI metric JSON files, any Stage-3 run is clearly marked as secondary, all evaluations use the declared test sets, paired statistics are regenerated from per-image rows, deviations from the paper are documented, and the final tables plus command manifest are saved under `results/paper_metric_comparison/`.
