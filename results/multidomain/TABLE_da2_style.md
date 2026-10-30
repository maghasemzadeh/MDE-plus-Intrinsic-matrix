# DA2-style results table — v1 ViT-L multi-domain models

Absolute metric depth, no per-image scale/shift alignment, mean over the test
images; every model is scored on exactly the same images. **Bold** = best value
in the column (ties at three decimals are all bold). All models are ViT-L.

- **Proposed P**: camera-aware (intrinsics token), trained on NYU + KITTI +
  VKITTI2 + Cityscapes with camera augmentation (`experiments/md_v1/P`).
- **Control C**: identical training, intrinsics withheld (`experiments/md_v1/C`).
- **Control C0**: intrinsics withheld and no camera augmentation
  (`experiments/md_v1/C0`); the strongest camera-blind model.
- **Base DA2**: released Depth Anything V2 metric ViT-L, zero-shot on every set;
  Hypersim checkpoint on indoor sets, Virtual KITTI 2 checkpoint on KITTI,
  Cityscapes and DIODE outdoor. Run with our evaluation code: the DA2 paper has
  no numbers for this setting (its Table 2 is relative depth after per-image
  alignment, and its metric NYU/KITTI models were never released).

Test-set sizes: NYU 654, KITTI 652, Cityscapes 500, DIODE indoor 325, DIODE
outdoor 446, Middlebury 23, SUN RGB-D 200 per sensor. P, C and C0 were trained
on the training splits of NYU, KITTI and Cityscapes; everything in (b) and (c)
is unseen by all four models.

"Significant" below: paired Wilcoxon signed-rank on per-image values, two-sided
p < 0.05, counted only when the per-image ranks favour the same model as the
mean.

## 1. Standard test images

**(a) Test splits of the training datasets**

| Model | NYU | | | KITTI | | | Cityscapes | | |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ |
| Proposed P | 0.088 | 0.945 | **0.309** | 0.156 | 0.811 | 4.063 | 0.109 | 0.894 | 5.417 |
| Control C | 0.133 | 0.856 | 0.395 | 0.175 | 0.787 | 4.403 | 0.136 | 0.851 | 5.938 |
| Control C0 | **0.087** | **0.947** | 0.311 | **0.090** | **0.924** | **2.857** | **0.104** | **0.903** | **5.291** |
| Base DA2 | 0.207 | 0.692 | 0.600 | 0.146 | 0.820 | 3.893 | 0.243 | 0.577 | 7.569 |

**(b) Unseen datasets (zero-shot)**

| Model | DIODE-in | | | DIODE-out | | | Middlebury | | |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ |
| Proposed P | 0.268 | 0.431 | 1.233 | **0.517** | 0.118 | 8.322 | 0.677 | 0.270 | 1.816 |
| Control C | 0.240 | **0.509** | **0.982** | 0.606 | **0.529** | **6.810** | **0.235** | **0.527** | **0.834** |
| Control C0 | 0.275 | 0.396 | 1.204 | 0.531 | 0.434 | 6.968 | 0.245 | 0.476 | 0.882 |
| Base DA2 | **0.235** | 0.507 | 1.029 | 0.740 | 0.339 | 7.474 | 0.384 | 0.142 | 1.190 |

**(c) Unseen SUN RGB-D sensors (zero-shot)**

| Model | Kinect v1 | | | Kinect v2 | | | RealSense | | | Xtion | | |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ |
| Proposed P | **0.169** | **0.766** | **0.369** | **0.104** | **0.906** | **0.351** | 0.093 | 0.935 | **0.234** | 0.176 | 0.770 | 0.430 |
| Control C | 0.314 | 0.478 | 0.532 | 0.209 | 0.677 | 0.470 | 0.125 | 0.862 | 0.290 | 0.216 | 0.697 | 0.493 |
| Control C0 | 0.173 | 0.752 | 0.375 | 0.135 | 0.860 | 0.352 | 0.114 | 0.883 | 0.290 | **0.145** | **0.818** | **0.403** |
| Base DA2 | 0.196 | 0.749 | 0.396 | 0.246 | 0.524 | 0.616 | **0.092** | **0.941** | 0.265 | 0.178 | 0.784 | 0.472 |

**Standard test images: Proposed P vs each other model, over 30 columns**

| P vs | P better | of which p<0.05 | P worse | of which p<0.05 | tie |
|:--|--:|--:|--:|--:|--:|
| Control C | 22 | 21 | 8 | 8 | 0 |
| Control C0 | 13 | 8 | 17 | 13 | 0 |
| Base DA2 | 17 | 14 | 13 | 9 | 0 |

## 2. Camera shift: centre crop ×1.5 (narrower field of view, K updated for P)

**(a) Test splits of the training datasets**

| Model | NYU | | | KITTI | | | Cityscapes | | |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ |
| Proposed P | **0.097** | **0.927** | **0.346** | **0.123** | 0.874 | 4.055 | **0.087** | **0.935** | **4.763** |
| Control C | 0.118 | 0.877 | 0.393 | 0.124 | **0.875** | **3.841** | 0.108 | 0.894 | 5.128 |
| Control C0 | 0.207 | 0.450 | 0.701 | 0.265 | 0.207 | 6.068 | 0.223 | 0.408 | 8.287 |
| Base DA2 | 0.164 | 0.673 | 0.547 | 0.249 | 0.286 | 6.489 | 0.178 | 0.675 | 9.083 |

**(b) Unseen datasets (zero-shot)**

| Model | DIODE-in | | | DIODE-out | | | Middlebury | | |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ |
| Proposed P | **0.256** | **0.571** | **1.080** | **0.521** | 0.128 | 9.068 | 0.813 | 0.172 | 2.110 |
| Control C | 0.298 | 0.338 | 1.200 | 0.551 | **0.513** | **6.533** | **0.293** | **0.394** | **0.994** |
| Control C0 | 0.382 | 0.171 | 1.664 | 0.523 | 0.248 | 8.516 | 0.320 | 0.295 | 1.092 |
| Base DA2 | 0.381 | 0.187 | 1.547 | 0.627 | 0.317 | 8.164 | 0.515 | 0.056 | 1.542 |

**(c) Unseen SUN RGB-D sensors (zero-shot)**

| Model | Kinect v1 | | | Kinect v2 | | | RealSense | | | Xtion | | |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ | AbsRel↓ | δ1↑ | RMSE↓ |
| Proposed P | 0.238 | 0.630 | 0.397 | **0.106** | **0.915** | **0.314** | 0.138 | **0.846** | 0.312 | 0.221 | 0.659 | 0.547 |
| Control C | 0.295 | 0.534 | 0.460 | 0.113 | 0.890 | 0.335 | **0.130** | 0.842 | **0.307** | **0.190** | **0.736** | **0.485** |
| Control C0 | 0.189 | 0.608 | 0.420 | 0.148 | 0.741 | 0.454 | 0.249 | 0.303 | 0.564 | 0.225 | 0.436 | 0.688 |
| Base DA2 | **0.185** | **0.635** | **0.394** | 0.142 | 0.773 | 0.399 | 0.267 | 0.251 | 0.558 | 0.206 | 0.523 | 0.608 |

**Camera shift ×1.5: Proposed P vs each other model, over 30 columns**

| P vs | P better | of which p<0.05 | P worse | of which p<0.05 | tie |
|:--|--:|--:|--:|--:|--:|
| Control C | 18 | 14 | 12 | 8 | 0 |
| Control C0 | 24 | 20 | 6 | 5 | 0 |
| Base DA2 | 22 | 20 | 8 | 4 | 0 |

Generated from `results/multidomain/{md1_P,md1_C,md1_C0,released_hypersim_vitl,released_vkitti_vitl}[_zoom1.5]/<dataset>.json`.
