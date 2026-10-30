# ViT-S pilot — zoom 1.5

Absolute metric depth, no per-image alignment. Cells: AbsRel / δ1 / RMSE (m). Best AbsRel per row in bold.

| dataset | role | n | S-P2 | S-C0 |
|---|---|---:|---|---|
| nyu | indoor, train camera | 654 | **0.125 / 0.862 / 0.42** | 0.210 / 0.448 / 0.71 |
| kitti | outdoor, train camera | 652 | **0.131 / 0.836 / 4.34** | 0.288 / 0.167 / 6.70 |
| cityscapes | outdoor, train camera | 500 | **0.130 / 0.869 / 6.09** | 0.216 / 0.455 / 8.16 |
| diode_indoor | indoor, zero-shot | 325 | **0.290 / 0.478 / 1.17** | 0.377 / 0.241 / 1.59 |
| diode_outdoor | outdoor, zero-shot | 446 | **0.485 / 0.238 / 7.94** | 0.513 / 0.263 / 8.87 |
| middlebury | indoor, zero-shot | 23 | 1.283 / 0.117 / 3.38 | **0.357 / 0.286 / 1.23** |
| sunrgbd_kv1 | indoor, zero-shot | 200 | 0.335 / 0.480 / 0.54 | **0.213 / 0.575 / 0.45** |
| sunrgbd_kv2 | indoor, zero-shot | 200 | **0.148 / 0.818 / 0.43** | 0.156 / 0.707 / 0.47 |
| sunrgbd_realsense | indoor, zero-shot | 200 | **0.187 / 0.731 / 0.40** | 0.238 / 0.369 / 0.54 |
| sunrgbd_xtion | indoor, zero-shot | 200 | 0.262 / 0.614 / 0.62 | **0.235 / 0.417 / 0.72** |

## Paired: S-P2 vs S-C0

| dataset | n | AbsRel S-P2 | AbsRel S-C0 | Δ AbsRel | p (Wilcoxon) | S-P2 wins | δ1 S-P2 | δ1 S-C0 | RMSE S-P2 | RMSE S-C0 | p RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nyu | 654 | 0.1249 | 0.2101 | -0.0851 | 1.7e-52 | 81% | 0.862 | 0.448 | 0.421 | 0.712 | 3.3e-68 |
| kitti | 652 | 0.1308 | 0.2876 | -0.1567 | 6.3e-105 | 99% | 0.836 | 0.167 | 4.345 | 6.702 | 1.2e-101 |
| cityscapes | 500 | 0.1299 | 0.2161 | -0.0862 | 2.9e-62 | 86% | 0.869 | 0.455 | 6.089 | 8.160 | 1.9e-52 |
| diode_indoor | 325 | 0.2896 | 0.3766 | -0.0870 | 1.3e-22 | 78% | 0.478 | 0.241 | 1.171 | 1.592 | 2.4e-35 |
| diode_outdoor | 446 | 0.4853 | 0.5128 | -0.0275 | 7.5e-08 | 61% | 0.238 | 0.263 | 7.940 | 8.868 | 5.1e-13 |
| middlebury | 23 | 1.2832 | 0.3565 | +0.9267 | 1.5e-04 | 17% | 0.117 | 0.286 | 3.385 | 1.228 | 3.0e-04 |
| sunrgbd_kv1 | 200 | 0.3352 | 0.2129 | +0.1223 | 2.2e-12 | 32% | 0.480 | 0.575 | 0.544 | 0.449 | 1.1e-04 |
| sunrgbd_kv2 | 200 | 0.1483 | 0.1560 | -0.0078 | 8.9e-02 | 56% | 0.818 | 0.707 | 0.431 | 0.474 | 1.2e-02 |
| sunrgbd_realsense | 200 | 0.1874 | 0.2384 | -0.0509 | 2.7e-06 | 67% | 0.731 | 0.369 | 0.405 | 0.540 | 4.2e-08 |
| sunrgbd_xtion | 200 | 0.2615 | 0.2348 | +0.0268 | 6.5e-01 | 52% | 0.614 | 0.417 | 0.620 | 0.718 | 2.2e-02 |

S-P2 has the lower mean AbsRel on 7 of 10 datasets.
