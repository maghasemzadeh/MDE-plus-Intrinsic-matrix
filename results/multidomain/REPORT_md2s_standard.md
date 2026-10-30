# ViT-S pilot — standard

Absolute metric depth, no per-image alignment. Cells: AbsRel / δ1 / RMSE (m). Best AbsRel per row in bold.

| dataset | role | n | S-P2 | S-C0 |
|---|---|---:|---|---|
| nyu | indoor, train camera | 654 | 0.116 / 0.884 / 0.39 | **0.100 / 0.917 / 0.33** |
| kitti | outdoor, train camera | 652 | 0.119 / 0.881 / 3.61 | **0.100 / 0.913 / 3.35** |
| cityscapes | outdoor, train camera | 500 | 0.135 / 0.842 / 6.46 | **0.129 / 0.853 / 6.16** |
| diode_indoor | indoor, zero-shot | 325 | **0.271 / 0.460 / 1.16** | 0.309 / 0.387 / 1.20 |
| diode_outdoor | outdoor, zero-shot | 446 | **0.484 / 0.189 / 7.80** | 0.543 / 0.432 / 7.31 |
| middlebury | indoor, zero-shot | 23 | 0.670 / 0.381 / 1.81 | **0.308 / 0.413 / 1.08** |
| sunrgbd_kv1 | indoor, zero-shot | 200 | 0.233 / 0.635 / 0.46 | **0.208 / 0.694 / 0.42** |
| sunrgbd_kv2 | indoor, zero-shot | 200 | **0.112 / 0.883 / 0.36** | 0.142 / 0.848 / 0.39 |
| sunrgbd_realsense | indoor, zero-shot | 200 | 0.134 / 0.847 / 0.31 | **0.108 / 0.889 / 0.29** |
| sunrgbd_xtion | indoor, zero-shot | 200 | 0.209 / 0.708 / 0.50 | **0.153 / 0.808 / 0.42** |

## Paired: S-P2 vs S-C0

| dataset | n | AbsRel S-P2 | AbsRel S-C0 | Δ AbsRel | p (Wilcoxon) | S-P2 wins | δ1 S-P2 | δ1 S-C0 | RMSE S-P2 | RMSE S-C0 | p RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nyu | 654 | 0.1159 | 0.1003 | +0.0156 | 3.0e-37 | 26% | 0.884 | 0.917 | 0.390 | 0.334 | 2.2e-55 |
| kitti | 652 | 0.1186 | 0.0998 | +0.0188 | 2.3e-73 | 16% | 0.881 | 0.913 | 3.605 | 3.347 | 1.3e-37 |
| cityscapes | 500 | 0.1347 | 0.1289 | +0.0057 | 2.3e-12 | 35% | 0.842 | 0.853 | 6.458 | 6.160 | 1.1e-24 |
| diode_indoor | 325 | 0.2708 | 0.3090 | -0.0382 | 7.2e-11 | 67% | 0.460 | 0.387 | 1.160 | 1.198 | 1.0e-08 |
| diode_outdoor | 446 | 0.4840 | 0.5426 | -0.0586 | 1.8e-01 | 48% | 0.189 | 0.432 | 7.803 | 7.315 | 6.6e-08 |
| middlebury | 23 | 0.6700 | 0.3078 | +0.3622 | 4.5e-02 | 30% | 0.381 | 0.413 | 1.807 | 1.081 | 1.0e-01 |
| sunrgbd_kv1 | 200 | 0.2326 | 0.2081 | +0.0245 | 1.1e-08 | 32% | 0.635 | 0.694 | 0.457 | 0.420 | 9.5e-07 |
| sunrgbd_kv2 | 200 | 0.1121 | 0.1418 | -0.0297 | 7.1e-09 | 70% | 0.883 | 0.848 | 0.361 | 0.395 | 1.3e-02 |
| sunrgbd_realsense | 200 | 0.1336 | 0.1080 | +0.0257 | 2.5e-05 | 39% | 0.847 | 0.889 | 0.314 | 0.286 | 4.5e-03 |
| sunrgbd_xtion | 200 | 0.2094 | 0.1525 | +0.0568 | 3.3e-14 | 30% | 0.708 | 0.808 | 0.499 | 0.418 | 1.1e-07 |

S-P2 has the lower mean AbsRel on 3 of 10 datasets.
