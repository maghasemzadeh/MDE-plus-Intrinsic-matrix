---
name: paper-eval-protocol
description: DA2 paper zero-shot evaluation protocol and verified reproduction numbers. Use when running or modifying evaluate_paper.py, reproducing DA2 Table 2 (NYU/KITTI AbsRel/δ1), comparing results against paper numbers, or writing about that reproduction in the thesis.
---

# Paper-Protocol Evaluation (verified against DA2 paper)

`evaluate_paper.py` reproduces the DA2 paper's zero-shot relative-depth protocol
(Table 2) with the released *basic* checkpoints. Verified results (2026-07, lingotube):

| run | ours (AbsRel/δ1) | paper |
|---|---|---|
| NYU vitl | 0.043 / 0.979 | 0.045 / 0.979 |
| NYU vits | 0.051 / 0.973 | 0.053 / 0.973 |
| KITTI vitl (no crop) | 0.076 / 0.948 | 0.074 / 0.946 |
| KITTI vits (no crop) | 0.083 / 0.934 | 0.078 / 0.936 |

(KITTI with the garg crop instead: vitl 0.070/0.956, vits 0.077/0.944 — the two
crop variants bracket the paper numbers; remaining differences are within normal
reproduction tolerance for this benchmark.)

Protocol essentials (all implemented in `evaluate_paper.py` + `src/metrics.py`):
- NYU: eigen 654 split (canonical indices from splits.mat), **rawDepths** GT
  (not the inpainted `depths` field), eigen crop, caps 0.001–10 m.
- KITTI: DA2's 652-image eigen-val filelist (`datasets/raw_data/kitti/da2_val.txt`
  on the server; images under `kitti_raw/`), **no crop**, caps 0.001–80 m.
- Basic model: per-image scale+shift alignment in inverse-depth space, then
  predictions clipped to the eval depth range (without clipping, near-zero
  aligned disparities explode and wreck the mean).
- Paper Table 4 (metric NYU/KITTI) uses in-domain fine-tuned checkpoints that
  were never released — it is NOT reproducible with the public
  hypersim/vkitti metric checkpoints.

