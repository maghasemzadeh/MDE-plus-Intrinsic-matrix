"""
Pre-GPU sanity check for the focal-jitter data pipeline.

Constructs NYUTrainingDataset with focal_jitter enabled and pulls samples,
verifying that:
  - alpha varies per sample (fx spread across draws of the SAME item),
  - fy/fx stays fixed while fx varies (only f scaled, not cx/cy),
  - depth scales with fx (correlation of median depth vs fx across draws),
  - the valid mask keeps alpha-scaled far points (mask fraction stable).

Runs on CPU, needs the NYU .mat (server).
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.training_datasets import NYUTrainingDataset

MAT = os.path.join(PROJECT_ROOT, 'datasets', 'raw_data', 'nyu', 'nyu_depth_v2_labeled.mat')

ds = NYUTrainingDataset(MAT, 'train', size=(518, 518), focal_jitter=1.6)
print(f"\nfocal_jitter={ds.focal_jitter}, intrinsics_aug={ds.intrinsics_aug}")

np.random.seed(0)
N_DRAWS = 30
fx_vals, cx_vals, med_depths, mask_fracs, fyfx = [], [], [], [], []
for _ in range(N_DRAWS):
    s = ds[0]  # same item, fresh augmentation draws
    K = s['intrinsics'].numpy()
    d = s['depth'].numpy()
    m = s['valid_mask'].numpy()
    fx_vals.append(K[0, 0])
    cx_vals.append(K[0, 2])
    fyfx.append(K[1, 1] / K[0, 0])
    med_depths.append(float(np.median(d[m])) if m.any() else float('nan'))
    mask_fracs.append(float(m.mean()))

fx = np.array(fx_vals)
md = np.array(med_depths)
print(f"fx across {N_DRAWS} draws of item 0: min={fx.min():.1f} max={fx.max():.1f} "
      f"ratio={fx.max()/fx.min():.2f}")
print(f"fy/fx (should be ~constant): std={np.std(fyfx):.5f}")
print(f"median valid depth: min={np.nanmin(md):.2f} max={np.nanmax(md):.2f}")
print(f"valid-mask fraction: min={min(mask_fracs):.3f} max={max(mask_fracs):.3f}")
corr = np.corrcoef(np.log(fx), np.log(md))[0, 1]
print(f"corr(log fx, log median depth) across draws: {corr:.3f}")

ok = True
# geometric_aug alone varies fx by up to ~2.2x; jitter adds 1.6^2 more.
# Requiring >2.5x confirms the jitter contributes beyond the zoom-crop.
if fx.max() / fx.min() < 2.5:
    ok = False
    print("FAIL: fx spread too small - jitter not applied?")
# fy/fx wobbles ~0.15% because _geometric_aug rounds win_h and win_w
# independently; the jitter itself scales fx and fy by the same alpha.
if np.std(fyfx) > 5e-3:
    ok = False
    print("FAIL: fy/fx not constant - K corrupted")
if not (corr > 0.3):
    ok = False
    print("FAIL: depth does not track fx - jitter depth scaling broken? "
          "(some correlation is expected: alpha scales both)")
if max(mask_fracs) - min(mask_fracs) > 0.2:
    ok = False
    print("FAIL: valid-mask fraction varies wildly with alpha - mask threshold bug?")

print("\nSANITY " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
