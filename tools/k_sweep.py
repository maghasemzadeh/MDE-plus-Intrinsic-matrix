"""
K-sensitivity diagnostic ("K-sweep").

For a set of eigen-test NYU images, run a checkpoint with fx, fy scaled by a
range of factors (image pixels and cx, cy untouched) and measure how the
predicted depth responds. Under the pinhole model u = f*X/Z, a model that
actually consumes intrinsics should scale its predicted depth proportionally
to f (log-log slope ~= 1); a model that ignores K stays flat (slope ~= 0).

Usage (on lingotube, venv active, repo root):
    python tools/k_sweep.py \
        --checkpoint revised=models/.../revised_.../best.pth \
        --checkpoint control=models/.../base_.../best.pth \
        --encoder vitl --model-type metric \
        --num-images 20 --output-dir results/k_sweep --tag round2

Each --checkpoint is LABEL=PATH and may be repeated. Outputs a JSON with
per-image mean/median predictions per factor and fitted slopes, plus a
matplotlib figure (relative predicted depth vs focal factor, log-log).
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import DatasetConfig, NYUDataset
from models import DepthAnythingV2RevisedWrapper


def fit_loglog_slope(factors, values):
    """Least-squares slope of log(values) vs log(factors)."""
    x = np.log(np.asarray(factors, dtype=np.float64))
    y = np.log(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(y)):
        return float('nan')
    A = np.stack([x, np.ones_like(x)], axis=1)
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def main():
    parser = argparse.ArgumentParser(description='K-sensitivity sweep on NYU eigen-test images')
    parser.add_argument('--checkpoint', action='append', required=True,
                        help='LABEL=PATH of a checkpoint to sweep; repeatable')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--model-type', default='metric', choices=['metric', 'basic'])
    parser.add_argument('--num-images', type=int, default=20)
    parser.add_argument('--factors', type=float, nargs='+',
                        default=[0.6, 0.8, 1.0, 1.25, 1.6])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output-dir', default='results/k_sweep')
    parser.add_argument('--tag', default=None, help='Slug for output filenames')
    args = parser.parse_args()

    checkpoints = {}
    for spec in args.checkpoint:
        if '=' not in spec:
            parser.error(f"--checkpoint must be LABEL=PATH, got: {spec}")
        label, path = spec.split('=', 1)
        checkpoints[label] = path

    dataset = NYUDataset(DatasetConfig(split='test', max_items=args.num_images))
    items = dataset.find_items()[:args.num_images]
    if not items:
        print("No NYU eigen-test items found.")
        sys.exit(1)
    print(f"K-sweep over {len(items)} eigen-test images, factors {args.factors}")

    results = {}
    for label, ckpt in checkpoints.items():
        print(f"\n=== {label}: {ckpt} ===")
        model = DepthAnythingV2RevisedWrapper({
            'model_type': args.model_type,
            'encoder': args.encoder,
            'checkpoint_path': ckpt,
            'device': args.device,
            'max_depth': 20.0,
        })
        model.load_model()
        uses_k = bool(getattr(model, 'use_camera_intrinsics', False))
        print(f"  uses_camera_intrinsics = {uses_k}")

        per_image = []
        for item in items:
            image = dataset.load_image(item) if dataset.is_cached_dataset() else None
            if image is None:
                import cv2
                image = cv2.imread(item.image_path)
            K0 = dataset.load_intrinsic(item)
            if K0 is None:
                print(f"  {item.item_id}: no intrinsics, skipped")
                continue
            means, medians = [], []
            for f in args.factors:
                K = np.asarray(K0, dtype=np.float64).copy()
                K[0, 0] *= f
                K[1, 1] *= f
                pred = model.infer_image(image, input_size=args.input_size,
                                         intrinsics=K)
                pred = np.asarray(pred, dtype=np.float64)
                means.append(float(np.mean(pred)))
                medians.append(float(np.median(pred)))
            per_image.append({
                'item_id': item.item_id,
                'mean_pred': means,
                'median_pred': medians,
                'slope_mean': fit_loglog_slope(args.factors, means),
                'slope_median': fit_loglog_slope(args.factors, medians),
            })
            print(f"  {item.item_id}: median pred per factor = "
                  + ', '.join(f'{v:.3f}' for v in medians)
                  + f" | slope={per_image[-1]['slope_median']:.3f}")

        slopes = [r['slope_median'] for r in per_image if np.isfinite(r['slope_median'])]
        results[label] = {
            'checkpoint': ckpt,
            'uses_camera_intrinsics': uses_k,
            'factors': args.factors,
            'per_image': per_image,
            'slope_median_mean': float(np.mean(slopes)) if slopes else float('nan'),
            'slope_median_std': float(np.std(slopes)) if slopes else float('nan'),
        }
        # Free GPU memory before the next checkpoint
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'model':<20} {'log-log slope (median pred vs f)':<36} uses_K")
    print("-" * 70)
    for label, r in results.items():
        print(f"{label:<20} {r['slope_median_mean']:.4f} +/- {r['slope_median_std']:.4f}"
              f"{'':<12} {r['uses_camera_intrinsics']}")
    print("=" * 70)
    print("Interpretation: slope ~= 1.0 -> model rescales depth with f exactly as"
          " u = f*X/Z requires; slope ~= 0 -> model ignores K.")

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(args.output_dir, f'k_sweep_{tag}.json')
    with open(json_path, 'w') as fjson:
        json.dump({'config': vars(args), 'results': results}, fjson, indent=2)
    print(f"Saved JSON: {json_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, r in results.items():
        curves = []
        for img in r['per_image']:
            med = np.asarray(img['median_pred'], dtype=np.float64)
            ref_idx = int(np.argmin(np.abs(np.asarray(r['factors']) - 1.0)))
            if med[ref_idx] > 0:
                curves.append(med / med[ref_idx])
        if not curves:
            continue
        rel = np.exp(np.mean(np.log(np.asarray(curves)), axis=0))  # geometric mean
        ax.plot(r['factors'], rel, marker='o',
                label=f"{label} (slope {r['slope_median_mean']:.2f})")
    ax.plot(args.factors, args.factors, 'k--', alpha=0.4, label='ideal (slope 1)')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('focal length factor')
    ax.set_ylabel('relative predicted depth (median, geo-mean over images)')
    ax.set_title('K-sensitivity sweep (NYU eigen-test)')
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, f'k_sweep_{tag}.png')
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure: {fig_path}")


if __name__ == '__main__':
    main()
