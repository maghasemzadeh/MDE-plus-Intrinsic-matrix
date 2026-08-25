"""
Causal-dependence ("knockout") test for an intrinsics-aware checkpoint.

Evaluates one checkpoint on the NYU eigen-654 split under several
K-conditions and reports absolute-depth metrics per condition:

  correct   the dataset's true K (the standard eval)
  withheld  no intrinsics passed at all (K-path disabled)
  scaled:X  fx, fy multiplied by X (deliberately wrong focal length)

If the model merely *tolerates* K, all conditions score the same. If it
*causally relies* on K, withholding it hurts, and a wrong focal length must
hurt in the specific way the pinhole model predicts: predictions scale by
~X^slope, so AbsRel degrades toward |X^slope - 1|.

Usage (server, venv, repo root):
    python tools/k_knockout.py \
        --checkpoint models/.../revised_.../best.pth \
        --encoder vitl --model-type metric \
        --scales 0.7 1.4 --output-dir results/k_knockout --tag focal
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
from tqdm import tqdm

from datasets import DatasetConfig, NYUDataset
from models import DepthAnythingV2RevisedWrapper
from src.metrics import compute_depth_metrics

METRIC_KEYS = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']


def main():
    parser = argparse.ArgumentParser(description='K causal-dependence test on NYU eigen-654')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--model-type', default='metric', choices=['metric', 'basic'])
    parser.add_argument('--scales', type=float, nargs='+', default=[0.7, 1.4])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--min-depth-eval', type=float, default=0.001)
    parser.add_argument('--max-depth-eval', type=float, default=10.0)
    parser.add_argument('--max-items', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output-dir', default='results/k_knockout')
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    conditions = ['correct', 'withheld'] + [f'scaled:{s:g}' for s in args.scales]

    dataset = NYUDataset(DatasetConfig(split='test', max_items=args.max_items))
    dataset.depth_field = 'rawDepths'
    items = dataset.find_items()
    if args.max_items:
        items = items[:args.max_items]

    model = DepthAnythingV2RevisedWrapper({
        'model_type': args.model_type,
        'encoder': args.encoder,
        'checkpoint_path': args.checkpoint,
        'device': args.device,
        'max_depth': 20.0,
    })
    model.load_model()
    if not getattr(model, 'use_camera_intrinsics', False):
        print('checkpoint does not use camera intrinsics — knockout test is meaningless')
        sys.exit(1)
    is_metric = model.is_metric()

    per_condition = {c: [] for c in conditions}
    skipped = 0
    for item in tqdm(items, desc='knockout', unit='img'):
        image = dataset.load_image(item) if dataset.is_cached_dataset() else cv2.imread(item.image_path)
        if image is None:
            skipped += 1
            continue
        gt = np.squeeze(np.asarray(dataset.load_gt_depth(item.gt_path, item), dtype=np.float32))
        K0 = dataset.load_intrinsic(item)
        if K0 is None:
            skipped += 1
            continue
        gt_eval = gt.copy()
        gt_eval[~np.isfinite(gt_eval)] = np.nan
        gt_eval[(gt_eval < args.min_depth_eval) | (gt_eval > args.max_depth_eval)] = np.nan

        for cond in conditions:
            if cond == 'withheld':
                K = None
            else:
                K = np.asarray(K0, dtype=np.float64).copy()
                if cond.startswith('scaled:'):
                    s = float(cond.split(':')[1])
                    K[0, 0] *= s
                    K[1, 1] *= s
            pred = model.infer_image(image, input_size=args.input_size, intrinsics=K)
            pred = np.squeeze(np.asarray(pred, dtype=np.float32))
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            m = compute_depth_metrics(pred, gt_eval, is_metric_model=is_metric,
                                      outputs_inverse_depth=model.outputs_inverse_depth())
            if m['n_valid'] < 10 or not np.isfinite(m['abs_rel']):
                continue
            m['item_id'] = item.item_id
            per_condition[cond].append(m)

    summary = {}
    for cond, rows in per_condition.items():
        summary[cond] = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
        summary[cond]['n_images'] = len(rows)

    print('\n' + '=' * 78)
    print(f"{'condition':<14} " + ' '.join(f'{k:>8}' for k in METRIC_KEYS))
    print('-' * 78)
    for cond in conditions:
        s = summary[cond]
        print(f"{cond:<14} " + ' '.join(f"{s[k]:8.4f}" for k in METRIC_KEYS) + f"  (n={s['n_images']})")
    print('=' * 78)
    print('If the model causally relies on K: withheld/scaled must be clearly '
          'worse than correct, with scaled:X degrading toward |X^slope - 1| AbsRel.')

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.output_dir, f'k_knockout_{tag}.json')
    with open(out, 'w') as f:
        json.dump({'config': vars(args), 'summary': summary,
                   'per_condition': per_condition, 'n_skipped': skipped}, f, indent=2)
    print(f'Saved JSON: {out}')


if __name__ == '__main__':
    main()
