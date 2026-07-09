"""
FoV-varied metric evaluation on the NYU eigen-654 test split.

For each test image, evaluate at several deterministic center zoom-crops with
exactly-updated intrinsics: crop the central (H/z, W/z) window from the image
and the GT depth and shift the principal point (cx -= x0, cy -= y0). Focal
length and depth values are unchanged (same scene points, narrower FoV). No
resampling is done here — the model wrapper resizes internally and its
intrinsics normalization is relative to the input frame, so a pure crop is
geometrically exact.

An intrinsics-aware model should hold its absolute-depth accuracy across
zooms (it can read the changed FoV from K); a K-blind model sees a
zoomed-in image whose semantic scale cues suggest a closer scene and
degrades.

Usage (on lingotube, venv active, repo root):
    python tools/eval_fov_varied.py \
        --checkpoint revised=models/.../revised_.../best.pth \
        --checkpoint control=models/.../base_.../best.pth \
        --encoder vitl --model-type metric \
        --zooms 1.0 1.4 1.8 --output-dir results/fov_varied --tag focal
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


def center_zoom_crop(image, gt, K, zoom):
    """Central (H/zoom, W/zoom) crop of image + GT with exact K update."""
    H, W = image.shape[:2]
    win_h, win_w = int(round(H / zoom)), int(round(W / zoom))
    y0 = (H - win_h) // 2
    x0 = (W - win_w) // 2
    image = image[y0:y0 + win_h, x0:x0 + win_w]
    gt = gt[y0:y0 + win_h, x0:x0 + win_w]
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    return image, gt, K


def main():
    parser = argparse.ArgumentParser(description='FoV-varied NYU eigen-654 metric evaluation')
    parser.add_argument('--checkpoint', action='append', required=True,
                        help='LABEL=PATH of a checkpoint to evaluate; repeatable')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--model-type', default='metric', choices=['metric', 'basic'])
    parser.add_argument('--zooms', type=float, nargs='+', default=[1.0, 1.4, 1.8])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--min-depth-eval', type=float, default=0.001)
    parser.add_argument('--max-depth-eval', type=float, default=10.0)
    parser.add_argument('--max-items', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output-dir', default='results/fov_varied')
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    checkpoints = {}
    for spec in args.checkpoint:
        if '=' not in spec:
            parser.error(f"--checkpoint must be LABEL=PATH, got: {spec}")
        label, path = spec.split('=', 1)
        checkpoints[label] = path

    dataset = NYUDataset(DatasetConfig(split='test', max_items=args.max_items))
    items = dataset.find_items()
    if args.max_items:
        items = items[:args.max_items]
    print(f"FoV-varied eval: {len(items)} eigen-test images, zooms {args.zooms}")

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
        is_metric = model.is_metric()
        print(f"  uses_camera_intrinsics = {uses_k}")

        per_zoom = {f'{z:g}': [] for z in args.zooms}
        skipped = 0
        for item in tqdm(items, desc=f'{label}', unit='img'):
            image = dataset.load_image(item) if dataset.is_cached_dataset() else cv2.imread(item.image_path)
            if image is None:
                skipped += 1
                continue
            gt = np.squeeze(np.asarray(dataset.load_gt_depth(item.gt_path, item), dtype=np.float32))
            K0 = dataset.load_intrinsic(item)
            if K0 is None:
                skipped += 1
                continue

            for z in args.zooms:
                img_z, gt_z, K_z = center_zoom_crop(image, gt, K0, z)
                pred = model.infer_image(img_z, input_size=args.input_size,
                                         intrinsics=K_z)
                pred = np.squeeze(np.asarray(pred, dtype=np.float32))
                if pred.shape != gt_z.shape:
                    pred = cv2.resize(pred, (gt_z.shape[1], gt_z.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                gt_eval = gt_z.copy()
                gt_eval[~np.isfinite(gt_eval)] = np.nan
                gt_eval[(gt_eval < args.min_depth_eval) | (gt_eval > args.max_depth_eval)] = np.nan

                m = compute_depth_metrics(pred, gt_eval, is_metric_model=is_metric,
                                          outputs_inverse_depth=model.outputs_inverse_depth())
                if m['n_valid'] < 10 or not np.isfinite(m['abs_rel']):
                    continue
                m['item_id'] = item.item_id
                per_zoom[f'{z:g}'].append(m)

        summary = {}
        for z_key, rows in per_zoom.items():
            summary[z_key] = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
            summary[z_key]['n_images'] = len(rows)
        results[label] = {
            'checkpoint': ckpt,
            'uses_camera_intrinsics': uses_k,
            'summary': summary,
            'per_zoom': per_zoom,
            'n_skipped': skipped,
        }

        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    header = f"{'model':<14} {'zoom':<6} " + ' '.join(f'{k:>8}' for k in METRIC_KEYS)
    print(header)
    print("-" * 78)
    for label, r in results.items():
        for z_key, s in r['summary'].items():
            print(f"{label:<14} {z_key:<6} " + ' '.join(f"{s[k]:8.4f}" for k in METRIC_KEYS)
                  + f"  (n={s['n_images']})")
    print("=" * 78)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(args.output_dir, f'fov_varied_{tag}.json')
    with open(json_path, 'w') as fjson:
        json.dump({'config': vars(args), 'results': results}, fjson, indent=2)
    print(f"Saved JSON: {json_path}")


if __name__ == '__main__':
    main()
