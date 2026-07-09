"""
Output-difference visualization for a revised-vs-control checkpoint pair.

For N sample eigen-test images, produce:
  - side-by-side ABSOLUTE depth maps (shared color scale in meters, not
    per-image normalized) for both models plus GT,
  - |revised - control| difference maps,
  - per-image scale-ratio (median pred/GT) histograms for both models.

Usage (on lingotube, venv active, repo root):
    python tools/output_difference_viz.py \
        --checkpoint revised=models/.../revised_.../best.pth \
        --checkpoint control=models/.../base_.../best.pth \
        --encoder vitl --num-images 6 --output-dir results/output_diff --tag focal
"""

import argparse
import os
import sys
import json
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from datasets import DatasetConfig, NYUDataset
from models import DepthAnythingV2RevisedWrapper


def main():
    parser = argparse.ArgumentParser(description='Revised-vs-control output difference figures')
    parser.add_argument('--checkpoint', action='append', required=True,
                        help='LABEL=PATH; exactly two expected (e.g. revised=..., control=...)')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--model-type', default='metric', choices=['metric', 'basic'])
    parser.add_argument('--num-images', type=int, default=6)
    parser.add_argument('--num-ratio-images', type=int, default=100,
                        help='Images used for the scale-ratio histograms')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output-dir', default='results/output_diff')
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    checkpoints = {}
    for spec in args.checkpoint:
        label, path = spec.split('=', 1)
        checkpoints[label] = path
    labels = list(checkpoints.keys())
    if len(labels) != 2:
        parser.error('exactly two --checkpoint entries expected')

    n_load = max(args.num_images, args.num_ratio_images)
    dataset = NYUDataset(DatasetConfig(split='test', max_items=n_load))
    items = dataset.find_items()[:n_load]

    preds = {lab: [] for lab in labels}   # aligned with items order
    ratios = {lab: [] for lab in labels}  # median pred/GT per image
    gts, images = [], []

    for lab in labels:
        model = DepthAnythingV2RevisedWrapper({
            'model_type': args.model_type,
            'encoder': args.encoder,
            'checkpoint_path': checkpoints[lab],
            'device': args.device,
            'max_depth': 20.0,
        })
        model.load_model()
        print(f"{lab}: uses_camera_intrinsics = {getattr(model, 'use_camera_intrinsics', False)}")
        for idx, item in enumerate(tqdm(items, desc=lab, unit='img')):
            image = dataset.load_image(item) if dataset.is_cached_dataset() else cv2.imread(item.image_path)
            gt = np.squeeze(np.asarray(dataset.load_gt_depth(item.gt_path, item), dtype=np.float32))
            K = dataset.load_intrinsic(item)
            pred = model.infer_image(image, input_size=args.input_size, intrinsics=K)
            pred = np.squeeze(np.asarray(pred, dtype=np.float32))
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            valid = np.isfinite(gt) & (gt > 0.001) & (gt <= 10.0)
            if valid.sum() > 10:
                ratios[lab].append(float(np.median(pred[valid]) / np.median(gt[valid])))
            if idx < args.num_images:
                preds[lab].append(pred)
                if lab == labels[0]:
                    gts.append(gt)
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or datetime.now().strftime('%Y%m%d_%H%M%S')

    n = len(gts)
    vmax = max(float(np.nanpercentile(g, 99)) for g in gts)
    fig, axes = plt.subplots(n, 5, figsize=(22, 3.6 * n), squeeze=False)
    for i in range(n):
        pA, pB, gt = preds[labels[0]][i], preds[labels[1]][i], gts[i]
        diff = np.abs(pA - pB)
        panels = [
            (images[i], None, 'image', None),
            (gt, 'turbo', 'GT depth (m)', (0, vmax)),
            (pA, 'turbo', f'{labels[0]} (m)', (0, vmax)),
            (pB, 'turbo', f'{labels[1]} (m)', (0, vmax)),
            (diff, 'magma', f'|{labels[0]} - {labels[1]}| (m)', (0, max(0.5, float(np.nanpercentile(diff, 99))))),
        ]
        for j, (data, cmap, title, clim) in enumerate(panels):
            ax = axes[i][j]
            if cmap is None:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=cmap, vmin=clim[0], vmax=clim[1])
                fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
    fig.suptitle(f'Absolute depth maps, shared scale 0-{vmax:.1f} m ({tag})')
    fig.tight_layout()
    maps_path = os.path.join(args.output_dir, f'depth_maps_{tag}.png')
    fig.savefig(maps_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {maps_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.5, 1.5, 41)
    for lab in labels:
        ax.hist(ratios[lab], bins=bins, alpha=0.55, label=f'{lab} (median {np.median(ratios[lab]):.3f})')
    ax.axvline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('per-image scale ratio: median(pred)/median(GT)')
    ax.set_ylabel('images')
    ax.set_title(f'Scale-ratio distribution over {len(ratios[labels[0]])} eigen-test images ({tag})')
    ax.legend()
    fig.tight_layout()
    hist_path = os.path.join(args.output_dir, f'scale_ratio_hist_{tag}.png')
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {hist_path}")

    stats = {lab: {'ratios': ratios[lab],
                   'median': float(np.median(ratios[lab])),
                   'mean': float(np.mean(ratios[lab]))} for lab in labels}
    json_path = os.path.join(args.output_dir, f'output_diff_{tag}.json')
    with open(json_path, 'w') as fjson:
        json.dump({'config': {k: v for k, v in vars(args).items()}, 'scale_ratio_stats': stats}, fjson, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
