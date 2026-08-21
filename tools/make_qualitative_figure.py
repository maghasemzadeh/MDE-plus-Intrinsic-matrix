"""Pre-registered qualitative depth comparison for the thesis.

Section 4.6 of the thesis states that no qualitative figure was included because
no auditable protocol existed: any hand-picked sample would be selective and
unverifiable.  This script implements exactly the protocol that section asks
for, in two mandatory phases.

Phase 1 -- lock (no model is loaded, no depth map is ever seen)::

    python tools/make_qualitative_figure.py lock \
        --num-images 6 --seed 20260821 \
        --manifest results/qualitative/manifest.json

    Selects the sample deterministically from the seed, writes the chosen image
    identifiers to a manifest, and prints a SHA-256 digest of that list.  Record
    the digest (e.g. paste it in the thesis caption) *before* running phase 2.

Phase 2 -- render (inference on exactly the locked images, nothing else)::

    python tools/make_qualitative_figure.py render \
        --manifest results/qualitative/manifest.json \
        --checkpoint proposed=models/.../revised_.../best.pth \
        --checkpoint control=models/.../base_.../best.pth \
        --encoder vitl --output-dir results/qualitative

    Refuses to run if the manifest is absent or its digest does not match its
    own contents, so the sample cannot be silently changed after the outputs
    have been inspected.

Both models are rendered with a single shared colormap and a single shared
depth range per row, so no visual advantage can come from per-image
normalisation.  Outputs are vector PDFs ready for \\includegraphics.

Run on lingotube with the venv active, from the repository root.
"""

import argparse
import hashlib
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
import cv2
from tqdm import tqdm

from datasets import DatasetConfig, NYUDataset
from models import DepthAnythingV2RevisedWrapper

# Standard NYUv2 evaluation caps, matching the rest of the evaluation chain.
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0


def _item_id(item, index):
    """Stable identifier for a dataset item, independent of load order."""
    for attr in ('item_id', 'image_path'):
        value = getattr(item, attr, None)
        if value:
            return str(value)
    return f'index:{index}'


def _digest(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def cmd_lock(args):
    """Choose the sample from the seed alone, before any model is loaded."""
    dataset = NYUDataset(DatasetConfig(split='test'))
    items = dataset.find_items()
    ids = [_item_id(item, i) for i, item in enumerate(items)]
    if args.num_images > len(ids):
        raise SystemExit(f'only {len(ids)} test images available')

    rng = np.random.default_rng(args.seed)
    chosen_idx = sorted(rng.choice(len(ids), size=args.num_images, replace=False).tolist())
    chosen_ids = [ids[i] for i in chosen_idx]

    manifest = {
        'protocol': 'pre-registered qualitative sample; ids fixed before any inference',
        'created': datetime.now().isoformat(timespec='seconds'),
        'split': 'test',
        'population_size': len(ids),
        'seed': args.seed,
        'num_images': args.num_images,
        'indices': chosen_idx,
        'item_ids': chosen_ids,
        'sha256': _digest(chosen_ids),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)) or '.', exist_ok=True)
    with open(args.manifest, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print(f'Locked {args.num_images} of {len(ids)} images (seed {args.seed}).')
    print(f'Manifest: {args.manifest}')
    print(f'SHA-256 : {manifest["sha256"]}')
    print('Record this digest before rendering.')


def _load_manifest(path):
    if not os.path.exists(path):
        raise SystemExit(f'manifest not found: {path}\nRun the "lock" phase first.')
    with open(path) as fh:
        manifest = json.load(fh)
    expected = _digest(manifest['item_ids'])
    if expected != manifest.get('sha256'):
        raise SystemExit(
            'manifest digest mismatch: the locked sample was modified after locking.\n'
            f'  stored:   {manifest.get("sha256")}\n  recomputed: {expected}')
    return manifest


def cmd_render(args):
    manifest = _load_manifest(args.manifest)
    checkpoints = {}
    for spec in args.checkpoint:
        if '=' not in spec:
            raise SystemExit(f'--checkpoint must be LABEL=PATH, got: {spec}')
        label, path = spec.split('=', 1)
        checkpoints[label] = path
    if len(checkpoints) != 2:
        raise SystemExit('exactly two --checkpoint entries expected (proposed and control)')
    labels = list(checkpoints)

    dataset = NYUDataset(DatasetConfig(split='test'))
    all_items = dataset.find_items()
    by_id = {_item_id(it, i): it for i, it in enumerate(all_items)}
    missing = [i for i in manifest['item_ids'] if i not in by_id]
    if missing:
        raise SystemExit(f'{len(missing)} locked image ids are not present in the split: {missing[:3]}')
    items = [by_id[i] for i in manifest['item_ids']]

    images, gts = [], []
    preds = {lab: [] for lab in labels}
    ratios = {lab: [] for lab in labels}

    ratio_items = all_items[:args.num_ratio_images] if args.num_ratio_images else []

    for lab in labels:
        model = DepthAnythingV2RevisedWrapper({
            'model_type': args.model_type,
            'encoder': args.encoder,
            'checkpoint_path': checkpoints[lab],
            'device': args.device,
            'max_depth': 20.0,
        })
        model.load_model()
        print(f'{lab}: uses_camera_intrinsics = {getattr(model, "use_camera_intrinsics", False)}')

        for idx, item in enumerate(tqdm(items, desc=f'{lab} (locked sample)', unit='img')):
            image = dataset.load_image(item) if dataset.is_cached_dataset() else cv2.imread(item.image_path)
            gt = np.squeeze(np.asarray(dataset.load_gt_depth(item.gt_path, item), dtype=np.float32))
            K = dataset.load_intrinsic(item)
            pred = np.squeeze(np.asarray(
                model.infer_image(image, input_size=args.input_size, intrinsics=K), dtype=np.float32))
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            preds[lab].append(pred)
            if lab == labels[0]:
                gts.append(gt)
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        for item in tqdm(ratio_items, desc=f'{lab} (scale ratio)', unit='img'):
            image = dataset.load_image(item) if dataset.is_cached_dataset() else cv2.imread(item.image_path)
            gt = np.squeeze(np.asarray(dataset.load_gt_depth(item.gt_path, item), dtype=np.float32))
            K = dataset.load_intrinsic(item)
            pred = np.squeeze(np.asarray(
                model.infer_image(image, input_size=args.input_size, intrinsics=K), dtype=np.float32))
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            valid = np.isfinite(gt) & (gt > MIN_DEPTH) & (gt <= MAX_DEPTH)
            if valid.sum() > 10:
                ratios[lab].append(float(np.median(pred[valid]) / np.median(gt[valid])))

        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- panel figure: one shared depth range per row, identical for all models ---
    n = len(gts)
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 2.9 * n), squeeze=False)
    for i in range(n):
        gt = gts[i]
        valid = np.isfinite(gt) & (gt > MIN_DEPTH) & (gt <= MAX_DEPTH)
        vmin = 0.0
        vmax = float(np.nanpercentile(gt[valid], 99)) if valid.any() else MAX_DEPTH
        panels = [
            (images[i], None, 'RGB input'),
            (np.where(valid, gt, np.nan), 'turbo', 'ground truth (m)'),
            (preds[labels[1]][i], 'turbo', f'{labels[1]} (m)'),
            (preds[labels[0]][i], 'turbo', f'{labels[0]} (m)'),
        ]
        for j, (data, cmap, title) in enumerate(panels):
            ax = axes[i][j]
            if cmap is None:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                if j == 3:
                    fig.colorbar(im, ax=ax, fraction=0.046)
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.axis('off')
        axes[i][0].set_ylabel(f'#{i + 1}', fontsize=9)
    fig.suptitle(
        f'Pre-registered sample, seed {manifest["seed"]}, digest {manifest["sha256"][:12]}  '
        f'(shared colormap and depth range per row)', fontsize=10)
    fig.tight_layout()
    maps_pdf = os.path.join(args.output_dir, 'qualitative_depth_maps.pdf')
    fig.savefig(maps_pdf, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', maps_pdf)

    # --- scale-ratio distribution: the real distribution behind the reported medians ---
    hist_pdf = None
    if ratios[labels[0]]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        bins = np.linspace(0.5, 1.6, 45)
        colours = {labels[0]: '#1F6FB2', labels[1]: '#C0392B'}
        for lab in labels:
            ax.hist(ratios[lab], bins=bins, alpha=0.55, color=colours[lab],
                    label=f'{lab} (median {np.median(ratios[lab]):.3f})')
        ax.axvline(1.0, color='k', ls='--', alpha=0.6, label='ideal (1.000)')
        ax.set_xlabel('per-image scale ratio:  median(pred) / median(GT)')
        ax.set_ylabel('number of images')
        ax.legend()
        fig.tight_layout()
        hist_pdf = os.path.join(args.output_dir, 'scale_ratio_distribution.pdf')
        fig.savefig(hist_pdf, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', hist_pdf)

    report = {
        'manifest': manifest,
        'checkpoints': checkpoints,
        'encoder': args.encoder,
        'model_type': args.model_type,
        'figures': {'depth_maps': maps_pdf, 'scale_ratio': hist_pdf},
        'scale_ratio_stats': {
            lab: {
                'n': len(ratios[lab]),
                'median': float(np.median(ratios[lab])) if ratios[lab] else None,
                'mean': float(np.mean(ratios[lab])) if ratios[lab] else None,
            } for lab in labels
        },
    }
    report_path = os.path.join(args.output_dir, 'qualitative_report.json')
    with open(report_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print('Saved:', report_path)
    print('\nCopy the two PDFs into thesis/figures/ and use the snippets in '
          'thesis/figures/pending/qualitative_figures.tex')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    lock = sub.add_parser('lock', help='fix the sample before any inference')
    lock.add_argument('--num-images', type=int, default=6)
    lock.add_argument('--seed', type=int, default=20260821)
    lock.add_argument('--manifest', default='results/qualitative/manifest.json')
    lock.set_defaults(func=cmd_lock)

    render = sub.add_parser('render', help='render exactly the locked sample')
    render.add_argument('--manifest', default='results/qualitative/manifest.json')
    render.add_argument('--checkpoint', action='append', required=True,
                        help='LABEL=PATH; exactly two (e.g. proposed=..., control=...)')
    render.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    render.add_argument('--model-type', default='metric', choices=['metric', 'basic'])
    render.add_argument('--input-size', type=int, default=518)
    render.add_argument('--num-ratio-images', type=int, default=654,
                        help='images used for the scale-ratio histogram (0 disables it)')
    render.add_argument('--device', default=None)
    render.add_argument('--output-dir', default='results/qualitative')
    render.set_defaults(func=cmd_render)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
