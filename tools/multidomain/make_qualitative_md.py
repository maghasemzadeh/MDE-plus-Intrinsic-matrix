#!/usr/bin/env python3
"""Locked qualitative sample for the multi-domain test sets.

`lock` chooses the images from a seed alone (no model is loaded and no depth map
is read) and records a SHA-256 digest of the chosen ids. Predictions are then
dumped for exactly those images with eval_related.py --dump-manifest, and the
ground truth / RGB panels with `dump-inputs`. `make_qualitative_md_render.py`
draws the figures from those files, without a GPU.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from datasets.multidomain import EVAL_READERS  # noqa: E402


def digest(sets):
    lines = [f'{n}:{i}' for n in sorted(sets) for i in sets[n]['ids']]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def cmd_lock(args):
    if os.path.exists(args.manifest):
        raise SystemExit(f'{args.manifest} exists: a locked sample is never overwritten')
    rng = np.random.default_rng(args.seed)
    sets = {}
    for name in args.datasets.split(','):
        reader = EVAL_READERS[name]()
        chosen = rng.choice(len(reader), size=args.per_set, replace=False).tolist()
        sets[name] = {'population': len(reader), 'indices': chosen,
                      'ids': [reader.item_id(i) for i in chosen]}
    manifest = {'protocol': 'ids fixed from the seed before any inference',
                'created': datetime.now().isoformat(timespec='seconds'),
                'seed': args.seed, 'per_set': args.per_set, 'sets': sets,
                'sha256': digest(sets)}
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=1)
    print('SHA-256', manifest['sha256'])
    for name, s in sets.items():
        print(name, s['population'], s['ids'])


def cmd_dump_inputs(args):
    from eval_related import center_zoom
    manifest = json.load(open(args.manifest))
    assert digest(manifest['sets']) == manifest['sha256'], 'manifest was modified after locking'
    for name, s in manifest['sets'].items():
        reader = EVAL_READERS[name]()
        lo, hi = reader.eval_range
        d = os.path.join(args.dump_dir, 'inputs', name)
        os.makedirs(d, exist_ok=True)
        for i in s['indices']:
            rgb, gt, K = reader.load(i)
            gt = np.asarray(gt, dtype=np.float32)
            region = reader.eval_mask(gt.shape)
            valid = np.isfinite(gt) & (gt > lo) & (gt < hi) & region
            np.savez_compressed(os.path.join(d, f'{i}.npz'), rgb=rgb, gt=gt.astype(np.float16),
                                valid=valid)
            z_rgb, z_gt, z_region, _ = center_zoom(rgb, gt, region, np.asarray(K), args.zoom)
            z_valid = np.isfinite(z_gt) & (z_gt > lo) & (z_gt < hi) & z_region
            np.savez_compressed(os.path.join(d, f'{i}_zoom{args.zoom:g}.npz'), rgb=z_rgb,
                                gt=z_gt.astype(np.float16), valid=z_valid)
    print('INPUTS_DONE')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    lock = sub.add_parser('lock')
    lock.add_argument('--seed', type=int, default=20261106)
    lock.add_argument('--per-set', type=int, default=6)
    lock.add_argument('--datasets', default='nyu,sunrgbd_kv1,sunrgbd_kv2,cityscapes')
    lock.add_argument('--manifest', default='results/qualitative_md/manifest.json')
    lock.set_defaults(func=cmd_lock)
    dump = sub.add_parser('dump-inputs')
    dump.add_argument('--manifest', default='results/qualitative_md/manifest.json')
    dump.add_argument('--dump-dir', default='results/qualitative_md')
    dump.add_argument('--zoom', type=float, default=1.5)
    dump.set_defaults(func=cmd_dump_inputs)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
