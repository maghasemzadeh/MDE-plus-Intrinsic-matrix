"""
Compare per-iteration train/loss between a revised/base checkpoint pair's
TensorBoard event files. With --seed both arms saw identical batches, so the
per-iter losses are directly comparable pairwise.

Usage:
    python tools/compare_pair_losses.py <pair_dir> [--last-frac 0.33]

<pair_dir> contains one revised_* and one base_* run directory (train.py
layout). Prints head/tail means and the pairwise win rate of revised over
base in the last fraction of iterations.
"""

import argparse
import glob
import os
import sys

import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_loss(run_dir):
    ev_files = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
    if not ev_files:
        raise FileNotFoundError(f'no event file in {run_dir}')
    acc = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    acc.Reload()
    if 'train/loss' not in acc.Tags()['scalars']:
        raise KeyError(f"train/loss not in {run_dir}: {acc.Tags()['scalars']}")
    events = acc.Scalars('train/loss')
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    order = np.argsort(steps)
    return steps[order], vals[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_dir')
    parser.add_argument('--last-frac', type=float, default=0.33)
    args = parser.parse_args()

    revised_dirs = sorted(glob.glob(os.path.join(args.pair_dir, 'revised_*')))
    base_dirs = sorted(glob.glob(os.path.join(args.pair_dir, 'base_*')))
    if not revised_dirs or not base_dirs:
        print(f'need one revised_* and one base_* dir under {args.pair_dir}')
        sys.exit(1)
    r_steps, r_loss = load_loss(revised_dirs[-1])
    b_steps, b_loss = load_loss(base_dirs[-1])

    n = min(len(r_loss), len(b_loss))
    r_loss, b_loss = r_loss[:n], b_loss[:n]
    r_steps = r_steps[:n]
    if not np.array_equal(r_steps, b_steps[:n]):
        print('WARNING: step indices differ between arms; comparison is approximate')

    k = max(1, int(n * args.last_frac))
    head = slice(0, k)
    tail = slice(n - k, n)

    def stats(x, sl):
        return float(np.mean(x[sl]))

    print(f'iterations compared: {n}')
    print(f'{"":<12} {"first-third mean":>18} {"last-third mean":>18}')
    print(f'{"revised":<12} {stats(r_loss, head):>18.4f} {stats(r_loss, tail):>18.4f}')
    print(f'{"base":<12} {stats(b_loss, head):>18.4f} {stats(b_loss, tail):>18.4f}')
    diff = b_loss[tail] - r_loss[tail]
    win = float(np.mean(diff > 0))
    print(f'\nlast-third: revised better on {win*100:.1f}% of iters, '
          f'mean(base - revised) = {np.mean(diff):+.4f}')
    finite = np.isfinite(r_loss).all() and np.isfinite(b_loss).all()
    decreasing = stats(r_loss, tail) < stats(r_loss, head) and stats(b_loss, tail) < stats(b_loss, head)
    print(f'losses finite: {finite}; both arms decreasing: {decreasing}')


if __name__ == '__main__':
    main()
