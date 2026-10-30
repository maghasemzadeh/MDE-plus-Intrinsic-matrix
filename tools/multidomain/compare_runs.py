#!/usr/bin/env python3
"""Paired comparison of eval_multidomain.py result directories.

For every dataset present in both directories, images are matched by id and
the per-image metrics compared: means, mean difference with a 95% bootstrap
CI, Wilcoxon signed-rank p-value and the fraction of images the first run
wins. Prints a markdown table; --json also writes it machine-readable.

    python tools/multidomain/compare_runs.py RUN_A RUN_B [--names A,B]
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

LOWER_BETTER = {'abs_rel', 'rmse', 'sq_rel', 'rmse_log', 'log10', 'silog'}


def load(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.json') and fn != 'summary.json':
            r = json.load(open(os.path.join(d, fn)))
            out[r['dataset']] = {m['id']: m for m in r['per_image']}
    return out


def boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_a')
    p.add_argument('run_b')
    p.add_argument('--names', default='A,B')
    p.add_argument('--metrics', default='abs_rel,rmse,d1,silog')
    p.add_argument('--json', default=None)
    a = p.parse_args()
    na, nb = a.names.split(',')
    ra, rb = load(a.run_a), load(a.run_b)
    rows, report = [], {}
    print(f"| dataset | n | metric | {na} | {nb} | diff ({na}-{nb}) [95% CI] | Wilcoxon p | {na} wins |")
    print("|---|---:|---|---:|---:|---|---:|---:|")
    for ds in [d for d in ra if d in rb]:
        ids = sorted(set(ra[ds]) & set(rb[ds]))
        report[ds] = {'n': len(ids)}
        for k in a.metrics.split(','):
            xa = np.array([ra[ds][i][k] for i in ids])
            xb = np.array([rb[ds][i][k] for i in ids])
            diff = xa - xb
            lo, hi = boot_ci(diff)
            try:
                pval = float(wilcoxon(xa, xb).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                pval = float('nan')
            better_a = (diff < 0) if k in LOWER_BETTER else (diff > 0)
            win = float(better_a.mean())
            report[ds][k] = {'a': float(xa.mean()), 'b': float(xb.mean()), 'diff': float(diff.mean()),
                             'ci': [float(lo), float(hi)], 'p': pval, 'a_win_frac': win}
            print(f"| {ds} | {len(ids)} | {k} | {xa.mean():.4f} | {xb.mean():.4f} | "
                  f"{diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {pval:.2e} | {win:.0%} |")
    if a.json:
        with open(a.json, 'w') as f:
            json.dump({'a': a.run_a, 'b': a.run_b, 'names': [na, nb], 'results': report}, f, indent=1)


if __name__ == '__main__':
    main()
