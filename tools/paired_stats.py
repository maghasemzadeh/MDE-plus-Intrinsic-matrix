"""
Paired per-image statistics between two evaluate_paper.py output JSONs.

Matches per-image metrics by item_id and reports, per metric: means, paired
mean difference with a bootstrap 95% CI, and the Wilcoxon signed-rank p-value.

Usage:
    python tools/paired_stats.py A.json B.json --label-a revised --label-b control
"""

import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

METRIC_KEYS = ['abs_rel', 'rmse', 'd1', 'silog', 'log10', 'rmse_log']
HIGHER_BETTER = {'d1', 'd2', 'd3'}


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {row['item_id']: row for row in data['per_image']}


def bootstrap_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_a')
    parser.add_argument('json_b')
    parser.add_argument('--label-a', default='A')
    parser.add_argument('--label-b', default='B')
    args = parser.parse_args()

    a, b = load(args.json_a), load(args.json_b)
    common = sorted(set(a) & set(b))
    print(f'paired images: {len(common)} '
          f'(A has {len(a)}, B has {len(b)})\n')
    print(f'{"metric":<10} {args.label_a:>10} {args.label_b:>10} {"diff(A-B)":>11} '
          f'{"95% CI":>22} {"wilcoxon p":>11}  verdict')
    print('-' * 90)
    for key in METRIC_KEYS:
        va = np.array([a[i][key] for i in common], dtype=np.float64)
        vb = np.array([b[i][key] for i in common], dtype=np.float64)
        diff = va - vb
        lo, hi = bootstrap_ci(diff)
        try:
            p = wilcoxon(va, vb).pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            p = float('nan')
        sig = p < 0.05 and (lo > 0 or hi < 0)
        if not sig:
            verdict = 'no significant difference'
        else:
            a_better = (diff.mean() > 0) == (key in HIGHER_BETTER)
            verdict = f'{args.label_a if a_better else args.label_b} better'
        print(f'{key:<10} {va.mean():>10.4f} {vb.mean():>10.4f} {diff.mean():>+11.4f} '
              f'[{lo:>+9.4f},{hi:>+9.4f}] {p:>11.2e}  {verdict}')


if __name__ == '__main__':
    main()
