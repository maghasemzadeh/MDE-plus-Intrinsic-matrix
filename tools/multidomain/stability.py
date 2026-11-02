#!/usr/bin/env python3
"""Cross-dataset stability of eval_multidomain.py runs.

Dataset-level view (one AbsRel per test set) plus a paired bootstrap over
images within each set. Reports, per run: mean / std / coefficient of
variation / worst-case AbsRel across sets, and the paired per-set change
between a "standard" and a "shifted" (center-zoom) evaluation of the same
checkpoint. With --ref, also the per-set ratio run/ref and a bootstrap CI of
the mean log-ratio over sets (resampling both sets and images).

    python tools/multidomain/stability.py RUN[:ZOOMRUN] ... [--ref RUN] [--json out]
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon


def load(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.json') and fn != 'summary.json':
            r = json.load(open(os.path.join(d, fn)))
            out[r['dataset']] = np.array([m['abs_rel'] for m in r['per_image']]), [m['id'] for m in r['per_image']]
    return out


def means(run, sets):
    return np.array([run[s][0].mean() for s in sets])


def sig_counts(a, b, sets, alpha=0.01):
    """Sets where a is significantly better / worse than b on AbsRel
    (paired Wilcoxon on shared image ids, two-sided, alpha) and not different."""
    better = worse = 0
    for s in sets:
        ia = dict(zip(a[s][1], a[s][0]))
        ib = dict(zip(b[s][1], b[s][0]))
        ids = sorted(set(ia) & set(ib))
        x = np.array([ia[i] for i in ids])
        y = np.array([ib[i] for i in ids])
        if wilcoxon(x, y).pvalue < alpha:
            if x.mean() < y.mean():
                better += 1
            else:
                worse += 1
    return better, worse


def boot_logratio(a, b, sets, n=5000, seed=0):
    """Mean over sets of log(AbsRel_a / AbsRel_b); resample sets, then images
    (paired by id) within each chosen set."""
    rng = np.random.default_rng(seed)
    paired = {}
    for s in sets:
        ia = {k: v for k, v in zip(a[s][1], a[s][0])}
        ib = {k: v for k, v in zip(b[s][1], b[s][0])}
        ids = sorted(set(ia) & set(ib))
        paired[s] = (np.array([ia[i] for i in ids]), np.array([ib[i] for i in ids]))
    vals = []
    for _ in range(n):
        chosen = rng.choice(len(sets), len(sets))
        lr = []
        for c in chosen:
            x, y = paired[sets[c]]
            j = rng.integers(0, len(x), len(x))
            lr.append(np.log(x[j].mean() / y[j].mean()))
        vals.append(np.mean(lr))
    return np.percentile(vals, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='RUN_DIR or RUN_DIR:SHIFTED_RUN_DIR')
    p.add_argument('--ref')
    p.add_argument('--json')
    a = p.parse_args()

    runs = {}
    for spec in a.runs:
        std, _, shift = spec.partition(':')
        runs[os.path.basename(std)] = (load(std), load(shift) if shift else None)
    ref = load(a.ref) if a.ref else None
    sets = sorted(set.intersection(*[set(r[0]) for r in runs.values()] + ([set(ref)] if ref else [])))
    print('sets (%d): %s\n' % (len(sets), ', '.join(sets)))

    res = {}
    print('| run | mean AbsRel | std | CV | worst set | mean shifted | mean shift change |')
    print('|---|---:|---:|---:|---|---:|---:|')
    for name, (std, shift) in runs.items():
        m = means(std, sets)
        row = dict(mean=m.mean(), std=m.std(ddof=1), cv=m.std(ddof=1) / m.mean(),
                   worst=sets[int(m.argmax())], worst_val=m.max(), per_set=dict(zip(sets, m)))
        s_txt, c_txt = '-', '-'
        if shift:
            ms = means(shift, sets)
            row.update(shift_mean=ms.mean(), shift_change=float(np.mean((ms - m) / m)),
                       shift_per_set=dict(zip(sets, ms)))
            s_txt, c_txt = '%.3f' % ms.mean(), '%+.1f%%' % (100 * row['shift_change'])
        res[name] = row
        print('| %s | %.3f | %.3f | %.2f | %s (%.3f) | %s | %s |' % (
            name, row['mean'], row['std'], row['cv'], row['worst'], row['worst_val'], s_txt, c_txt))

    if ref:
        print('\nper-set AbsRel ratio run/ref (<1 = run better), bootstrap CI of mean log-ratio over sets+images')
        mr = means(ref, sets)
        for name, (std, _) in runs.items():
            ratio = means(std, sets) / mr
            lo, hi = boot_logratio(std, ref, sets)
            res[name]['vs_ref'] = dict(ratio=dict(zip(sets, ratio)), geo_mean=float(np.exp(np.mean(np.log(ratio)))),
                                       ci=[float(np.exp(lo)), float(np.exp(hi))], wins=int((ratio < 1).sum()))
            sb, sw = sig_counts(std, ref, sets)
            res[name]['vs_ref'].update(sig_better=sb, sig_worse=sw)
            print('%s: geometric-mean ratio %.3f [%.3f, %.3f], lower mean on %d/%d sets, '
                  'significantly better %d / worse %d (Wilcoxon p<0.01)' % (
                name, res[name]['vs_ref']['geo_mean'], np.exp(lo), np.exp(hi), (ratio < 1).sum(), len(sets), sb, sw))
            print('   ' + ', '.join('%s %.2f' % (s, r) for s, r in zip(sets, ratio)))
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
