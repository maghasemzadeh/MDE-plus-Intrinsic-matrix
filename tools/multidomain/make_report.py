#!/usr/bin/env python3
"""Markdown report over eval_multidomain.py result directories.

    python tools/multidomain/make_report.py \
        --runs P=results/multidomain/md1_P,C=...,C0=...,DA2-hypersim=...,DA2-vkitti=... \
        --pairs P:C,P:C0 --out results/multidomain/REPORT_md1.md

Table 1: AbsRel / delta1 / RMSE of every run on every dataset (absolute metric
depth, no alignment). Table 2: paired per-image tests for each requested pair
(mean difference, Wilcoxon signed-rank p, share of images won).
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ORDER = ['nyu', 'kitti', 'cityscapes', 'diode_indoor', 'diode_outdoor', 'middlebury',
         'sunrgbd_kv1', 'sunrgbd_kv2', 'sunrgbd_realsense', 'sunrgbd_xtion']
ROLE = {'nyu': 'indoor, train camera', 'kitti': 'outdoor, train camera',
        'cityscapes': 'outdoor, train camera', 'diode_indoor': 'indoor, zero-shot',
        'diode_outdoor': 'outdoor, zero-shot', 'middlebury': 'indoor, zero-shot',
        'sunrgbd_kv1': 'indoor, zero-shot', 'sunrgbd_kv2': 'indoor, zero-shot',
        'sunrgbd_realsense': 'indoor, zero-shot', 'sunrgbd_xtion': 'indoor, zero-shot'}


def load_run(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if fn.endswith('.json') and fn != 'summary.json':
            r = json.load(open(os.path.join(d, fn)))
            out[r['dataset']] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, help='name=dir,name=dir,...')
    p.add_argument('--pairs', default='', help='A:B,A:C (paired tests)')
    p.add_argument('--title', default='Multi-domain metric depth')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    runs = {}
    for part in a.runs.split(','):
        name, d = part.split('=', 1)
        runs[name] = load_run(d)
    datasets = [d for d in ORDER if any(d in r for r in runs.values())]
    lines = [f'# {a.title}', '', 'Absolute metric depth, no per-image alignment. '
             'Cells: AbsRel / δ1 / RMSE (m). Best AbsRel per row in bold.', '']
    lines.append('| dataset | role | n | ' + ' | '.join(runs) + ' |')
    lines.append('|---|---|---:|' + '---|' * len(runs))
    for ds in datasets:
        cells, absrels = [], []
        for name, r in runs.items():
            if ds in r:
                s = r[ds]['summary']
                absrels.append(s['abs_rel'])
                cells.append(f"{s['abs_rel']:.3f} / {s['d1']:.3f} / {s['rmse']:.2f}")
            else:
                absrels.append(np.inf)
                cells.append('–')
        best = int(np.argmin(absrels))
        cells[best] = f'**{cells[best]}**'
        n = next(r[ds]['summary']['n_images'] for r in runs.values() if ds in r)
        lines.append(f"| {ds} | {ROLE.get(ds, '')} | {n} | " + ' | '.join(cells) + ' |')

    for pair in [x for x in a.pairs.split(',') if x]:
        A, B = pair.split(':')
        lines += ['', f'## Paired: {A} vs {B}', '',
                  f'| dataset | n | AbsRel {A} | AbsRel {B} | Δ AbsRel | p (Wilcoxon) | {A} wins | '
                  f'δ1 {A} | δ1 {B} | RMSE {A} | RMSE {B} | p RMSE |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        wins = 0
        for ds in datasets:
            if ds not in runs[A] or ds not in runs[B]:
                continue
            ia = {m['id']: m for m in runs[A][ds]['per_image']}
            ib = {m['id']: m for m in runs[B][ds]['per_image']}
            ids = sorted(set(ia) & set(ib))
            xa = np.array([ia[i]['abs_rel'] for i in ids])
            xb = np.array([ib[i]['abs_rel'] for i in ids])
            ra = np.array([ia[i]['rmse'] for i in ids])
            rb = np.array([ib[i]['rmse'] for i in ids])
            da = np.mean([ia[i]['d1'] for i in ids])
            db = np.mean([ib[i]['d1'] for i in ids])
            pa = wilcoxon(xa, xb).pvalue if np.any(xa != xb) else 1.0
            pr = wilcoxon(ra, rb).pvalue if np.any(ra != rb) else 1.0
            wins += xa.mean() < xb.mean()
            lines.append(f'| {ds} | {len(ids)} | {xa.mean():.4f} | {xb.mean():.4f} | '
                         f'{xa.mean() - xb.mean():+.4f} | {pa:.1e} | {np.mean(xa < xb):.0%} | '
                         f'{da:.3f} | {db:.3f} | {ra.mean():.3f} | {rb.mean():.3f} | {pr:.1e} |')
        lines.append('')
        lines.append(f'{A} has the lower mean AbsRel on {wins} of '
                     f'{sum(ds in runs[A] and ds in runs[B] for ds in datasets)} datasets.')
    text = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    open(a.out, 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
