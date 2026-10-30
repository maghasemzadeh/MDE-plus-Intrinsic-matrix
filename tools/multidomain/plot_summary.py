#!/usr/bin/env python3
"""Dot-plot summary of eval_multidomain.py runs: AbsRel per test set for each
model, standard test images vs the centre-zoom camera shift, shared log axis.

    python tools/multidomain/plot_summary.py --runs P=md1_P,C=md1_C,C0=md1_C0 \
        --zoom-suffix _zoom1.5 --out results/multidomain/figures/md1_summary.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'results', 'multidomain')
ROWS = [('nyu', 'NYU'), ('kitti', 'KITTI'), ('cityscapes', 'Cityscapes'),
        ('diode_indoor', 'DIODE indoor'), ('diode_outdoor', 'DIODE outdoor'),
        ('middlebury', 'Middlebury-H'), ('sunrgbd_kv1', 'SUN Kinect v1'),
        ('sunrgbd_kv2', 'SUN Kinect v2'), ('sunrgbd_realsense', 'SUN RealSense'),
        ('sunrgbd_xtion', 'SUN Xtion')]
# reference categorical order, slots 1-3 (validated all-pairs, light surface);
# shapes are the secondary encoding the aqua slot's contrast WARN requires
STYLE = [('#2a78d6', 'o'), ('#eb6834', 's'), ('#1baf7a', '^')]
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def absrel(run):
    s = json.load(open(os.path.join(R, run, 'summary.json')))['results']
    return {k: v['abs_rel'] for k, v in s.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, help='label=run_dir,...')
    p.add_argument('--zoom-suffix', default='_zoom1.5')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    runs = [part.split('=', 1) for part in a.runs.split(',')]
    panels = [('Standard test images', ''), ('Camera shift: centre-zoom ×1.5', a.zoom_suffix)]

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'text.color': INK,
                         'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True, facecolor=SURFACE)
    ys = list(range(len(ROWS)))[::-1]
    for ax, (title, suffix) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        vals = [absrel(run + suffix) for _, run in runs]
        for y, (key, _) in zip(ys, ROWS):
            row = [v[key] for v in vals if key in v]
            ax.plot([min(row), max(row)], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle='round')
        for (label, _), v, (color, marker) in zip(runs, vals, STYLE):
            ax.scatter([v[k] for k, _ in ROWS], ys, s=46, color=color, marker=marker, zorder=3,
                       edgecolors=SURFACE, linewidths=1.2, label=label)
        ax.axhline(6.5, color=INK2, lw=0.6, ls=(0, (2, 3)))
        ax.set_xscale('log')
        ax.set_xticks([0.1, 0.2, 0.4, 0.8])
        ax.set_xticklabels(['0.1', '0.2', '0.4', '0.8'])
        ax.set_xlim(0.075, 1.0)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.tick_params(length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.set_xlabel('AbsRel (log scale, lower is better)')
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([name for _, name in ROWS])
    axes[0].text(0.077, 6.62, 'training cameras', fontsize=7.5, color=INK2, va='bottom')
    axes[0].text(0.077, 6.38, 'zero-shot cameras', fontsize=7.5, color=INK2, va='top')
    axes[1].legend(loc='lower right', frameon=False, fontsize=8.5, handletextpad=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print('saved', a.out)


if __name__ == '__main__':
    main()
