#!/usr/bin/env python3
"""Depth-map and error-map images of the two qualitative slides (no GPU needed).

One row per test set (NYUv2, SUN RGB-D Kinect v2, Cityscapes): the first image of
each set in thesis/figures/qualitative_md_v2/manifest.json, i.e. images selected
after evaluation. Both images have the same size and the same panel boxes, so the
two slides show the same scenes at the same places:

  assets/img/qual-rows.jpg    input | reference depth | depth of the three models
  assets/img/error-rows.jpg   input | reference depth | predicted / reference ratio

Drawing rules are those of tools/multidomain/make_qualitative_md_render.py: one
colour scale per row taken from the reference depth alone, colour linear in
inverse depth, AbsRel of the image in every model panel. The script prints the
column centres and row centres (fractions) used by the slide markup.

  python3 make_qual_images.py --font-dir <folder with Vazirmatn-SemiBold.ttf>
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import LogNorm, Normalize
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'tools', 'multidomain'))
import make_qualitative_md_render as R  # noqa: E402

ROWS = ['nyu', 'sunrgbd_kv2', 'cityscapes']
# layout in pixels of the 1920 x 1080 slide canvas; the files are written at SCALE times that
W, PANELS, GAP, ROW_GAP = 1400, 5, 6, 16
CBAR_GAP, CBAR_W, TICKS_W = 10, 16, 56
SCALE = 2
PANEL_W = (W - GAP * (PANELS - 1) - CBAR_GAP - CBAR_W - TICKS_W) / PANELS


def layout(cases):
    heights = [PANEL_W * c['gt'].shape[0] / c['gt'].shape[1] for c in cases]
    total = sum(heights) + ROW_GAP * (len(cases) - 1)
    tops, y = [], 0.0
    for h in heights:
        tops.append(y)
        y += h + ROW_GAP
    return heights, tops, total


class Canvas:
    def __init__(self, cases):
        self.heights, self.tops, self.h = layout(cases)
        self.fig = plt.figure(figsize=(W / 100, self.h / 100), dpi=100 * SCALE)

    def axes(self, x, top, w, h):
        ax = self.fig.add_axes([x / W, 1 - (top + h) / self.h, w / W, h / self.h])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    def panel(self, row, col):
        return self.axes(col * (PANEL_W + GAP), self.tops[row], PANEL_W, self.heights[row])

    def save(self, path):
        png = path[:-4] + '.png'
        self.fig.savefig(png, dpi=100 * SCALE, facecolor='white')
        plt.close(self.fig)
        Image.open(png).convert('RGB').save(path, quality=92, subsampling=0)
        os.remove(png)


def tag(ax, text, best, fonts):
    ax.text(0.035, 0.06, text, transform=ax.transAxes, ha='left', va='bottom',
            fontproperties=fonts['bold' if best else 'regular'], fontsize=13.5, color='#0d1b33',
            bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.25', fc='white', ec='none', alpha=0.9))


def depth_cmap():
    cmap = plt.get_cmap('Spectral').copy()
    cmap.set_bad(R.NO_GT)
    return cmap


def draw_reference(canvas, r, case, px):
    lo, hi = R.depth_scale(case)
    norm = Normalize(vmin=-1 / lo, vmax=-1 / hi)
    canvas.panel(r, 0).imshow(R.shrink(case['rgb'], px), aspect='auto', interpolation='none')
    gt = R.shrink_masked(case['gt'], case['valid'], px)
    canvas.panel(r, 1).imshow(-1 / gt, cmap=depth_cmap(), norm=norm, aspect='auto', interpolation='none')
    return lo, hi, norm


def draw_depth(cases, out, fonts, px):
    canvas = Canvas(cases)
    for r, case in enumerate(cases):
        lo, hi, norm = draw_reference(canvas, r, case, px)
        scores = {a: case[a + '_absrel'] for a in R.ARMS}
        best = min(scores, key=scores.get)
        for c, arm in enumerate(R.ARMS, start=2):
            ax = canvas.panel(r, c)
            pred = R.shrink(np.maximum(case[arm], 1e-3), px)
            ax.imshow(-1 / pred, cmap=depth_cmap(), norm=norm, aspect='auto', interpolation='none')
            tag(ax, R.fa(scores[arm]), arm == best, fonts)
        h = canvas.heights[r]
        cax = canvas.axes(PANELS * PANEL_W + (PANELS - 1) * GAP + CBAR_GAP, canvas.tops[r] + 0.07 * h, CBAR_W, 0.86 * h)
        cb = canvas.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=depth_cmap()), cax=cax)
        ticks = np.array([lo, np.sqrt(lo * hi), hi])
        cb.set_ticks(-1 / ticks)
        cb.set_ticklabels([R.fa(t, '{:.1f}' if hi < 20 else '{:.0f}') for t in ticks],
                          fontproperties=fonts['regular'], fontsize=12.5, color='#33415c')
        cax.tick_params(length=3, pad=3, width=0.6, color='#33415c')
        cb.outline.set_linewidth(0.5)
    canvas.save(out)
    return canvas


def draw_error(cases, out, fonts, px):
    """Same boxes as the depth image; the colour-bar strip stays empty (one fixed scale, drawn on the slide)."""
    canvas = Canvas(cases)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(R.NO_GT)
    norm = LogNorm(vmin=1 / 1.5, vmax=1.5)
    for r, case in enumerate(cases):
        draw_reference(canvas, r, case, px)
        scores = {a: case[a + '_absrel'] for a in R.ARMS}
        best = min(scores, key=scores.get)
        for c, arm in enumerate(R.ARMS, start=2):
            ratio = np.maximum(case[arm], 1e-3) / np.maximum(case['gt'], 1e-3)
            ax = canvas.panel(r, c)
            ax.imshow(R.shrink_masked(ratio, case['valid'], px), cmap=cmap, norm=norm,
                      aspect='auto', interpolation='none')
            tag(ax, R.fa(scores[arm]), arm == best, fonts)
    canvas.save(out)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=os.path.join(REPO, 'results', 'qualitative_md_v2', 'candidates'))
    p.add_argument('--manifest', default=os.path.join(REPO, 'thesis', 'figures', 'qualitative_md_v2', 'manifest.json'))
    p.add_argument('--out-dir', default=os.path.join(HERE, 'assets', 'img'))
    p.add_argument('--font-dir', required=True, help='folder with Vazirmatn-SemiBold.ttf and Vazirmatn-Bold.ttf')
    args = p.parse_args()

    manifest = json.load(open(args.manifest))
    fonts = {'regular': fm.FontProperties(fname=os.path.join(args.font_dir, 'Vazirmatn-SemiBold.ttf')),
             'bold': fm.FontProperties(fname=os.path.join(args.font_dir, 'Vazirmatn-Bold.ttf'))}
    cases = []
    for name in ROWS:
        index, item_id = manifest['sets'][name]['indices'][0], manifest['sets'][name]['ids'][0]
        case = R.load_case(args.root, name, index, '')
        assert all(case[a + '_id'] == item_id for a in R.ARMS), (name, index)
        cases.append(case)
        print(name, item_id, {a: round(case[a + '_absrel'], 4) for a in R.ARMS})
    px = int(round(PANEL_W * SCALE))
    canvas = draw_depth(cases, os.path.join(args.out_dir, 'qual-rows.jpg'), fonts, px)
    draw_error(cases, os.path.join(args.out_dir, 'error-rows.jpg'), fonts, px)
    print('size (css px)', W, round(canvas.h, 1))
    print('column centres %', [round(100 * (c * (PANEL_W + GAP) + PANEL_W / 2) / W, 2) for c in range(PANELS)])
    print('row centres %', [round(100 * (t + h / 2) / canvas.h, 2) for t, h in zip(canvas.tops, canvas.heights)])


if __name__ == '__main__':
    main()
