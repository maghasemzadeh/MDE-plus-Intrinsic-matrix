#!/usr/bin/env python3
"""Draw the thesis qualitative figures from the locked multi-domain sample.

Needs no GPU: reads results/qualitative_md/{manifest.json,inputs,pred} written by
make_qualitative_md.py and eval_related.py. The depth figures show every locked
image of a set; the error and camera-shift figures show the first two of each set
in locked draw order. Counts over all locked images go to geometry.json.

In every row, the ground truth and all models share one colour scale, computed
from the ground truth alone. Colour is linear in inverse depth, as in the Depth
Anything figures. Each prediction panel carries that image's AbsRel, in bold
when it is the lowest of the row. Column headers are typeset by LaTeX at the
fractions printed by this script and stored in geometry.json.
"""
import argparse
import hashlib
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import LogNorm, Normalize

matplotlib.rcParams['pdf.fonttype'] = 42

ARMS = ['base', 'control', 'proposed']
SET_NAMES = {'nyu': 'NYUv2', 'sunrgbd_kv1': 'Kinect v1', 'sunrgbd_kv2': 'Kinect v2',
             'cityscapes': 'Cityscapes'}
FIG_W = 6.2           # inches; matches \textwidth closely, the PDF is scaled to it
LABEL_W, CBAR_W, CBAR_TICKS_W, GAP = 0.030, 0.012, 0.042, 0.004
NO_GT = '#D9D9D9'
FA = str.maketrans('0123456789.', '۰۱۲۳۴۵۶۷۸۹٫')


def fa(value, fmt='{:.3f}'):
    return fmt.format(value).translate(FA)


def load_case(root, name, index, suffix):
    inp = np.load(os.path.join(root, 'inputs', name, f'{index}{suffix}.npz'))
    case = {'rgb': inp['rgb'], 'gt': inp['gt'].astype(np.float32), 'valid': inp['valid']}
    for arm in ARMS:
        d = np.load(os.path.join(root, 'pred', arm, name, f'{index}{suffix}.npz'))
        case[arm] = d['pred'].astype(np.float32)
        case[arm + '_absrel'] = float(d['abs_rel'])
        case[arm + '_id'] = str(d['id'])
    return case


def shrink(img, width_px):
    h, w = img.shape[:2]
    if w <= width_px:
        return img
    return cv2.resize(img, (width_px, int(round(h * width_px / w))), interpolation=cv2.INTER_AREA)


def shrink_masked(values, valid, width_px):
    """Area-average valid pixels only, so masks do not bleed into the values."""
    filled = np.where(valid, values, 0.0).astype(np.float32)
    weight = shrink(valid.astype(np.float32), width_px)
    small = shrink(filled, width_px)
    return np.where(weight > 0.25, small / np.maximum(weight, 1e-6), np.nan)


class Grid:
    """Rows of equal-width panels; row height follows that row's image aspect."""

    def __init__(self, cases, n_panels):
        self.n_panels = n_panels
        self.panel_w = (1 - LABEL_W - CBAR_W - CBAR_TICKS_W - GAP * (n_panels + 1)) / n_panels
        panel_in = self.panel_w * FIG_W
        heights = [panel_in * c['gt'].shape[0] / c['gt'].shape[1] for c in cases]
        gap_in = GAP * FIG_W
        total = sum(heights) + gap_in * (len(cases) - 1)
        self.fig = plt.figure(figsize=(FIG_W, total))
        self.rows, y = [], total
        for h in heights:
            y -= h
            self.rows.append((y / total, h / total))
            y -= gap_in
        self.total_in = total
        self.centres = [LABEL_W + j * (self.panel_w + GAP) + self.panel_w / 2 for j in range(n_panels)]

    def panel(self, row, col):
        bottom, h = self.rows[row]
        ax = self.fig.add_axes([LABEL_W + col * (self.panel_w + GAP), bottom, self.panel_w, h])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    def colorbar_axis(self, row):
        bottom, h = self.rows[row]
        x = LABEL_W + self.n_panels * (self.panel_w + GAP)
        return self.fig.add_axes([x, bottom + 0.06 * h, CBAR_W, 0.88 * h])

    def group_label(self, first, last, text, font):
        top = self.rows[first][0] + self.rows[first][1]
        centre = (top + self.rows[last][0]) / 2
        self.fig.text(LABEL_W * 0.42, centre, text, rotation=90, ha='center', va='center',
                      fontproperties=font, fontsize=8)


def tag(ax, text, best, fonts):
    ax.text(0.03, 0.05, text, transform=ax.transAxes, ha='left', va='bottom',
            fontproperties=fonts['bold' if best else 'regular'], fontsize=7.5, color='black',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))


def depth_scale(case):
    """Row colour scale from the ground truth alone: never from a prediction."""
    g = case['gt'][case['valid']]
    return float(np.percentile(g, 1)), float(np.percentile(g, 99))


def draw_depth_figure(cases, labels, out, fonts, width_px):
    grid = Grid(cases, 5)
    cmap = plt.get_cmap('Spectral').copy()
    cmap.set_bad(NO_GT)
    for r, case in enumerate(cases):
        lo, hi = depth_scale(case)
        # Spectral runs red -> blue with increasing value: inverse depth is negated
        # so near is red and far is blue, as in the Depth Anything figures.
        norm = Normalize(vmin=-1 / lo, vmax=-1 / hi)
        grid.panel(r, 0).imshow(shrink(case['rgb'], width_px), aspect='auto', interpolation='none')
        gt = shrink_masked(case['gt'], case['valid'], width_px)
        grid.panel(r, 1).imshow(-1 / gt, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        scores = {a: case[a + '_absrel'] for a in ARMS}
        best = min(scores, key=scores.get)
        for c, arm in enumerate(ARMS, start=2):
            ax = grid.panel(r, c)
            pred = shrink(np.maximum(case[arm], 1e-3), width_px)
            ax.imshow(-1 / pred, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
            tag(ax, fa(scores[arm]), arm == best, fonts)
        cax = grid.colorbar_axis(r)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = grid.fig.colorbar(sm, cax=cax)
        ticks = np.array([lo, np.sqrt(lo * hi), hi])
        cb.set_ticks(-1 / ticks)
        cb.set_ticklabels([fa(t, '{:.1f}' if hi < 20 else '{:.0f}') for t in ticks],
                          fontproperties=fonts['regular'], fontsize=6.5)
        cax.tick_params(length=1.5, pad=1, width=0.4)
        cb.outline.set_linewidth(0.3)
    for first, last, text in labels:
        grid.group_label(first, last, text, fonts['latin'])
    grid.fig.savefig(out, dpi=300)
    plt.close(grid.fig)
    return grid


def draw_error_figure(cases, labels, out, fonts, width_px):
    """RGB + signed pred/GT ratio on valid pixels, one fixed scale for all rows."""
    grid = Grid(cases, 4)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(NO_GT)
    norm = LogNorm(vmin=1 / 1.5, vmax=1.5)
    for r, case in enumerate(cases):
        grid.panel(r, 0).imshow(shrink(case['rgb'], width_px), aspect='auto', interpolation='none')
        scores = {a: case[a + '_absrel'] for a in ARMS}
        best = min(scores, key=scores.get)
        for c, arm in enumerate(ARMS, start=1):
            ratio = np.maximum(case[arm], 1e-3) / np.maximum(case['gt'], 1e-3)
            ax = grid.panel(r, c)
            ax.imshow(shrink_masked(ratio, case['valid'], width_px), cmap=cmap, norm=norm,
                      aspect='auto', interpolation='none')
            tag(ax, fa(scores[arm]), arm == best, fonts)
    cax = grid.fig.add_axes([LABEL_W + 4 * (grid.panel_w + GAP), 0.25, CBAR_W, 0.5])
    cb = grid.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = [1 / 1.5, 0.8, 1.0, 1.25, 1.5]
    cb.minorticks_off()
    cb.set_ticks(ticks)
    cb.set_ticklabels([fa(t, '{:.2f}') for t in ticks], fontproperties=fonts['regular'], fontsize=6.5)
    cax.tick_params(length=1.5, pad=1, width=0.4, which='both')
    cb.outline.set_linewidth(0.3)
    for first, last, text in labels:
        grid.group_label(first, last, text, fonts['latin'])
    grid.fig.savefig(out, dpi=300)
    plt.close(grid.fig)
    return grid


def collect(manifest, root, spec, suffix=''):
    """spec: [(set, k)]. Takes the first k images of each set in locked order."""
    cases, labels, meta = [], [], []
    for name, k in spec:
        s = manifest['sets'][name]
        first = len(cases)
        for index, item_id in list(zip(s['indices'], s['ids']))[:k]:
            case = load_case(root, name, index, suffix)
            assert all(case[a + '_id'] == item_id for a in ARMS), (name, index)
            cases.append(case)
            meta.append({'set': name, 'index': index, 'id': item_id,
                         **{a: round(case[a + '_absrel'], 4) for a in ARMS}})
        labels.append((first, len(cases) - 1, SET_NAMES[name]))
    return cases, labels, meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='results/qualitative_md')
    p.add_argument('--out-dir', default='thesis/figures/qualitative_md')
    p.add_argument('--font-dir', default='thesis/font')
    p.add_argument('--width-px', type=int, default=400, help='stored width of each panel')
    args = p.parse_args()

    manifest = json.load(open(os.path.join(args.root, 'manifest.json')))
    lines = [f'{n}:{i}' for n in sorted(manifest['sets']) for i in manifest['sets'][n]['ids']]
    assert hashlib.sha256('\n'.join(lines).encode()).hexdigest() == manifest['sha256'], \
        'manifest was modified after locking'
    fonts = {
        'regular': fm.FontProperties(fname=os.path.join(args.font_dir, 'IRLotusICEE.ttf')),
        'bold': fm.FontProperties(fname=os.path.join(args.font_dir, 'IRLotusICEE_Bold.ttf')),
        'latin': fm.FontProperties(fname=os.path.join(args.font_dir, 'LiberationSerif-Regular.ttf')),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    two_each = [('nyu', 2), ('sunrgbd_kv1', 2), ('sunrgbd_kv2', 2), ('cityscapes', 2)]
    figures = {
        'qual_md_nyu': (draw_depth_figure, [('nyu', 6)], ''),
        'qual_md_kv1': (draw_depth_figure, [('sunrgbd_kv1', 6)], ''),
        'qual_md_kv2': (draw_depth_figure, [('sunrgbd_kv2', 6)], ''),
        'qual_md_cityscapes': (draw_depth_figure, [('cityscapes', 6)], ''),
        'qual_md_error': (draw_error_figure, two_each, ''),
        'qual_md_shift': (draw_depth_figure, two_each, '_zoom1.5'),
    }
    geometry = {'all_locked': {}}
    for view, suffix in (('standard', ''), ('shift', '_zoom1.5')):
        _, _, meta = collect(manifest, args.root, [(n, manifest['per_set']) for n in manifest['sets']], suffix)
        per_set = {}
        for name in manifest['sets']:
            rows = [m for m in meta if m['set'] == name]
            per_set[name] = {a: sum(min(ARMS, key=lambda x: m[x]) == a for m in rows) for a in ARMS}
        total = {a: sum(per_set[n][a] for n in per_set) for a in ARMS}
        geometry['all_locked'][view] = {'lowest_absrel_per_set': per_set, 'lowest_absrel_total': total}
        print(view, 'lowest AbsRel over all locked images', total, per_set)
    for stem, (draw, spec, suffix) in figures.items():
        cases, labels, meta = collect(manifest, args.root, spec, suffix)
        out = os.path.join(args.out_dir, stem + '.pdf')
        grid = draw(cases, labels, out, fonts, args.width_px)
        wins = {a: sum(min(ARMS, key=lambda x: m[x]) == a for m in meta) for a in ARMS}
        geometry[stem] = {'centres': [round(c, 4) for c in grid.centres],
                          'height_over_width': round(grid.total_in / FIG_W, 4),
                          'rows': meta, 'lowest_absrel_rows': wins, 'rule': spec, 'view': suffix or 'standard'}
        print(stem, f'{os.path.getsize(out) / 1e6:.1f} MB', 'centres', geometry[stem]['centres'],
              'h/w', geometry[stem]['height_over_width'], 'wins', wins)
    all_locked = geometry.pop('all_locked')
    with open(os.path.join(args.out_dir, 'geometry.json'), 'w') as f:
        json.dump({'seed': manifest['seed'], 'sha256': manifest['sha256'],
                   'all_locked': all_locked, 'figures': geometry}, f, indent=1)


if __name__ == '__main__':
    main()
