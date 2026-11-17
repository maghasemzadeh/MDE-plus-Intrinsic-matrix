#!/usr/bin/env python3
"""Qualitative figures from images CHOSEN AFTER evaluation (not a random sample).

Unlike make_qualitative_md.py, which fixes the images from a seed before any model
is run, this tool ranks the test images by how far the proposed model is ahead of
both other models and lets a person pick the final ones by eye. Figures made this
way are illustrations; they are not evidence of how often the proposed model wins.

  candidates   rank from the per-image results in results/multidomain, keep the
               top N of each set                       -> <root>/manifest.json
  dump-inputs  RGB, ground truth and valid mask of the manifest images
  (predictions: eval_related.py --dump-manifest <root>/manifest.json --only-manifest)
  sheets       contact sheets of all manifest images, for choosing by eye
  final        write the manifest of the chosen images (order = row order)
  render       thesis figures from a final manifest (no GPU)

A candidate must have the proposed model best on AbsRel, RMSE and d1 at once, an
AbsRel of the proposed model no worse than the set median, and enough valid ground
truth. Candidates are ordered by min(AbsRel control, AbsRel released) - AbsRel proposed.
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

SETS = ['nyu', 'sunrgbd_kv1', 'sunrgbd_kv2', 'cityscapes']
RUNS = {'proposed': lambda ds: 'md1_P', 'control': lambda ds: 'md1_C',
        'base': lambda ds: 'released_vkitti_vitl' if ds == 'cityscapes' else 'released_hypersim_vitl'}
PROTOCOL = ('chosen after inference: proposed model best on AbsRel, RMSE and d1, ranked by '
            'AbsRel margin over both other models, final images picked by eye')


def digest(sets):
    lines = [f'{n}:{i}' for n in sorted(sets) for i in sets[n]['ids']]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def per_image(results, arm, ds):
    with open(os.path.join(results, RUNS[arm](ds), f'{ds}.json')) as f:
        return {m['id']: m for m in json.load(f)['per_image']}


def write_manifest(path, sets, extra):
    manifest = {'protocol': PROTOCOL, 'created': datetime.now().isoformat(timespec='seconds'),
                **extra, 'sets': sets, 'sha256': digest(sets)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=1)


def cmd_candidates(args):
    from datasets.multidomain import EVAL_READERS
    sets = {}
    for ds in args.datasets.split(','):
        P, C, B = (per_image(args.results, a, ds) for a in ('proposed', 'control', 'base'))
        ids = [i for i in P if i in C and i in B]
        p_med = np.median([P[i]['abs_rel'] for i in ids])
        nv_min = np.percentile([P[i]['n_valid'] for i in ids], args.min_valid_percentile)
        rows = []
        for i in ids:
            p, c, b = P[i], C[i], B[i]
            if not (p['abs_rel'] < min(c['abs_rel'], b['abs_rel'])
                    and p['rmse'] < min(c['rmse'], b['rmse'])
                    and p['d1'] >= max(c['d1'], b['d1'])):
                continue
            if p['abs_rel'] > p_med or p['n_valid'] < nv_min:
                continue
            rows.append((min(c['abs_rel'], b['abs_rel']) - p['abs_rel'], i))
        rows.sort(reverse=True)
        rows = rows[:args.per_set]
        reader = EVAL_READERS[ds]()
        index_of = {reader.item_id(k): k for k in range(len(reader))}
        sets[ds] = {'population': len(reader), 'eligible': len(rows),
                    'indices': [index_of[i] for _, i in rows], 'ids': [i for _, i in rows],
                    'margin': [round(m, 4) for m, _ in rows]}
        print(ds, len(rows), 'candidates, margin', sets[ds]['margin'][0], '..', sets[ds]['margin'][-1])
    write_manifest(args.manifest, sets, {'stage': 'candidates', 'per_set': args.per_set})


def cmd_dump_inputs(args):
    from datasets.multidomain import EVAL_READERS
    manifest = json.load(open(args.manifest))
    root = os.path.dirname(os.path.abspath(args.manifest))
    for name, s in manifest['sets'].items():
        reader = EVAL_READERS[name]()
        lo, hi = reader.eval_range
        d = os.path.join(root, 'inputs', name)
        os.makedirs(d, exist_ok=True)
        for i in s['indices']:
            rgb, gt, _ = reader.load(i)
            gt = np.asarray(gt, dtype=np.float32)
            valid = np.isfinite(gt) & (gt > lo) & (gt < hi) & reader.eval_mask(gt.shape)
            np.savez_compressed(os.path.join(d, f'{i}.npz'), rgb=rgb, gt=gt.astype(np.float16),
                                valid=valid)
    print('INPUTS_DONE')


def fonts_from(font_dir):
    from matplotlib import font_manager as fm
    files = {'regular': 'IRLotusICEE.ttf', 'bold': 'IRLotusICEE_Bold.ttf',
             'latin': 'LiberationSerif-Regular.ttf'}
    return {k: fm.FontProperties(fname=os.path.join(font_dir, v)) for k, v in files.items()}


def check_against_tables(manifest, root, results):
    """Dumped predictions must be the ones behind the tables: same per-image AbsRel."""
    worst = 0.0
    for ds, s in manifest['sets'].items():
        for arm in RUNS:
            table = per_image(results, arm, ds)
            for index, item_id in zip(s['indices'], s['ids']):
                d = np.load(os.path.join(root, 'pred', arm, ds, f'{index}.npz'))
                assert str(d['id']) == item_id, (ds, arm, index)
                worst = max(worst, abs(float(d['abs_rel']) - table[item_id]['abs_rel']))
    print(f'max |AbsRel dump - AbsRel table| = {worst:.2e}')
    return worst


def cmd_sheets(args):
    import make_qualitative_md_render as R
    manifest = json.load(open(args.manifest))
    root = os.path.dirname(os.path.abspath(args.manifest))
    check_against_tables(manifest, root, args.results)
    fonts = fonts_from(args.font_dir)
    out_dir = os.path.join(root, 'sheets')
    os.makedirs(out_dir, exist_ok=True)
    for ds, s in manifest['sets'].items():
        for start in range(0, len(s['indices']), args.rows):
            cases, labels = [], []
            for rank in range(start, min(start + args.rows, len(s['indices']))):
                cases.append(R.load_case(root, ds, s['indices'][rank], ''))
                labels.append((len(cases) - 1, len(cases) - 1, str(rank)))
            out = os.path.join(out_dir, f'{ds}_{start:02d}.png')
            R.draw_depth_figure(cases, labels, out, fonts, args.width_px)
            print(out)


def cmd_final(args):
    cand = json.load(open(args.candidates))
    sets = {}
    for spec in args.pick:
        ds, ranks = spec.split('=')
        ranks = [int(r) for r in ranks.split(',')]
        s = cand['sets'][ds]
        sets[ds] = {'population': s['population'], 'indices': [s['indices'][r] for r in ranks],
                    'ids': [s['ids'][r] for r in ranks], 'margin': [s['margin'][r] for r in ranks],
                    'candidate_rank': ranks}
    write_manifest(args.manifest, sets, {'stage': 'final', 'per_set': len(ranks),
                                         'candidates_sha256': cand['sha256']})
    print(json.dumps({k: v['ids'] for k, v in sets.items()}, indent=1))


def cmd_render(args):
    import make_qualitative_md_render as R
    manifest = json.load(open(args.manifest))
    assert digest(manifest['sets']) == manifest['sha256']
    root = args.root or os.path.dirname(os.path.abspath(args.manifest))
    worst = check_against_tables(manifest, root, args.results)
    fonts = fonts_from(args.font_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    n = manifest['per_set']
    figures = {
        'qual_md_nyu': (R.draw_depth_figure, [('nyu', n)]),
        'qual_md_kv1': (R.draw_depth_figure, [('sunrgbd_kv1', n)]),
        'qual_md_kv2': (R.draw_depth_figure, [('sunrgbd_kv2', n)]),
        'qual_md_cityscapes': (R.draw_depth_figure, [('cityscapes', n)]),
        'qual_md_error': (R.draw_error_figure, [(ds, 2) for ds in SETS]),
    }
    geometry = {}
    for stem, (draw, spec) in figures.items():
        cases, labels, meta = R.collect(manifest, root, spec, '')
        out = os.path.join(args.out_dir, stem + '.pdf')
        grid = draw(cases, labels, out, fonts, args.width_px)
        geometry[stem] = {'centres': [round(c, 4) for c in grid.centres],
                          'height_over_width': round(grid.total_in / R.FIG_W, 4),
                          'rows': meta, 'rule': spec}
        print(stem, f'{os.path.getsize(out) / 1e6:.1f} MB', 'centres', geometry[stem]['centres'])
    with open(os.path.join(args.out_dir, 'geometry.json'), 'w') as f:
        json.dump({'protocol': manifest['protocol'], 'sha256': manifest['sha256'],
                   'max_absrel_difference_to_tables': worst, 'figures': geometry}, f, indent=1)
    with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    root = 'results/qualitative_md_v2'

    c = sub.add_parser('candidates')
    c.add_argument('--results', default='results/multidomain')
    c.add_argument('--datasets', default=','.join(SETS))
    c.add_argument('--per-set', type=int, default=20)
    c.add_argument('--min-valid-percentile', type=float, default=40)
    c.add_argument('--manifest', default=f'{root}/candidates/manifest.json')
    c.set_defaults(func=cmd_candidates)

    d = sub.add_parser('dump-inputs')
    d.add_argument('--manifest', default=f'{root}/candidates/manifest.json')
    d.set_defaults(func=cmd_dump_inputs)

    s = sub.add_parser('sheets')
    s.add_argument('--manifest', default=f'{root}/candidates/manifest.json')
    s.add_argument('--results', default='results/multidomain')
    s.add_argument('--font-dir', default='thesis/font')
    s.add_argument('--rows', type=int, default=5)
    s.add_argument('--width-px', type=int, default=400)
    s.set_defaults(func=cmd_sheets)

    f = sub.add_parser('final')
    f.add_argument('--candidates', default=f'{root}/candidates/manifest.json')
    f.add_argument('--manifest', default=f'{root}/candidates/final_manifest.json')
    f.add_argument('pick', nargs='+', help='set=rank,rank,... (candidate ranks, in row order)')
    f.set_defaults(func=cmd_final)

    r = sub.add_parser('render')
    r.add_argument('--manifest', default=f'{root}/candidates/final_manifest.json')
    r.add_argument('--root', help='folder with inputs/ and pred/ (default: next to the manifest)')
    r.add_argument('--results', default='results/multidomain')
    r.add_argument('--out-dir', default='thesis/figures/qualitative_md_v2')
    r.add_argument('--font-dir', default='thesis/font')
    r.add_argument('--width-px', type=int, default=400)
    r.set_defaults(func=cmd_render)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
