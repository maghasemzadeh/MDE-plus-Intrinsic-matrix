"""Extract TensorBoard scalars into thesis-ready CSV and a pgfplots snippet.

The thesis describes a silent training collapse (section 3.7.4 / 3.9.8): the
loss curve looks entirely normal while the head collapses to a constant zero
output, and only the frozen validation metric reveals it.  That story is far
more convincing as a plotted curve than as prose, but the run logs live on the
server.  This script turns them into vector-plottable data.

Usage (on lingotube, venv active, repository root)::

    # list what is available first
    python tools/tensorboard_to_thesis.py --runs runs/ --list

    # export selected runs/tags
    python tools/tensorboard_to_thesis.py \
        --runs runs/ \
        --run collapsed=vitl_basic_depth_target \
        --run fixed=vitl_basic_disparity_target \
        --tag train/loss --tag val/absrel \
        --output-dir thesis/figures/curves

Produces, per tag, a CSV (step,value per run) and a ready-to-paste pgfplots
`\\addplot table` block that uses the thesis colour palette.  The CSVs are small
text files, so they can be committed and the figure rebuilt without the server.

Only depends on tensorboard's own event reader; no torch import required.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

PALETTE = ['thProposed', 'thControl', 'thAccent', 'thIdeal', 'thBase']


def _load_reader():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        return EventAccumulator
    except ImportError:
        raise SystemExit(
            'tensorboard is not installed in this environment.\n'
            'Install it with:  pip install tensorboard')


def _accumulate(path, EventAccumulator):
    acc = EventAccumulator(path, size_guidance={'scalars': 0})
    acc.Reload()
    return acc


def _find_runs(root):
    """Return {run_name: event_dir} for every directory holding event files."""
    runs = OrderedDict()
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(f.startswith('events.out.tfevents') for f in filenames):
            name = os.path.relpath(dirpath, root)
            runs[name if name != '.' else os.path.basename(os.path.abspath(root))] = dirpath
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', default='runs', help='root directory holding TensorBoard runs')
    parser.add_argument('--list', action='store_true', help='list runs and available scalar tags, then exit')
    parser.add_argument('--run', action='append', default=[],
                        help='LABEL=RUN_NAME to export; repeatable. Default: every run found.')
    parser.add_argument('--tag', action='append', default=[],
                        help='scalar tag to export; repeatable. Default: every shared tag.')
    parser.add_argument('--output-dir', default='thesis/figures/curves')
    args = parser.parse_args()

    EventAccumulator = _load_reader()
    if not os.path.isdir(args.runs):
        raise SystemExit(f'runs directory not found: {args.runs}')

    found = _find_runs(args.runs)
    if not found:
        raise SystemExit(f'no TensorBoard event files under {args.runs}')

    if args.list:
        for name, path in found.items():
            acc = _accumulate(path, EventAccumulator)
            tags = acc.Tags().get('scalars', [])
            print(f'\n{name}\n  {path}')
            for t in tags:
                print(f'    {t}  ({len(acc.Scalars(t))} points)')
        return

    if args.run:
        selected = OrderedDict()
        for spec in args.run:
            if '=' not in spec:
                raise SystemExit(f'--run must be LABEL=RUN_NAME, got: {spec}')
            label, name = spec.split('=', 1)
            if name not in found:
                raise SystemExit(f'run not found: {name}\navailable: {", ".join(found)}')
            selected[label] = found[name]
    else:
        selected = OrderedDict((n, p) for n, p in found.items())

    accs = {label: _accumulate(path, EventAccumulator) for label, path in selected.items()}

    if args.tag:
        tags = args.tag
    else:
        tagsets = [set(a.Tags().get('scalars', [])) for a in accs.values()]
        tags = sorted(set.intersection(*tagsets)) if tagsets else []
        if not tags:
            raise SystemExit('no scalar tag is common to all selected runs; pass --tag explicitly')

    os.makedirs(args.output_dir, exist_ok=True)

    for tag in tags:
        slug = tag.replace('/', '_').replace(' ', '_')
        rows = {}
        for label, acc in accs.items():
            if tag not in acc.Tags().get('scalars', []):
                print(f'  (skipping {label}: tag {tag} absent)')
                continue
            for ev in acc.Scalars(tag):
                rows.setdefault(ev.step, {})[label] = ev.value
        if not rows:
            continue

        labels = [l for l in accs if any(l in v for v in rows.values())]
        csv_path = os.path.join(args.output_dir, f'{slug}.csv')
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=' ')
            writer.writerow(['step'] + labels)
            for step in sorted(rows):
                writer.writerow([step] + [rows[step].get(l, 'nan') for l in labels])
        print('Saved:', csv_path)

        tex_path = os.path.join(args.output_dir, f'{slug}.tex')
        with open(tex_path, 'w') as fh:
            fh.write('%% تولیدشده توسط tools/tensorboard_to_thesis.py -- در فصل مربوطه \\input کنید\n')
            fh.write('\\begin{figure}[htbp]\n    \\centering\n    \\begin{tikzpicture}\n')
            fh.write('    \\begin{axis}[\n        thesisplot,\n')
            fh.write('        xlabel={گام آموزش},\n')
            fh.write('        ylabel={%s},\n' % tag.replace('_', r'\_'))
            fh.write('        legend style={at={(0.97,0.97)}, anchor=north east},\n    ]\n')
            for i, label in enumerate(labels):
                colour = PALETTE[i % len(PALETTE)]
                fh.write('    \\addplot[draw=%s, mark=none] table[x=step, y=%s, col sep=space]\n'
                         '        {figures/curves/%s};\n' % (colour, label, os.path.basename(csv_path)))
                fh.write('    \\addlegendentry{%s}\n' % label)
            fh.write('    \\end{axis}\n    \\end{tikzpicture}\n')
            fh.write('    \\caption{}\n    \\label{fig:curve_%s}\n\\end{figure}\n' % slug)
        print('Saved:', tex_path)

    print('\nCopy the CSVs next to the .tex snippets under thesis/figures/curves/ '
          'and \\input the snippet where you want the figure.')


if __name__ == '__main__':
    main()
