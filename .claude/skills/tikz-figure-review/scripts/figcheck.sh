#!/usr/bin/env bash
# Render the thesis's TikZ/pgfplots figures on their own, outside the shared build.
#
# usage: figcheck.sh <workdir> [fig:label ...]
#
# First run copies thesis/ into <workdir>/thesis (an isolated copy; edit the figures
# there). Every run compiles only the figure environments that contain a tikzpicture,
# with the real class and preamble, and writes one PNG per figure to <workdir>/out/.
# With labels given, only those figures are compiled.
set -uo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <workdir> [fig:label ...]" >&2; exit 2; }
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$script_dir/../../../.." && pwd)"
work="$(mkdir -p "$1" && cd "$1" && pwd)"; shift
export PATH="$HOME/texlive/2026/bin/x86_64-linux:$PATH"

if [[ ! -d "$work/thesis" ]]; then
    rsync -a --exclude '*.pdf' --exclude 'figures/qualitative_md*' \
        --exclude 'figures/samples' --exclude 'main.*' "$repo/thesis/" "$work/thesis/"
fi
python3 "$script_dir/mkfigs.py" "$work/thesis" "$@" > "$work/labels.txt"
cd "$work/thesis"
if ! xelatex -interaction=nonstopmode -halt-on-error figs.tex > "$work/figs.log" 2>&1; then
    grep -n -A6 "^!" "$work/figs.log" | head -40
    echo "BUILD FAILED (see $work/figs.log)"; exit 1
fi
echo "overfull boxes: $(grep -c Overfull "$work/figs.log")"
mkdir -p "$work/out"
python3 - "$work" <<'PY'
import sys, pymupdf
work = sys.argv[1]
labels = [l.split()[1] for l in open(f"{work}/labels.txt").read().split("\n") if l.strip()]
doc = pymupdf.open(f"{work}/thesis/figs.pdf")
assert len(doc) == len(labels), (len(doc), len(labels))
for page, label in zip(doc, labels):
    box = None
    for block in page.get_text("blocks"):
        r = pymupdf.Rect(block[:4]); box = r if box is None else box | r
    for drawing in page.get_drawings():
        box = drawing["rect"] if box is None else box | drawing["rect"]
    box = (box + (-12, -12, 12, 12)) & page.rect
    name = label.replace("fig:", "")
    page.get_pixmap(dpi=140, clip=box).save(f"{work}/out/{name}.png")
    print(f"{work}/out/{name}.png  {round(box.width)}x{round(box.height)}pt")
PY
