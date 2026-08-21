#!/usr/bin/env python3
"""Guard: reported numbers in the thesis must not change across edits.

Compares every numeric token in the thesis .tex sources between a git
reference (default: HEAD) and the working tree, ignoring numbers that carry
no result meaning (citation keys, labels/refs, figure paths, English gloss
footnotes, LaTeX lengths).  Exits non-zero if any surviving number differs.

    python tools/check_thesis_numbers.py                # working tree vs HEAD
    python tools/check_thesis_numbers.py --ref bee2892  # vs another commit
    python tools/check_thesis_numbers.py --verbose      # list every number
"""
import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GLOBS = ["thesis/tex/*.tex", "thesis/main.tex", "thesis_sections.tex"]

# Digit families that show up in a Persian thesis.
DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})
DIGIT_MAP[ord("٫")] = "."  # Arabic decimal separator

# Commands whose *arguments* are bookkeeping, not results.
IGNORED_ARG_CMDS = (
    r"cite|citep|citet|nocite|ref|autoref|eqref|pageref|cref|Cref|label|"
    r"enfootnote|glsadd|gls|acrshort|acrlong|includegraphics|input|include|"
    r"bibliography|usepackage|documentclass|hypersetup|url|href"
)
CMD_ARG_RE = re.compile(r"\\(?:%s)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}" % IGNORED_ARG_CMDS)
OPT_ARG_RE = re.compile(r"\\includegraphics\s*\[[^\]]*\]")
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
# LaTeX lengths / spacing / column specs carry no result meaning.
LENGTH_RE = re.compile(
    r"-?\d*\.?\d+\s*(?:pt|cm|mm|em|ex|in|bp|sp|\\textwidth|\\linewidth|\\baselineskip)"
)
# Drawing geometry inside tikz/pgfplots: (x,y) coordinates and [option=val] keys
# are layout, not results.  Node/label text in {...} is kept.
TIKZ_ENV_RE = re.compile(
    r"\\begin\{(tikzpicture|axis|scope|pgfpicture)\}.*?\\end\{\1\}", re.DOTALL
)
COORD_RE = re.compile(r"\(\s*-?\d[^()]*?\)")
OPTS_RE = re.compile(r"\[[^\[\]{}]*\]")
# Model / dataset / architecture names carry version digits that are not results.
NAME_RE = re.compile(
    r"Depth[\s-]*Anything[\s-]*V?\d+|DepthAnythingV\d+|\bDA[-\s]?[23]\b|"
    r"DINOv\d+|NYU(?:\s?Depth)?[\s-]?[vV]?\d+|VKITTI\s?\d+|KITTI\s?\d+|"
    r"ViT-?[A-Za-z]\d*|MiDaS\s?v?\d*|ResNet-?\d+|EfficientNet-?[Bb]\d+|"
    r"COCO\s?\d{4}|ImageNet-?\d*[Kk]?|\b[23]D\b|CVPR\s?\d{4}|ICCV\s?\d{4}|ECCV\s?\d{4}",
    re.IGNORECASE,
)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _strip_geometry(m):
    body = m.group(0)
    body = OPTS_RE.sub(" ", body)
    body = COORD_RE.sub(" ", body)
    return body


def strip_noise(text: str) -> str:
    text = text.translate(DIGIT_MAP)
    text = COMMENT_RE.sub("", text)
    for _ in range(3):  # nested/adjacent commands
        text = CMD_ARG_RE.sub(" ", text)
    text = OPT_ARG_RE.sub(" ", text)
    text = TIKZ_ENV_RE.sub(_strip_geometry, text)
    text = NAME_RE.sub(" ", text)
    text = LENGTH_RE.sub(" ", text)
    return text


def numbers(text: str):
    return NUM_RE.findall(strip_noise(text))


def numbers_with_context(text: str):
    stripped = strip_noise(text)
    out = []
    for m in NUM_RE.finditer(stripped):
        lo = max(0, m.start() - 45)
        ctx = stripped[lo:m.end() + 45].replace("\n", " ")
        out.append((m.group(), " ".join(ctx.split())))
    return out


def git_show(ref: str, relpath: str):
    r = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:{relpath}"],
        capture_output=True, text=True,
    )
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="HEAD", help="git ref to compare against (default HEAD)")
    ap.add_argument("--files", nargs="*", default=None, help="explicit files instead of the defaults")
    ap.add_argument("--verbose", action="store_true", help="print every number found per file")
    ap.add_argument(
        "--allow-additions", action="store_true",
        help="pass as long as no existing number was removed or altered "
             "(new numbers from newly written content are permitted)",
    )
    args = ap.parse_args()

    if args.files:
        paths = [Path(f).resolve() for f in args.files]
    else:
        paths = sorted({p for g in DEFAULT_GLOBS for p in REPO.glob(g)})

    changed, checked, missing = [], 0, []
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        old_text = git_show(args.ref, rel)
        if old_text is None:
            missing.append(rel)
            continue
        new_text = path.read_text(encoding="utf-8")
        old, new = numbers(old_text), numbers(new_text)
        checked += 1
        if args.verbose:
            print(f"  {rel}: {len(new)} numbers")
        if args.allow_additions:
            sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
            lost = [old[i] for tag, i1, i2, _, _ in sm.get_opcodes()
                    if tag in ("delete", "replace") for i in range(i1, i2)]
            if lost:
                changed.append((rel, old, new, new_text))
        elif old != new:
            changed.append((rel, old, new, new_text))

    print(f"Checked {checked} file(s) against {args.ref}.")
    for rel in missing:
        print(f"  note: {rel} not present at {args.ref} (new file, skipped)")

    if not changed:
        print("OK: no existing number was removed or altered."
              if args.allow_additions else "OK: no reported number changed.")
        return 0

    print(f"\nNUMBERS CHANGED in {len(changed)} file(s):\n")
    for rel, old, new, new_text in changed:
        print(f"--- {rel}")
        for line in difflib.unified_diff(old, new, args.ref, "worktree", lineterm="", n=1):
            print(f"    {line}")
        ctx = dict(numbers_with_context(new_text))
        for n in set(new) - set(old):
            if n in ctx:
                print(f"    context for new {n!r}: …{ctx[n]}…")
        print()
    print("Review each change: it must be an intentional, re-validated result.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
