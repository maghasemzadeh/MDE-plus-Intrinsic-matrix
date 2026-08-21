#!/usr/bin/env python3
"""Structural checks on the thesis sources that do not need a LaTeX run.

Catches the failures that actually break this document's build: a figure whose
image file was never produced, a \\ref with no \\label, a duplicated label, an
unbalanced environment, and a tabular row whose cell count disagrees with its
column specification.

    python3 tools/check_thesis_structure.py
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
THESIS = REPO / "thesis"
GRAPHICS_EXT = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")
# mirrors \graphicspath in tex/commands.tex
GRAPHICS_ROOTS = (THESIS / "img",)

COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:ref|autoref|eqref|cref|Cref|pageref)\{([^}]*)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
BEGIN_RE = re.compile(r"\\begin\{([^}]*)\}")
END_RE = re.compile(r"\\end\{([^}]*)\}")
TABULAR_RE = re.compile(
    r"\\begin\{(tabular|tabularx)\}\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\}\s*)?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)


class ThesisStructureChecker:
    """Collects structural problems across every .tex source of the thesis."""

    def __init__(self):
        self.problems = []
        self.labels = defaultdict(list)
        self.refs = []

    @staticmethod
    def strip(text):
        return COMMENT_RE.sub("", text)

    @staticmethod
    def sources():
        files = [THESIS / "main.tex"]
        files += sorted((THESIS / "tex").glob("*.tex"))
        files += sorted((THESIS / "figures").rglob("*.tex"))
        return [f for f in files if f.exists()]

    def report(self, path, message):
        self.problems.append(f"{path.relative_to(REPO)}: {message}")

    def check_environments(self, path, text):
        stack = []
        for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", text):
            kind, name = m.group(1), m.group(2)
            line = text[: m.start()].count("\n") + 1
            if kind == "begin":
                stack.append((name, line))
            else:
                if not stack:
                    self.report(path, f"line {line}: \\end{{{name}}} with no matching \\begin")
                elif stack[-1][0] != name:
                    open_name, open_line = stack[-1]
                    self.report(path, f"line {line}: \\end{{{name}}} closes \\begin{{{open_name}}} from line {open_line}")
                    stack.pop()
                else:
                    stack.pop()
        for name, line in stack:
            self.report(path, f"line {line}: \\begin{{{name}}} never closed")

    @staticmethod
    def resolve_graphic(target):
        """Mimic graphicx lookup: relative to thesis/, then each \\graphicspath root,
        trying the known extensions case-insensitively (the template ships .PNG)."""
        target = target.strip().lstrip("./")
        for root in (THESIS, *GRAPHICS_ROOTS):
            for ext in GRAPHICS_EXT:
                candidate = root / (target + ext)
                if candidate.exists():
                    return True
                parent = candidate.parent
                if parent.is_dir():
                    wanted = candidate.name.lower()
                    if any(f.name.lower() == wanted for f in parent.iterdir()):
                        return True
        return False

    def check_graphics(self, path, text):
        for m in GRAPHIC_RE.finditer(text):
            line = text[: m.start()].count("\n") + 1
            if not self.resolve_graphic(m.group(1)):
                self.report(path, f"line {line}: missing image target '{m.group(1)}'")

    def check_inputs(self, path, text):
        for m in INPUT_RE.finditer(text):
            target = m.group(1).strip().lstrip("./")
            line = text[: m.start()].count("\n") + 1
            if not any((THESIS / (target + ext)).exists() for ext in ("", ".tex")):
                self.report(path, f"line {line}: missing \\input target '{m.group(1)}'")

    def check_tabulars(self, path, text):
        for m in TABULAR_RE.finditer(text):
            spec, body = m.group(2), m.group(3)
            cols = len(re.findall(r"[lcrXpmb]", re.sub(r"\{[^{}]*\}", "", spec)))
            if cols == 0:
                continue
            start_line = text[: m.start()].count("\n") + 1
            for row in body.split(r"\\"):
                clean = re.sub(r"\\multicolumn\s*\{(\d+)\}\s*\{[^{}]*\}\s*\{[^{}]*\}", lambda r: "&" * (int(r.group(1)) - 1), row)
                clean = re.sub(r"\\(hline|toprule|midrule|bottomrule|cmidrule|rowcolor)\b[^&]*", "", clean)
                if not clean.strip() or clean.strip().startswith("%"):
                    continue
                n = clean.count("&") - clean.count(r"\&") + 1
                if n > cols:
                    self.report(path, f"line ~{start_line}: row has {n} cells but the column spec allows {cols}")
                    break

    def run(self):
        for path in self.sources():
            text = self.strip(path.read_text(encoding="utf-8"))
            self.check_environments(path, text)
            self.check_graphics(path, text)
            self.check_inputs(path, text)
            self.check_tabulars(path, text)
            for m in LABEL_RE.finditer(text):
                self.labels[m.group(1)].append(path.relative_to(REPO).as_posix())
            for m in REF_RE.finditer(text):
                self.refs.append((m.group(1), path.relative_to(REPO).as_posix(),
                                  text[: m.start()].count("\n") + 1))

        for name, where in sorted(self.labels.items()):
            if len(where) > 1:
                self.problems.append(f"duplicate \\label{{{name}}} in {', '.join(where)}")
        for name, where, line in self.refs:
            if name not in self.labels:
                self.problems.append(f"{where}: line {line}: \\ref to undefined label '{name}'")

        print(f"Checked {len(self.sources())} source file(s); "
              f"{len(self.labels)} labels, {len(self.refs)} references.")
        if not self.problems:
            print("OK: no structural problem found.")
            return 0
        print(f"\n{len(self.problems)} problem(s):\n")
        for problem in self.problems:
            print(f"  {problem}")
        return 1


if __name__ == "__main__":
    sys.exit(ThesisStructureChecker().run())
