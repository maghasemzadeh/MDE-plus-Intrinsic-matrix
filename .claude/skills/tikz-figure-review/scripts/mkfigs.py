#!/usr/bin/env python3
"""Extract every figure environment that contains a tikzpicture into figs.tex."""
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
only = set(sys.argv[2:])
out = [r"""\documentclass[twoside,msc,irfonts]{./kntu-thesis}
\input{./tex/commands}
\input{./tex/faTitle}
\input{./tex/enTitle}
\input{./tex/glossaries-settings}
\input{./tex/words}
\input{./tex/acronyms.tex}
\begin{document}
\pagestyle{empty}
"""]
labels = []
for ch in ["chapter1", "chapter2", "chapter3", "chapter4"]:
    src = (root / "tex" / f"{ch}.tex").read_text(encoding="utf-8")
    for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", src, re.S):
        blk = m.group(0)
        if "tikzpicture" not in blk:
            continue
        lab = re.search(r"\\label\{(fig:[^}]*)\}", blk).group(1)
        if only and lab not in only:
            continue
        blk = re.sub(r"\\begin\{figure\}\[[^\]]*\]", r"\\begin{figure}[p]", blk)
        labels.append(lab)
        out.append("%% " + lab + "\n" + blk + "\n\\clearpage\n")
out.append("\\end{document}\n")
(root / "figs.tex").write_text("\n".join(out), encoding="utf-8")
print("\n".join(f"{i+1} {l}" for i, l in enumerate(labels)))
