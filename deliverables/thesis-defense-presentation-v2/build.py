#!/usr/bin/env python3
"""Assemble index.html from src/shell.html, src/slides/*.html and content/fa.json.

Every Persian string on the slides lives in content/fa.json and is referenced from
the slide partials as {{key}}.  Run:  python3 build.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fa = json.loads((ROOT / "content" / "fa.json").read_text(encoding="utf8"))

parts = [fa[f"part.{i}"] for i in range(1, 8)]
config = {"parts": parts, "backupLabel": fa["b.label"], "partLabel": fa["ui.part"], "of": "از"}

slides = "".join(p.read_text(encoding="utf8") for p in sorted((ROOT / "src" / "slides").glob("*.html")))
html = (ROOT / "src" / "shell.html").read_text(encoding="utf8").replace("<!--SLIDES-->", slides)
html = html.replace("{{CONFIG}}", json.dumps(config, ensure_ascii=False))
# a slide whose notes have a quick-glance summary (key <slide>.sum) also carries it, for the presenter's summary switch
html = re.sub(r'<aside class="notes">\{\{([A-Za-z0-9_]+)\.notes\}\}</aside>',
              lambda m: m.group(0) + (f'<aside class="notes-sum">{{{{{m.group(1)}.sum}}}}</aside>' if f"{m.group(1)}.sum" in fa else ""),
              html)
template = html

missing = set()


def sub(m):
    key = m.group(1)
    if key not in fa:
        missing.add(key)
        return m.group(0)
    return fa[key].replace("&", "&amp;").replace("<", "&lt;")


html = re.sub(r"\{\{([A-Za-z0-9_.]+)\}\}", sub, html)
if missing:
    sys.exit("missing content keys: " + ", ".join(sorted(missing)))

(ROOT / "index.html").write_text(html, encoding="utf8")
n_main = len(re.findall(r'<section class="slide(?![^"]*backup)', html))
n_backup = len(re.findall(r'<section class="slide[^"]*backup', html))
unused = sorted(k for k in fa if "{{" + k + "}}" not in template and not k.startswith(("part.", "ui.", "meta.", "b.label")))
print(f"index.html written: {n_main} main slides + {n_backup} backup slides")
if unused:
    print("unused content keys:", ", ".join(unused))
