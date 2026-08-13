#!/usr/bin/env bash

set -euo pipefail

thesis_dir="$(cd "$(dirname "$0")" && pwd)"
build_dir="$thesis_dir/../.build/thesis"

# \include writes per-chapter auxiliary files below tex/, so mirror that
# directory in the build tree before compiling.
mkdir -p "$build_dir/tex"

# BibTeX resolves the document's `./tex/MyReferences` path relative to the
# auxiliary directory, so expose the bibliography at the matching build path.
cp "$thesis_dir/tex/MyReferences.bib" "$build_dir/tex/MyReferences.bib"

cd "$thesis_dir"
build_status=0
latexmk -bibtex -pdf main.tex || build_status=$?

# Keep the convenient, human-facing PDF beside main.tex. All other generated
# files remain under .build/thesis and are ignored by Git. Copy a newly emitted
# PDF even when LaTeX reports source errors, while preserving the failing status.
if [[ -f "$build_dir/main.pdf" ]]; then
    cp "$build_dir/main.pdf" "$thesis_dir/main.pdf"
fi

exit "$build_status"
