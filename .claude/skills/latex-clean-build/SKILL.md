---
name: latex-clean-build
description: Configure or repair clean, out-of-tree LaTeX builds so auxiliary files do not clutter or get archived with a thesis, paper, report, or book source tree. Use when users ask to move TeX compile artifacts such as .aux, .log, .toc, .synctex.gz, bibliography, index, acronym, or glossary outputs elsewhere; keep only a convenient PDF; add Git ignore rules; clean existing generated files; or align latexmk, build scripts, and editor recipes with a separate output directory.
---

# LaTeX Clean Build

Keep generated LaTeX state outside the document source tree while preserving the project's existing engine, bibliography, index, glossary, acronym, and editor workflows.

## Inspect before editing

1. Read applicable `CLAUDE.md` files and inspect Git status. Preserve unrelated user changes.
2. Locate the root TeX document, `latexmkrc` or `.latexmkrc`, build scripts or Makefiles, editor settings, bibliography files, and custom index/glossary commands.
3. List existing generated files and check whether any are tracked. Do not assume every PDF is generated; figures and guides may be source assets.
4. Infer the requested PDF policy: output directory only, copied beside the root TeX file, tracked, or ignored. If the user wants a convenient PDF but does not specify, copy the final document PDF beside the root TeX file and ignore that one generated PDF.

## Configure the build

Prefer `latexmk` for orchestration when the project already uses it or has multiple passes.

1. Choose an ignored output directory outside the document source folder, normally a repository-level `.build/<document>/`.
2. Set both latexmk output and auxiliary directories to it. Account for the invocation working directory: relative `latexmkrc` paths are resolved from where latexmk runs, not necessarily from the rc file location.
3. Add or update a small build wrapper that:
   - resolves its own absolute directory;
   - creates the output directory and any subdirectories required by `\include` auxiliary files;
   - runs from the source directory so relative class, font, image, and TeX paths still work;
   - invokes latexmk with the project's existing engine and custom dependencies;
   - copies only the final PDF to the requested convenient location;
   - quotes every path and works when repository paths contain spaces;
   - returns the real build status. If TeX emits a usable PDF despite source errors, it may copy that PDF before returning failure.
4. Preserve existing BibTeX/Biber, MakeIndex, Xindy, glossary, and acronym behavior. Out-of-tree builds can change their working directory. If an auxiliary file contains an explicit relative bibliography path such as `./tex/refs`, make that path resolvable from the output directory or safely adjust the bibliography search/configuration; verify it with a real build.
5. Update editor recipes when necessary so they call latexmk or the wrapper. Warn that a direct `xelatex main.tex` command bypasses the clean configuration and can recreate clutter.

Do not replace a working custom multi-pass setup with a generic command unless the resulting build is verified to retain all outputs.

## Ignore generated output

Ignore the chosen output directory at repository scope. Also add narrowly scoped fallback patterns under the document tree for common by-products in case an editor bypasses the wrapper, including relevant forms of:

`.acn`, `.acr`, `.alg`, `.aux`, `.bbl`, `.bcf`, `.blg`, `.blo`, `.bls`, `.brf`, `.fdb_latexmk`, `.fls`, `.glo`, `.glg`, `.gls`, `.idx`, `.ilg`, `.ind`, `.ist`, `.loa`, `.lof`, `.log`, `.lol`, `.lot`, `.mw`, `.out`, `.run.xml`, `.synctex.gz`, `.tdo`, `.toc`, and generated `.xdy`.

Scope PDF ignores to the generated main PDF. Never use a broad `*.pdf` rule in a LaTeX project because source figures, guides, and included papers may be PDFs.

Remember that `.gitignore` does not untrack files already committed. Report tracked generated files and only untrack them when that action is within the user's request.

## Clean safely

Enumerate exact candidates before deletion. Remove only known generated artifacts inside the document tree, preserve the requested PDF, and leave source assets untouched. Treat user-authored or ambiguous files as source. State that deleted build artifacts are reproducible by rebuilding.

## Verify

1. Run shell/config syntax checks and `git diff --check` scoped to files changed by this task.
2. Run a real build when the toolchain is available.
3. Confirm auxiliary files are produced only in the external output directory and the final PDF is where promised.
4. Use `git check-ignore -v` on representative auxiliary, output-directory, and PDF paths.
5. Inspect Git status and ensure unrelated source modifications remain untouched.
6. Separate build-layout failures from pre-existing TeX content errors. Report both accurately; do not fix document content unless asked.

In the handoff, give the exact build command, output location, PDF location and Git policy, cleanup performed and recoverability, verification result, and any unrelated TeX errors that still prevent a clean exit.
