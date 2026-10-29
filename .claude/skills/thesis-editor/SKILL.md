---
name: thesis-editor
description: Edit the Persian LaTeX thesis source (without rendering the PDF) when the user requests a thesis change, an academic or professional revision, or a thesis update based on project outputs or results. Apply the relevant LaTeX, academic-writing, and figure-making skills, while making only the requested thesis changes. Read project evidence as needed, but write only under thesis/; never edit thesis-gemini/ or other project folders.
---

# Thesis editor

Use this skill for every request that changes the thesis or asks to incorporate the
project's outputs, results, metrics, figures, or experiment conclusions into the
thesis. It also applies when the user asks to validate, commit, or push a
thesis update.

## Scope and safety

- Treat `thesis/` as the only editable source tree. The source files are primarily
  `thesis/main.tex` and `thesis/tex/*.tex`, with thesis-owned assets under
  `thesis/img/`, `thesis/figures/`, and `thesis/code/`.
- `thesis-gemini/` is out of scope and must never be edited, staged, or committed by
  this skill. Do not use it as a substitute source tree.
- Read `outputs/`, `results/`, `logs/`, `tests/`, and project code when they are
  evidence for a thesis update, but treat them as read-only inputs. Do not rewrite
  experiment results or generated project artifacts as part of a thesis edit.
- Before editing, record `git status --short` and inspect the relevant diff. Preserve
  unrelated pre-existing changes, especially changes outside `thesis/`; never use
  `git add -A`, `git clean`, reset, or force-push to hide them.
- If the requested change explicitly targets `thesis-gemini/` or another folder, say
  that it is outside this skill's scope and leave that folder unchanged.

## Required supporting skills

Before editing, identify every installed supporting skill that applies to the
requested operation and follow its `SKILL.md` instructions alongside this skill.
Use the following routing:

- `paper-writing-codex-skill` for compile-safe LaTeX revision, formulas, captions,
  and paper/thesis editing workflows.
- `latex-clean-build` for clean, out-of-tree LaTeX builds and generated-file
  hygiene when build configuration or compilation artifacts are involved.
- `scientific-writing` for research-backed academic prose, claims, citations,
  structure, and publication-style wording.
- `academic-humanizer` when the user asks to remove AI-like patterns while
  preserving scientific claims, numbers, citations, and uncertainty. Use the
  generic `humanizer` skill only when its broader prose rewrite is specifically
  requested; do not apply either skill to unrelated text.
- `academic-paper-review` when the request is to review readiness, weaknesses,
  novelty, rigor, or revision priorities rather than to make a narrowly specified
  source edit.
- `tikz-diagrams` for LaTeX-native diagrams, plots, schematics, and other
  vector figures that belong in the thesis source.
- `excalidraw-diagram` for conceptual or workflow diagrams when Excalidraw is the
  appropriate deliverable.
- `imagegen` only when a requested figure genuinely requires a raster illustration,
  texture, or other bitmap asset; use the thesis figure directory and preserve
  provenance where applicable.

Use all applicable skills when several apply, but do not invoke unrelated skills or
load overlapping prose skills without a concrete reason. When instructions differ,
the user's request and the narrowest task-specific constraint take precedence.

## Change isolation and concurrent edits

- Convert the user's request into an explicit list of requested changes before
  writing. Edit only the files and lines necessary for those items, plus the
  minimum directly required LaTeX dependencies (for example, a label/reference,
  glossary entry, or figure asset required to compile the requested change).
- Do not perform opportunistic cleanup, reformatting, terminology-wide rewrites,
  citation changes, renaming, or unrelated fixes. Preserve all other thesis
  content exactly as it was.
- Record the initial status and relevant diff, then re-check the target file's
  current contents immediately before applying a patch. If another person or
  agent has changed the file during the task, reload the latest version and apply
  a minimal patch on top of it; never overwrite or discard their work. If the
  concurrent edit conflicts with the requested change and cannot be resolved
  safely, stop that edit and report the conflict rather than guessing.
- After editing, inspect `git diff -- thesis` and verify that every hunk serves an
  item in the request or is a directly necessary compile dependency. Correct any
  accidental unrelated hunk before building. Preserve unrelated pre-existing
  changes and never use reset, clean, or forceful overwrite commands to conceal
  them.

## Academic and professional editing

1. Locate the authoritative evidence before changing a claim. For numerical results,
   record the experiment/configuration, split, sample count, metric definition,
   uncertainty or statistical test, and the file or command that produced it. Never
   invent a value or upgrade an association into a causal claim.
2. Keep the argument precise and reproducible: distinguish relative from metric
   depth, report the evaluation protocol, preserve units and sample sizes, and cite
   the appropriate bibliography entry. Use a neutral academic tone, remove
   unsupported superlatives, and keep terminology consistent across chapters,
   captions, tables, abstract, and conclusion.
3. When introducing or renaming a term, update the thesis glossary/acronym source
   (`thesis/tex/words.tex` or `thesis/tex/acronyms.tex`) when appropriate. Search the
   whole `thesis/` tree for stale names, old numbers, duplicate claims, and broken
   references before building.
4. Express mathematics with the document's existing LaTeX conventions: use proper
   math delimiters and environments, define symbols before use, and add or preserve
   `\label`/`\ref` links for equations, figures, and tables. Keep formulas in LaTeX
   rather than substituting Unicode operators or prose approximations.

## Figures and diagrams

When the requested change needs a new or revised figure, create or edit it with the
appropriate supporting figure skill rather than approximating it in prose. Prefer
`tikz-diagrams` for figures that should compile as part of the LaTeX source; use
`excalidraw-diagram` for conceptual or workflow artwork; use `imagegen` only for a
bitmap asset that cannot reasonably be represented as LaTeX/vector source. Keep
source and generated assets under the thesis-owned directories, do not fabricate
experimental data, and update only the caption, label, references, and placement
that the requested figure requires. Check the figure source, caption, label, and placement in the
`.tex` (the user will render the PDF).

## Persian and template invariants

Preserve the working template and its options unless the requested change truly
requires a template change. In particular, do not replace XeLaTeX with pdfLaTeX,
remove `irfonts`, reorder packages in `thesis/tex/commands.tex`, or bypass the
existing `xepersian`/`xindy` setup. Reuse the RTL/LTR direction helpers and macros
already defined there, including project macros such as `\lr{}`, `\rl{}`,
`\enfootnote{}`, `\fadecimal{}`, `\proposedmodel{}`, `\controlmodel{}`,
`\basemodel{}`, and `\calibrationmatrix{}`. Keep Persian text right-to-left,
Latin terms explicitly left-to-right, and mixed Persian/Latin captions and TikZ
labels direction-safe. Keep Persian digits and decimal formatting consistent with
the existing `displaymathdigits=persian` and `\fadecimal` conventions.

Do not add a new package, font, bibliography style, or ad-hoc formatting workaround
when an existing command or option already provides the needed behavior. If a
template-level change is unavoidable, explain it in the task summary and review all
affected source files after the change.

## No PDF rendering

The user renders and updates `thesis/main.pdf` themselves. Never run
`./thesis/build.sh`, `latexmk`, `xelatex`, or any other command that regenerates
`thesis/main.pdf` (or `.build/thesis/`), and never edit, copy over, or stage
`thesis/main.pdf`, unless the user explicitly asks for a build in that message.

Verify changes from the source instead: re-read the edited `.tex` regions, check
that braces, environments, `\begin`/`\end` pairs, `\label`/`\ref`/`\cite` keys,
and project macros (`\lr{}`, `\fadecimal{}`, etc.) are valid, and inspect
`git diff -- thesis`. Do not hand-edit `.aux`, `.log`, `.toc`, glossary, index, or
other generated files. In the final report, state that no build was run and the
PDF was left for the user to render.

## Commit and push

Only when the user asks to commit or push:

1. Review `git diff -- thesis` and confirm that every staged source change serves the
   requested task. Do not stage `thesis/main.pdf` (the user updates it); leave
   `.build/thesis/` and other generated files untracked/ignored.
2. Stage only the intended thesis files by path. Never stage `thesis-gemini/`, result
   directories, logs, credentials, or unrelated pre-existing edits.
3. Commit with a concise message describing the thesis update, then push the current
   branch to its configured remote. Do not rewrite history or force-push. If the push
   is rejected, stop and report the remote/rebase issue rather than discarding work.
4. Report the edited files, evidence used, commit hash, and push result. If any required verification or push step cannot be completed,
   state exactly what remains instead of claiming success.

## Required completion report

End every task with a concise, formal English report that lists:

1. each file changed and the exact requested item addressed there;
2. any evidence, figure source, or supporting skill used;
3. source-level verification results (no PDF build was run), plus commit/push status when applicable;
4. this explicit scope statement, filled with the actual itemized list:

   > **I changed only the items you requested:** [itemized list]. **No unrelated
   > thesis content was modified.**

If concurrent edits were detected, identify which pre-existing or concurrent changes
were preserved and distinguish them clearly from this task's changes.
