---
name: thesis-editor
description: Edit the Persian LaTeX thesis source and render the PDF with thesis/build.sh before asking for review, when the user requests a thesis change, an academic or professional revision, or a thesis update based on project outputs or results. Apply the relevant LaTeX, academic-writing, and figure-making skills, while making only the requested thesis changes. Read project evidence as needed, but write only under thesis/; never edit thesis-gemini/ or other project folders. Every edit is written as visible review markup (green added, yellow new words, red-highlighted removals and deletions with the text itself left black and readable), then the user is asked to finalize or revert it.
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

## Change-tracking markup (mandatory for every thesis edit)

Every change this skill makes to thesis prose, headings, captions, tables, or
figures is written as **visible review markup**, never as a silent edit. The user
reads the marked-up PDF, then tells you to finalize or revert. The markup applies
to every update, including small wording fixes, unless the user explicitly asks for
an unmarked edit in that message.

### Colour code

| Kind of change | How it must look | Macro / environment |
|---|---|---|
| New paragraph, section, sentence, caption, table, or figure | Green highlight | `\edadd{...}`, `edadded` env, `\edaddbox{...}`, `\edaddrow`, `\edaddcell` |
| Edited paragraph — words removed | Red highlight only (text stays black, no line-through) | old half of `\edrep{old}{new}` |
| Edited paragraph — words inserted | Yellow highlight | new half of `\edrep{old}{new}`, or `\edins{...}` for a pure insertion inside an edited sentence |
| Deleted part (sentence, paragraph, section, row) | Red highlight only (text stays black, no line-through) | `\eddel{...}`, `eddeleted` env, `\eddelbox{...}`, `\eddelrow` |

Rules for choosing the kind:

- A paragraph or section that did not exist before is **added** (green) as a whole;
  do not mix yellow into it.
- A paragraph that existed and is partly rewritten is **edited**: mark only the
  changed words/phrases with `\edrep{old}{new}` (or `\edins{}` for pure insertions,
  `\edrep{old}{}` for pure removals inside the paragraph). Keep unchanged words
  outside the macros. Mark at phrase granularity — do not wrap a whole paragraph in
  one `\edrep` unless nearly every word changed.
- Content removed as a whole unit (a sentence, paragraph, list item, section,
  table row, figure) is **deleted**: keep it in the source wrapped in `\eddel{}` /
  `eddeleted` so the user can see it in the PDF. Never actually remove text until
  the user finalizes.
- Numbers changed inside an existing table cell or sentence are edits:
  `\edrep{\fadecimal{0.081}}{\fadecimal{0.079}}`.
- Section headings: wrap the heading text, e.g. `\section{\edadd{...}}` or
  `\section{...\edrep{old}{new}...}`. Because headings also feed the table of
  contents and bookmarks, use the optional short argument with the plain text:
  `\section[plain new title]{\edadd{plain new title}}`.
- Labels, `\cite` keys, and `\ref`s stay functional: never put a `\label` inside a
  deleted block that is still referenced elsewhere — move the `\label` or mark the
  referencing text too.

### Word-level colour discipline

For an edited paragraph, sentence, caption, table cell, or heading, apply colour only to the exact changed words or phrases. Keep unchanged text outside review macros. Use `\edrep{old}{new}` when replacing wording, `\edins{...}` for inserted words, and `\edrep{old}{}` for removed words. Do not wrap an entire existing paragraph in a yellow, green, or red environment merely because one sentence changed. A whole paragraph may be green only when the paragraph is entirely new; a whole deleted unit may be red only when the unit itself is removed. The same word-level rule applies consistently to yellow insertions, red deletions, and any green additions inside existing prose.

## Markup macros

The macros live in `thesis/tex/commands.tex`. Before the first marked-up edit,
check whether `\edadd` is already defined there; if it is not, insert the block
below **once**, directly after the existing `\showedits` block (the legacy yellow
`\edhl`/`\edpar`/`edited`/`\edhlbox`/`\edrow`/`\edcell` macros) and before
`xepersian` is loaded. `ulem` is loaded with `normalem` so `\emph` is unchanged.
Adding this block is a direct compile dependency of the requested edit and is
allowed under the template rules; do not reorder other packages.

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% نشانه‌گذاری بازبینی: سبز = افزوده، زرد = واژه تازه، قرمز خط‌خورده = حذف
%%% \showreviewfalse همه نشانه‌ها را پنهان می‌کند (متن تازه می‌ماند، متن حذفی نه)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage[normalem]{ulem}
% در متن راست‌به‌چپ، XeTeX ترتیب جعبه‌ها را وارونه می‌کند و نوار رنگی ulem روی واژه
% کشیده می‌شود و آن را می‌پوشاند؛ در حالت RTL ترتیب را برمی‌گردانیم تا رنگ پشت متن بماند.
\makeatletter
\def\UL@putbox{\ifx\UL@start\@empty \else % not inner
   \vrule\@width\z@ \LA@penalty\@M
   \if@RTL
     \dimen@\wd\UL@box \box\UL@box
     {\UL@skip\dimen@ \kern-\UL@skip \UL@leaders}%
   \else
     {\UL@skip\wd\UL@box \UL@leaders \kern-\UL@skip}%
     \box\UL@box
   \fi
  \fi}
\makeatother
\definecolor{thRevAdd}{HTML}{B9F6CA}  % افزوده: سبز
\definecolor{thRevNew}{HTML}{FFF176}  % واژه تازه در بند ویرایش‌شده: زرد
\definecolor{thRevDel}{HTML}{D32F2F}  % قرمز پررنگ (برای متن به کار نمی‌رود)
\definecolor{thRevDelBG}{HTML}{EF9A9A}% زمینه حذف: قرمز؛ متن سیاه می‌ماند
\newif\ifshowreview
\showreviewtrue
\ifshowreview
 % برجسته‌سازی شکست‌پذیر میان سطرها (ulem رنگ را پشت متن می‌کشد)
 \newcommand{\edhlwith}[2]{\bgroup\markoverwith{\textcolor{#1}{\rule[-0.55ex]{2pt}{2.6ex}}}\ULon{#2}}
 \newcommand{\edadd}[1]{\edhlwith{thRevAdd}{#1}}
 \newcommand{\edins}[1]{\edhlwith{thRevNew}{#1}}
 \newcommand{\edold}[1]{\edhlwith{thRevDelBG}{#1}}
 \newcommand{\edrep}[2]{\edold{#1}\ifx\relax#2\relax\else\ \edins{#2}\fi}
 % زمینه قرمز روشن + خط قرمز روی متن در یک نشانه (ulem تودرتو پشتیبانی نمی‌شود)
 \newcommand{\eddel}[1]{\edhlwith{thRevDelBG}{#1}}
 % بندها و بخش‌های افزوده/حذف‌شده؛ میان صفحه‌ها شکسته می‌شوند
 \newenvironment{edadded}{\colorlet{shadecolor}{thRevAdd}\begin{shaded}}{\end{shaded}}
 \newenvironment{eddeleted}{\colorlet{shadecolor}{thRevDelBG}\begin{shaded}}{\end{shaded}}
 % جدول یا شکل کامل افزوده/حذف‌شده (درون محیط شناور)
 \newcommand{\edaddbox}[1]{\colorbox{thRevAdd}{\begin{minipage}{\dimexpr\linewidth-2\fboxsep\relax}\centering #1\end{minipage}}}
 \newcommand{\eddelbox}[1]{\colorbox{thRevDelBG}{\begin{minipage}{\dimexpr\linewidth-2\fboxsep\relax}\centering #1\end{minipage}}}
 % ردیف/خانه جدول
 \newcommand{\edaddrow}{\rowcolor{thRevAdd}}
 \newcommand{\edaddcell}{\cellcolor{thRevAdd}}
 \newcommand{\ednewcell}{\cellcolor{thRevNew}}
 \newcommand{\eddelrow}{\rowcolor{thRevDelBG}}
\else
 \newcommand{\edadd}[1]{#1}
 \newcommand{\edins}[1]{#1}
 \newcommand{\edold}[1]{}
 \newcommand{\edrep}[2]{#2}
 \newcommand{\eddel}[1]{}
 \newenvironment{edadded}{}{}
 \newenvironment{eddeleted}{\setbox0\vbox\bgroup}{\egroup}
 \newcommand{\edaddbox}[1]{#1}
 \newcommand{\eddelbox}[1]{}
 \newcommand{\edaddrow}{}
 \newcommand{\edaddcell}{}
 \newcommand{\ednewcell}{}
 \newcommand{\eddelrow}{}
\fi
```

Usage notes:

- `\edadd`, `\edins`, `\edrep`, `\eddel` are for text **within one paragraph**
  (they line-break but cannot contain `\par`, blank lines, `\begin…\end`, or
  display math). For whole paragraphs/sections use the `edadded` / `eddeleted`
  environments; inside `eddeleted`, also wrap each paragraph's text in `\eddel{}`
  so the red highlight is drawn behind every line. For whole floats use `\edaddbox` / `\eddelbox` inside
  the `figure`/`table` environment around the body and caption (do not put
  `shaded` inside a float).
- Never nest the inline review macros inside each other (`ulem` cannot nest
  underlines); split the text into adjacent macro calls instead.
- Inside the `ulem` macros, wrap fragile or boxed content — `\lr{...}`, `\cite`,
  `\ref`, inline math, `\fadecimal{}` — as a single braced unit, e.g.
  `\edadd{متن {\lr{DA2}} متن}`; braces keep `ulem` from splitting it.
- A deleted table row keeps its cells: `\eddelrow \eddel{a} & \eddel{b} \\`.
  An added row: `\edaddrow a & b \\`.
- Display equations: wrap the whole `equation` in `\edaddbox{...}`/`\eddelbox{...}`
  or, for an edited equation, show old and new versions as one deleted box plus
  one added box.
- If a source-level check shows `ulem` is incompatible with a particular construct,
  fall back to a background-only highlight (`\colorbox{thRevDelBG}{...}` for
  removals) for that construct and mention it in the report; never drop the
  marking silently.
- Removed and deleted text must stay readable: mark it **only** with the red
  background highlight. Never colour the text itself red (`\color{thRevDel}`,
  `\textcolor{thRevDel}`) and never add a line-through (`\sout`, strike rules).
  If `thesis/tex/commands.tex` still has an older red-text or strike-through
  definition of `\edold`, `\eddel`, or `eddeleted`, replace it with the block
  above before marking new deletions.
- The `\UL@putbox` override in the block is required: without it every inline
  highlight (`\edadd`, `\edins`, `\edold`, `\eddel`) in Persian (RTL) text is
  painted *over* the words and hides them. Keep it if you touch the block, and
  when you are asked to render, look at one highlighted Persian line in the PDF
  to confirm the text shows through.
- The legacy yellow macros (`\edhl`, `\edpar`, `edited`, `\edhlbox`, `\edrow`,
  `\edcell`, toggle `\showedits`) belong to the earlier 2026-10-30 update. Leave
  them as they are unless the user asks to finalize/revert that update too; do not
  use them for new edits.

### Pending-review ledger

Record every marked-up update in `thesis/REVIEW_PENDING.md` (create it if absent).
Each update gets an ID (`R1`, `R2`, … continuing the highest existing number) and
an entry listing: date, short description, and every file + section/label +
approximate line touched, with the kind of each change (added / edited / deleted).
Also put the ID in a LaTeX comment on the line before each marked region,
`% REVIEW R3`, so finalize/revert can find every region with `grep -n 'REVIEW R3'`
even after other edits shift line numbers. Several updates may be pending at once;
never mix markup of different IDs inside one macro.

### Ask the user after every update

After each marked-up update (and after the completion report), ask the user — with
`AskUserQuestion` when available — whether they have read the marked sections,
naming the sections **edited in this session** and describing them in plain words
(for example "the §4.3.3 rewrite you just asked for"); an update ID alone is not
enough, because the user does not track the IDs. Ask only about the exact changes
made in the current session. Offer:

1. **Read — make final** (finalize),
2. **Not accepted — revert** (revert),
3. **Not read yet — keep highlighted** (leave markup and ledger entry pending).

Never finalize or revert on your own initiative, and never treat silence or an
unrelated reply as approval. If the user starts a new thesis request while updates
are pending, keep the old markup and ledger entries and mark the new update with a
new ID, but do **not** ask about updates left over from earlier sessions or earlier
requests: ask only about the regions this session changed. When the session's
change is a further edit inside a region of an older pending ID (for example a
rewrite of text that update added), keep that ID, and still phrase the question as
"the changes made now" in that section, never as the whole older update. Older
pending entries stay in the ledger untouched until the user brings them up.

### Finalize (user read and accepted)

For the chosen ID(s), in every region tagged `% REVIEW Rn`:

- `\edadd{X}` → `X`; `\edins{X}` → `X`; `\edrep{old}{new}` → `new`;
- `\begin{edadded}` … `\end{edadded}` → keep the body, drop the environment lines;
- `\edaddbox{X}` → `X`; delete `\edaddrow`, `\edaddcell`, `\ednewcell` tokens;
- `\eddel{X}` → removed entirely (with surrounding whitespace tidied);
  `eddeleted` blocks, `\eddelbox{...}`, and rows starting with `\eddelrow` →
  removed entirely, **permanently**;
- remove the `% REVIEW Rn` comment lines and the ledger entry (delete
  `thesis/REVIEW_PENDING.md` when it becomes empty).

### Revert (user did not accept)

For the chosen ID(s), restore the pre-update text exactly:

- `\edadd{X}`, `\edins{X}`, `edadded` blocks, `\edaddbox{X}`, and rows starting with
  `\edaddrow` → removed entirely;
- `\edrep{old}{new}` → `old`; `\eddel{X}` → `X`; `eddeleted` block → its body with
  the inner `\eddel{}` wrappers unwrapped; `\eddelbox{X}` → `X`; `\eddelrow` row →
  the row with its cell `\eddel{}` wrappers unwrapped;
- also revert any compile dependency added for that update (new glossary entry,
  label, bibliography key, figure asset) unless another pending or final change
  still uses it;
- remove the `% REVIEW Rn` comments and the ledger entry.

For both finalize and revert: use exact, region-by-region edits (not a blind global
regex), keep the macro definitions in `commands.tex` for future updates, check
`grep -n 'REVIEW Rn\|\\edadd\|\\edins\|\\edrep\|\\eddel\|edadded\|eddeleted' thesis/`
afterwards to confirm no markup of that ID remains, and inspect `git diff -- thesis`
so that only that ID's regions changed. A revert must leave the affected regions
byte-identical to their state before the update (compare with
`git show <commit-before-update>:<file>` when the update was committed).

## Codex pass for Persian prose edits (mandatory)

Every edit that changes Persian prose in the thesis goes through the Codex CLI
(`codex`, on PATH) before it is rendered and reported. This covers rewriting or
smoothing a paragraph, writing a new paragraph or summary, merging or splitting
paragraphs, shortening text, and rewriting table cells or captions. It does not
cover pure LaTeX/structure changes with no wording change (moving a block, changing
a `\section` level, fixing a label or reference, removing review markup).

1. Make the edit in the source first, with the review markup this skill requires.
2. Run Codex read-only from `thesis/` and tell it exactly where the edit is and what
   to compare it with (the original wording, the tables or figures it describes, and
   the neighbouring paragraphs or the section it leads into):

   ```bash
   cd thesis && codex exec --sandbox read-only --skip-git-repo-check "<prompt>"
   ```

   The prompt must name the file and a searchable anchor (label or opening words) and
   ask Codex to report, briefly and in English: (a) claims not supported by the
   cited tables/text or contradicting nearby sections, and any changed meaning,
   number, caveat or citation compared with the original; (b) Persian grammar,
   spelling and half-space (ZWNJ) problems; (c) awkward or unclear sentences with a
   suggested Persian rewrite. State any constraint the user gave (for example "no
   numbers in this paragraph") and end with "Do not modify any file."
3. Apply the findings you agree with by editing the source yourself. Codex never
   writes thesis files. Keep this project's terminology and macros (`\proposedmodel{}`,
   `\controlmodel{}`, `\lr{}`, `\fadecimal{}`, انکودر) even when Codex's rewrite
   uses plain words, and do not adopt a suggestion that changes a claim or number.
4. Then build with `thesis/build.sh` as usual.
5. In the completion report, say that the Codex pass ran, list what it changed, and
   mention any suggestion you rejected and why. If `codex` is unavailable or fails,
   say so explicitly instead of skipping the step silently.

Running Codex sends the quoted thesis text to an external service; the author has
asked for this for thesis prose edits.

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

## Running evaluations for a thesis update (GPU)

When a thesis update needs a new evaluation, inference, or any other GPU run, do it
on lingotube through the `gpu` skill (the local machine has no usable GPU). The
server's RTX 4090 is normally occupied by an LLM container from the `~/llm` stack
(`llm-qwen`, or `llm-gemma` / `llm-nemotron`, ~22.7 GB VRAM), so free it first and
bring it back afterwards:

1. **Check what holds the GPU** before launching anything:

   ```bash
   ssh lingotube 'docker ps --filter "name=^llm-" --format "{{.Names}}"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
   ```

   Record the exact names of the running `llm-*` containers (there may be none).
   If the GPU is held by something other than an `llm-*` container (for example a
   training job), do **not** stop it; report it and wait or ask the user.
2. **Stop only those `llm-*` containers**, then confirm the VRAM is free:

   ```bash
   ssh lingotube 'docker stop llm-qwen && sleep 5 && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'
   ```

   Use `docker stop`/`docker start` on the existing container. Do not use
   `docker compose down`, `docker rm`, or `--profile ... up` (they recreate or
   remove containers), and never stop the other stack services (`open-webui`,
   `gateway`, `docgen`, `nginx-tls`, `docling`, `searxng`).
3. **Run the evaluation** in tmux as the `gpu` skill describes, and chain the
   restart onto the same tmux command so the container comes back even if the
   evaluation fails or this session disconnects:

   ```bash
   <EVAL_COMMAND> 2>&1 | tee $LOG_FILE; echo EXIT_CODE=$?; docker start llm-qwen
   ```

   (List every container you stopped in that `docker start`.)
4. **Verify the LLM is back** after the run finishes: the container should be
   `Up` and, once the model has loaded (allow a few minutes), `healthy`:

   ```bash
   ssh lingotube 'docker ps --filter "name=^llm-" --format "{{.Names}} {{.Status}}"'
   ```

   If the restart failed or the container is not healthy after ~5 minutes, run
   `docker start <name>` again and check `docker logs --tail 50 <name>`; if it
   still fails, tell the user explicitly rather than leaving it down silently.
5. If the evaluation is cancelled or abandoned before it runs, restart the stopped
   containers immediately; never end the task with an `llm-*` container that was
   running before left stopped.

In the completion report, state which `llm-*` container was stopped (or that none
was running), and that it was restarted and confirmed up/healthy.

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

## Render the PDF before asking for review

After a marked-up edit, and **before** asking the finalize / revert / keep question,
build the PDF with `./thesis/build.sh` so the user can read the highlighted result
in `thesis/main.pdf` and answer from it. Never ask the review question about
changes that have not been rendered. Do not use any other build path: no `xelatex`
or `pdflatex` calls, no copying over `thesis/main.pdf` by hand. Never stage
`thesis/main.pdf`.

If the build fails, fix the source (only within the requested change) and rebuild;
if it still fails, report the error instead of asking the review question.
Pre-existing warnings elsewhere (for example "Missing character" on pages the edit
did not touch) are not caused by the edit; mention them only if they appear in the
edited sections.

Also verify from the source: re-read the edited `.tex` regions, check that braces,
environments, `\begin`/`\end` pairs, `\label`/`\ref`/`\cite` keys, and project
macros (`\lr{}`, `\fadecimal{}`, etc.) are valid, and inspect `git diff -- thesis`.
Do not hand-edit `.aux`, `.log`, `.toc`, glossary, index, or other generated files.
In the final report, state that the PDF was built and give the printed page of each
edited section so the user can find it.

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
3. source-level verification results and the PDF build result (with the printed page of each edited section), plus commit/push status when applicable;
   if a GPU evaluation was run, which `llm-*` container was stopped and confirmation
   that it was restarted and is up/healthy again;
4. this explicit scope statement, filled with the actual itemized list:

   > **I changed only the items you requested:** [itemized list]. **No unrelated
   > thesis content was modified.**

If concurrent edits were detected, identify which pre-existing or concurrent changes
were preserved and distinguish them clearly from this task's changes.

Also state the review ID of this update, the colour legend (green = added,
yellow = new words, red highlight = removed words and deleted parts),
then ask the finalize / revert / keep-pending question described in
"Ask the user after every update", limited to what was changed in this session.

## Persian prose quality rules

For Persian thesis prose, every sentence must have a complete grammatical ending, preferably an explicit verb or a complete predicate; do not end a sentence with an unexplained number, parenthesis, or fragment. Whenever a number appears, state what it measures and which model, dataset, metric, ratio, confidence interval, or test it belongs to. Do not leave a bare ratio or count as the subject of a sentence. Split long sentences when several comparisons or statistical qualifications are present. Prefer plain terms such as «از نظر آماری قابل اتکا» when the intended meaning is statistical evidence, and avoid unexplained or awkward uses of «معنادار».
