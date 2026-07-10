# Thesis: pending content re-integration

**Context:** on 2026-07-10, `thesis/` was reverted to its 2026-06-19 state (commit
`74bee36`) because the thesis still would not compile on `latex.sharif.edu` (TexReady)
after five straight fix attempts — see "Build-fix attempts (discarded)" below. That revert
would have deleted real writing (the experimental-results integration into chapters 3-5),
so this file plus `thesis_pending_reintegration.patch` (repo root) preserve it for
re-integration once the platform issue is sorted out some other way (e.g. setting the
compiler manually in the site's project settings instead of relying on autodetection).

Nothing here was lost — it's just parked outside `thesis/tex/*.tex` until you're ready.

## How to reapply

`thesis_pending_reintegration.patch` is `git diff 74bee36 HEAD -- thesis/tex/chapter3.tex
thesis/tex/chapter4.tex thesis/tex/chapter5.tex thesis/tex/commands.tex`, taken right before
the revert. As long as you haven't otherwise edited those four files since the revert, it
applies cleanly:

```bash
git apply thesis_pending_reintegration.patch
```

If it doesn't apply cleanly (because you've since edited those chapters by hand), use
`git apply --3way thesis_pending_reintegration.patch` to get conflict markers instead of a
hard failure, or cherry-pick individual hunks with `git apply --include=... -p1` /
manual copy-paste — use the subsection map below to find the right spot by
`\label{...}`.

You can also pull any piece straight from git history instead of the patch, e.g.
`git show 65a5b2d -- thesis/tex/chapter4.tex`.

## What's in the patch, by topic

### Chapter 3 (`thesis/tex/chapter3.tex`)

| Topic | Commit(s) | Anchor |
|---|---|---|
| Camera descriptor rewritten as a 9-dim vector (added FoV angles, aspect ratio, mean focal length, H·W) instead of the original 4-dim `[fx,fy,cx,cy]`; note that K and (H,W) must share the same coordinate frame | 76db955 | (edit to §3.5/3.6 intro, no new label) |
| Camera encoder redescribed as token injection into the **encoder** (2-layer MLP + 4 transformer blocks + zero-init output gate) instead of the original decoder-concatenation description; architecture figure and training-algorithm pseudocode updated to match | 76db955, 6018172 | (edit to §3.6, architecture figure/algorithm) |
| Leak-free NYU train/test split: original 80/20 split over all 1449 images leaked 527/654 (80.6%) of the Eigen test set into training; fixed by holding out the 654 first | 76db955 | `\subsection{تفکیک صحیح مجموعه‌های آموزش و آزمون در NYUv2}` |
| Disparity target space: relative-model training on raw depth targets silently collapsed to constant-zero output within ~18 steps (dead ReLU head interacting with scale/shift-invariant loss); fixed by training on `1/depth` instead | 76db955 | `\label{subsec:disparity_target_space}` |
| FoV-aware geometric augmentation: exact zoom-crop with K updated in lockstep (`cx' = cx - x0`, `fx' = fx · Wnet/wwin`, ...), replacing a crop that silently invalidated K | 76db955 | `\label{subsec:fov_augmentation}` |
| Focal-jitter augmentation: pinhole-exact `(f,Z) → (αf, αZ)` rescaling, `α ~ exp(U(±ln 1.6))`, makes scale unrecoverable from image content alone so the K-blind control has an irreducible error floor; seed-locking so both arms of the controlled study see identical data | 76db955 (method write-up), e3870d8 (round-3 refinements) | `\label{subsec:focal_jitter}` |
| Debugging-lessons subsection: the three silent-failure bugs (dead-ReLU collapse, stale K under crop, metric-head architecture mismatch causing ~4x depth over-estimation) written up as a methodology note | 76db955 | `\label{subsec:debugging_lessons}` |
| Three overfull-hbox paragraphs reworded (no content change) | 852b084 | — |

### Chapter 4 (`thesis/tex/chapter4.tex`)

New top-level section **"مطالعهٔ کنترل‌شده بر روی پروتکل استاندارد NYUv2"** (`\label{sec:controlled_nyu_study}`), containing:

| Subsection | Anchor | Result |
|---|---|---|
| Pipeline verification (reproduce DA2 paper's own numbers first) | `subsec:pipeline_verification`, `tab:paper_reproduction` | confirms the eval harness itself is correct before trusting any comparison |
| Zero-shot fine-tuning failure | `subsec:zeroshot_failure` | naive fine-tuning on a new domain provably *hurts* zero-shot numbers |
| Basic/relative protocol four-way comparison (ViT-S/L × with/without K) | `subsec:basic_protocol_results`, tables `nyu_basic_vits`/`nyu_basic_vitl`/`nyu_basic_significance` | paired t-test + Wilcoxon over the 654-image Eigen test set |
| Augmentation effect + an informative negative result | `subsec:aug_basic_results`, `tab:nyu_basic_aug` | plain zoom-crop augmentation alone doesn't isolate the K-contribution — motivates focal jitter |
| Metric-protocol results | `subsec:metric_protocol_results`, tables `nyu_metric_results`/`nyu_metric_significance` | intrinsics-aware metric model beats control with dose-response vs. FoV variation |
| Round 3 — focal-jitter results | `subsec:focal_jitter_results`, `tab:nyu_metric_focal` | all six eigen-654 metrics win at p≤1e-24; K-sweep slope 0.06→0.93 (ideal 1.0) |
| Ablation: focal jitter alone vs. λ change | `subsec:focal_ablation`, `tab:nyu_metric_focal_ablation` | AbsRel 0.0834 vs 0.1065, p≤1e-39, K-sweep slope 0.97 — effect fully attributed to jitter, not the λ=0.5→0.3 SiLog change |
| K-sweep diagnostic (replaces placeholder tikz figure with real data) | `tab:k_sweep_slopes` | log-log slope of predicted-scale vs. injected-α, per model variant |
| K-knockout: withhold K / feed wrong K | `subsec:k_knockout`, `tab:k_knockout` | withholding K degrades AbsRel 59%; f×0.7 or f×1.4 degrades to match `|α^slope − 1|` exactly as the pinhole model predicts — direct causal proof the model reads K, not just correlates with it |
| Protocol/capacity dependence discussion | `subsec:protocol_capacity_dependence` | why the effect size differs between basic/metric protocols and ViT-S/L |
| Controlled-study summary | `subsec:controlled_study_summary` | — |
| Paired statistical-test methodology note (t-test description generalized from "used" to "used in cross-dataset comparisons", since the controlled study uses paired tests instead) | 76db955 | (edit near top of ch4 stats section) |

Sources: 76db955 (initial campaign + pipeline verification + zero-shot + basic/metric protocol results), e3870d8 (focal-jitter round-3 results + K-sweep real data), 65a5b2d (ablation + K-knockout), 6018172 (table formatting only — restructured two significance tables to long format because they were up to 122pt overfull; resized architecture figure; made long CLI tokens breakable).

### Chapter 5 (`thesis/tex/chapter5.tex`)

- New subsection **"دستاوردهای روش‌شناختی و تشخیصی"** (methodological/diagnostic contributions) — `\label{subsec:methodological_contributions}` (76db955)
- Findings summary, innovations list, limitations, and future-work sections revised across all three content commits to cite the statistics and diagnostics above: 76db955 (initial refined findings + methodological contributions + limitations/future work), e3870d8 (focal jitter + K-sweep + seed-locked pairing added to innovations list), 65a5b2d (summary + limitations updated with the ablation/K-knockout results)

### `thesis/tex/commands.tex`

Guarded no-op shim for `\UseMathForPositioningText` (removed from `array.sty` in TeX Live 2026 but still referenced by bidi's tabular patch); inert on older distributions. 8 lines, from 6018172 — purely a compatibility shim, not content, but included in the patch since it travels with the same file.

## Build-fix attempts (discarded — for reference, don't repeat these blindly)

Five commits tried to fix the same underlying problem: `latex.sharif.edu` was falling back
to `pdflatex` instead of `xelatex`, which fails immediately because `fontspec`/`xepersian`
require XeTeX or LuaTeX. None of them resolved it (that's why you asked for the revert):

1. **`6018172`** — untracked all committed LaTeX build artifacts (`main.aux`, `.log`, `.fls`,
   `.pdf`, `.xdv`, chapter `.aux` files, ~10,400 lines) and added `thesis/.gitignore`. Root
   cause identified: a stale committed `main.aux` written by a different bidi/kernel version
   broke fresh compiles (`\bidi@finalfootnoteperpage` undefined) on other TeX distributions.
   **This part of the fix was real and is being kept** — the revert does not re-commit those
   build artifacts, so that specific stale-`main.aux` bug does not come back.
2. **`8ed02a9`** — added `% !TeX program = xelatex` (Overleaf's standard autodetect comment)
   as the literal first line of `main.tex`, alongside the pre-existing TeXstudio variant
   `% !TEX TS-program = xelatex`.
3. **`24083f5`** — the same fontspec/pdflatex error recurred, so added a third variant
   `%!TEX program = xelatex` (Overleaf's documented exact casing) plus an explicit UTF-8
   directive.
4. **`e81331b`** — reasoned the *original* `% !TEX TS-program = xelatex` first line (before
   any of this started) was actually what the site's parser wanted, and that the earlier
   commits pushing it down to line 2+ broke it. Reverted `main.tex`'s header back to the
   06-19 original. Also re-committed freshly regenerated `main.{bbl,bls,gls,acr,ind}` so the
   site could produce a complete document via xelatex passes alone, without needing
   `bibtex8`/`xindy` with Persian modules.
5. **`ccb27e6`** — reasoned that `latex.sharif.edu` is a ShareLaTeX fork that reads
   `latexmkrc` only from the *project root* of the uploaded project; if the project was
   created from a whole-repo zip, `thesis/latexmkrc` (which has always remapped `$pdflatex`
   to `xelatex`) would sit one directory too deep and never get read. Added copies at the
   repo root (`latexmkrc`, `.latexmkrc`) and as `thesis/.latexmkrc`.

**Still broken after all five.** Since none of the autodetection-comment or latexmkrc-copy
theories panned out, the next thing to try is probably just setting the compiler to XeLaTeX
**manually** in the site's own project settings UI, rather than relying on any form of
autodetection — worth checking whether that setting exists and sticks before writing more
detection heuristics.

## What was NOT reverted

- `REPORT_INTRINSICS_EXPERIMENT.md` (repo root) — outside `thesis/`, left untouched. It has
  its own copy of the ablation/K-knockout numbers from `65a5b2d`.
- `thesis/latexmkrc` (no leading dot) — the original file, predates this whole episode,
  untouched.
- `thesis/.gitignore` — kept (in the untracked-build-artifacts form from `6018172`, minus
  the `main.{bbl,bls,gls,acr,ind}` tracking exceptions `e81331b` added, since those files
  were removed again in this revert).
