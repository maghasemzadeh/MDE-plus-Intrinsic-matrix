# Pending thesis review updates

Colour legend: green = added, yellow = new words in an edited paragraph,
red highlight (black text, no strike) = removed words and deleted parts,
gray = moved without word changes.
Find every region of an update with `grep -n 'REVIEW R<n>' thesis/tex/*.tex`.

## Finalized — 2026-11-18 — all earlier updates (branch `refactor/final`)

At the author's request, every review mark that was still pending was made final before the last
full read of the thesis: R21, R25 and R26 in chapter 5, R30, R31 and R32 in chapter 3 and the English
abstract, and the gray R7 `edmoved` wrapper of §5.2.2. New text kept, deleted text and the old
`tab:hypothesis_verdicts_old` table dropped, `% REVIEW` lines removed. The source text before these
updates is in git history (R26: 1f2833a; R30–R32: b9c7246). The macros in `tex/commands.tex` stay.

## R33 — 2026-11-18 — final full read: confirmed defects only

Whole thesis read section by section (abstracts, chapters 1–5, figure includes, front and back matter)
and checked against the tables, the stored results (`results/multidomain/`) and the rendered PDF.
Text changes are highlighted; figure, reference and layout fixes cannot carry highlights and are
tagged with a `% REVIEW R33` comment where the source allows it. Persian wording of the ch. 2 /
acknowledgement edits came from Codex (read-only) on a fixed brief.

| # | File | Location (printed page) | Change | Kind |
|---|---|---|---|---|
| 1 | chapter1.tex | `fig:thesis_roadmap` (p. 9) | dashed link from chapter 1 lost its first segment (`--` missing since commit 68d3c49) — restored; note moved below the line (it was struck through by it) | figure fix |
| 2 | chapter2.tex | `fig:pinhole_model` (p. 15) | three dimension lines f, Z, Y had lost `--` (same commit) — restored | figure fix |
| 3 | chapter2.tex | `fig:ch2_taxonomy` (p. 18) | figure ran 81 pt past the margin (right box cut off) and its Persian labels were in reversed word order — scaled to the text width, labels wrapped in `\rl{}` | figure fix |
| 4 | chapter2.tex | DPT decoder equation (p. 26) | equation was wider than the text block (number pushed to its own line) — split over two lines | layout fix |
| 5 | chapter2.tex | §2.2.1 item 2 (p. 18) | garbled «نقشه عمق رنگی عمق (RGB-D)» → «نقشه عمق حسگر RGB-D» | edited |
| 6 | chapter2.tex | §2.2.6 (p. 21) | first mention of Depth Pro had no citation — `\cite{Bochkovskii2025DepthPro}` added | reference fix |
| 7 | chapter2.tex | captions of `fig:absolute_model_comparison` (p. 21) and `fig:depthanything_teaser` (p. 23) | both images come from the Depth Anything V2 repository — «(منبع: [cite])» added | added |
| 8 | chapter2.tex | §2.3.1 (p. 22) | sentence said both the relative and the metric released checkpoints are the starting weights; chapters 3–4 start only from the relative version and use the metric versions as the comparison reference — corrected | edited |
| 9 | chapter2.tex, chapter3.tex | 5 places | «نقطه بازرسی» → «نقطه وارسی» (2 + item 8) and «تک‌چشمی» → «تک‌دوربینه» (3), the forms used in the rest of the thesis and in the glossary | edited |
| 10 | chapter3.tex | after the first equation of §3.2 (p. 32) | symbol `$K$` was missing in «که در آن، I تصویر ورودی، … ماتریس کالیبراسیون دوربین» | inserted |
| 11 | chapter3.tex | §3.2.3 | `\label{subsec:camera_integration}` added (needed by item 19) | compile dependency |
| 12 | chapter3.tex | §3.3.2 (p. 42) | ImageNet «میانگین و واریانس» → «میانگین و انحراف معیار» (the formula divides by σ) | edited |
| 13 | chapter3.tex | §3.4.5 (p. 46–47) | orphan sentence «این زنجیره به‌صورت شماتیک تشریح می‌شود.» (no schematic follows) | deleted |
| 14 | chapter3.tex | §3.4.5 (p. 47) | narrowest field of view at zoom 2.2: «۳۳ درجه» → «۳۱ درجه» (2·atan(0.6167/2.2) = 31.3°; Table 4.2 already says 31) | edited |
| 15 | chapter3.tex | `tab:loss_overview` (p. 52) | `(releative)` → `(relative)` | typo |
| 16 | chapter4.tex | `tab:basic_model_silog` (p. 63) | DrivingStereo was used without a citation — `\cite{Yang2019DrivingStereo}` added | reference fix |
| 17 | chapter4.tex | `tab:nyu_absolute_focal`, `fig:effect_sizes` | `SILog` → `SiLog` (spelling used everywhere else) | typo |
| 18 | chapter4.tex | `fig:dose_response` (p. 71) | dashed p = 0.05 line had lost `--` — restored; upper y-axis ticks printed as «8·10⁻²» → fixed-point | figure fix |
| 19 | chapter4.tex | §4.4.5 (p. 71) | «نمونه آزمون‌های زوجی بخش …» pointed to §3.4.6 (augmentation definition) → `subsec:focal_jitter_results` | reference fix |
| 20 | chapter4.tex | `fig:k_knockout` (p. 77) | dashed baseline line had lost `--` — restored; legend `|X^s−1|` → `|c^s−1|` (symbol used in the text) | figure fix |
| 21 | chapter4.tex | §4.6.2 (p. 83) | ViT-L absolute reductions «بین ۱۳ تا ۵۰ درصد» → «بین ۱۱ تا ۵۰ درصد» (KITTI, −10.9 %, is one of the seven reliable sets) | edited |
| 22 | chapter4.tex | §4.6.2 (p. 84) | coefficient of variation of AbsRel over the eight sets: 0.85 → 0.80 (proposed), 0.72 → 0.67 (control); recomputed from `results/multidomain/md1_{P,C}` (sample std / mean); released model 0.79 unchanged; ordering and conclusion unchanged | edited |
| 23 | chapter5.tex | §5.2.3 (p. 101) | the four-method comparison is in the text of §3.2.3, not in Figure 3.2 — «در شکل …» → «در زیربخش …» | edited |
| 24 | chapter5.tex | §5.3 first limitation; last item of the §5.4 list | missing space before «(بخش …)»; list ended with «؛» → «.» | punctuation |
| 25 | faTitle.tex | acknowledgement | «دانشته» → «دانسته‌ام», doubled «را», «آقادکتر» → «آقای دکتر», «هم دانشگاهیان», «همراهی شان», «ثابت قدم نموده است» → «ثابت‌قدم نموده‌اند» | edited |
| 26 | thesis_preamble.tex | acknowledgement page | stray letter «A» printed at the bottom of the page (`A\abstractPage`) | removed |
| 27 | glossaries-settings.tex | both glossaries, abbreviation list | header row repeated as a body entry with a blank cell (the 24 "Missing character" warnings) — two rows removed; `EXIF` row removed (not used in the thesis); running head of these pages said «فهرست جداول» → own titles | layout fix |
| 28 | main.tex | English abstract, second page | carried the Persian running head — `\pagestyle{empty}` like its first page | layout fix |

- Finalize R33: `\edrep{old}{new}` → new, `\edins{X}`/`\edadd{X}` → X (un-brace `{\lr{…}}`, `{\cite{…}}`, `{\ref{…}}`, `{\fadecimal{…}}`, `{$K$}`), drop the `\eddel{…}` sentence in chapter 3, remove the `% REVIEW R33` lines. Items without markup need nothing.
- Revert R33: `\edrep{old}{new}` → old, remove `\edins`/`\edadd`, `\eddel{X}` → X; for the unmarked items restore the files from `git show 670b1b1:<file>` (the commit before this update on `refactor/final`).

## R35 — 2026-11-18 — hypothesis 1 without metric names (author's choice)

| # | File | Location | Change | Kind |
|---|---|---|---|---|
| 1 | chapter1.tex | `sec:research_hypotheses`, hypothesis 1 (p. 5) | «در معیارهای مطلق (AbsRel و RMSE)» → «در ارزیابی مطلق», so that the hypothesis matches its evidence table (`tab:basic_model_silog`, SiLog) | edited |

- Finalize: `\edrep{old}{new}` → new; remove the `% REVIEW R35` line. Revert: → old (un-brace the two `{\lr{…}}`).

R34 (§4.6.3, «بیش از ده مدل با دو اندازه انکودر») was accepted by the author on 2026-11-18 and is final.

## Noted in the final read, left unchanged (author decisions)

- The first paragraph of the Persian abstract is general, the English one names CAM-Convs / Metric3D / UniDepth / Depth Pro.
- Abstract: «از نظر علی ضروری»; §4.7.2: the findings «ادعای لزوم … را اثبات نمی‌کنند» (model-level reliance vs necessity in principle).
- `alg:training_pipeline` writes the camera encoder as `MLP(d_cam)`; the text defines it as MLP + four transformer blocks + gate.
- §4.6 opening: «این زنجیره همان AbsRel برابر ۰٫۲۰۷ را … (زیربخش 4.4.3)» — that subsection does not print 0.207.
