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

## R36 — 2026-11-18 — external judge changed (author's request)

| # | File | Location | Change | Kind |
|---|---|---|---|---|
| 1 | faTitle.tex | `\externaljudge` (committee page, p. 6) | «دکتر محمدرضا محمدی» → «دکتر محمدحسین رهبان»; rank «استادیار» unchanged | metadata, unmarked |
| 2 | faTitle.tex | acknowledgement | same name change, marked with `\edrep` | edited |

- Finalize: `\edrep{old}{new}` → new; remove the `% REVIEW R36` line. Revert: → old, and restore `\externaljudge{دکتر محمدرضا محمدی}`.

## R37 — 2026-11-18 — abstract rewritten to one page, keywords updated (author's request)

Author's brief: motivate the problem, state current solutions and their gap briefly (without
discussing earlier methods that take camera intrinsics), say what was used, how much it improved and
what the benefit is; one page in the final state. Content and numbers fixed by Claude from chapters
4–5; Persian wording by Codex (two wording rounds and one read-only review), English written to match.

| # | File | Location (printed page) | Change | Kind |
|---|---|---|---|---|
| 1 | faTitle.tex | `\faabstract` (p. 11–12 while marked) | three old paragraphs → three new paragraphs (motivation and gap; method; results and benefit) | deleted + added |
| 2 | faTitle.tex | `\keywords` | «MDE» → «تخمین عمق تک‌دوربینه»; «یادگیری عمیق بدون نظارت» removed (§3.4 says the fine-tuning is supervised); «ماتریس کالیبراسیون دوربین» added; «هندسه تصویربرداری» → «مدل بنیادین» | edited |
| 3 | enTitle.tex | `\enabstract` (English abstract, end of the PDF) | same rewrite in English | deleted + added |
| 4 | enTitle.tex | `\latinkeywords` | same keyword changes; each marked item is wrapped in `\texorpdfstring{markup}{final text}` because this field also feeds the PDF keywords | edited |

- Numbers used, all already in the thesis: AbsRel 0.1065 → 0.0834, about 22 % (`tab:nyu_absolute_focal`); lower AbsRel than the control on 8 of 8 test sets with ViT-S and ViT-L (`tab:md_encoder_protocol`); NYUv2 0.088 / 0.133 / 0.207 and the four-set, three-metric comparison (`tab:multidomain_main`).
- Left out on purpose to reach one page: the 0.06 → 0.97 slope and the 111 % removal test as numbers (the pinhole sentence stays in words), the four training-set names, the nine-dimensional descriptor, CAM-Convs / Metric3D / UniDepth / Depth Pro (English), «از نظر علی ضروری».
- This update settles the first two items of "Noted in the final read" above (the two abstracts now say the same thing; «ضروری» is gone).
- One-page check: with `\showreviewfalse` in an isolated copy, the Persian abstract with its keywords ends on one page (19 text lines, about one line to spare) and the English one on one page. While marked, old and new text together take more than one page.
- Finalize: drop both `eddeleted` blocks, keep the bodies of the `edadded` blocks without the environment lines; in `\keywords`: `\edrep{old}{new}` → new, drop `\eddel{…}`, `\edins{X}` → X; in `\latinkeywords`: each `\texorpdfstring{…}{X}` → X (the one with an empty second argument is dropped); remove the four `% REVIEW R37` lines.
- Revert: drop the `edadded` blocks, unwrap the `eddeleted` blocks (`\eddel{X}` → X, un-brace `{\lr{…}}`, `{\fadecimal{…}}`, `{$\delta_1$}`), restore both keyword lists; reference commit for the old text: ec0e0ad.

## R38 — 2026-11-18 — first-of-its-kind parenthetical removed from the conclusion (author's request)

| # | File | Location | Change | Kind |
|---|---|---|---|---|
| 1 | chapter5.tex | chapter 5 conclusion, sentence starting «روش پیشنهادی … ماتریس کالیبراسیون معلوم را» (line ≈ 218) | parenthetical «(تا آنجا که نویسنده اطلاع دارد، برای نخستین‌بار در یک مدل بنیادین عمق تک‌تصویری)» removed; the rest of the sentence is unchanged | deleted (inline, `\edrep{old}{}`) |

- Finalize: delete the `\edrep{…}{}` call and one of the two spaces around it; remove the `% REVIEW R38` line. Revert: `\edrep{old}{}` → old; remove the `% REVIEW R38` line.

## R39 — 2026-11-18 — DIODE outdoor scale error removed from the limitations (author's request)

Author's instruction: the DIODE outdoor scale-error discussion must not be in the limitations, and the
thesis should not keep pointing at it.

| # | File | Location | Change | Kind |
|---|---|---|---|---|
| 1 | chapter5.tex | §5.3 limitations, paragraph «مهم‌ترین محدودیت مدل نهایی…» | clause «؛ در DIODE بیرونی نیز خطای مقیاس سراسری با نسبت مقیاس ۰٫۶۱ دیده شد که علت آن تعیین نشد» removed; the sentence now ends at «هم‌تراز بود.» | deleted (inline, `\edrep{old}{}`) |
| 2 | chapter5.tex | `tab:limitations_future` | row «خطای مقیاس در DIODE بیرونی» removed | deleted row (`\eddelrow`) |

| 3 | chapter5.tex | §5.4 future work, paragraph «آزمایش چندمجموعه‌ای … مسیر مشخص دیگری نیز پیش می‌گذارد» | the two-item list is dropped: its DIODE item («بررسی منشا خطای مقیاس سراسری … بر DIODE بیرونی…») is removed and its first item (repeat with several seeds) is folded into one sentence; «چند مسیر مشخص دیگر» → «مسیر مشخص دیگری» | edited + deleted list (`eddeleted`) |
| 4 | chapter4.tex | §4.6, last sentence of the paragraph «در DIODE بیرونی، رتبه‌بندی سه مدل…» | «علت این خطای مقیاس در این پژوهش تعیین نشد و بررسی آن به کارهای آینده در بخش … واگذار می‌شود.» removed | deleted (inline, `\edrep{old}{}`) |

- Items 3 and 4 were chosen by the author in a follow-up question the same day. Still in the thesis by the author's choice: the rest of the §4.6 paragraph that explains the DIODE outdoor row of `tab:multidomain_all` (scale ratio 0.61) and the coefficient-of-variation sentence after `tab:md_encoder_protocol`.
- Codex read-only pass on item 3: meaning of the first list item kept, grammar and half-spaces fine. Its rewrite was not adopted (it moved «به‌ویژه» onto «حسگرهای دیده‌نشده» and dropped «مستقیما»). It noted that the sentence overlaps with the preceding paragraph on multiple seeds; that overlap was already there with the list.
- Finalize: items 1 and 4: delete the `\edrep{…}{}` call (item 4 with the space before it); item 2: delete the three `\eddelrow` row lines; item 3: `\edrep{old}{new}` → new (twice) and delete the whole `eddeleted` block; remove the five `% REVIEW R39` lines.
- Revert: items 1 and 4: `\edrep{old}{}` → old (un-brace `{\lr{DIODE}}`, `{\fadecimal{۰٫۶۱}}`, `{بخش~\ref{…}}`); item 2: drop `\eddelrow`, unwrap the three `\eddel{…}` cells; item 3: `\edrep{old}{new}` → old (twice), unwrap the `eddeleted` block and its two `\eddel{…}` (un-brace `{\proposedmodel{}}`, `{\lr{DIODE}}`); remove the five `% REVIEW R39` lines.

## R40 — 2026-11-18 — figures with overlapping boxes, labels or legends redrawn (author's request)

The author reported charts whose text and shapes ran into each other. Every vector figure was rendered
and checked; nine were redrawn. No number, label wording, colour or caption changed; only layout.
Figures are not colour-highlighted (same convention as the R33 figure fixes): each redrawn drawing
starts with a `% REVIEW R40 (figure redrawn: …)` comment line.

| # | File | Location (printed page) | Change | Kind |
|---|---|---|---|---|
| 1 | commands.tex | before the `thblock` style block | `\tikzset{every picture/.append style={execute at begin picture={\setstretch{1}}}}`: single line spacing inside every tikzpicture (the body's double spacing made multi-line nodes taller than their layout). Also tightens the multi-line nodes of the figures that were not redrawn (1.1, 2.3, 4.1) | added |
| 2 | chapter2.tex | `fig:scale_ambiguity`, Fig. 2.1 (p. 12) | panel titles moved inside taller panels; camera boxes placed relative to each other; equation no longer on the panel edge; «تصویر ثبت‌شده یکسان» broken as two lines | edited (layout) |
| 3 | chapter3.tex | `fig:proposed_architecture`, Fig. 3.1 (p. 37) | camera branch stacked below the encoder with relative positioning (boxes overlapped before); token-sequence caption above the tokens; loss note above the decoder; γ note beside the gate with a dotted connector; `\resizebox` replaced by `adjustbox` `max width=\textwidth` | edited (layout) |
| 4 | chapter3.tex | `fig:conditioning_mechanisms`, Fig. 3.2 (p. 38) | three equal panels 4.6 cm wide and tall enough for the notes (notes ran over the panel frame before) | edited (layout) |
| 5 | chapter3.tex | `fig:focal_jitter`, Fig. 3.4 (p. 48) | the α distribution sits on the branch between the input and the two samples (it covered both sample boxes before); arms placed below the samples | edited (layout) |
| 6 | chapter4.tex | `fig:eval_protocols`, Fig. 4.2 (p. 67) | three separate columns (the two note boxes covered the left and right columns before); protocol names on the horizontal run of each branch; orthogonal dashed connector to the absolute branch | edited (layout) |
| 7 | chapter4.tex | `fig:fov_generalization`, Fig. 4.4 (p. 70) | y ticks as fixed decimals (were «۹·۱۰⁻²»); legend moved to the empty lower-right corner (it covered the control line); marker fill follows the line colour | edited (layout) |
| 8 | chapter4.tex | `fig:intrinsics_sensitivity`, Fig. 4.8 (p. 76) | legend moved above the axis (it covered the response line); marker fill follows the line colour | edited (layout) |
| 9 | chapter4.tex | `fig:k_knockout`, Fig. 4.9 (p. 78) | legend moved above the axis in two columns; the floating label «خطای پایه مدل پیشنهادی در حالت صحیح» is now a legend entry for the dashed line; `ymax` 0.60 → 0.50 (headroom for the old legend no longer needed). The pending `% REVIEW R33` comment inside this figure is kept | edited (layout) |
| 10 | chapter4.tex | `fig:multidomain_absrel`, Fig. 4.10 (p. 80) | bars 18 pt wide with 3 pt gaps so each value label fits over its own bar (labels ran into the neighbouring bars); one swatch per legend entry | edited (layout) |

- Checked and left as they were: Figs. 1.1, 2.2, 2.3, 4.1, 4.3, 4.5 and the external PDF of Fig. 4.7.
- Method: each figure was drawn and rendered in an isolated copy with
  `.claude/skills/tikz-figure-review/scripts/figcheck.sh`, then `thesis/build.sh` was run and all fifteen
  vector figures were inspected in `main.pdf` (149 pages, no LaTeX error).
- Finalize: remove the ten `% REVIEW R40` comment lines (one in `commands.tex`, nine above the drawings).
- Revert: restore the drawing of each figure (the lines between `\centering` and `\caption`) and remove
  the `\tikzset{every picture…}` line with its comments; reference commit for the old drawings: 7e087c3.
