# Pending thesis review updates

Colour legend: green = added, yellow = new words in an edited paragraph,
red highlight (black text, no strike) = removed words and deleted parts,
gray = moved without word changes.
Find every region of an update with `grep -n 'REVIEW R<n>' thesis/tex/*.tex`.

## Finalized — 2026-11-17 — everything except chapter 5

Requested by the author: make every earlier change final (no highlights), except the recent chapter 5
edits. Resolved in the abstracts, chapters 1–4 and `figures/pending/*`: all review markup of R1–R25
(R1, R2, R3, R4, R5, R6, R7, R9, R10, R11, R12, R13, R14, R15, R18, R19, R20, R21, R24, R25) and the
one-word edits R28 and R29. New text kept; deleted text, red blocks, deleted rows/tables/figures and the
`% REVIEW` comment lines dropped; yellow figure frames (R19) and the commented-out control plot removed.
Table structure was rebuilt after the deletions: `tab:multidomain_all` now has one metric block
(3 columns, 8 test sets), `tab:md_encoder_protocol` and `tab:datasets_overview` lost the DIODE indoor
and Middlebury rows, `tab:training_hyperparams` has only the shared values, and `fig:dataset_samples`
shows NYUv2 centred above Cityscapes and KITTI. The build (`./thesis/build.sh`) succeeds without errors
or undefined references. Chapter 5 was left exactly as the parallel sessions had it (R27 was finalized by
its own session); its remaining marks are listed below. The macros in `tex/commands.tex` stay.

## Still pending (chapter 5 only)

- R30 (entry below): §3.2.4 one-word fix «مقیاس واقعی» → «هندسه تصویربرداری».
- R26 (full entry below): §5.2.2 research-question answers and Table 5.1.
- Older chapter 5 marks (R1, R2, R6, R7, R20) were finalized on 2026-11-18 at the author's request; only the marks of the last two days stay pending in `chapter5.tex`: R21 (four inline marks: summary paragraph, deleted sentence in the R26 block, future-work item), R25 (item 8, one sentence) and R26 (below).
  Finalize rule: `\edrep{old}{new}` → new, `\edadd{X}`/`\edins{X}`/`\edmove{X}` → X, drop `\eddel`,
  `eddeleted` blocks, `\eddelrow` rows and `\eddelbox` tables, unwrap `edadded`/`edmoved`, remove the
  `% REVIEW` lines.
- The last chapter 5 pass (R27, §5.3 limitations) is already final; its original wording is in git
  history (commit 0d2d87d).

## R26 — 2026-11-17 — §5.2.2 «پاسخ نهایی به سوالات تحقیق»: questions restated, answers rewritten, Table 5.1 shortened

Author requests (three, same session): (1) before each answer, write the complete research
question again on its own line; (2) check the answers against the whole thesis so that they match,
shorten the table `tab:hypothesis_verdicts`, show every hypothesis as accepted and do not use
«مشروط» in the table; (3) rewrite the answers so that they are fluent and easy to understand.

What was wrong in the old text (checked against chapter 1 and the current chapter 4): answer 1 gave
no evidence and answered only the first half of question 1; answer 2 used only the NYUv2 ViT-L pair
and ignored the multi-domain repetition; answer 3 said «مثبت اما مشروط», listed the relative-protocol
result as a condition although question 3 is about absolute evaluation, cited
`subsec:revised_cross_dataset` for a comparison that subsection no longer contains, and did not
mention the sensitivity tests. The content of each new answer and table cell was fixed first; Codex
(read-only) wrote the Persian wording and proofread the result.

Wording notes: the hypothesis-3 cell is about the ViT-L multi-domain relative pair (lower AbsRel
than the control on all eight sets, `tab:md_encoder_protocol`), because chapter 4 also reports two
small reliable deficits elsewhere (`tab:nyu_basic_aug`; two sets of the ViT-S relative pair).

All text regions are inside the gray R7 `edmoved` block of this subsection; the tables are outside it.

| # | File | Location (label) | Change | Kind |
|---|---|---|---|---|
| 1 | chapter5.tex | `sec:chapter5_research_questions_answers`, intro sentence | «بر اساس شواهد گزارش‌شده در فصل چهارم» → «بر پایه شواهد فصل چهارم»; «… را در کنار … قرار می‌دهد» → «…، شاهد اصلی و داوری نهایی هر فرضیه را نشان می‌دهد» | edited |
| 2 | chapter5.tex | old table, label renamed `tab:hypothesis_verdicts_old` | whole float wrapped in `\eddelbox` (its R1/R6 markup inside is untouched) | deleted table |
| 3 | chapter5.tex | `tab:hypothesis_verdicts` (new float after the old one) | short three-row table, every verdict «تأیید می‌شود» | added table |
| 4 | chapter5.tex | before each answer | three lines «سوال اول: …», «سوال دوم: …», «سوال سوم: …» (word for word from `sec:research_questions_ch1`) | added |
| 5 | chapter5.tex | after each question line | three new answer paragraphs | added |
| 6 | chapter5.tex | the three old answer paragraphs | each wrapped whole in `\eddel{}` | deleted |
| 7 | chapter5.tex | paragraph «بخشی از قید دوم …» | deleted (content is in the new answer 3); it is three adjacent `\eddel{}`: the first and third are R26, the middle one («و در آزمون جابه‌جایی دوربین …») is R21's and was not changed | deleted |

- Finalize R26: `\edrep{old}{new}` → new; delete the old table float (from the comment «REVIEW R26:
  old table deleted» to its `\end{table}`); `\edaddbox{X}` → X for the new table; `\edadd{X}` → X
  (un-brace `{\lr{…}}`, `{\ref{…}}`, `{\fadecimal{…}}`, `{\proposedmodel{}}` etc.); remove the four
  `\eddel{…}` paragraphs (for item 7 also the middle R21 `\eddel`, and R21's comment line above it);
  remove the `% REVIEW R26` lines.
- Revert R26: `\edrep{old}{new}` → old; remove `\eddelbox{%` and its closing `}% REVIEW R26: end of
  \eddelbox`, rename the label back to `tab:hypothesis_verdicts`; delete the new table float; delete
  the six `\edadd{…}` paragraphs (three questions, three answers); `\eddel{X}` → X for the three old
  answers and for the first and third piece of item 7 (un-brace the fragile units; the middle piece
  stays as R21 markup); remove the `% REVIEW R26` lines. Check against
  `git show 1f2833a:thesis/tex/chapter5.tex`.

## R31 — 2026-11-18 — §3 «مدیریت و نرمال‌سازی پارامترهای درونی دوربین»: duplicate 4-D d_cam removed, points to the 9-D descriptor

- `tex/chapter3.tex`, subsec:camera_descriptor (~l.76): added `\label{eq:camera_descriptor}` to the 9-D d_cam equation (compile dependency, no markup).
- `tex/chapter3.tex`, «مدیریت و نرمال‌سازی…» (~l.340–375):
  - edited: end of the "به‌منظور جلوگیری از ناپایداری عددی…" sentence now refers to §subsec:camera_descriptor and Eq. eq:camera_descriptor.
  - deleted: duplicate normalization equation (f̃x=fx/W …), its "که در آن W و H…" sentence, "بردار نهایی پارامترهای دوربین به‌صورت", the 4-D d_cam equation, and "تعریف شده و همراه با … تشکیل می‌دهد.".
  - edited: «این بردار» → «بردار d_cam».
- Note: the two deleted equations were `equation` (numbered); while pending they are shown as `equation*` inside `\eddelbox` so they take no number. On REVERT restore them as numbered `equation` environments.


## R30 — 2026-11-18 — §3.2.4 «مزایای مورد انتظار از ادغام صریح پارامترهای درونی»: camera "scale" → camera imaging geometry

| # | File | Location | Change | Kind |
|---|---|---|---|---|
| 1 | chapter3.tex | §3.2.4, first paragraph (~line 249) | «با در نظر گرفتن مقیاس واقعی دوربین» → «با در نظر گرفتن هندسه تصویربرداری دوربین» | edited |

Only use of «مقیاس» in the thesis that attributed a "scale" to the camera; all other uses (absolute/physical/scene/depth scale, scale alignment) were checked and left unchanged.
- Finalize: `\edrep{مقیاس واقعی}{هندسه تصویربرداری}` → `هندسه تصویربرداری`; remove the `% REVIEW R30` line.
- Revert: → `مقیاس واقعی`; remove the `% REVIEW R30` line.


## R32 — 2026-11-18 — remove every mention of Depth Anything 3 (author: the work does not use DA3 or its ideas)

| # | File | Location | Change | Kind |
|---|---|---|---|---|
| 1 | tex/chapter3.tex | §3.2.3, paragraph «این طراحی با راهکارهای پیشین تفاوت ساختاری دارد» (~l.116) | last sentence «ساختار این انکودر از CameraEnc در Depth Anything 3 … افزوده شده است.» | deleted |
| 2 | tex/enTitle.tex | English abstract, first paragraph (~l.30) | clause ", and Depth Anything 3 uses a camera token only for cross-view consistency with multi-view input" removed; " and" inserted before "UniDepth and Depth Pro" to close the list | deleted + edited |

The bib entry `DepthAnything3` in MyReferences.bib is left in place: with no `\cite` and no `\nocite{*}` it is not printed.
- Finalize: drop the `\eddel{…}` in both files (tidy the space before it), `\edins{ and}` → ` and`; remove the `% REVIEW R32` lines.
- Revert: `\eddel{X}` → X (un-brace `{\lr{…}}` / `~{\cite{…}}` in chapter 3); delete `\edins{ and}`; remove the `% REVIEW R32` lines.
