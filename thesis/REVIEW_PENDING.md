# Pending thesis review updates

Colour legend: green = added, yellow = new words in an edited paragraph,
red highlight (black text, no strike) = removed words and deleted parts.
Find every region of an update with `grep -n 'REVIEW R<n>' thesis/tex/*.tex`.


models with the proposed model's margin; zero-shot behaviour; control worse than
the released base; fixes raised earlier (Kinect v1 "unseen sensor" wording,
source of the base-model row, KITTI JPEG re-encoding disclosure); a fair,
defensible account of the test sets without progress; future work with the
time/compute limits; the extent of the experiments.

md5s_P5a, md5s_P6a, md7s_P7, released_*; `_zoom1.5` = camera-shift test;
`md1_P_kitti_jpeg95`, `*_kscale0.7/1.4`), paired Wilcoxon tests on per-image
metrics, training logs `logs/md*-*.log` on lingotube.

Marking note: paragraphs, table rows and abstracts that still carry the legacy
yellow background of the 2026-10-30 update (`edited`, `\edhlbox`, `\edrow`,
`\edcell`, `\edpar`) show new words in **green** (`\edadd`) rather than yellow,
because yellow is invisible on that background. There, a replacement is written
`\edold{old}\ \edadd{new}`: finalize → `new`; revert → `old` (drop the `\ `).

| File | Location (approx. line) | Change | Kind |
|---|---|---|---|
| chapter3.tex | §مجموعه‌داده‌های مورد استفاده, multi-domain paragraph (447) | sentence listing the six additional test sets | added words |
| chapter3.tex | same list, item «آزمون صفرنمونه» (456) | "new sensors" → "new data and sensors"; Kinect v1 = NYUv2's sensor model, new scenes/calibration; Kinect v2/RealSense/Xtion absent from training | edited |
| chapter3.tex | tab:datasets_overview rows KITTI, DIODE, Middlebury, Hypersim, SUN RGB-D (480–489) | roles updated (multi-domain test, Hypersim ViT-S test); SUN sensors list; "unseen sensors" → "unseen data" | edited cells |
| chapter3.tex | tab:multidomain_datasets (507–513) | KITTI test count 652; rows RealSense, Xtion, DIODE indoor/outdoor, Middlebury | added cell / rows |
| chapter4.tex | §sec:multidomain_results base-row paragraph (1067) | base row is our evaluation of the released checkpoints (paper's metric table = unreleased fine-tuned models; pipeline validated by tab:paper_reproduction) | added words |
| chapter4.tex | same section, test-set paragraph (1070) | "not seen in training" → no image used in training; Kinect v1 = NYUv2's sensor model, Kinect v2 absent | edited |
| chapter4.tex | same section (1077) | "two unseen sensors" → "unseen data" | edited |
| chapter4.tex | new subsection subsec:md_all_sets (1086–1178) | text, tab:multidomain_all (10 test sets × standard/shift), zero-shot, control-vs-base, sets without progress | added |
| chapter4.tex | new subsection subsec:md_campaign (1221–1258) | extent of experiments, screening variants, tab:md_runs | added |
| chapter4.tex | §جمع‌بندی فصل نتایج, multi-domain paragraph (1678) | wording fix; summary of the new findings | edited |
| chapter5.tex | limitations, multi-domain paragraph (573) | wording fix | edited |
| chapter5.tex | future work, multi-domain paragraph (707) | wording fix; Hypersim ViT-S test outcome | edited |
| chapter5.tex | future work, new block (713) | time/compute limits and five concrete next steps | added |
| chapter5.tex | final summary, multi-domain paragraph (789) | wording fix; replication across both pairs | edited |

No compile dependency was added (all labels new to this update are defined in
it; all cited keys already exist in MyReferences.bib).

## R2 — 2026-11-02 — §4.3.3 rewritten: cross-dataset stability of the proposed model

Requested by the author: rewrite §4.3.3 (`subsec:revised_cross_dataset`)
completely with enough data, and remove stale or duplicated cross-dataset /
zero-shot statements elsewhere. The per-set numbers are not repeated: they stay
in `tab:multidomain_all` (R1); §4.3.3 adds only set-level aggregates and tests.

Evidence: `results/multidomain/` on lingotube — main pair `md1_P`, `md1_C`,
released DA2 (`released_mix` = hypersim for indoor sets, vkitti for outdoor,
as in `tab:multidomain_all`) and `_zoom1.5` camera-shift counterparts;
supplementary K-blind arm `md1_C0`. Statistics from
`tools/multidomain/stability.py` (geometric-mean AbsRel ratio over the 10 sets
with a 5000-resample bootstrap over sets then paired images; per-set paired
Wilcoxon, alpha 0.01, uncorrected); outputs
`results/multidomain/stability_md1_final.json`. Paired Wilcoxon/bootstrap per
set from `tools/multidomain/compare_runs.py`.

| File | Location (approx. line) | Change | Kind |
|---|---|---|---|
| chapter4.tex | §4.3.3 body after the heading | old two paragraphs (t-test / "historical, cannot be judged") | deleted; author approved, removed permanently on 2026-11-06 |
| chapter4.tex | §4.3.3 (~278–345) | scope, method, `tab:stability_summary`, `tab:stability_paired`, results vs control/base, dispersion, camera shift, supplementary K-blind arm, verdict | added |
| chapter4.tex | discussion chapter, quantitative-summary subsection (~1481) | "significant reduction of sensitivity to dataset change" → accuracy gain without more uniform error | edited words |
| chapter4.tex | discussion chapter, answers to research questions (~1539) | "cross-dataset comparison not yet reproduced" → reproduced only in the multi-domain experiment | edited words |
| chapter5.tex | summary (~146) | "absolute stability maintained across indoor/outdoor" → mean accuracy gain, error not more uniform | edited words |

Revision 2026-11-06: the author asked for the green and yellow text of §4.3.3 to be
rewritten in plain, fluent Persian. All numbers, tests and claims are unchanged;
sentences were split, the «بازو»/«نقاب» wording replaced, the reference to the
deleted "historical report" dropped, and the legacy yellow opening sentence
(`edited` block) rewritten. The two tables were not touched.

Not touched (reported to the author instead): the R1 restatements of the
SUN RGB-D / NYUv2 numbers in the ch4 discussion and chapter summaries, ch5
summary/contributions and the abstracts.

## R3 — 2026-11-09 — §4.3.3 shortened (further edit inside the R2 text)

Requested by the author: make §4.3.3 shorter and drop the tests that make the
proposed model look worse. Done as a condensation: removed the post-hoc
«without Middlebury» sensitivity paragraph, the percent-change camera-shift
paragraph and the two «mean after shift» / «relative change» columns of
`tab:stability_summary` (both are unreferenced after the cut and, per the text
itself, are not a clean degradation measure), and shortened the setup, the
control comparison, the shift-vs-published paragraph and the supplementary
K-blind-model paragraphs. Deliberately kept, in one sentence each (they bound
the headline claim and are cited in chapter 5): CV 0.85 vs 0.59/0.69 with
Middlebury as its main cause, the CI [0.67, 1.20] including 1, losses to the
control on DIODE indoor and Middlebury, the camera-shift tie with the control
(1.03), and the supplementary model without scale augmentation (1.14 on
original images).

Evidence: unchanged from R2 (no new numbers; every figure in the new text was
already in R2).

| File | Location | Change | Kind |
|---|---|---|---|
| chapter4.tex | §4.3.3 opening block (~282) | four R2 paragraphs replaced by three shorter ones | deleted + added |
| chapter4.tex | `tab:stability_summary` (~305–316) | caption tail and the last two columns | deleted cells |
| chapter4.tex | after the table (~319) | two R2 paragraphs on original-image results replaced by one | deleted + added |
| chapter4.tex | dispersion / shift / supplementary model (~358–381) | paragraphs 2–5 replaced by two shorter ones | deleted + added |

Revert note: reverting R3 unwraps the red `eddeleted` paragraphs to plain text;
re-wrap them in `% REVIEW R2` / `\begin{edadded}` so they stay green as before.

## R4 — 2026-11-06 — Qualitative depth figures for the multi-domain campaign

Requested by the author: image comparisons of our model against the control, the
base model and other models, in the style of Depth Anything V2 and related work.
Only proposed / control / released base are included; other published models were
not run (executing their third-party code on the server was not approved).

Evidence: locked sample `thesis/figures/qualitative_md/manifest.json` (seed
20261106, 6 images per set, SHA-256 4c11bcdc9b21…); predictions
`results/qualitative_md/pred/` on lingotube and locally (`tools/multidomain/eval_related.py`);
figures from `tools/multidomain/make_qualitative_md_render.py`; panel values and
win counts in `thesis/figures/qualitative_md/geometry.json`; median scale ratios
in `logs/qual-dump.log` on lingotube. All 144 dumped AbsRel values match
`results/multidomain/{md1_P,md1_C,released_mix}{,_zoom1.5}` within 2.5e-6.

| File | Location (approx. line) | Change | Kind |
|---|---|---|---|
| chapter4.tex | §sec:qualitative_results, new subsection subsec:qualitative_multidomain (1340) | protocol, reading guide, results, detail caveat, camera-shift result | added |
| figures/pending/qualitative_md_figures.tex | new file, `\input` from the subsection | fig:qual_md_nyu, _kv1, _kv2, _cityscapes, _error, _shift | added |
| figures/qualitative_md/ | new directory | six figure PDFs, manifest.json, geometry.json | added (compile dependency) |

## R5 — 2026-11-13 — Paragraph-level Persian editing from §4.4.7 onward

Requested by the author: start at §۴.۴.۷ «پیکربندی لرزش کانونی: جهش سهم پارامترهای دوربین» and check every subsequent thesis paragraph one by one with the paragraph-editor workflow, changing only text that needs clearer or more fluent Persian.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter4.tex | §۴.۴.۷ and integrated configuration summary through §۴.۴.۹ | agreement, statistical plurality, fluency, repeated wording, and spelling corrections | edited |
| chapter4.tex | camera-sensitivity section through the multi-domain section | spelling, punctuation, grammatical agreement, and one malformed clause repaired | edited |
| chapter4.tex | multi-domain results through chapter conclusion | agreement, spelling, terminology, and CityScapes limitation wording clarified | edited |

All scientific claims, numbers, metrics, tests, citations, equations, labels, and section structure were preserved. Review markers are tagged `% REVIEW R5` in `thesis/tex/chapter4.tex`. The mandated `./thesis/build.sh` validation was attempted after editing but could not run because `latexmk` is unavailable in the environment (`command not found`).

## R6 — 2026-11-13 — Paragraph-level Persian editing through the end of the thesis

Requested by the author: continue the paragraph-editor review after Chapter 4 through the end of the included thesis source. The review covered all of `thesis/tex/chapter5.tex`, from the introduction through the final summary, checking each paragraph in sequence with its section context.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter5.tex | Introduction, main findings, and final research-question answers | spelling, agreement, punctuation, and wording corrections | edited |
| chapter5.tex | Final interpretation, innovations, and limitations | spelling, agreement, and fluency corrections | edited |
| chapter5.tex | Future work and final chapter summary | spelling, agreement, and fluency corrections | edited |

All scientific claims, numbers, metrics, citations, equations, labels, and existing review markup were preserved. New review markers are tagged `% REVIEW R6` in `thesis/tex/chapter5.tex`. The appendices remain excluded because `main.tex` comments out their include lines. The mandated `./thesis/build.sh` validation remains unavailable because `latexmk` is not installed in the environment.

## R7 — 2026-11-15 — Restructuring of chapters 2–5 per `thesis/table-edit.md`

Requested by the author: apply the target structure and per-section instructions of
`thesis/table-edit.md` (compared against `thesis/table.md`), add the ViT-S / ViT-L,
absolute / relative multi-domain results, and mark everything with the colour rule.

Author decisions taken on 2026-11-15 (asked before editing):
- Older marks (R1–R6, legacy yellow) stay where the text is untouched; inside blocks
  that R7 moves or deletes they were flattened to their accepted form.
- **Gray** (`\edmove`, `edmoved`, `\edmovebox`, `\edmoverow`; new macros in
  `commands.tex`) = text moved without word changes, shown only at its new place.
  Green / yellow / red keep their meaning. Deleted headings are starred
  (`\section*{\eddel{…}}`) so the PDF shows the target numbering.
- All new or reworded Persian (and the English abstract sentence) was written by Codex.

Evidence: `results/multidomain/VITS_VS_VITL_MULTIDOMAIN.md` and
`vits_vs_vitl_multidomain.csv`; per-image files `results/multidomain/<run>/<set>.json`
`*_noK`); paired Wilcoxon p-values recomputed and equal to the CSV; per-set threshold
0.01, uncorrected. Training settings in `tab:experiment_settings` come from the logged
argument dictionaries and checkpoint configs on lingotube (`logs/train-nyu-pair.log`,
`train-vitl-disp.log`, `train-vitl-aug.log`, `metric-pair2.log`,
`train-focal-ablation-pair.log`, `md1-*.log`, `md2s-*.log`, `md7-*.log`, `mdrel.log`,
`mdrels-train-*.log`).

| File | Location | Change | Kind |
|---|---|---|---|
| commands.tex | review-macro block | gray moved-text macros | added (compile dependency) |
| chapter2.tex | 2.1.1 | `fig:scale_ambiguity`, metric definitions from old 3.8 (`sec:eval_metrics`) | moved |
| chapter2.tex | 2.1.2 | `fig:pinhole_model`, lens-distortion and homogeneous-component sentences | moved |
| chapter2.tex | 2.1.3 | old 3.9.1 / 3.9.2 merged; SiLog form without the radical removed | moved + deleted |
| chapter2.tex | new section 2.3 (`sec:ch2_base_model`), 2.3.1–2.3.8 | old 2.2.7 split under eight new headings, with sentences from old 3.2, 3.7.2, 3.7.5; new label `subsec:ch2_base_output` | added headings + moved |
| chapter3.tex | intro, summary | roadmap and summary agree with the new layout | edited |
| chapter3.tex | old 3.2, 3.3, 3.4, 3.6.2, 3.7.3, 3.8, 3.9.1, 3.9.2, 3.9.6 | removed sections (content moved as the plan says) | deleted |
| chapter3.tex | design section start; 3.2.3; 3.2.4 | f(I,K)→D definition; fixed-injection and future-work sentences; RMSE reference | moved + added + edited |
| chapter3.tex | 3.3.1, datasets table, R1 paragraph | wording per plan; Cityscapes case-study phrase and KITTI JPEG-95 sentence | edited + deleted |
| chapter3.tex | 3.4 opening, 3.4.1–3.4.4 | controlled-study wording, retitle, script/option names removed | edited |
| chapter3.tex | 3.5 heading and opening, 3.5.1, 3.5.2 | retitle, sentence from old 3.9.6, reference updates, SiLog and RMSE-stability sentences (`subsec:rmse_sensitivity`) | edited + moved |
| chapter4.tex | 4.1 | roadmap paragraph naming the two stages | added |
| chapter4.tex | 4.2 opening, 4.2.1 seed sentence | wording; seed fixed only in the focal-jitter configuration and multi-domain training | edited |
| chapter4.tex | 4.2.2 | retitled; schedule paragraph (no warmup) and table intro rewritten; `tab:training_hyperparams` limited to shared values; new `tab:experiment_settings` | edited + deleted + added |
| chapter4.tex | old 4.2.3, 4.2.4 | code structure and debugging subsections | deleted |
| chapter4.tex | new 4.3 (`sec:experiment_path`), 4.3.1, 4.3.2 | new heading; old 4.4.1 (journey table row 2 removed, seven → six steps, figure updated) and old 4.4.2 | added + moved + edited + deleted |
| chapter4.tex | 4.4 | retitled, subsection heading dissolved, old 4.3.2 with its table and figure | edited + deleted |
| chapter4.tex | 4.5 heading; old 4.4.3; 4.5.3; 4.5.4; new 4.5.5 (`subsec:qualitative_controlled`); 4.5.6 | retitle; zero-shot subsection; table reference fix; dose-response block and effect-size figure merged, uniformity claim softened; qualitative text re-homed; zero-shot sentence | edited + deleted + moved + added |
| chapter4.tex | 4.6 | «فصل سوم» → «فصل دوم»; two last subsections | edited + deleted |
| chapter4.tex | 4.7 heading and intro | «گام پایانی:»; `tab:multidomain_main` and its paragraph; two CityScapes sentences | edited + moved |
| chapter4.tex | 4.7.2 `subsec:revised_cross_dataset` | retitled; new text and `tab:md_encoder_protocol`; old text and `tab:stability_summary`, `tab:stability_paired` | added + deleted |
| chapter4.tex | 4.7.3 `tab:md_runs` and text | rows 11–12 (relative pairs), 13 → 17 models | edited + added |
| chapter4.tex | old qualitative section, CityScapes section, old 4.9.1, 4.9.2, 4.9.5 | headings and text | deleted (parts moved to 4.5.5, 4.7 intro, chapter 5) |
| chapter4.tex | 4.8 opening, 4.8.1; 4.9 | wording, multi-domain sentences; summary reduced to one paragraph | edited + added + deleted |
| chapter5.tex | 5.1 | shortened; structure sentence | deleted + added + edited |
| chapter5.tex | new 5.2 «نتیجه‌گیری» (`sec:chapter5_conclusion`), 5.2.1–5.2.3 | old 5.2, 5.3, 5.5 become subsections; sentences from chapter 4 and old 5.4 / 5.8; multi-domain paragraph rewritten; hypothesis table cell; novelty subsections merged | added + moved + edited + deleted |
| chapter5.tex | old 5.4, 5.6.1, 5.8 | headings and remaining text | deleted |
| chapter5.tex | 5.3, `tab:limitations_future`; 5.4 | retitles; merged limitation paragraphs; table row; multi-camera paragraph rewritten; list items | edited + deleted + added |

Labels added: `sec:experiment_path`, `subsec:qualitative_controlled`,
`tab:experiment_settings`, `tab:md_encoder_protocol`, `sec:chapter5_conclusion`,
`subsec:ch2_base_output`. Labels kept only on red blocks (they disappear on finalize,
`tab:da2_revised_comparison`, `fig:cross_dataset_silog`, `tab:zeroshot_failure`.

Not built: there is no TeX on the editing machine; source-level checks only (brace
and environment balance, no nested review macros, no undefined or duplicate labels,
all cited keys present).

Revert of R7: nothing was committed, so
`git checkout edd45b2 -- thesis/tex/chapter2.tex thesis/tex/chapter3.tex thesis/tex/chapter4.tex thesis/tex/chapter5.tex thesis/tex/faTitle.tex thesis/tex/enTitle.tex thesis/tex/commands.tex`
restores the pre-update text exactly (the R7 moves leave no trace at the old places,
so a region-by-region revert is not possible). Finalize: as for other IDs, plus
`\edmove{X}` → X, `edmoved` → body, `\edmovebox{X}` → X, drop `\edmoverow` /
`\edmovecell`, and turn nothing back for starred red headings (they are removed).

R7 additions after the author's answers (2026-11-15, same ID):

| File | Location | Change | Kind |
|---|---|---|---|
| chapter3.tex | 3.4.3, last sentence | «گرم‌سازی خطی» → no warmup (every run logged `warmup_epochs: 0`) | edited |
| chapter3.tex | 3.4.7 multi-domain paragraph (legacy yellow block) | «برابری بیت‌به‌بیت هزینه گام صفر» → agreement to at least four significant digits | edited (`\edold` + `\edadd`) |
| chapter4.tex | 4.5.1 opening | ViT-L pair trained with the inverse-depth target, ViT-S pair with the direct depth target | edited |
| chapter4.tex | 4.5.3 | three sentences → one: AbsRel 0.0817 vs 0.0826 only; «هر سه معیار» and the released-model claim dropped | edited |

Open points reported to the author, not changed: wording of `tab:datasets_overview`
cells («نهایی», «مرحله اکتشافی», «ارزیابی صفرنمونه»); the chapter-1 roadmap figure
node for chapter 3; mixed depth notation in 2.1.1; the 4.5 intro «در مرحله دوم»;
caption «زوج اصلی» of `tab:multidomain_all`; row 2 text of `tab:md_runs`; the
`tab:limitations_future` cell on repeating the controlled study on KITTI; position of
the closing sentence of 5.2.

## R8 — 2026-11-16 — Section 1.7 «روش انجام تحقیق» restored

Requested by the author: the section had been a bare heading since commit `076d15d`
(2026-10-25), which moved its only paragraph into 1.3 «تعریف موضوع تحقیق».

| File | Location | Change | Kind |
|---|---|---|---|
| chapter1.tex | 1.3 → 1.7 | paragraph «…خلا مورد هدف این پژوهش…» moved back under «روش انجام تحقیق» (gray, `edmoved`); nothing left at 1.3 | moved |
| chapter1.tex | 1.7, opening | «با توجه به این مساله و فقدان تفکیک سهم علی ماتریس کالیبراسیون دوربین در پژوهش‌های پیشین،» → «بر این اساس،» (the pre-`076d15d` opening) | edited (`\edrep`) |

The rest of the paragraph keeps its current wording («مطلق», «انکودر \lr{DINOv2}»),
not the October wording («متریک», «هسته یک مدل بنیادین»). Not built.

Also in the working tree, no review ID (build fix, no visible change): `chapter4.tex`
`tab:training_hyperparams` caption given a plain short form for the list of tables.

- Finalize R8: drop the `edmoved` wrapper and the `% REVIEW R8` line; `\edrep{old}{new}` → new.
- Revert R8: move the paragraph back to the end of 1.3 with its old opening.

## R9 — 2026-11-16 — Section 4.4 merged into 4.3 as subsection 4.3.3

Requested by the author: the baseline-behaviour section «نقطه شروع…» was a separate
top-level section right after the experiment-path section 4.3; it is now its third
subsection.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter4.tex | old 4.4 → 4.3.3 | `\section` → `\subsection`; title unchanged (still marked `\edrep`); body untouched | structure |
| chapter4.tex | 4.3.3 opening | «در این بخش» / «در این بخش» → «زیربخش» (two places, `\edrep`) | edited |
| chapter4.tex | 4.3 intro | new closing sentence (`\edadd`): the baseline behaviour is reported at the end of 4.3 as a starting point, not a final ranking | added |
| chapter4.tex | 4.1 roadmap | the sentence listing the order of sections no longer treats the baseline as its own section (inside the existing `edadded`) | edited |

Side effect: all later chapter 4 sections shift down by one (old 4.5→4.4 … old 4.9→4.8).
All cross-references use `\ref`, so they follow automatically; no text hard-codes these numbers.

- Finalize R9: drop the `% REVIEW R9` lines and the two `\edrep{بخش}{زیربخش}` wrappers (keep «زیربخش»).
- Revert R9: change `\subsection` back to `\section` at `sec:quantitative_results`, remove the added 4.3 sentence, restore «بخش» and the 4.1 roadmap wording.

## R10 — 2026-11-16 — Chapter 3 intro: two «فصل دوم آمده است» sentences merged

| File | Location | Change | Kind |
|---|---|---|---|
| chapter3.tex | 3.1, «در ادامه این فصل…» paragraph | «فرمول‌بندی دقیق مساله و مدل هندسی دوربین در فصل دوم آمده است. جزییات عمومی مدل پایه نیز در فصل دوم آمده است؛» → «فرمول‌بندی دقیق مساله، مدل هندسی دوربین و جزییات عمومی مدل پایه در فصل دوم آمده است؛» (new words `\edins`) | edited |

- Finalize R10: drop the `\edins{…}` wrapper. Revert R10: restore the two sentences with the earlier `\edins{…}` pieces.

## Finalized — 2026-11-16 — everything up to the end of subsection 2.3.6

Requested by the author («تا انتهای صفحه ۴۳ پی‌دی‌اف»: page 43 of `last-highlight.pdf`, which ends inside 2.3.6).
All review markup was resolved in: the Persian and English abstracts (`faTitle.tex`, `enTitle.tex`),
chapter 1 (including R8) and chapter 2 up to the end of 2.3.6 (new text kept, deleted text and
red-background blocks dropped, `% REVIEW` comment lines removed). Entries R1, R7 and R8 above are
therefore settled for that range; their rows for chapter 2 sections from 2.3.7 on, and for chapters
3–5, are still pending. R9 and R10 are untouched.

## R11 — 2026-11-16 — Section 3.3.1: the two dataset tables merged into one

Requested by the author: the dataset overview table (old 3.2) and the multi-domain
dataset table (old 3.3) described the same datasets, so they were merged.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter3.tex | 3.3.1 | new merged table `tab:datasets_overview` («مجموعه‌داده‌های پژوهش: مشخصات دوربین، اندازه بخش‌ها و نقش هر یک»): environment + depth source, `f_y/H`, horizontal FoV, train/val/test sizes, role; 14 rows (all datasets of both tables, DIODE and SUN RGB-D split per sensor/scene, Hypersim as checkpoint row) (green `\edaddbox`) | added |
| chapter3.tex | 3.3.1 | old tables kept in place as red `\eddelbox` blocks (no labels, no footnote); they vanish in the final build | deleted |
| chapter3.tex | table intro | words «اندازه بخش‌های آموزش، اعتبارسنجی و آزمون،» and a sentence that all datasets provide the camera matrix (`\edadd`) | added |

Dropped as unimportant: the «دسترسی به ماتریس کالیبراسیون دوربین» column (every row was «دارد»; the per-row notes «ثابت برای کل مجموعه», «از فایل کالیبراسیون», «دو نمای کالیبره‌شده», «برای هر تصویر»), and the separate «نوع محیط» column (merged into «محیط و منبع عمق مرجع»). Role texts were shortened.
Cross-references: `\ref{tab:multidomain_datasets}` (3 places: chapter3 ×2, chapter4 ×1) now point to `tab:datasets_overview`; chapter4.tex changed only in that label.

- Finalize R11: drop `\edaddbox` wrapper and the two `\eddelbox` tables.
- Revert R11: remove the new table, unwrap the two old tables (restore their labels, the `\footnotemark` and `\LTRfootnotetext`), restore `tab:multidomain_datasets` refs.

## R12 — 2026-11-16 — Section 3.3.1: the five reasons for the dataset combination summarized

Requested by the author: summarize the five reasons («این ترکیب به پنج دلیل انتخاب شد»), keeping all five, and drop unimportant detail
and anything the dataset overview table already shows. The Persian wording was written with codex.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter3.tex | 3.3.1, list after «این ترکیب به پنج دلیل انتخاب شد:» | the original five-item list kept as a red `eddeleted` block (the R1 markup inside item 5 was flattened to its proposed text because `ulem` cannot nest; original below) | deleted |
| chapter3.tex | same place | shorter five-item list (green `edadded`): `f_y/H` and field-of-view ranges, image counts (۳۷۸۶۰, ۶۵۴), per-source listing of depth sensors, B3DO/exclusion details and the Kinect v2 field-of-view comparison removed because the table shows them | added |
| chapter3.tex | same place | `\end{edited}` moved from after the list to right after «این ترکیب به پنج دلیل انتخاب شد:» (a shaded box cannot be nested inside the legacy yellow `edited` box) | structure |

- Finalize R12: delete the `eddeleted` block, drop the `edadded` wrapper, remove `% REVIEW R12`.
- Revert R12: delete the `edadded` block, remove the `\end{edited}` that follows the intro line, and put the original lines below back (the last `\end{edited}` returns after `\end{itemize}`). Full original source (with the R1 markup in item 5) is saved in the session backup and reproduced here:

```latex
\begin{itemize}
 \item تنوع واقعی دوربین: نسبت $f_y/H$ در چهار دوربین آموزشی از \fadecimal{۱٫۰۸} تا \fadecimal{۲٫۲۱} و میدان دید افقی آن‌ها از حدود ۴۹ تا ۸۲ درجه تغییر می‌کند. بنابراین بردار توصیف‌گر دوربین، برخلاف آموزش تک‌دوربینه \lr{NYUv2}، مقادیر واقعا متفاوتی می‌گیرد و تنوع هندسی فقط از راه افزونه‌سازی شبیه‌سازی نمی‌شود.
 \item پوشش هم‌زمان صحنه‌های داخلی و بیرونی: \lr{NYUv2} صحنه‌های داخلی را با عمق تا حدود ۱۰ متر و سه مجموعه دیگر صحنه‌های جاده‌ای و شهری را با عمق تا ۸۰ متر پوشش می‌دهند.
 \item مکمل‌بودن منابع عمق مرجع: حسگر \lr{Kinect} در \lr{NYUv2}، لایدار در \lr{KITTI}، تطبیق استریو در \lr{Cityscapes} و عمق رندرشده دقیق و چگال در \lr{Virtual KITTI 2}؛ حجم بالای داده مصنوعی (۳۷۸۶۰ تصویر) کم‌بودن برچسب چگال در داده‌های واقعی بیرونی را جبران می‌کند.
 \item امکان مقایسه با ادبیات: بخش آزمون ۶۵۴ تصویری \lr{NYUv2} همان پروتکل استاندارد مقاله مرجع است~\cite{Eigen2014Depth,DepthAnythingV2}.
% REVIEW R1
 \item آزمون صفرنمونه بر \edold{حسگرهای تازه}\ \edadd{داده‌ها و حسگرهای تازه}: تصاویر \lr{Kinect v1} از بخش \lr{B3DO} مجموعه‌داده \lr{SUN RGB-D} برداشته شدند و تصاویری که این مجموعه از \lr{NYUv2} وام گرفته است کنار گذاشته شدند تا هیچ هم‌پوشانی با داده آموزش پیش نیاید. \edadd{حسگر {\lr{Kinect v1}} همان مدل حسگری است که {\lr{NYUv2}} با آن ثبت شده است؛ بنابراین در این مجموعه، صحنه‌ها و کالیبراسیون دوربین تازه‌اند ({$f_y/H$} برابر {\fadecimal{۱٫۲۲}} در برابر {\fadecimal{۱٫۰۸}})، نه خود حسگر. سه حسگر {\lr{Kinect v2}}، {\lr{RealSense}} و {\lr{Xtion}} در هیچ‌یک از مجموعه‌داده‌های آموزشی وجود ندارند.} حسگر \lr{Kinect v2} با $f_y/H\approx1.00$ میدان دید عمودی حدود ۵۳ درجه دارد که از همه دوربین‌های آموزشی (حداکثر حدود ۵۰ درجه در \lr{NYUv2}) بازتر است.
\end{itemize}
\end{edited}
```

R12, second part — the dense paragraph after the dataset table («در جدول ۳.۲، تصاویر آموزشی NYUv2…»), also in 3.3.1:

| File | Location | Change | Kind |
|---|---|---|---|
| chapter3.tex | 3.3.1, paragraph after the tables | original paragraph kept as a red `eddeleted` block (its R1/R7 markup flattened; the already-deleted KITTI JPEG sentence is included in it) | deleted |
| chapter3.tex | same place | shorter paragraph (green `edadded`): NYUv2 80/20 split, held-out validation parts, test sets and protocols (raw depth + crop, KITTI list from the reference code, Middlebury ×2 downscaling), Cityscapes disparity depth and hood exclusion, depth ranges; removed: counts shown in the table, scene/drive/city names, the list of later-added test sets | added |
| chapter3.tex | same place | `\begin{edited}`…`\end{edited}` around the old paragraph replaced by the two blocks | structure |

- Finalize: delete the `eddeleted` block, drop the `edadded` wrapper, remove `% REVIEW R12/R1/R7` lines there.
- Revert: delete both new blocks and put back this original (with its R1/R7 markup):

```latex
\begin{edited}
% REVIEW R1
% REVIEW R7
\end{edited}
```

## R12 — 2026-11-16 — Section 3.6 «جمع‌بندی فصل»: second paragraph merged into the first

| File | Location | Change | Kind |
|---|---|---|---|
| chapter3.tex | 3.6 | the separate paragraph «در گام پایانی، همین چارچوب به آموزش چندمجموعه‌ای…» (was after the sentence announcing chapter 4) moved into the first paragraph, just before «فصل چهارم بر پایهٔ این چارچوب…»; its last verb «ممکن می‌کنند» → «ممکن کردند» to match the past tense of the paragraph (`\edins`) | edited |

- Finalize R12: drop the `\edins` wrapper. Revert R12: move the sentence back to its own yellow paragraph after the chapter-4 sentence, with «ممکن می‌کنند».

## Finalized — 2026-11-16 — chapters 1–3 (front matter, chapter 1, chapter 2, chapter 3)

Requested by the author after their review («همه موارد کاملا درست بود»). All review markup in the abstracts,
chapter 1, chapter 2 and chapter 3 is resolved (new text kept; deleted text, red blocks and the two old dataset
tables of R11 dropped; `% REVIEW` lines removed). This settles R1, R7 and R8 and also R10, R11 and R12 for those
chapters. **Still pending:** everything in chapter 4 (including R9) and chapter 5, plus the markup macros in
`tex/commands.tex` (kept for those chapters).

## R13 — 2026-11-16 — «دوربین‌های واقعی» → «دوربین‌های متفاوت» for the multi-domain training

Reason (author): the multi-domain training still applies the simulated camera augmentations and
Virtual KITTI 2 is rendered, so «واقعی» overstated it.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter3.tex | 3.3.1, 3.4.7 title and opening, 3.6 | «دوربین‌های واقعی (متفاوت)» → «دوربین‌های متفاوت»; «ماتریس کالیبراسیون دوربین واقعی خود» → «…دوربین خود»; 3.4.7 opening now says the real calibration differences are added on top of the same augmentations | plain edits (chapter already finalized) |
| chapter4.tex | 4.1 roadmap (inside existing green block) | «با دوربین‌های واقعی» → «با دوربین‌های متفاوت» | edited |
| chapter4.tex | 4.6 intro | «چند دوربین واقعی» → «چند دوربین متفاوت» (`\edrep`) | edited |
| chapter5.tex | contributions table, methodological row | «با دوربین‌های واقعی» → «با دوربین‌های متفاوت» (`\edrep`) | edited |

Left unchanged on purpose: «چهار دوربین واقعی و مصنوعی» / «دوربین‌های واقعی و مصنوعی متفاوت» (already accurate),
«اثرات ثانویه دوربین‌های واقعی» (lens effects), «نه از دوربین‌های واقعی متفاوت» in the limitations, «به دوربین واقعی بازگردانده می‌شود»
(canonical-focal), and text inside deleted blocks.

## R14 — 2026-11-16 — Subsection 4.3.1 (experiment path) rewritten to match the current chapter

Requested by the author: the path table no longer matched the thesis.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter4.tex | 4.3.1 title | back to «…مسیر رسیدن به آزمایش نهایی» (the path now ends at the multi-domain step) | edited |
| chapter4.tex | 4.3.1 intro | rewritten: eight steps, in the order of the chapter's sections (green block) | rewritten |
| chapter4.tex | `tab:experiment_journey` | all rows rewritten (green rows), each pointing to the section that reports it: 1 audit + paper reproduction; 2 baseline instability (new); 3 relative four-way comparison (ViT-S small gain, ViT-L none); 4 zoom-crop in the relative protocol; 5 absolute evaluation (0.0817 vs 0.0826); 6 focal jitter (gain 0.0009→0.0231, sweep slope 0.06→0.97); 7 matrix removal (+111%); 8 multi-domain training (new) | rewritten |
| chapter4.tex | phase summary + `fig:experiment_phases` | three phases → four (added «تعمیم», row 8); caption rewritten (green) | rewritten |

Conflicts removed: the old zoom-crop row described a crop/matrix mismatch bug (the chapter introduces zoom-crop to give
the constant camera descriptor some variety); the old absolute-evaluation row claimed a trend over «تنوع میدان دید» (the
chapter reports one small gain); the baseline step and the final multi-domain step were missing.
**Not shown in the PDF:** the previous rows, intro and phase paragraph were replaced outright, not kept as red text
(they are in git at `c60d338` and in the pre-R7 history). The closing «چهار شرط» paragraph is unchanged.

- Finalize R14: drop `\edaddrow`, the `edadded` wrappers, `\edaddbox` and `\edmove` on the title.
- Revert R14: restore the subsection from `git show c60d338:thesis/tex/chapter4.tex` (lines of `subsec:checkpoint_audit`).

## R15 — 2026-11-16 — Bar chart of the final multi-domain result (AbsRel)

Requested by the author: add a bar chart like the old cross-dataset one, in the right section, for the final result.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter4.tex | §sec:multidomain_results, after the paragraph that introduces `tab:multidomain_main` (~1196) | one-sentence pointer to the chart | added (green block) |
| chapter4.tex | same place | `fig:multidomain_absrel`: grouped bars, AbsRel of the published base, control and proposed models (ViT-L, absolute, no alignment) on NYUv2, Cityscapes, SUN RGB-D Kinect v1 and Kinect v2; values copied from `tab:multidomain_main` | added (green box) |

The old chart `fig:cross_dataset_silog` (red, deleted in R7) is not restored: its proposed-model numbers did not reproduce.

- Finalize R15: drop `edadded`, `\edadd` and `\edaddbox` wrappers and the `% REVIEW R15` comments.
- Revert R15: delete the `edadded` block and the whole `figure` (both tagged `% REVIEW R15`).

## R15 — 2026-11-16 — Section 4.4.1: closing paragraph for the four-way comparison tables

| File | Location | Change | Kind |
|---|---|---|---|
| chapter4.tex | 4.4.1, after `tab:nyu_basic_vitl` | new paragraph (green block), without numbers: proposed beats the published model in both sizes; after fine-tuning, proposed and control perform almost the same (ViT-S slightly better, ViT-L tied); likely reason is the constant camera matrix on single-camera NYUv2, so zoom-crop augmentation was added next (4.4.2) | added |

- Finalize R15: drop the `edadded` wrapper. Revert R15: delete the paragraph.

## R18 — 2026-11-17 — Evaluation metrics made consistent: AbsRel (+ δ1) is the main metric

Requested by the author: remove the contradictions about AbsRel / RMSE / SiLog and use AbsRel as the main metric for absolute models.
Evidence: `src/metrics.py` (AbsRel, SqRel, RMSE, log10, δ, SiLog with λ=0.5), `train.py:104` (SiLog λ=0.5). Codex pass ran. Tag R16 is already used (unlogged), R17 was finalized and removed.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter2.tex | 2.1.1 (`sec:eval_metrics`), AbsRel sentence (~91) | «به مقیاس مطلق صحنه حساسیت کمتری» → unitless / independent of depth range; new sentences: AbsRel = \|s−1\| for a global scale error, main metric, asymmetric, so δ1 is reported with it | edited |
| chapter2.tex | 2.1.1, RMSE paragraph (~95) | «حساس به پارامترهای درونی دوربین / فاصله کانونی» → in metres, grows with depth range, compared only within one dataset; focal-length scale error shows in both metrics | edited |
| chapter2.tex | 2.1.1, after δ definition (~101) | only δ1 is reported (as in DA2) | edited |
| chapter2.tex | 2.1.1, after δ definition | new paragraph + equation defining SqRel and log10 (used only in some chapter 4 tables) | added |
| chapter3.tex | 3.2.4 (~249) | sentence pointing to «نتایج تجربی ... تحلیل RMSE» in 2.1.1 (which has no results) | deleted |
| chapter3.tex | 3.5.2 `subsec:evaluation_protocol` (~610) | «همراه با δ1» + reason AbsRel is the main metric | edited |
| chapter3.tex | 3.5.2, SiLog paragraph (~614) | «تغییرناپذیری ذاتی نسبت به مقیاس» → with λ=0.5 only partly scale-invariant (full only with λ=1); «برای این دسته از مدل‌ها» removed | edited |
| chapter3.tex | 3.5.2, `subsec:rmse_sensitivity` paragraph (~618) | old «RMSE as stability measure» paragraph (red) replaced by new paragraph: RMSE in metres, within-dataset only; cross-dataset and camera-stability claims rest on AbsRel/δ1 and the K-sweep/knockout tests | deleted + added |
| chapter3.tex | 3.6 summary (~640) | «RMSE برای سنجش حساسیت به مقیاس … AbsRel برای مقایسه نرمال‌شده» → AbsRel + δ1 main, RMSE within-dataset complement | edited |
| chapter4.tex | 4.3.3 `sec:quantitative_results` (~388) | SiLog «به‌عنوان معیار مطلق اصلی» → «در این مقایسه»; added: AbsRel + δ1 main in the rest of the chapter | edited |
| chapter4.tex | 4.3.3 (~412) | «خروجی عمق … و در نتیجه به پارامترهای دوربین» → «خطای … به نوع دیتاست»; added caveat that datasets differ in more than the camera, camera role tested directly in `sec:camera_sensitivity` | edited |

- Finalize R18: `\edrep{old}{new}` → new, `\edins{X}` → X, drop `edadded` wrapper, delete `\eddel{}` sentence and the `eddeleted` paragraph, remove `% REVIEW R18` lines.
- Revert R18: `\edrep{old}{new}` → old, delete `\edins{}` and the `edadded` blocks, unwrap `\eddel{}` / `eddeleted`, remove `% REVIEW R18` lines.

## R19 — 2026-11-17 — §4.5: knockout test extended to the zoom-crop model; control line removed from the sweep chart

Evidence: new run `results/lambda05_focal/k_knockout_zoomcrop.json` on lingotube (tools/k_knockout.py, NYUv2 eigen-654, rawDepths, 0.001–10 m; checkpoint `metric_finetuning_aug2/revised_metric_vitl_nyu_intrinsics_lr5e-6_bs2_20260706_224730/best.pth`, sweep slope 0.06). Zoom-crop AbsRel: correct 0.0819, removed 0.1007, f×0.7 0.0807, f×1.4 0.0916. Focal-jitter numbers unchanged (`k_knockout_lam05.json`). Persian drafted and reviewed with Codex.

| File | Location | Change | Kind |
|---|---|---|---|
| chapter4.tex | §4.5.1 after `tab:k_sweep_slopes` note | paragraph: control slope is zero by construction, not drawn; informative contrast is the two proposed configs | added |
| chapter4.tex | `fig:intrinsics_sensitivity` | control `\addplot` + legend commented out; figure framed in yellow | edited |
| chapter4.tex | §4.5.2 method paragraph | sentence: same test run on zoom-crop model | edited (yellow) |
| chapter4.tex | `tab:k_knockout` | config header row + three zoom-crop columns; wrapped in adjustbox | added cells |
| chapter4.tex | §4.5.2 results paragraph (R5) | «برای پیکربندی لرزش کانونی» inserted | edited (yellow) |
| chapter4.tex | after R5 paragraph | new paragraph interpreting zoom-crop results and the out-of-distribution effect of removal | added |
| chapter4.tex | sentence before `fig:k_knockout`, its caption | two short insertions | edited (yellow) |
| chapter4.tex | `fig:k_knockout` | paired bars (zoom-crop added), stars shifted to focal bars; yellow frame | edited |

- Finalize R19: drop `edadded`/`\edadd`/`\edins` wrappers, `\edaddrow`/`\edaddcell` tokens, the two `\colorbox{thRevNew}{%` … `}` frames, the commented control plot lines, and the `% REVIEW R19` comments.
- Revert R19: delete the added paragraphs/insertions, restore `tab:k_knockout` to 4 columns without adjustbox, restore single-series bars at x=1..4 (bar width 20pt) with stars at x=3,4 and legend «خطای مشاهده‌شده», uncomment the control plot, remove the frames.

## R20 — 2026-11-17 — Section 4.6 «گام پایانی: تعمیم بین‌دامنه‌ای با آموزش چندمجموعه‌ای» shortened

Requested by the author (two messages): the final multi-domain section is too long; shorten it.
Rules given: no explanation of DIODE / Middlebury (left to the future-work section of chapter 5,
which already has that item); the trial runs get one paragraph of overall history with no details;
keep the emphasis on the results and the zero-shot behaviour; existing highlights stay as they
are; new text is written by Codex.

New text (items 3, 6, 7, 10) was written by Codex (`codex exec` through `codex_prose.sh`, read-only)
from fact lists taken from the old paragraphs; every number in it was checked against the old text. Two terms in the Codex output were corrected by hand to the thesis vocabulary: «مقیاس کلی» → «مقیاس سراسری» (item 3) and «نسبت مقیاس قاب» → «نسبت ابعاد قاب» (item 10; the old text says aspect ratio).
All other items are marked deletions. No table value changed. Not built: no TeX on this machine.
(This update was first tagged R19 locally; renumbered R20 because the author's commit b1281ad uses R19.)
Older marks (R1, R3, R7, R15) were not finalized or reverted; where an R1/R7 green block had to be
split around a deleted paragraph, the block was closed and reopened with its original tag.

| # | File | Location (label) | Change | Kind |
|---|---|---|---|---|
| 1 | chapter4.tex | `sec:multidomain_results`, first paragraph | sentence on mixed precision, batch size 4 and three hours per arm | deleted sentence |
| 2 | chapter4.tex | same, paragraph after `tab:multidomain_main` | «در هر دوازده ستون جدول بهترین مقدار را دارد …» (repeats the paragraph before the table) | deleted sentence |
| 3 | chapter4.tex | `subsec:md_all_sets`, «در مقابل …» | lead paragraph and the three-item list on DIODE indoor / DIODE outdoor / Middlebury deleted; one new short paragraph (author asked for it after the first pass, for the defence): not better than the control on these three sets; error is mainly global scale; DIODE outdoor scale ratios 0.61 vs 1.20 / 1.27 and why AbsRel and δ1 rank differently; Middlebury 1.47 vs 0.95, 23 images, 34° field of view against 49–82° in training; cause left to `sec:chapter5_future_work` | deleted paragraph + list, added paragraph |
| 4 | chapter4.tex | `subsec:md_all_sets`, last paragraph | first sentence (RealSense / Xtion), already stated above | deleted sentence |
| 6 | chapter4.tex | `subsec:revised_cross_dataset`, absolute-models paragraph | the two sentences on DIODE indoor / Middlebury (12 percent, 0.677 vs 0.235, «در زیربخش … بررسی می‌شوند») replaced by one sentence pointing to future work | edited (`\edrep`) |
| 7 | chapter4.tex | `subsec:revised_cross_dataset` | three paragraphs (relative models; removal of the matrix at test time; NYUv2 ViT-S reconciliation) replaced by one paragraph with the same aggregate facts; dropped: set names of the ViT-S relative wins/losses, the Middlebury relative values, the repeated 0.088 → 1.31 sentence | deleted paragraphs, added paragraph |
| 8 | chapter4.tex | `subsec:revised_cross_dataset`: set-up paragraph, last paragraph, footnote of `tab:md_encoder_protocol` | the three references to rows of `tab:md_runs` | deleted words |
| 9 | chapter4.tex | `subsec:md_campaign` | all four paragraphs and the table `tab:md_runs` (R1 `\edaddbox` → `\eddelbox`) | deleted paragraphs + table |
| 11 | chapter4.tex | `subsec:qualitative_multidomain`, first paragraph | clause repeating which released checkpoint is used | deleted words |
| 12 | chapter4.tex | `subsec:qualitative_multidomain`, last paragraph | paragraph on the camera-shift figure | deleted paragraph |
| 13 | figures/pending/qualitative_md_figures.tex | figure `fig:qual_md_shift` | whole figure: R3 `\edaddbox` → `\eddelbox` | deleted figure |
| 14 | chapter5.tex | `sec:chapter5_future_work`, multi-domain paragraph (~614) | pointer «(زیربخش …)» to `subsec:md_campaign` after the Hypersim ViT-S trial sentence: that subsection no longer gives the per-set outcome; the sentence itself stays | deleted words |
| 15 | chapter5.tex | `sec:chapter5_future_work`, compute-limits sentence (~622) | pointer «زیربخش `subsec:md_campaign`» → «زیربخش‌های `subsec:multidomain_training` و `subsec:md_campaign`» (the frozen-encoder / 12-epoch settings are in chapter 3; the single RTX 4090 is in the campaign paragraph). Reference wording only, not sent to Codex | edited (`\edrep`) |

Checked and left unchanged: chapter 5 limitations (~483) still points to `subsec:md_all_sets` (the ten-set table is there) and keeps the DIODE outdoor / Middlebury scale ratios; chapter 3 `tab:datasets_overview` keeps Middlebury's 23 images and 34-degree field of view and the Hypersim row; no other file refers to `tab:md_runs`, `subsec:md_campaign` or `fig:qual_md_shift`.


- Finalize R20: remove every `\eddel{…}`, `eddeleted` block, the `tab:md_runs` table and the `fig:qual_md_shift` figure tagged R20; `\edrep{old}{new}` (item 6) → new; drop the `edadded` wrappers of items 3, 7, 10; merge the split halves of the footnote `\edins` (item 5); in chapter 5 drop the `\eddel{…}` of item 14 and keep the new half of the `\edrep` of item 15; remove the `% REVIEW R20` lines.
- Revert R20: in chapter 5 unwrap the `\eddel{…}` of item 14 and keep the old half of the `\edrep` of item 15; restore the regions from commit b1281ad (`git show b1281ad:thesis/tex/chapter4.tex`, lines of `sec:multidomain_results` … before `\section{بحث و تفسیر نتایج}`; and `git checkout b1281ad -- thesis/figures/pending/qualitative_md_figures.tex`) — b1281ad is the author's commit this update was merged with; its wording changes inside the deleted blocks (the «بهتر در» sentence, the camera-shift figure sentence, the short runs-table caption) are kept inside those blocks.

## R21 — 2026-11-17 — Camera-shift test, DIODE indoor and Middlebury removed from the whole thesis

Requested by the author: remove the «جابه‌جایی دوربین» (camera-shift) test and the two test sets on
which the proposed model has no best value in any of AbsRel, δ1, RMSE (DIODE indoor, Middlebury)
from `tab:multidomain_all`, and remove every trace of them from the thesis. The multi-domain
evaluation is now reported on eight test sets (NYUv2, Cityscapes, KITTI, four SUN RGB-D sensors,
DIODE outdoor; 3052 images = 3400 − 325 − 23).

New Persian text was written by Codex (`codex_prose.sh`, read-only): items 9, 14 (ch. 4) and 20, 22
(ch. 5). All other items are marked deletions or count changes. Counts were recomputed from
`tab:md_encoder_protocol` after dropping the two sets: absolute ViT-S lower AbsRel on 8 of 8 (was 9 of
10), reliable on 7; absolute ViT-L 8 of 8 (was 8 of 10), reliable on 7; relative ViT-L 8 of 8 (was 9
of 10), reliable on 5, gap ≤ 0.0006; relative ViT-S reliably better on 3, reliably worse on 2 (was 3).
Item 14 contains a **new derived statistic**: the coefficient of variation of AbsRel over the eight
sets, computed (sample standard deviation / mean) from the per-set values printed in
`tab:multidomain_all` (0.85 / 0.72 / 0.79); it replaces the ten-set values 0.85 / 0.59 / 0.69 and keeps
the chapter 5 sentence «خطای آن میان مجموعه‌ها یکنواخت‌تر نمی‌شود» supported.

Older pending marks (R1, R3, R7, R14, R15, R20) were not finalized or reverted. Where a region that
R21 deletes as a whole contained older inline marks, they were flattened to their new text inside the
R21 deletion (items 2, 5, 10, 14 and the KITTI camera-shift cells of item 6); where R21 changes words
inside an older `\edins{…}`, that `\edins` was split around the change (items 17, 18).

| # | File | Location (label) | Change | Kind |
|---|---|---|---|---|
| 1 | chapter4.tex | experiment-path table, row 8 (~270) and pipeline figure node `o4` (~313) | «ده مجموعه آزمون» → «هشت»; the TikZ node is changed without an inline mark | edited |
| 2 | chapter4.tex | `sec:multidomain_results`, paragraph «بر NYUv2، Cityscapes و Kinect v2، برتری … با تغییر دوربین» | whole camera-shift paragraph; the legacy `edited` block now starts at the next paragraph | deleted paragraph |
| 3 | chapter4.tex | `tab:multidomain_camera_shift` | whole table (`\edhlbox` → `\eddelbox`) | deleted table |
| 4 | chapter4.tex | `subsec:md_all_sets`, first paragraph | «برای تصویری کامل» dropped; six → four other sets; DIODE indoor and Middlebury dropped from the list; «هر ده مجموعه را با تصاویر اصلی و در آزمون جابه‌جایی دوربین» → «هر هشت مجموعه را»; seven → five unseen sets | edited |
| 5 | chapter4.tex | `tab:multidomain_all` | caption ten → eight; group headers «تصاویر اصلی» and «جابه‌جایی دوربین» and the three camera-shift columns (header + all 30 rows); the three rows of DIODE indoor and the three rows of Middlebury | deleted cells / rows, edited caption |
| 6 | chapter4.tex | same table, KITTI rows | R7 `\edrep` in the three camera-shift cells flattened to the new value inside `\eddel` | deleted cells |
| 7 | chapter4.tex | `subsec:md_all_sets`, SUN RGB-D paragraph | «تصاویر اصلی» in «هر دوازده ستون تصاویر اصلی» | deleted words |
| 8 | chapter4.tex | `subsec:md_all_sets`, «در مقابل، با تصاویر اصلی …» (R20 paragraph on DIODE indoor / outdoor / Middlebury) | whole paragraph | deleted paragraph |
| 9 | chapter4.tex | same place | new paragraph on DIODE outdoor only: ranking depends on the metric, scale ratios 0.61 vs 1.20 / 1.27, why AbsRel and δ1 differ, error mainly global scale, cause left to `sec:chapter5_future_work` | added paragraph (Codex) |
| 10 | chapter4.tex | `subsec:md_all_sets`, last paragraph | remaining sentence on the camera-shift test; the block (R1 `edadded`) is now `eddeleted` | deleted paragraph |
| 11 | chapter4.tex | `subsec:revised_cross_dataset`, set-up paragraph | «هر یک از ده مجموعه آزمون» → eight | edited |
| 12 | chapter4.tex | `tab:md_encoder_protocol` | caption ten → eight; rows DIODE indoor and Middlebury in both blocks | deleted rows, edited caption |
| 13 | chapter4.tex | `subsec:revised_cross_dataset`, absolute and relative paragraphs | counts (see above); «در هیچ مجموعه‌ای … بدتر نیست» and the R20 sentence on DIODE indoor / Middlebury deleted (the R20 `\edrep{old}{new}` became `\edrep{old}{}\eddel{new}`) | edited, deleted sentences |
| 14 | chapter4.tex | `subsec:revised_cross_dataset`, «برای زوج اصلی مطلق ViT-L، دو یافته دیگر …» | paragraph (ten-set CV + camera-shift ratio 1.03) deleted; one new paragraph with the eight-set CV | deleted paragraph, added paragraph (Codex) |
| 15 | chapter4.tex | `subsec:md_campaign` | «هر ده … ۳۴۰۰ تصویر، با تصاویر اصلی و در آزمون جابه‌جایی دوربین» → «هشت … ۳۰۵۲ تصویر»; «یا در آزمون جابه‌جایی دوربین» | edited, deleted words |
| 16 | chapter4.tex | `subsec:qualitative_multidomain`, first paragraph | «هر ۱۴۴ پیش‌بینی … (سه مدل، تصاویر اصلی و برش مرکزی)» → «هر ۷۲ پیش‌بینی … (سه مدل)» (24 images × 3 models) | edited |
| 17 | chapter4.tex | discussion, protocol-dependence paragraph (~1949) | «از ده مجموعه» → eight | edited |
| 18 | chapter4.tex | discussion, capacity paragraph (~1965) | relative ViT-S «سه … بدتر» → two; «به‌جز Middlebury» | edited, deleted words |
| 19 | chapter4.tex | `sec:results_summary` | ten → eight; clause on DIODE indoor / Middlebury | edited |
| 20 | chapter5.tex | summary of findings (~141) | 9 of 10 and 8 of 10 → all 8 sets with both encoder sizes; clause on DIODE indoor / Middlebury; «جز بر Middlebury» | edited (Codex wording) |
| 21 | chapter5.tex | answer to the research question (~229) | clause «و در آزمون جابه‌جایی دوربین نیز … باقی می‌ماند» | deleted words |
| 22 | chapter5.tex | limitations (~485) | DIODE + Middlebury → DIODE outdoor only (lowest AbsRel, weaker than the control in δ1 and RMSE; scale ratio 0.61) | edited (Codex wording) |
| 23 | chapter5.tex | future work list (~628) | «بر DIODE و Middlebury … همین مجموعه‌ها» → «بر DIODE بیرونی … همین مجموعه» | edited |
| 24 | chapter3.tex | datasets subsection, first sentence | Middlebury removed from the list and from the `\cite` | edited |
| 25 | chapter3.tex | same, multi-domain paragraph | «برای گزارشی کامل، شش» → «چهار»; DIODE indoor and Middlebury dropped | edited |
| 26 | chapter3.tex | `tab:datasets_overview` | DIODE indoor role: «؛ آزمون صفرنمونه» removed (the row stays for the exploratory stage); Middlebury row | deleted words / row |
| 27 | chapter3.tex | paragraph after the table | DIODE «داخلی و بیرونی» → «بیرونی»; Middlebury 2014, its down-scaling sentence and its 20 m range | edited, deleted sentence |
| 28 | figures/pending/dataset_samples.tex | `fig:dataset_samples` | Middlebury panel: caption marked deleted, image kept until finalize | deleted panel |

Checked and left unchanged: the NYUv2 zoom sweep with a deterministic centre crop in the
focal-jitter study (`subsec:focal_jitter_results`, ~782) is a different experiment; DIODE in the
exploratory stage (`tab:basic_model_silog`, «NYUv2 vs DIODE») stays; DIODE outdoor stays in every
table. Older pending red blocks (R7, R20) still contain camera-shift / Middlebury text until those
updates are finalized. Not in the PDF and not touched: `thesis/figures/samples/middlebury.jpg`,
`thesis/figures/middlebury_camera_audit.pdf`, `thesis/table*.md`, the bibliography entry
`Scharstein2014Middlebury`.

- Finalize R21: remove every `\eddel{…}` / `eddeleted` block / `\eddelbox` table tagged R21;
  `\edrep{old}{new}` → new; drop the `edadded` wrappers of items 9 and 14. In `tab:multidomain_all`
  (item 5) also change the column spec `rrcccccc` → `rrccc`, drop the first header row with its
  `\cline{3-8}` (the `\multirow{2}` cells become plain cells), `\multicolumn{8}` → `\multicolumn{5}`,
  and remove the six deleted rows with their `\multirow` lines. In `tab:md_encoder_protocol` (item 12)
  move the `\hline` of each deleted Middlebury row to the DIODE outdoor row above it. In
  `tab:datasets_overview` (item 26) move the `\midrule` of the deleted Middlebury row to the DIODE
  outdoor row. In `fig:dataset_samples` (item 28) remove `middlebury.jpg` and its caption and centre
  the NYUv2 image with `\multicolumn{2}{c}{…}`. Merge the split `\edins` halves of items 17 and 18.
  Remove the `% REVIEW R21` lines.
- Revert R21: restore the tagged regions from commit 7d5f0e8 (`git diff 7d5f0e8 -- thesis/tex
  thesis/figures/pending/dataset_samples.tex` shows exactly this update while nothing else is
  committed on top).


| # | File | Location (label) | Change | Kind |
|---|---|---|---|---|

- Finalize R22: remove the `\eddel{…}` and the `% REVIEW R22` line.
- Revert R22: unwrap the `\eddel{…}` (un-brace `\lr{…}`) and remove the `% REVIEW R22` line.

## R25 — 2026-11-17 — §4.7 «بحث و تفسیر نتایج»: references to other papers removed; §4.8 chapter summary rewritten in simple words

Requested by the author: (1) in the discussion section, remove the references to other articles
and state only this thesis's own findings; (2) in «جمع‌بندی فصل نتایج», drop the sentence
«بنابراین مقیاس مطلق آن بین دیتاست‌ها پایدار نیست», concentrate on the achievements, and explain
simply, without numbers and dataset names. New wording checked/written with Codex (read-only).
(This update was first tagged R23, then R24; both IDs were taken by the qualitative-figures update
of a parallel session, so it is R25.)

| # | File | Location (label) | Change | Kind |
|---|---|---|---|---|
| 1 | chapter4.tex | `sec:discussion`, intro sentence | «مقایسه غیرمستقیم با ادبیات موجود در حوزه MDE» → «بیان یافته‌های اصلی و دامنه ادعای پژوهش»; the R7 `\edrep` new half was shortened by «مقایسه غیرمستقیم با» | edited |
| 2 | chapter4.tex | `subsec:protocol_capacity_dependence`, «این تبیین، فرضیه‌ای …» | «و نه یافته‌ای نقل‌شده از ادبیات؛ کارهایی که … گزارش نکرده‌اند» + 3 citations removed | edited |
| 3 | chapter4.tex | `subsec:literature_comparison`, heading | «مقایسه غیرمستقیم با ادبیات موضوع» → «یافته‌های اصلی و دامنه ادعای پژوهش» (label unchanged) | edited |
| 4 | chapter4.tex | same subsection, first two paragraphs | «مقایسه مستقیم اعداد این فصل …» and «در جریان غالب مطالعات پیشین …» (4 citations) | deleted |
| 5 | chapter4.tex | same subsection, «سه یافته اصلی …» | CAM-Convs, Metric3D, diffusion/FOV and UniDepth comparisons replaced by the thesis's own three findings with table references | edited |
| 6 | chapter4.tex | same subsection, «در عین حال …» | «این هم‌جهتی … نمی‌کند» → «این یافته‌ها … نمی‌کنند»; Depth Pro sentence and «بنابراین» removed | edited |
| 7 | chapter4.tex | `sec:results_summary` | whole summary paragraph replaced by a simple one (no numbers, dataset names or table references); on the author's later request it opens with the base-model sentence, subject named explicitly («مقیاس مطلق این مدل پایه …») | deleted + added |
| 8 | chapter5.tex | «تفاوت روش پیشنهادی با چهار روش …» (~302) | pointer «و زیربخش … (فصل چهارم)» removed (that subsection no longer compares with those methods); the R7 `\edins` was split in two | deleted words |

Absorbed older marks (their text is now inside R25 deletions, written with the already-corrected
spelling): R5 `\edrep{تایید می‌کند}{تأیید می‌کند}` (item 4), R5 `\edrep{ارایه می‌کند}{ارائه می‌کند}`
(item 5), and the R7 `edadded` block of the summary with its two R21 `\edrep`s (item 7).

- Finalize R25: `\edrep{old}{new}` → new; remove the `\eddel{…}` pieces and both `eddeleted` blocks;
  drop the `edadded` wrapper of the new summary; heading becomes
  `\subsection{یافته‌های اصلی و دامنه ادعای پژوهش}`; item 1: merge the remaining R7 `\edrep` with the
  new words; item 8: merge the two `\edins` halves; remove the `% REVIEW R25` lines.
- Revert R25: `\edrep{old}{new}` → old, `\eddel{X}` → X, un-brace `{\lr{…}}`, `{\cite{…}}`, `{\ref{…}}`,
  and restore these older marks exactly:
  item 1 → `…و ظرفیت مدل و مقایسه غیرمستقیم با} ادبیات موجود در حوزه \lr{MDE} است.` (inside the R7 `\edrep`);
  item 4 → `% REVIEW R5` + `ماتریس کالیبراسیون دوربین نابیناست، همین نکته را از منظر ارزیابی \edrep{تایید می‌کند}{تأیید می‌کند}.`;
  item 5 → `% REVIEW R5` + `\lr{Metric3D} برای انگیزه‌بخشی به فضای دوربین متعارف \edrep{ارایه می‌کند}{ارائه می‌کند}~\cite{Yin2023Metric3D}.`;
  item 8 → one `\edins{…}` again, without the `\eddel`;
  item 7 → delete both R25 blocks and put back the block below, verbatim, after `\label{sec:results_summary}`:

```latex
% REVIEW R7
\begin{edadded}
% REVIEW R21 (ده -> هشت؛ جمله DIODE داخلی و Middlebury)
خطای \basemodel{} منتشرشده در هر سه مقایسه بین‌دیتاستی تفاوتی از نظر آماری قابل اتکا داشت (جدول~\ref{tab:basic_model_silog})؛ بنابراین مقیاس مطلق آن بین دیتاست‌ها پایدار نیست. در مطالعه کنترل‌شده تک‌دوربینه روی \lr{NYUv2}، در پروتکل نسبی، سهم خالص ماتریس کالیبراسیون دوربین در برابر \controlmodel{} هم‌داده فقط برای انکودر کوچک از نظر آماری قابل اتکا بود (بخش~\ref{sec:controlled_nyu_study}). در ارزیابی مطلق بدون تراز، \proposedmodel{} در هر دو پیکربندی آموزشی از \controlmodel{} بهتر بود و شکاف \lr{AbsRel} از \fadecimal{۰٫۰۰۰۹} در \zoomcropconfig{} به \fadecimal{۰٫۰۲۳۱} در \focaljitterconfig{} افزایش یافت (جدول~\ref{tab:dose_response_summary}). آزمون‌های حساسیت نیز نشان دادند که مدل واقعاً فاصله کانونی را می‌خواند (بخش~\ref{sec:camera_sensitivity}). در گام پایانی، \proposedmodel{} \lr{ViT-L} آموزش‌دیده با چهار دیتاست در هر سه معیار روی \lr{NYUv2}، \lr{Cityscapes} و دو حسگر \lr{Kinect} در \lr{SUN RGB-D} از \controlmodel{} و \publishedbasemodel{} دقیق‌تر بود (بخش~\ref{sec:multidomain_results}؛ جدول~\ref{tab:multidomain_main}). در مدل‌های مطلق با هر دو اندازه انکودر، \proposedmodel{} در هفت مجموعه از \edrep{ده}{هشت} مجموعه آزمون از نظر آماری قابل اتکا بهتر از \controlmodel{} بود\edrep{؛ در {\lr{DIODE}} داخلی و {\lr{Middlebury}}، {\controlmodel{}} با انکودر {\lr{ViT-L}} دقیق‌تر بود و}{ و} در مدل‌های نسبی تفاوت پایداری مشاهده نشد (جدول~\ref{tab:md_encoder_protocol}). فصل پنجم این یافته‌ها را جمع‌بندی می‌کند، به سوالات تحقیق پاسخ می‌دهد و محدودیت‌ها و کارهای آینده را بیان می‌کند.
\end{edadded}
```

## R24 — 2026-11-17 — §4.6 qualitative figures: the 24 images replaced by images chosen after evaluation

Requested by the author: show the best-looking outputs of the proposed model in the qualitative
figures of `subsec:qualitative_multidomain` (the four depth figures and the ratio-map figure), and
do not say in the captions that the images were chosen.

How the images were chosen (`tools/multidomain/select_qualitative_md.py`, new file): from the
per-image results in `results/multidomain/{md1_P,md1_C,released_hypersim_vitl,released_vkitti_vitl}`,
keep images where the proposed model is best on AbsRel, RMSE and δ1 at once, its AbsRel is not above
the set median and the ground truth is dense enough; rank by min(AbsRel control, AbsRel released) −
AbsRel proposed; dump predictions for the top 20 of each set on lingotube; pick six per set by eye
from contact sheets. Chosen list: `thesis/figures/qualitative_md_v2/manifest.json` (its `protocol`
field says the images were chosen after inference). Dumped AbsRel equals the table's per-image
value within 1.8e-6 for all 72 predictions. Same checkpoints and same drawing code as before.

Because the images are no longer a seed-locked sample, every sentence that said so is removed
(seed, SHA-256, «قفل‌شده», «بدون حذف»), and the counts that only described the old 24 images
(rows won per figure, 20/3/1 of 24, 7/1 of 8, the 0.035 vs 0.197 example) are removed. They are
replaced by counts over the full four test sets, which are new derived numbers:
lowest AbsRel of the three models on 1554 images — proposed 1205, control 189, released 160;
per set, proposed 501/654 (NYUv2), 122/200 (Kinect v1), 145/200 (Kinect v2), 437/500 (Cityscapes).
The median scale ratios are now over all images of each set (proposed 0.935–1.080, control
1.090–1.250, released 1.146–1.243; from `summary.median_scale_ratio` of the same result files).
The captions say «شش تصویر از …» and do not mention how the images were picked; the body text
calls them «تصویر نمونه» and says they do not replace the table. Codex (read-only) checked the new
wording; three of its suggestions were applied.

All text edits are inside the pending green R3 block (same convention as R21's edits there).

| # | File | Location (label) | Change | Kind |
|---|---|---|---|---|
| 1 | chapter4.tex | `subsec:qualitative_multidomain`, paragraph 1 | locked-protocol / seed / SHA-256 sentences → «از هر یک از چهار مجموعه آزمون این جدول شش تصویر نمونه آورده شده است. خروجی … بر این ۲۴ تصویر نشان داده می‌شود» | edited |
| 2 | chapter4.tex | same, paragraph 2 | «هر … بدون حذف» → «این»; sentence with rows won per figure | edited + deleted sentence |
| 3 | chapter4.tex | same, paragraph 3 | 20/3/1-of-24 sentences → full-set counts; median scale ratio of six images → of all images (six numbers); «به‌ترتیب قفل‌شده» → «در چهار شکل عمق پیشین»; sentence «در این هشت ردیف …» | edited + deleted sentence |
| 4 | chapter4.tex | same, paragraph 4 | last sentence (three images where the released model is best, 0.035 vs 0.197) | deleted sentence |
| 5 | figures/pending/qualitative_md_figures.tex | `fig:qual_md_nyu`, `_kv1`, `_kv2`, `_cityscapes`, `_error` | old figure: `\edaddbox` (R3) → `\eddelbox`, label renamed `…_old`; new figure in `\edaddbox` from `figures/qualitative_md_v2/`, original label, caption without «قفل‌شده» | deleted figure + added figure (×5) |
| 6 | figures/qualitative_md_v2/ | five PDFs, `geometry.json`, `manifest.json` | new assets | added |

Not touched: `fig:qualitative_locked` (the NYUv2-only locked sample of the focal-jitter study) and
its text; `fig:qual_md_shift` (already deleted by R20/R21).

- Finalize R24: `\edrep{old}{new}` → new; remove the `\eddel{…}` sentences; in the figures file
  delete the five `\eddelbox` figures tagged R24 (labels `…_old`) and unwrap nothing else (the new
  figures stay in `\edaddbox` only if R3 is still pending, otherwise unwrap them); remove the
  `% REVIEW R24` lines. `thesis/figures/qualitative_md/qual_md_{nyu,kv1,kv2,cityscapes,error}.pdf`
  are then unused.
- Revert R24: `\edrep{old}{new}` → old, `\eddel{X}` → X (un-brace `{\lr{…}}`, `{\fadecimal{…}}`,
  `{\ref{…}}`, `{\proposedmodel{}}` etc.); in the figures file delete the five new figures tagged
  «R24 (new images)», turn the five R24 `\eddelbox` back into `\edaddbox` and rename the labels
  back (drop `_old`); delete `thesis/figures/qualitative_md_v2/`; remove the `% REVIEW R24` lines
  and the R24 comment in the file header.

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

## R27 — 2026-11-17 — §5.3 «محدودیت‌های تحقیق»: three paragraphs summarized into two, data/time limitation added, table shortened

Author request: review the section again, reword what is unclear, summarize the paragraph
«در پیکربندی زوم‌ـ‌برش … / مطالعه کنترل‌شده دو محدودیت داشت …» and the paragraph after it, say that
training data and training time were limited (so the final model can still improve, most visibly at
object boundaries, where the released base model is sharper), and make `tab:limitations_future`
very short and data-rich. All Persian text was written by Codex (read-only) from a fact list and
checked by a second Codex pass.

Evidence: `tab:k_knockout` (0.0836 → 0.1764 = +111 %, f×0.7 0.2898, f×1.4 0.4332);
`tab:dose_response_summary` (zoom-crop AbsRel −0.0009, p = 0.040); `tab:multidomain_all`
(KITTI 0.156 vs released 0.146; RealSense 0.093 vs 0.092; Xtion 0.176 vs 0.178; DIODE outdoor scale
ratio 0.61); `subsec:multidomain_training` (4 datasets, 12 epochs × 6000 = 72000 samples, frozen
DINOv2); `subsec:md_campaign` (8 test sets, 3052 images, one RTX 4090); `sec:ch2_base_model`
(595 K synthetic + 62 M real images). Object boundaries: visual reading of the qualitative figures
under `thesis/figures/qualitative_md_v2/` (released base sharper than both the proposed and the
control model); no boundary metric was computed, so the cause is stated only as probable.

| # | File | Location | Change | Kind |
|---|---|---|---|---|
| 1 | chapter5.tex | `sec:chapter5_limitations`, 1st limitation (calibration) | knockout sentence made precise: focal-jitter configuration, AbsRel, f×0.7 / f×1.4 | edited |
| 2 | chapter5.tex | same, 2nd limitation (reference depth) | «بر مقادیر مطلق خطاهای مطلق اثرگذار باشد» → clearer clause | edited |
| 3 | chapter5.tex | same, architecture paragraph (three places) | "one specific model" → DA2 with ViT-S and ViT-L; "different encoders" → encoders outside this family; "the architecture used" → this family | edited |
| 4 | chapter5.tex | same, the two R7 paragraphs and the R1/R21 paragraph | kept as a red `eddeleted` block (R1/R7/R21 markup flattened to its current new wording) | deleted |
| 5 | chapter5.tex | same place | two short paragraphs (green `edadded`): controlled-study limits and their partial resolution; data/time limitation, sets where the model is not the best, object boundaries | added |
| 6 | chapter5.tex | `tab:limitations_future` | five rows → red deleted rows (legacy `\edcell` dropped inside them) | deleted rows |
| 7 | chapter5.tex | same table | six short rows with numbers | added rows |

- Finalize R27: `\edrep{old}{new}` → new, `\edins{X}` → X (un-brace `{\lr{…}}`, `{\fadecimal{…}}`,
  `{\focaljitterconfig{}}`); delete the `eddeleted` block and the five `\eddelrow` rows; drop the
  `edadded` environment lines and the six `\edaddrow` tokens; remove the `% REVIEW R27` lines. This
  also ends the R7 / R1 / R21 marks that were on the three replaced paragraphs.
- Revert R27: items 1–3 `\edrep{old}{new}` → old, remove the `\edins{…}`; delete the green block and
  the six `\edaddrow` rows; replace the red paragraph block and the red rows by the original text
  below (it carries the earlier R7 / R1 / R21 marks and the legacy `\edcell` cells); remove the
  `% REVIEW R27` lines. There is no commit to restore from: R26 was uncommitted in the same file.

Original paragraphs (before R27):

```latex
% REVIEW R7
\begin{edadded}
در \zoomcropconfig{}، برتری \proposedmodel{} در \lr{RMSE} از نظر آماری قابل اتکا بود، اما \lr{AbsRel} تنها در مرز اتکاپذیری آماری بود. این محدودیت مربوط به \zoomcropconfig{} است؛ در \focaljitterconfig{} تمام معیارها به‌طور قابل اتکا بهبود یافتند. بنابراین قدرت شواهد به میزان ابهام هندسی ایجادشده در داده وابسته است.

مطالعه کنترل‌شده دو محدودیت داشت: تنوع دوربین فقط با افزونه‌سازی زوم‌ـ‌برش شبیه‌سازی شده بود و اثرات ثانویه دوربین‌های واقعی را پوشش نمی‌داد؛ همچنین مقایسه‌ها به یک مجموعه‌داده درون‌دامنه‌ای، یعنی \lr{NYUv2}، محدود بود. آزمایش چندمجموعه‌ای این دو محدودیت را تا حدی برطرف کرد (بخش~\ref{sec:multidomain_results}): یک مدل واحد بر چهار مجموعه‌داده با دوربین‌های واقعی و مصنوعی متفاوت آموزش دید و برتری آن در صحنه‌های داخلی \lr{NYUv2}، صحنه‌های شهری \lr{Cityscapes} و داده‌های دیده‌نشده دو حسگر \lr{Kinect} از \lr{SUN RGB-D} نشان داده شد. در این آموزش \lr{NYUv2} تنها منبع صحنه‌های داخلی بود؛ گسترش آن در بخش~\ref{sec:chapter5_future_work} پیشنهاد شده است.
\end{edadded}

% REVIEW R1
\begin{edadded}
% REVIEW R21 (محدودیت‌ها: فقط DIODE بیرونی)
در آزمایش چندمجموعه‌ای نیز محدودیت‌هایی باقی ماند (زیربخش~\ref{subsec:md_all_sets}). \edrep{بر {\lr{DIODE}} و {\lr{Middlebury}}، {\proposedmodel{}} با تصاویر اصلی از {\controlmodel{}} دقیق‌تر نبود و بر {\lr{RealSense}} و {\lr{Xtion}} در {\lr{AbsRel}} و {$\delta_1$} تنها هم‌تراز {\publishedbasemodel{}} بود.}{در {\lr{DIODE}} بیرونی، {\proposedmodel{}} کمترین {\lr{AbsRel}} را داشت، اما در {$\delta_1$} و {\lr{RMSE}} از {\controlmodel{}} ضعیف‌تر بود؛ در {\lr{RealSense}} و {\lr{Xtion}} نیز در {\lr{AbsRel}} و {$\delta_1$} تنها با {\publishedbasemodel{}} هم‌تراز بود.} خطای آن \edrep{بر {\lr{DIODE}} و {\lr{Middlebury}}}{در {\lr{DIODE}} بیرونی} عمدتا خطای مقیاس سراسری است، نه خطای شکل: نسبت مقیاس آن \edrep{بر {\lr{DIODE}} بیرونی {\fadecimal{۰٫۶۱}} و بر {\lr{Middlebury}} {\fadecimal{۱٫۴۷}}}{{\fadecimal{۰٫۶۱}}} است، در حالی که خطای شکل آن به دو مدل دیگر نزدیک است. علت این \edrep{خطاهای}{خطای} مقیاس در این پژوهش تعیین نشد.
\end{edadded}
```

Original table rows (before R27):

```latex
وابستگی به صحت ماتریس کالیبراسیون دوربین &
همان اتکای علی که مزیت مدل است، آن را به خطای کالیبراسیون حساس می‌کند &
مطالعه رفتار مدل تحت عدم قطعیت‌های واقع‌گرایانه کالیبراسیون؛ طراحی سازوکار مقاوم به نویز ماتریس کالیبراسیون دوربین \\
شبیه‌سازی تنوع دوربین با افزونه‌سازی &
اثرات ثانویه دوربین‌های واقعی مانند اعوجاج لنز، عمق میدان و نویز حسگر پوشش داده نشده است &
\edcell آموزش بر داده‌های واقعا چنددوربینه؛ با توجه به رشد سهم پارامترهای دوربین با شدت ابهام مقیاس، انتظار می‌رود سهم ماتریس کالیبراسیون دوربین در این حالت بزرگ‌تر شود؛ گام نخست این مسیر با آموزش چندمجموعه‌ای بر چهار دوربین برداشته شد (بخش~\ref{sec:multidomain_results}) \\
تمرکز بر یک مجموعه‌داده درون‌دامنه‌ای &
\edcell تعمیم یافته‌ها به صحنه‌های بیرونی با دامنه عمق بزرگ در مطالعه کنترل‌شده \lr{NYUv2} آزموده نشده بود؛ آزمایش چندمجموعه‌ای این تعمیم را بر \lr{Cityscapes} نشان داد &
\edcell تکرار مطالعه کنترل‌شده مطلق روی \lr{KITTI} یا مجموعه‌داده‌های بیرونی مشابه؛ افزودن مجموعه‌داده‌های داخلی بیشتر با دوربین‌های متنوع به ترکیب آموزشی \\
تک‌اجرا بودن هر بازو &
استنباط علی درون هر زوج قوی است، اما واریانس ناشی از تصادفی‌بودن کل فرآیند آموزش برآورد نشده است &
تکرار هر زوج با چند بذر مستقل و گزارش میانگین و انحراف معیار میان اجراها \\
تک‌معماری بودن ارزیابی &
نتایج در چارچوب خانواده \lr{Depth Anything V2} تفسیرپذیرند &
پیاده‌سازی همان چارچوب روی انکودرها و دیکودرهای متفاوت \\
```
