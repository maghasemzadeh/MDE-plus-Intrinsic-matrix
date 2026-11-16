# Pending thesis review updates

Colour legend: green = added, yellow = new words in an edited paragraph,
red highlight (black text, no strike) = removed words and deleted parts.
Find every region of an update with `grep -n 'REVIEW R<n>' thesis/tex/*.tex`.

## R1 — 2026-11-01 — Multi-domain campaign: full results, lossless KITTI, extent of work, wording fixes

Requested by the author: lossless-KITTI result for the proposed and control
models with the proposed model's margin; zero-shot behaviour; control worse than
the released base; fixes raised earlier (Kinect v1 "unseen sensor" wording,
source of the base-model row, KITTI JPEG re-encoding disclosure); a fair,
defensible account of the test sets without progress; future work with the
time/compute limits; the extent of the experiments.

Evidence: `results/multidomain/` (md1_P, md1_C, md7_P7, md7_C7, md2s_*, md4*_P4,
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
| chapter3.tex | paragraph after tab:multidomain_datasets (524) | sources of the added test sets, depth caps, KITTI JPEG-95 training copies disclosure | added words |
| chapter4.tex | §sec:multidomain_results base-row paragraph (1067) | base row is our evaluation of the released checkpoints (paper's metric table = unreleased fine-tuned models; pipeline validated by tab:paper_reproduction) | added words |
| chapter4.tex | same section, test-set paragraph (1070) | "not seen in training" → no image used in training; Kinect v1 = NYUv2's sensor model, Kinect v2 absent | edited |
| chapter4.tex | same section (1077) | "two unseen sensors" → "unseen data" | edited |
| chapter4.tex | camera-shift paragraph (1080) | pointer to the full and pair-2 camera-shift results | added words |
| chapter4.tex | new subsection subsec:md_all_sets (1086–1178) | text, tab:multidomain_all (10 test sets × standard/shift), zero-shot, control-vs-base, sets without progress | added |
| chapter4.tex | new subsection subsec:md_lossless_kitti (1182–1217) | JPEG finding, tab:md_lossless_kitti, pair-2 replication, fair caveat | added |
| chapter4.tex | new subsection subsec:md_campaign (1221–1258) | extent of experiments, screening variants, tab:md_runs | added |
| chapter4.tex | §پاسخ به سوالات تحقیق, multi-domain paragraph (1485) | wording fix; pair-2 replication and camera-shift caveat | edited |
| chapter4.tex | §جمع‌بندی فصل نتایج, multi-domain paragraph (1678) | wording fix; summary of the new findings | edited |
| chapter5.tex | §جمع‌بندی یافته‌ها (140) | wording fix; pair-2 replication, KITTI numbers, zero-shot, control vs base | edited |
| chapter5.tex | hypotheses table, hypothesis 2 cell (199) | wording fix; replicated in pair 2 | edited cell |
| chapter5.tex | contributions list, multi-domain item (423) | wording fix; pair-2 replication and KITTI | edited |
| chapter5.tex | limitations, multi-domain paragraph (573) | wording fix | edited |
| chapter5.tex | limitations, new paragraph (577) | sets without progress, unexplained scale offsets, pair-2 caveat | added |
| chapter5.tex | limitation four (582) | two separately trained pairs replicate the main comparison | edited (yellow) |
| chapter5.tex | future work, multi-domain paragraph (707) | wording fix; Hypersim ViT-S test outcome | edited |
| chapter5.tex | future work, new block (713) | time/compute limits and five concrete next steps | added |
| chapter5.tex | final summary, multi-domain paragraph (789) | wording fix; replication across both pairs | edited |
| faTitle.tex | Persian abstract, multi-domain paragraph (79) | wording fix; pair-2 KITTI result | edited |
| enTitle.tex | English abstract, multi-domain paragraph (33) | wording fix; pair-2 KITTI result | edited |

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
- KITTI: the best KITTI results are shown everywhere. For the ViT-L absolute pair the
  KITTI values come from the pair retrained with lossless KITTI images
  (`md7_P7` / `md7_C7`, seed 42) in `tab:multidomain_all` (original images and camera
  shift) and in the new `tab:md_encoder_protocol`, each with a one-sentence note.
- All new or reworded Persian (and the English abstract sentence) was written by Codex.

Evidence: `results/multidomain/VITS_VS_VITL_MULTIDOMAIN.md` and
`vits_vs_vitl_multidomain.csv`; per-image files `results/multidomain/<run>/<set>.json`
(`md1_P/C`, `md2s_P1/C`, `md7_P7/C7` and `_zoom1.5`, `mdrel_PR/CR`, `mdrels_PR/CR`,
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
| chapter4.tex | 4.7.1 `tab:multidomain_all` and text | KITTI rows from the lossless pair, bold, note; KITTI sentences | edited + added |
| chapter4.tex | old 4.6.2 lossless subsection, `tab:md_lossless_kitti` | whole subsection | deleted |
| chapter4.tex | 4.7.2 `subsec:revised_cross_dataset` | retitled; new text and `tab:md_encoder_protocol`; old text and `tab:stability_summary`, `tab:stability_paired` | added + deleted |
| chapter4.tex | 4.7.3 `tab:md_runs` and text | rows 11–12 (relative pairs), 13 → 17 models | edited + added |
| chapter4.tex | old qualitative section, CityScapes section, old 4.9.1, 4.9.2, 4.9.5 | headings and text | deleted (parts moved to 4.5.5, 4.7 intro, chapter 5) |
| chapter4.tex | 4.8 opening, 4.8.1; 4.9 | wording, multi-domain sentences; summary reduced to one paragraph | edited + added + deleted |
| chapter5.tex | 5.1 | shortened; structure sentence | deleted + added + edited |
| chapter5.tex | new 5.2 «نتیجه‌گیری» (`sec:chapter5_conclusion`), 5.2.1–5.2.3 | old 5.2, 5.3, 5.5 become subsections; sentences from chapter 4 and old 5.4 / 5.8; multi-domain paragraph rewritten; hypothesis table cell; novelty subsections merged | added + moved + edited + deleted |
| chapter5.tex | old 5.4, 5.6.1, 5.8 | headings and remaining text | deleted |
| chapter5.tex | 5.3, `tab:limitations_future`; 5.4 | retitles; merged limitation paragraphs; table row; multi-camera paragraph rewritten; list items | edited + deleted + added |
| faTitle.tex, enTitle.tex | multi-domain sentence of the abstract | second-pair sentence → KITTI from the lossless-retrained pair | edited |

Labels added: `sec:experiment_path`, `subsec:qualitative_controlled`,
`tab:experiment_settings`, `tab:md_encoder_protocol`, `sec:chapter5_conclusion`,
`subsec:ch2_base_output`. Labels kept only on red blocks (they disappear on finalize,
and the red chapter-3 JPEG sentence still points at the first): `subsec:md_lossless_kitti`,
`tab:md_lossless_kitti`, `tab:stability_summary`, `tab:stability_paired`,
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
| faTitle.tex, enTitle.tex | abstract | KITTI figures and the retraining remark removed; one sentence on the encoder-size / model-type comparison | deleted + added |

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
در جدول~\ref{tab:datasets_overview}، تصاویر آموزشی \lr{NYUv2}، ۶۳۶ تصویر از ۷۹۵ تصویر بیرون از بخش آزمون استاندارد هستند که مانند زیربخش~\ref{subsec:nyuv2_split} با نسبت ۸۰/۲۰ میان آموزش و اعتبارسنجی تقسیم شده‌اند؛ تصاویر \lr{KITTI} از درایوهای بخش آموزش رسمی آن، تصاویر \lr{Virtual KITTI 2} از صحنه‌های \lr{Scene01}، \lr{Scene06}، \lr{Scene18} و \lr{Scene20}، و تصاویر \lr{Cityscapes} از شهرهای بخش آموزش رسمی به‌جز زوریخ برداشته شده‌اند. داده اعتبارسنجی هر دامنه، ۵۰ تصویر از بخشی کنار گذاشته‌شده است: بخش اعتبارسنجی \lr{NYUv2}، دو درایو جداگانه از \lr{KITTI}، صحنه \lr{Scene02} از \lr{Virtual KITTI 2} و شهر زوریخ از \lr{Cityscapes}. مجموعه‌های آزمون عبارت‌اند از بخش آزمون استاندارد \lr{NYUv2} با نقشه‌های عمق خام و برش استاندارد، بخش اعتبارسنجی رسمی \lr{Cityscapes} و دو زیرمجموعه تصادفی ۲۰۰ تصویری از حسگرهای \lr{Kinect v1} و \lr{Kinect v2} در \lr{SUN RGB-D}. عمق مرجع \lr{Cityscapes} از نقشه‌های دیسپاریتی همان مجموعه‌داده و با خط پایه استریوی آن به‌دست می‌آید و ناحیه کاپوت خودرو از ارزیابی حذف می‌شود. \edadd{مجموعه‌های آزمون افزوده عبارت‌اند از فهرست ۶۵۲ تصویری ارزیابی {\lr{KITTI}} که همراه کد مقاله مرجع منتشر شده است، با عمق مرجع لایدار و درایوهایی که از داده آموزش کنار گذاشته شده‌اند؛ بخش اعتبارسنجی رسمی {\lr{DIODE}} با ۳۲۵ تصویر داخلی و ۴۴۶ تصویر بیرونی؛ ۲۳ صحنه {\lr{Middlebury}} نسخه ۲۰۱۴ که تصاویر و ماتریس کالیبراسیون دوربین آن‌ها با ضریب دو کوچک شده‌اند؛ و زیرمجموعه‌های تصادفی ۲۰۰ تصویری حسگرهای {\lr{RealSense}} و {\lr{Xtion}} از {\lr{SUN RGB-D}}. بازه ارزیابی عمق برای مجموعه‌های داخلی تا ۱۰ متر ({\lr{Middlebury}}: ۲۰ متر) و برای مجموعه‌های بیرونی تا ۸۰ متر است.} \eddel{تصاویر آموزشی {\lr{KITTI}}، برای کوچک نگه‌داشتن حجم انتقال به سرور آموزش، با فشرده‌سازی {\lr{JPEG}} با کیفیت ۹۵ ذخیره شدند، در حالی که تصاویر آزمون همه مجموعه‌ها پرونده‌های اصلی‌اند؛ اثر این انتخاب در زیربخش~{\ref{subsec:md_lossless_kitti}} بررسی شده است.}
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
| 5 | chapter4.tex | footnote of `tab:multidomain_all` | «(ردیف ۲ جدول …)»; the R7 `\edins` around the footnote is split in two around it | deleted words |
| 6 | chapter4.tex | `subsec:revised_cross_dataset`, absolute-models paragraph | the two sentences on DIODE indoor / Middlebury (12 percent, 0.677 vs 0.235, «در زیربخش … بررسی می‌شوند») replaced by one sentence pointing to future work | edited (`\edrep`) |
| 7 | chapter4.tex | `subsec:revised_cross_dataset` | three paragraphs (relative models; removal of the matrix at test time; NYUv2 ViT-S reconciliation) replaced by one paragraph with the same aggregate facts; dropped: set names of the ViT-S relative wins/losses, the Middlebury relative values, the repeated 0.088 → 1.31 sentence | deleted paragraphs, added paragraph |
| 8 | chapter4.tex | `subsec:revised_cross_dataset`: set-up paragraph, last paragraph, footnote of `tab:md_encoder_protocol` | the three references to rows of `tab:md_runs` | deleted words |
| 9 | chapter4.tex | `subsec:md_campaign` | all four paragraphs and the table `tab:md_runs` (R1 `\edaddbox` → `\eddelbox`) | deleted paragraphs + table |
| 10 | chapter4.tex | `subsec:md_campaign` | one new paragraph: 17 models (7 ViT-L, 10 ViT-S) on one RTX 4090, ten test sets / 3400 images, several changes tried mostly with ViT-S, most of them aimed at cues that identify the dataset instead of the focal length (e.g. the frame aspect ratio, constant within each training dataset), none adopted, KITTI from the lossless-retrained pair; pointer to `subsec:checkpoint_audit` | added paragraph |
| 11 | chapter4.tex | `subsec:qualitative_multidomain`, first paragraph | clause repeating which released checkpoint is used | deleted words |
| 12 | chapter4.tex | `subsec:qualitative_multidomain`, last paragraph | paragraph on the camera-shift figure | deleted paragraph |
| 13 | figures/pending/qualitative_md_figures.tex | figure `fig:qual_md_shift` | whole figure: R3 `\edaddbox` → `\eddelbox` | deleted figure |
| 14 | chapter5.tex | `sec:chapter5_future_work`, multi-domain paragraph (~614) | pointer «(زیربخش …)» to `subsec:md_campaign` after the Hypersim ViT-S trial sentence: that subsection no longer gives the per-set outcome; the sentence itself stays | deleted words |
| 15 | chapter5.tex | `sec:chapter5_future_work`, compute-limits sentence (~622) | pointer «زیربخش `subsec:md_campaign`» → «زیربخش‌های `subsec:multidomain_training` و `subsec:md_campaign`» (the frozen-encoder / 12-epoch settings are in chapter 3; the single RTX 4090 is in the campaign paragraph). Reference wording only, not sent to Codex | edited (`\edrep`) |

Checked and left unchanged: chapter 5 limitations (~483) still points to `subsec:md_all_sets` (the ten-set table is there) and keeps the DIODE outdoor / Middlebury scale ratios; chapter 3 `tab:datasets_overview` keeps Middlebury's 23 images and 34-degree field of view and the Hypersim row; no other file refers to `tab:md_runs`, `subsec:md_campaign` or `fig:qual_md_shift`.


- Finalize R20: remove every `\eddel{…}`, `eddeleted` block, the `tab:md_runs` table and the `fig:qual_md_shift` figure tagged R20; `\edrep{old}{new}` (item 6) → new; drop the `edadded` wrappers of items 3, 7, 10; merge the split halves of the footnote `\edins` (item 5); in chapter 5 drop the `\eddel{…}` of item 14 and keep the new half of the `\edrep` of item 15; remove the `% REVIEW R20` lines.
- Revert R20: in chapter 5 unwrap the `\eddel{…}` of item 14 and keep the old half of the `\edrep` of item 15; restore the regions from commit b1281ad (`git show b1281ad:thesis/tex/chapter4.tex`, lines of `sec:multidomain_results` … before `\section{بحث و تفسیر نتایج}`; and `git checkout b1281ad -- thesis/figures/pending/qualitative_md_figures.tex`) — b1281ad is the author's commit this update was merged with; its wording changes inside the deleted blocks (the «بهتر در» sentence, the camera-shift figure sentence, the short runs-table caption) are kept inside those blocks.
