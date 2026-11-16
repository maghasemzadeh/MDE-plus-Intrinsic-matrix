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
