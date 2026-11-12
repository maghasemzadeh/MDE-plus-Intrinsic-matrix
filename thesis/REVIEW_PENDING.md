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
