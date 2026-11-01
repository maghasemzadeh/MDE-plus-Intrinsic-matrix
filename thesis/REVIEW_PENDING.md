# Pending thesis review updates

Colour legend: green = added, yellow = new words in an edited paragraph,
red struck = removed words, red highlight + strike = deleted part.
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
