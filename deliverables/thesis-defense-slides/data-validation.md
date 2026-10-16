# Data validation for the defense deck

The deck now separates reproducible evidence from the one result that is still
protocol-dependent. The evidence ledger is the Evidence Status slide (kicker
31); the detailed source paths are in its speaker notes.

## Validated and represented

| Claim | Verdict | Evidence |
|---|---|---|
| Metric rounds 1–2 on NYU Eigen-654 | Valid with multiple-testing caveat | `results/fresh_validation_4_5_6_20260807/validation_report.md` reports PASS_WITH_MULTIPLE_TESTING_CAVEAT and reproduces the displayed means and paired tests. Round-2 AbsRel is significant before Holm correction (Wilcoxon `p=0.0399`) but not after it (`p=0.1595`); round-2 RMSE remains significant after Holm. |
| Focal-jitter round 3 | Valid | `results/section_4_5_7_validation/validation_report.md` reports PASS; all five means, confidence intervals, Wilcoxon tests, and Holm-adjusted results reproduce. |
| Focal-jitter headline values | Valid | Proposed/control AbsRel `0.0849/0.1030`, RMSE `0.3071/0.3431`, δ1 `0.9520/0.9226`, SILog `0.0942/0.1045`, and log10 `0.0363/0.0427`. |
| K-sweep | Valid with scope caveat | Slope `0.931966 ± 0.028589` over 20 images; one trained pair, so not a multi-seed estimate. |
| K-knockout | Valid with scope caveat | AbsRel `0.0924` (correct K), `0.1467` (K withheld), `0.2887` (`f×0.7`), `0.4274` (`f×1.4`), n=654. This is a within-checkpoint intervention under the no-Eigen-crop diagnostic protocol; it is not directly comparable to focal-jitter headline AbsRel `0.0849` (rawDepths + Eigen crop). |
| Field-of-view diagnostic | Valid | Fresh evaluation gives proposed `0.0924/0.0974/0.1065` and control `0.1102/0.1100/0.1148` for zoom `1.0/1.4/1.8`. |
| Qualitative scale-ratio distribution | Valid as descriptive evidence | `results/qualitative/qualitative_report.json`: proposed median `1.0146835`, control median `1.0411288`, SD `0.0775/0.0944`, n=654. The deck shows `1.015` and `1.041`. |
| NYU leak audit | Valid | The corrected split is 636 train / 159 validation / 654 test; 527 of 654 test images were present in the old training partition. |
| Table 4.9 ViT-S/ViT-L checks | Valid | `results/table_4_9_validation/validation_report.md` reports PASS. |

## Not confirmed

The cross-dataset §4.4 SiLog rows (`74.38%`, `54.78%`, `63.90%`, `70.42%`)
are **not confirmed** by the fresh full evaluation. The baseline means
reproduce, but all four proposed means and printed improvements mismatch. For
example, NYUv2 is `1.2839 → 0.3346` in the fresh historical-protocol
recomputation versus the printed `1.284 → 0.329` and `74.38%` improvement.

Those rows are excluded from the deck's headline evidence. They should be
reintroduced only after the checkpoint provenance and evaluation protocol are
reconciled. Source: `results/server-validation-4-4-2/validation_manifest_4_4_2.json`
and `results/server-validation-4-4-2/historical_protocol_recomputed_4_4_2.json`.
The thesis now labels the corresponding table and discussion as historical and
provisional, so the two artifacts make the same evidence-status claim.

## Arithmetic correction

The metric protocol slide now reports the implemented zoom-crop field-of-view ranges as approximately **40–63°** for round 1 and **31–63°** for round 2. These are computed from the NYU focal length and the implemented zoom range; the earlier printed lower endpoints (41° and 33°) were arithmetic approximations and have been corrected in the slide and speaker notes.
