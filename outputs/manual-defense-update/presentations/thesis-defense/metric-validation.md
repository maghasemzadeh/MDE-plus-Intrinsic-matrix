# Section 4.5.6 metric-evaluation validation

Overall status: **PASS_WITH_MULTIPLE_TESTING_CAVEAT**

## Recomputed aggregate table

| Evaluation | AbsRel | delta_1 | RMSE | Images |
|---|---:|---:|---:|---:|
| published | 0.207303271 | 0.691949992 | 0.606293804 | 654 |
| round1_control | 0.079589224 | 0.958349261 | 0.291893783 | 654 |
| round1_proposed | 0.079088772 | 0.959281434 | 0.288239228 | 654 |
| round2_control | 0.082612122 | 0.954611506 | 0.299833416 | 654 |
| round2_proposed | 0.081710758 | 0.956881806 | 0.294386832 | 654 |

## Recomputed paired tests

| Comparison | Metric | Proposed | Reference | Benefit | paired-t p | Wilcoxon p | Holm t p | Holm Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| round1_vs_published | AbsRel | 0.079088772 | 0.207303271 | +0.128214499 | 4.785e-133 | 8.444e-96 | 4.785e-133 | 8.444e-96 |
| round1_vs_published | delta_1 | 0.959281434 | 0.691949992 | +0.267331442 | 2.805e-75 | 1.562e-79 | 2.805e-75 | 1.562e-79 |
| round1_vs_published | RMSE | 0.288239228 | 0.606293804 | +0.318054576 | 6.659e-100 | 5.634e-87 | 6.659e-100 | 5.634e-87 |
| round1_vs_control | AbsRel | 0.079088772 | 0.079589224 | +0.000500452 | 0.2619 | 0.5251 | 0.5238 | 1.0000 |
| round1_vs_control | delta_1 | 0.959281434 | 0.958349261 | +0.000932173 | 0.3552 | 0.1788 | 0.5238 | 0.5363 |
| round1_vs_control | RMSE | 0.288239228 | 0.291893783 | +0.003654555 | 0.0049 | 0.0093 | 0.0244 | 0.0463 |
| round2_vs_control | AbsRel | 0.081710758 | 0.082612122 | +0.000901364 | 0.0892 | 0.0399 | 0.2906 | 0.1595 |
| round2_vs_control | delta_1 | 0.956881806 | 0.954611506 | +0.002270299 | 0.0726 | 0.8254 | 0.2906 | 1.0000 |
| round2_vs_control | RMSE | 0.294386832 | 0.299833416 | +0.005446583 | 6.468e-05 | 1.210e-04 | 3.881e-04 | 7.263e-04 |

Positive `Benefit` always means the proposed intrinsics-aware model is better.
Holm correction is applied separately to the six proposed-vs-control tests (two rounds × three metrics) for each test family.

## Audit conclusion

- 5 files passed protocol, finite-value, 654-ID pairing, zero-skipped-image, and stored-summary checks.
- All 15 aggregate cells reproduce the thesis table after four-decimal rounding.
- All 12 printed p-values reproduce within their displayed precision.
- The round-2 AbsRel Wilcoxon result is significant before correction (p=0.039871) but not after Holm correction (p=0.159485).
- Round-2 RMSE remains significant after Holm correction (paired-t adjusted p=0.000388).
