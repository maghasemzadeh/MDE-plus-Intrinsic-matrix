# Table 4.9 independent validation

Overall status: **PASS**

| Encoder | Reference | Metric | Reference mean | Proposed mean | Benefit | Paired t p | Wilcoxon p | Raw conclusion | Holm conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| ViT-S | published | AbsRel | 0.050816 | 0.050143 | +0.000673 | 0.396 | 0.0916 | not significant | not significant |
| ViT-S | published | RMSE | 0.241422 | 0.206406 | +0.035016 | 2.826e-15 | 4.961e-18 | proposed better | proposed better |
| ViT-S | published | delta_1 | 0.973258 | 0.974533 | +0.001275 | 0.342 | 0.0539 | not significant | not significant |
| ViT-S | control | AbsRel | 0.050738 | 0.050143 | +0.000595 | 0.00131 | 0.00502 | proposed better | proposed better |
| ViT-S | control | RMSE | 0.208767 | 0.206406 | +0.002362 | 1.052e-05 | 3.539e-05 | proposed better | proposed better |
| ViT-S | control | delta_1 | 0.974106 | 0.974533 | +0.000427 | 0.288 | 0.596 | not significant | not significant |
| ViT-L | published | AbsRel | 0.042546 | 0.037733 | +0.004812 | 2.478e-08 | 8.857e-06 | proposed better | proposed better |
| ViT-L | published | RMSE | 0.211858 | 0.191320 | +0.020538 | 1.038e-07 | 1.056e-09 | proposed better | proposed better |
| ViT-L | published | delta_1 | 0.978887 | 0.986142 | +0.007255 | 4.052e-06 | 1.193e-12 | proposed better | proposed better |
| ViT-L | control | AbsRel | 0.037773 | 0.037733 | +0.000040 | 0.772 | 0.179 | not significant | not significant |
| ViT-L | control | RMSE | 0.192236 | 0.191320 | +0.000916 | 0.289 | 0.496 | not significant | not significant |
| ViT-L | control | delta_1 | 0.986194 | 0.986142 | -0.000052 | 0.807 | 0.0355 | control/reference better | not significant |

`Benefit` is oriented so positive means the proposed model is better. Holm correction is applied separately to the 12 paired-t and 12 Wilcoxon tests.

All 24 printed p-values agree with recomputation within their displayed rounding.
All 6 input files passed protocol, sample-count, finite-value, ID-pairing, and stored-summary checks.
