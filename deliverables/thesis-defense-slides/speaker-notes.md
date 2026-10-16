# Thesis defense speaker notes

45 slides. Notes are exported from the HTML deck; the PDF contains slides only.

## 01. Title

Welcome the committee. State the one-sentence thesis: single images are metrically ambiguous; this work tests whether giving a depth model the camera's own intrinsic matrix resolves that ambiguity, and proves it with a causal experimental chain rather than a single benchmark table.

## 02. Roadmap

One sentence per section so the committee has a map before the detail starts. Emphasize up front that the campaign section is the longest because it is where the evidence actually lives.

## 03. The Scale-Ambiguity Problem

This is the entire thesis in one picture. A monocular image is a 2D projection; a bigger object further away and a smaller object closer produce pixel-identical images. Foundation depth models infer the missing scale from learned priors, while the camera's intrinsic matrix supplies a direct geometric signal. K is not sufficient to solve every monocular-scene ambiguity by itself, but it makes the focal-length/scene-scale ambiguity explicit and testable in the controlled setting studied here. That gap is the thesis.

## 04. Related Work Landscape

Survey prior approaches quickly, grouped from classical to foundation models. The point isn't to dwell on each — it's to set up the taxonomy on the next slide: almost none of these give the network K as an explicit input.

## 05. The Research Gap

This taxonomy is the actual argument for why the thesis exists: of four possible ways to handle K, almost all prior work picks 1-3. Explicit conditioning is rare in single-image depth (for example, CAM-Convs and UniDepth), while MVSNet uses K with multiple calibrated views. Strategy 4 is the thesis's position.

## 06. Research Questions

These three RQs, verbatim from chapter 1, are the spine of the whole talk. Read them slowly. Point out that a 4th, unstated but structurally necessary condition -- proving the model actually consumes K, not just correlates with an improvement -- becomes the K-sweep and K-knockout tests in section III; it is what makes RQ3's answer trustworthy rather than merely observed.

## 07. Section II — Method

_Section divider; no speaker notes._

## 08. Pinhole Model & the Intrinsic Matrix

Ground the audience in the exact geometry being exploited before showing the architecture. K is not a proxy or a learned embedding here — it's the literal calibration matrix.

## 09. The 9-D Camera Descriptor

The implemented descriptor is the normalized 9-vector [f̃x, f̃y, c̃x, c̃y, FoVh/π, FoVw/π, W/H, (fx+fy)/(2 max(H,W)), H·W/(1920·1080)]. It contains focal parameters, principal point, field of view, aspect ratio, normalized mean focal length, and image area. K and (H,W) must describe the same image frame; this is why crop/resize augmentation must update K exactly.

## 10. Architecture — Gated Camera-Token Injection

Walk the diagram left to right. K becomes a normalized 9-vector, then a two-layer MLP with GELU projects it to the DINOv2 embedding dimension; four transformer blocks refine the single camera token. A scalar output gate, initialized exactly at zero, multiplies that token before it is additively injected into the DINOv2 CLS token at the configurable injection layer (default first layer). At the start of fine-tuning the camera contribution is exactly zero, preserving pretrained behavior; the gate opens only if K reduces the loss. The DINOv2 backbone stays frozen, so the camera branch and DPT head adapt around the more-than-62-million-image prior.

## 11. Why Token Injection, Not Concatenation or FiLM

Anticipate the obvious committee question: why this conditioning mechanism? Compare against the two most natural alternatives and explain why both would corrupt the pretrained model, which the gated-token approach avoids by construction.

## 12. Training Strategy — Matched Pairs

Introduce the paired design that makes every later result interpretable: proposed and control are trained on identical data, seed, schedule, and only differ in whether the camera branch is active. Shared final protocol: initialize from released DA2 weights; leak-free NYUv2 split 636 train / 159 validation / 654 Eigen test; 40 epochs; learning rate 5×10⁻⁶; frozen DINOv2 encoder; head learning-rate multiplier 10; gradient clipping 1.0. The control is identically trained without camera intrinsics. This design lets every proposed-versus-control difference be attributed specifically to K, subject to the stated single-seed caveat. Protocol precision: the optimizer base/camera learning rate is 5×10⁻⁶; the DPT head uses 10× this rate, i.e. 5×10⁻⁵. Only trainable parameters are updated; DINOv2 stays frozen. Full seed-locking of sample order and augmentation draws is explicitly verified for the later focal-jitter pairs; earlier pairs are matched by their recorded recipe, without a multi-seed variance estimate. Knowledge distillation was disabled for the final controlled study.

## 13. Loss Functions

Two training regimes: basic/relative models train on disparity with a scale-shift-invariant loss (since output has no absolute scale); metric models train directly on depth with SiLog. Flag that the relative regime's alignment step is exactly what later erases K's signal — foreshadow.

## 14. Section III — Experimental Campaign

_Section divider; no speaker notes._

## 15. Scope of the Campaign

Set expectations: this was not one training run. The campaign audited about 20 pre-existing checkpoints, then ran more than 20 independent training runs across ViT-S and ViT-L, basic/relative and metric protocols, and multiple augmentation regimes. The checkpoint audit found most runs converged at in-domain validation AbsRel about 0.05–0.07, two runs collapsed to exactly 0.3698 across epochs, and two ViT-L basic checkpoints produced invalid NaN/garbage predictions on every image. The presentation focuses on the split leak and ViT-L collapse, two honest negative results, and the causal chain from correlation to mechanism to causation.

## 16. Pipeline Verification

Before trusting any comparison, the evaluation pipeline itself must be trustworthy. Reproduce the DA2 paper's own published Table 2 with the released checkpoints, independent of any of this project's own training code. Exact baseline values used later: NYU Eigen-654 rawDepths ViT-S AbsRel 0.0508, δ1 0.9733; ViT-L 0.0425, δ1 0.9789; KITTI 652 Eigen-val no-crop ViT-S 0.0829 / 0.9341; ViT-L 0.0758 / 0.9477. The rounded table shows our reproduction and paper values; all four are within normal reproduction tolerance.

## 17. Bug #1 — NYU Test-Set Leakage

This is the first bug found, and it invalidated every prior NYU result in the project. Emphasize the size of the contamination and that it's a bug pattern common in the field (naive random split instead of holding out the canonical eigen split first). The corrected split is documented in chapter 4 and the leak-free counts are 636 train, 159 validation, and 654 test images; treat the 527/654 count as a split-audit result, not a model-performance result. تقسیم قبلی روی همهٔ 1,449 جفت برچسب‌دار به‌صورت تصادفی 80/20 انجام می‌شد. راه‌حل، کنار گذاشتن اندیس‌های رسمی Eigen-654 پیش از هر تقسیم است؛ سپس فقط 795 نمونهٔ باقی‌مانده 80/20 می‌شوند: 636 train، 159 validation، 654 test. این اصلاح به split استاندارد مربوط است و به معنی تقسیم بر حسب کلاس یا مقدار برچسب عمق نیست. نتیجهٔ ViT-S بعدی فقط پس از این اصلاح قابل استناد است.

## 18. Negative Result — Zero-Shot Fine-Tuning Fails

An honest dead end, load-bearing for scientific credibility: fine-tuning on small multi-dataset mixes destroys the foundation model's zero-shot robustness, independent of the intrinsics input. KITTI: released DA2 0.0829 / 0.9341, camera-aware fine-tune 0.2139 / 0.6189, matched control 0.2132 / 0.6179 (AbsRel / δ1). NYU: released DA2 0.0508 / 0.9733; revised ViT-S trained without NYU 0.0657 / 0.9578. The 0.2139 versus 0.2132 gap is noise, so this strategy was abandoned and the campaign pivoted to the standard benchmark fine-tune protocol. Why the dataset mixture was problematic: different visual domains, depth ranges, cameras, ground-truth quality, and label conventions. The observed fine-tuning degradation is consistent with adaptation toward this small mixture; the separate contribution of each factor was not ablated. DA2 training uses 595K synthetic labelled images and 62M+ real unlabelled images converted to pseudo-labelled supervision (official source: https://depth-anything-v2.github.io/). روایت دفاع: به‌جای افزایش تنوع با مجموعه‌های کوچک و ناهمگون، برای مطالعهٔ کنترل‌شده از NYUv2 و تنوع هندسی با K صحیح استفاده کردیم.

## 19. The Pivot — Standard Fine-Tune Protocol

Explain the switch to DA2's own Table-4 style protocol: fine-tune on a benchmark's own training split, test on its held-out split, compare to the released model evaluated identically. This follows the released DA2 benchmark protocol and keeps the comparator and fine-tuned models on the same test split.

## 20. ViT-S — The First Clean Win

Present as the first genuinely clean, leak-free, statistically tested result. NYU Eigen-654, rawDepths GT, eigen crop, relative/basic protocol: released DA2 0.0508 / 0.9733 / 0.2414 / 0.0245 / 0.0222; no-intrinsics control 0.0507 / 0.9741 / 0.2088 / 0.0186 / 0.0218; proposed 0.0501 / 0.9745 / 0.2064 / 0.0182 / 0.0216 (AbsRel / δ1 / RMSE / sq_rel / log10). The proposed model is lower on all five displayed metrics versus the released baseline. Paired n=654: versus released DA2, RMSE mean benefit +0.0350, 95% bootstrap CI [+0.0270, +0.0438], Wilcoxon p=5×10⁻¹⁸; AbsRel benefit +0.0007, CI [−0.0008, +0.0022], p=0.09, not significant. Versus the matched control, AbsRel benefit +0.0006, CI [+0.0002, +0.0010], p=0.005; RMSE benefit +0.0024, CI [+0.0013, +0.0034], p=3.5×10⁻⁵. The independent Table 4.9 validation passes. Source: results/table_4_9_validation/validation_report.md.

## 21. Bug #2 — ViT-L Dead-ReLU Collapse

The most technically interesting bug: scaling to ViT-L collapsed both arms to a constant output. Walk through the root cause carefully — this is exactly the kind of debugging story a committee will respect. Emphasize it was invisible in the loss curve. روایت دفاع: وقتی از ViT-S به ViT-L رفتیم، هر دو بازوی کنترل و پیشنهادی هم‌زمان خراب شدند، در حالی که نمودار loss طبیعی بود. AbsRel اعتبارسنجی روی 0.346 ثابت ماند. کاهش head learning-rate multiplier از 10 به 1 و اضافه کردن warm-up مشکل را حل نکرد. بررسی مرحله‌به‌مرحله نشان داد سهم پیکسل‌های صفر در 18 گام از 0.001 به 1.0 رسید و خروجی در همه‌جا صفر شد. مدل ازپیش‌آموزش‌دیده disparity می‌داد: نقاط نزدیک مقدار بزرگ‌تری دارند؛ هدف آموزشی قبلی depth بود و همین جهت و دامنهٔ ناسازگار مشکل ایجاد کرد. ViT-S ظاهراً با بازآموزی head دوام آورده بود؛ این را به‌عنوان تفسیر تشخیصی، نه مزیت اثبات‌شدهٔ معماری کوچک، بیان کنید. با هدف 1/max(D_GT,d_min)، اعتبارسنجی در فضای inverse depth و ثبت basic_target_space=disparity قرارداد خروجی اصلاح شد. بازتولید 150 گامی پس از اصلاح: loss حدود 20–50 برابر کمتر و بدون پیکسل مرده. منبع تاریخی: git a51e35f:REPORT_INTRINSICS_EXPERIMENT.md؛ کد فعلی train.py و loss.py.

## 22. ViT-L, Post-Fix — Beats Baseline, Ties Control

Report honestly: leak-free NYU Eigen-654 basic/relative evaluation gives released DA2 ViT-L 0.0425 / 0.9789 / 0.2119 / 0.0195; matched control 0.0378 / 0.9862 / 0.1922 / 0.0142; proposed 0.0377 / 0.9861 / 0.1913 / 0.0143 (AbsRel / δ1 / RMSE / sq_rel). Versus released baseline, proposed AbsRel mean benefit +0.0048, 95% CI [+0.0032, +0.0066], Wilcoxon p=8.9×10⁻⁶; RMSE benefit +0.0205, CI [+0.0133, +0.0282], p=1.1×10⁻⁹. Versus matched control, this ViT-L pair is a statistical tie: AbsRel p=0.18 and RMSE p=0.50. NYU is captured with a single Kinect sensor, so K is constant; this motivates exact FoV variation and the metric protocol.

## 23. FoV Augmentation — An Informative Negative Result

This is a key conceptual slide: exact zoom-crop augmentation (zoom factor 1.0–2.2 depending on the run, fixed 4:3 window, 20% full-frame) updates K with c′x=(cx−x0)·Wnet/wwin, c′y=(cy−y0)·Hnet/hwin, f′x=fx·Wnet/wwin, f′y=fy·Hnet/hwin. NYU Eigen-654 basic/relative values: released DA2 0.0425 / 0.9789 / 0.2119; augmented control 0.0359 / 0.9872 / 0.1849; augmented proposed 0.0362 / 0.9869 / 0.1881 (AbsRel / δ1 / RMSE). Augmentation improves proposed from 0.0377 to 0.0362 (paired t p=1.2×10⁻⁸) and control from 0.0378 to 0.0359 (p=1.2×10⁻¹³). With correct varied K, proposed is still marginally behind its control: AbsRel difference +0.0003, p=0.0081; RMSE difference +0.0031, p=2×10⁻⁵; δ1 is a tie. Relative/basic evaluation aligns scale and shift per image before scoring, removing the global metric scale that K supplies; switch to metric evaluation to test that signal.

## 24. Augmentation Recipe — The Dataset Stayed Small, the Geometry Did Not

Make the accounting explicit. NYU Depth V2 contains 1,449 labelled RGB-depth pairs. The canonical Eigen test split holds out 654 images (45.1%) before training. The remaining 795 images are split 80/20 into 636 train and 159 validation images. Augmentation is online: the stored training set remains 636 unique images; each __getitem__ draw creates one stochastic view per image per epoch, so 40 epochs expose 25,440 training samples before batching, without claiming 25,440 distinct scenes. In the final FoV-aware recipe, 20% of draws keep the full frame and 80% sample a random 4:3 window with zoom s uniformly from 1.0 to 2.2. The image and depth are cropped together and resized to the fixed 686x518 network frame. K is updated exactly: sx=Wnet/wwin, sy=Hnet/hwin; cx'=sx*(cx-x0), cy'=sy*(cy-y0), fx'=sx*fx, fy'=sy*fy. Depth values are unchanged because the crop sees the same scene points. This fixes the earlier stale-K random crop and gives the camera branch varied, geometrically correct inputs. The train loop also flips RGB, depth and validity mask together with probability 0.5 and mirrors cx in K. The later focal-jitter stage draws alpha log-uniform in [0.625,1.6] (m=1.6), multiplies fx, fy and depth by alpha, and leaves RGB pixels unchanged. Validation/test use fixed geometry with no random augmentation. Both arms see the same sample order and augmentation draws using a shared seed. Sources: datasets/training_datasets.py, NYUTrainingDataset.__len__, _geometric_aug, _apply_focal_jitter; train.py data loaders and training flip; thesis-changes.md sections 2.6 and 2.8.

## 25. Metric Protocol — Where the Effect Appears

Now switch to metric evaluation: absolute depth, no per-image alignment, NYU Eigen-654. Released DA2 metric (Hypersim) is 0.2073 / 0.6919 / 0.6063. Round 1, FoV approximately 40–63°: control 0.0796 / 0.9583 / 0.2919; proposed 0.0791 / 0.9593 / 0.2882. Round 2, FoV approximately 31–63°: control 0.0826 / 0.9546 / 0.2998; proposed 0.0817 / 0.9569 / 0.2944 (AbsRel / δ1 / RMSE). Both fine-tunes are about 2.6× lower in AbsRel than the released metric model, but this is context, not the causal K comparison. Paired tests vs control: round 1 AbsRel t p=0.2619, Wilcoxon p=0.5251; RMSE t p=0.0049, Wilcoxon p=0.0093; δ1 t p=0.3552, Wilcoxon p=0.1788. Round 2 AbsRel t p=0.0892, Wilcoxon p=0.0399 (significant before Holm, adjusted Wilcoxon p=0.1595); RMSE t p=6.468×10⁻⁵, Wilcoxon p=1.210×10⁻⁴ (Holm-adjusted t p=3.881×10⁻⁴, Wilcoxon p=7.263×10⁻⁴); δ1 t p=0.0726, Wilcoxon p=0.8254. The effect grows as FoV variation widens. Source: results/fresh_validation_4_5_6_20260807/validation_report.md.

## 26. Focal-Jitter — Making Scale Unrecoverable Without K

The single most important methodological idea in the thesis. Walk through the pinhole invariance argument carefully: scaling focal length and depth by the same alpha leaves the image pixels identical, so a K-blind model faces literally contradictory labels for identical inputs and is mathematically forced into an error floor, while a K-aware model can resolve alpha exactly by reading f. Training precedes the K-sweep diagnostic: α is drawn log-uniformly in [0.625,1.6], and both focal lengths and depth labels are multiplied by α while RGB and principal point stay unchanged. This deliberately introduces label-scale ambiguity for the K-blind arm. Its loss-optimal compromise depends on the loss, so it need not equal the arithmetic mean of α. K supplies information to resolve the imposed factor within this controlled setting; this does not prove K alone resolves every monocular scene ambiguity. Training uses m=1.6, max_depth=20 m (10×1.6=16 m avoids clipping), SiLog λ=0.3, bs=2, accumulation=2, seed=42. The validity-mask ceiling is scaled with α. Same pixels with different focal factors at inference are then the K-sweep test; this observed response was not the cause of initiating training.

## 27. Focal-Jitter Results — Decisive on Every Metric

Headline focal-jitter result: metric ViT-L, NYU Eigen-654 rawDepths, eigen crop, [0.001,10] m, no scale/shift alignment, n=654, seed-locked proposed/control pair. Proposed versus control: AbsRel 0.084852 versus 0.103025, difference −0.018173, 95% bootstrap CI [−0.021049, −0.015305], Wilcoxon p=4.009×10⁻³¹, Holm p=1.604×10⁻³⁰, 17.64% benefit; RMSE 0.307074 versus 0.343147, difference −0.036074, CI [−0.043208, −0.028865], p=1.010×10⁻²⁴, Holm p=2.021×10⁻²⁴, 10.51% benefit; δ1 0.952031 versus 0.922592, difference +0.029439, CI [+0.022362, +0.036691], p=2.527×10⁻²⁴, Holm p=2.527×10⁻²⁴, 3.19% benefit; SILog 0.094218 versus 0.104463, difference −0.010245, CI [−0.011811, −0.008682], p=1.500×10⁻³⁴, Holm p=7.498×10⁻³⁴, 9.81% benefit; log10 0.036273 versus 0.042737, difference −0.006464, CI [−0.007584, −0.005347], p=2.840×10⁻²⁷, Holm p=8.520×10⁻²⁷, 15.13% benefit. All five confidence intervals exclude zero and all five remain significant after Holm correction. The absolute score is slightly worse than easier round 1 (0.0849 versus 0.0791), but the K evidence becomes decisive.

## 28. Where the Error Goes — Scale-Ratio Distributions

Show the generated per-image predicted/GT depth ratio distribution from results/qualitative/qualitative_report.json. Proposed median is 1.0146835 (shown as 1.015), control median is 1.0411288 (shown as 1.041), with SD 0.0775 versus 0.0944 across all 654 Eigen test images. This is a descriptive mechanism check; it is not a substitute for the paired metric tests.

## 29. Ablation — Isolating the Cause

Round 3 changed focal jitter and SiLog λ 0.5→0.3 together. A fourth independently-trained, seed-locked pair keeps λ=0.5 and adds focal jitter only: proposed/control AbsRel 0.0834 / 0.1065, RMSE 0.2987 / 0.3497, δ1 0.9554 / 0.9173, SILog 0.0918 / 0.1056. Wilcoxon p-values are 3.4×10⁻⁴⁵, 7.5×10⁻⁴⁰, 1.2×10⁻³⁹, and 2.2×10⁻⁵⁰ respectively. The K-sweep slope is 0.97±0.03. AbsRel 0.0834 is slightly better than the λ=0.3 focal-jitter result 0.0849, so the λ change was unnecessary and the augmentation is the causal driver. Source: thesis chapter 4 focal-ablation table. Validation scope: these round-4 means, Wilcoxon tests and 0.97±0.03 slope are corroborated by the thesis and historical experiment report, but the local fresh-inference archive covers rounds 1–3 only. Treat this as a supporting ablation, not an independently recomputed fourth seed. The between-λ score difference is descriptive; it does not establish that λ=0.5 is generally optimal.

## 30. The Dose-Response Pattern, Across Four Rounds

Across separately trained recipes, control minus proposed RMSE is 0.003654555 (round 1), 0.005446583 (round 2), 0.036073617 (focal-jitter λ=0.3), and 0.0510 (focal-jitter λ=0.5, thesis-reported). Relative RMSE reductions are 1.25%, 1.82%, 10.51%, and 14.58%. The plotted p-values are paired-t 0.0049 and 6.468e-5 for rounds 1–2, then Wilcoxon 1.010e-24 and 7.5e-40 for rounds 3–4. A p-value is not an effect size; the increase in the actual RMSE gap is the relevant trend. Round 4 changes λ while retaining focal jitter, so it is not a fourth level of FoV diversity. Each recipe has one seed per arm; this is evidence across settings, not repeated-seed replication.

## 31. K-Sweep — Does the Model Actually Read K?

The mechanistic diagnostic. Feed the SAME image with artificially rescaled fx/fy at inference time and measure the log-log slope of predicted depth vs injected focal scale. Ideal slope is 1.0 per pinhole geometry. Before focal-jitter training the model does not respond materially (slope 0.06); after focal-jitter training the fresh validation gives 0.931966 with SD 0.028589 over 20 images. Source: results/section_4_5_7_validation/validation_report.md. This is a diagnostic on one trained pair, not a multi-seed estimate.

## 32. K-Knockout — Causal Proof, Not Just Correlation

The strongest causal evidence in the thesis. The round-3 proposed checkpoint is evaluated on the full n=654 test set under four input conditions. Correct K: AbsRel 0.0924, δ1 0.9339, RMSE 0.3455. K withheld: 0.1467, 0.8449, 0.4826. f×0.7: 0.2887, 0.1310, 0.9153. f×1.4: 0.4274, 0.1209, 1.2118. Withholding K increases AbsRel by (0.1467/0.0924−1)=58.8%, rounded to 59%. With K-sweep slope s=0.93, the pinhole prediction |Xˢ−1| is about 0.28 for X=0.7 and 0.37 for X=1.4; observed wrong-K AbsRel is 0.289 and 0.427, including baseline error. This is a within-checkpoint intervention, not a multi-seed estimate. Its 0.0924 correct-K value belongs to the no-Eigen-crop diagnostic protocol used here; do not compare it directly with the 0.0849 focal-jitter headline, which uses rawDepths plus the Eigen crop. Source: thesis chapter 4 and results/section_4_5_7_validation/validation_report.md. Scope correction: |X^s−1| predicts a pure multiplicative scale error under ideal assumptions, not the full observed AbsRel with spatially varying baseline errors. The results establish causal dependence on the K input for this checkpoint; they do not uniquely prove that the network implements the pinhole equation. The 59% figure is an increase in error, not a 59% loss of accuracy. Raw per-image knockout arrays were not available locally for independent recomputation; the table is corroborated by the thesis and historical experiment report.

## 33. Held-Out FoV — The Proposed Model Wins at Every Zoom

Fresh section 4.5.7 field-of-view diagnostic on all 654 Eigen test images. Deterministic center zoom-crops at zoom 1.0, 1.4, and 1.8; K is updated exactly for each crop; metric AbsRel is reported without the Eigen crop, so compare within each row and do not treat these values as directly comparable to the focal-jitter headline table. Proposed/control AbsRel: zoom 1.0 = 0.092399869 / 0.110192786; zoom 1.4 = 0.097377528 / 0.110041302; zoom 1.8 = 0.106505191 / 0.114838025. The proposed model wins at every held-out zoom. Source: results/section_4_5_7_validation/validation_report.md and fov_varied_section_4_5_7_fresh.json.

## 34. Qualitative Results

The thesis uses a preregistered six-image qualitative set selected before inference, with seed 20260821 and selection hash prefix 8086cb124e40; all panels share common color scales. This slide shows one of those six images as an illustrative visual check. The primary evidence remains the paired quantitative statistics, K-sweep, and K-knockout already shown.

## 35. CityScapes — A Case Study, Not a Benchmark Row

Present CityScapes honestly as a methodological case study rather than another win/loss row. Its ground truth comes from stereo matching + 3D reconstruction, which is heterogeneous in quality, so raw error numbers here aren't directly comparable to NYU/VKITTI. This is another instance of the deck's intellectual-honesty theme: know the limits of your own ground truth.

## 36. What the Campaign Teaches, Beyond the Result

Zoom out to the methodology lesson: two consequential failures, one common signature — a plausible run can still be invalid. This is worth stating explicitly as its own contribution, independent of the headline finding.

## 37. Six Findings, Condensed

This is the campaign's own summary table — use it as a recap before moving to discussion. The rows summarize validated metric rounds, K-sweep, K-knockout, and the capacity result. The cross-dataset section 4.4 SiLog rows are intentionally not included because fresh validation did not reproduce all four proposed rows; see the Evidence Status slide.

## 38. Evidence Status — What Is Validated?

Use this slide to separate the controlled evidence from the exploratory cross-dataset comparison. The thesis reports metric rounds 1–3, focal-jitter, K-sweep 0.93 plus or minus 0.03, K-knockout, qualitative ratios 1.015 versus 1.041 with standard deviations 0.078 versus 0.094 on n=654, and the ViT-S/ViT-L NYU checks. Section 4.5.6 is PASS_WITH_MULTIPLE_TESTING_CAVEAT: round-2 AbsRel is significant before Holm correction (Wilcoxon p=0.0399) but not after it (Holm p=0.1595); round-2 RMSE remains significant after Holm (adjusted paired-t p=0.000388). The cross-dataset section 4.4 SiLog rows are not confirmed by fresh full evaluation: baseline means reproduce, but all four proposed means and printed gains mismatch, and the historical protocol is unit/provenance dependent. They are not used for headline claims. Sources: results/fresh_validation_4_5_6_20260807/validation_report.md, results/section_4_5_7_validation/validation_report.md, results/table_4_9_validation/validation_report.md, and results/server-validation-4-4-2/validation_manifest_4_4_2.json.

## 39. Section IV — Discussion & Conclusion

_Section divider; no speaker notes._

## 40. Answering the Research Questions

Come back to the three RQs from the opening and give honest, nuanced verdicts rather than simple yes/no. RQ2 in particular is a nice moment: it's answered by the AbsRel-vs-RMSE asymmetry seen throughout the campaign, and by the CityScapes GT-quality discussion.

## 41. Contributions — Theoretical & Practical

Two theoretical contributions, two practical. Keep this brisk since the underlying evidence was already shown at length in section III.

## 42. Contributions — Methodological & Diagnostic

This list is long because the campaign itself is the thesis's biggest single contribution. Each item was demonstrated at length in section III — this slide is the formal, named summary a defense expects, so keep descriptions to one line each and let the earlier slides carry the weight.

## 43. Limitations

Be honest and unhurried here — a defense committee respects a candidate who states limitations precisely rather than glossing over them. The K-accuracy dependency and the NYU-only scope are the two most likely to be probed in questions.

## 44. Future Work

End the technical content on a forward-looking note. The multi-camera and joint-calibration directions are the most exciting to mention if a committee member asks 'what would you do with more time?'

## 45. Thank You

Close warmly, invite questions. Mention the overview grid (press G) is available to jump directly to any slide a committee member asks about.
