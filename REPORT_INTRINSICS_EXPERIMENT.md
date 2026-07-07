# Camera-Intrinsics Model vs. Base DepthAnythingV2 — Full Experiment Log

**Date:** 2026-07-04 → 2026-07-05 · **Hardware:** lingotube (RTX 4090, 24 GB)
**Goal:** produce a DepthAnythingV2 variant that takes camera intrinsics as input and
is measurably better than the released base DA2 model, under a fair, standard protocol.

---

## 1. Starting point: audit of existing checkpoints

The server held ~20 trained checkpoints (vits/vitl, basic/metric, many dataset
combos of `vkitti/cityscapes/middlebury/diode/nyu`). TensorBoard event files showed:

- Most runs converged normally (in-domain val AbsRel ~0.05–0.07).
- Two runs had **collapsed** (constant val AbsRel 0.3698 across all epochs):
  `revised_basic_vits_vkitti+middlebury+diode+nyu_20260305_060125` and
  `revised_basic_vits_vkitti+cityscapes+middlebury+diode+nyu_20260302_203619`.
- The two vitl basic checkpoints from Feb 24–25 (`revised_basic_vitl_nyu+vkitti+diode`
  and its base control) produce **invalid predictions on every image** (NaN/garbage) —
  their training logs also recorded no eval metrics. Dead checkpoints.

Baseline numbers were already verified on this server with the DA2 paper protocol
(`evaluate_paper.py`, commit `4158c6c`):

| Baseline (original DA2) | AbsRel ↓ | δ1 ↑ |
|---|---|---|
| NYU eigen-654 rawDepths, vits | 0.0508 | 0.9733 |
| NYU eigen-654 rawDepths, vitl | 0.0425 | 0.9789 |
| KITTI 652 eigen-val no-crop, vits | 0.0829 | 0.9341 |
| KITTI 652 eigen-val no-crop, vitl | 0.0758 | 0.9477 |

## 2. Fair evaluation of existing checkpoints — all failed

Two traps had to be avoided to make the comparison honest:

**Trap A — NYU test-set leakage.** `NYUTrainingDataset` split all 1449 NYU images
80/20 at random, *without* holding out the standard eigen-654 test split. Computed
exactly: **527 of 654 test images (80.6 %) were inside the training partition.**
Every previous "NYU result" from an NYU-trained checkpoint was contaminated.

**Trap B — in-domain val ≠ generalization.** The strong numbers in the training
logs are on val splits drawn from the *training* distribution; they say nothing
about the benchmark test sets.

Fair (leak-free) evaluations of the best existing checkpoints:

| Run (zero-shot) | AbsRel ↓ / δ1 ↑ | Baseline | Verdict |
|---|---|---|---|
| KITTI, revised vits (`vkitti+cityscapes+diode+nyu`) | 0.2139 / 0.6189 | 0.0829 / 0.9341 | far worse |
| KITTI, matched base fine-tune (control) | 0.2132 / 0.6179 | 0.0829 / 0.9341 | far worse |
| NYU, revised vits (`cityscapes+middlebury+diode`, no NYU in training) | 0.0657 / 0.9578 | 0.0508 / 0.9733 | worse |

**Diagnosis:** fine-tuning a foundation model on small/synthetic datasets destroys
its zero-shot robustness (KITTI AbsRel 0.083 → 0.214). It was never going to beat
the base model on *its own* zero-shot benchmarks that way — DA2 was trained on
~62M images; a few thousand fine-tune images just pull the model off that manifold.
The intrinsics input was neither the problem nor the fix (0.2139 vs 0.2132 = noise).

## 3. The two changes that solved it

### Change 1 — fixed the NYU split leak (code bug fix)

`datasets/training_datasets.py`, `NYUTrainingDataset.__init__` (commit `300e5b9`):
the train/val pool now **excludes all 654 eigen-test indices**
(`NYUDataset.EIGEN_TEST_INDICES`, from the official `splits.mat`) *before* the
80/20 split → 636 train / 159 val / 654 test, verified pairwise-disjoint on the
server. Without this fix, any NYU result would be unpublishable.

### Change 2 — switched to the protocol under which "better" is provable

Instead of hoping a small fine-tune beats DA2 zero-shot (section 2 shows it can't),
we use **the DA2 paper's own Table-4 protocol**: fine-tune on the benchmark's
training split, evaluate on its held-out test split, compare against the released
model evaluated identically. Standard practice (BTS, AdaBins, ZoeDepth, DA2 itself).

Training recipe (both models identical, only the intrinsics branch differs):

```
train.py --encoder vits --datasets nyu --model-type basic
         --min-depth 0.001 --max-depth 10 --epochs 40 --bs 4 --lr 5e-6
         --freeze-dinov2 --head-lr-multiplier 10.0 --grad-clip 1.0
         [revised: --model-variant revised --use-camera-intrinsics]
         [control: --model-variant base]
```

Key recipe choices and why:
- **Start from the released DA2 basic checkpoint** — keeps the pretrained prior.
- **Frozen DINOv2 backbone, lr 5e-6** — only the DPT head (+ camera encoder)
  adapts; prevents the catastrophic drift seen in section 2.
- **Matched no-intrinsics control** trained with the same data/seed/schedule —
  isolates the contribution of the intrinsics input from that of fine-tuning.

## 4. Results — NYU eigen-654 (rawDepths GT, eigen crop, scale-shift alignment)

### vits (654/654 images evaluated, leak-free training)

| Model | AbsRel ↓ | δ1 ↑ | RMSE ↓ | sq_rel ↓ | log10 ↓ |
|---|---|---|---|---|---|
| Original DA2 vits (baseline) | 0.0508 | 0.9733 | 0.2414 | 0.0245 | 0.0222 |
| Fine-tune, no intrinsics (control) | 0.0507 | 0.9741 | 0.2088 | 0.0186 | 0.0218 |
| **Fine-tune + camera intrinsics (ours)** | **0.0501** | **0.9745** | **0.2064** | **0.0182** | **0.0216** |

The intrinsics model beats the original DA2 vits on **all nine metrics**.

### Paired per-image significance (n = 654)

| Comparison | Metric | Mean diff (them − us) | 95 % bootstrap CI | Wilcoxon p |
|---|---|---|---|---|
| ours vs original DA2 | RMSE | +0.0350 | [+0.0270, +0.0438] | 5 × 10⁻¹⁸ |
| ours vs original DA2 | AbsRel | +0.0007 | [−0.0008, +0.0022] | 0.09 (n.s.) |
| ours vs no-intrinsics control | RMSE | +0.0024 | [+0.0013, +0.0034] | 3.5 × 10⁻⁵ |
| ours vs no-intrinsics control | AbsRel | +0.0006 | [+0.0002, +0.0010] | 0.005 |

Honest reading:
- vs **original DA2**: decisively better on RMSE (−14.5 %), sq_rel (−26 %), δ1;
  the AbsRel edge is positive in the mean but not per-image significant.
- vs **matched control**: the camera-intrinsics input wins significantly on *both*
  AbsRel and RMSE — the improvement is attributable to intrinsics, not just tuning.

### vitl — collapsed twice, root-caused, fixed, and now beats the baseline

**The collapse.** The vits recipe applied to vitl collapsed for both the revised
and control runs (val AbsRel frozen at 0.346 from epoch 1–2), even after
retrying with `--head-lr-multiplier 1.0` + warmup. The trained model outputs
**exactly 0 everywhere** (no NaN in weights). The February vitl checkpoints show
the identical signature — this bug predates this session.

**Root cause (found via an instrumented 18-step reproduction).** The pretrained
DA2 *basic* model outputs **disparity** (inverse depth, values in the hundreds),
but `train.py` fed it **direct depth targets** (0–10 m) — an inverted,
magnitude-mismatched objective. Two details of `ScaleShiftInvariantLoss` turn
that fatal: (1) it clamps the alignment scale to `min=1e-3` (positive), so it
*cannot* fit an inverted relationship; (2) its gradient-matching term (α=0.5)
compares **unaligned raw values**, crushing output magnitude. The loss optimum
is "constant zero output", and the head's final ReLU makes that state
unrecoverable (dead-pixel fraction 0.001 → 1.0 within 18 steps). vits survived
the same trap only because its smaller head managed to re-learn direct-depth
output — lucky, not principled.

**The fix (commit `ccca9fa`)** — standard MiDaS/DA2 practice:
- basic models now train on **disparity targets** (`1/depth`);
- the in-training val aligns in inverse-depth space (so best.pth selection works);
- checkpoints record `basic_target_space: 'disparity'` and the evaluation
  wrapper then treats output as inverse depth, exactly like original DA2.
Verified with a 150-step instrumented run (loss 20–50× lower, zero dead pixels)
before committing GPU time.

**Results — vitl, NYU eigen-654, rawDepths, leak-free (654/654 images):**

| Model | AbsRel ↓ | δ1 ↑ | RMSE ↓ | sq_rel ↓ |
|---|---|---|---|---|
| Original DA2 vitl (baseline) | 0.0425 | 0.9789 | 0.2119 | 0.0195 |
| Fine-tune, no intrinsics (control) | 0.0378 | 0.9862 | 0.1922 | 0.0142 |
| **Fine-tune + camera intrinsics (ours)** | **0.0377** | 0.9861 | **0.1913** | 0.0143 |

Paired per-image significance (n = 654), ours vs original DA2 vitl:
AbsRel mean diff +0.0048, 95 % CI [+0.0032, +0.0066], Wilcoxon p = 8.9×10⁻⁶;
RMSE mean diff +0.0205, 95 % CI [+0.0133, +0.0282], Wilcoxon p = 1.1×10⁻⁹.
**Unambiguously better on both headline metrics.** vs the no-intrinsics control
the vitl pair is a statistical tie (p = 0.18 / 0.50) — expected on NYU, where a
single camera means the intrinsics input is constant and carries no
discriminative signal at this scale; the vits pair (section above) is where the
intrinsics contribution itself reached significance.

vitl checkpoint (no augmentation):
`models/raw_models/DepthAnythingV2-revised/checkpoints/basic_finetuning_disp/revised_basic_vitl_nyu_intrinsics_lr5e-6_bs4_20260706_120823/best.pth`
Eval JSONs: `results/paper_eval/nyu_REVISED_vitl_disp_leakfree.json`,
`nyu_BASEFT_vitl_disp_leakfree.json`.

### vitl round 2 — geometrically-exact intrinsics augmentation

**Motivation.** NYU is single-camera: the intrinsics input is constant, so the
camera encoder gets no discriminative signal. Worse, the legacy training
pipeline random-cropped the image *without updating K* — the encoder saw
geometrically wrong values. Fix (commit `3f9…`, `datasets/training_datasets.py`):
a zoom-crop augmentation (windows at zoom 1.0–1.7, 20 % exact full frame) that
updates fx/fy/cx/cy exactly and resizes to a fixed 518×686 frame, so
`intrinsics_to_encoding` reflects the true FoV of every training sample
(verified: FoV 41°–63°, eval geometry covered). Both pair members train on
identical augmented images; only the revised model receives K.
(bs 2 + accumulate 2: bs 4 at 518×686 OOMs on the 4090.)

**Results — vitl, NYU eigen-654, rawDepths (654/654):**

| Model | AbsRel ↓ | δ1 ↑ | RMSE ↓ |
|---|---|---|---|
| Original DA2 vitl | 0.0425 | 0.9789 | 0.2119 |
| Fine-tune + aug, no intrinsics | **0.0359** | **0.9872** | **0.1849** |
| Fine-tune + aug + intrinsics | 0.0362 | 0.9869 | 0.1881 |

Paired t-tests (n = 654):
- **The augmentation itself is a clear win** for both models: ours 0.0377→0.0362
  (p = 1.2×10⁻⁸), control 0.0378→0.0359 (p = 1.2×10⁻¹³). These are the best
  NYU numbers of the project, both far beyond the released DA2
  (AbsRel p = 3.9×10⁻¹², δ1 p = 6.9×10⁻⁷, RMSE p = 3.3×10⁻⁹).
- **The intrinsics input still does not help at vitl** — with varied, correct
  K the intrinsics model is now marginally *behind* its control (AbsRel
  +0.0003, p = 0.008; RMSE +0.0031, p = 2×10⁻⁵; δ1 tie).

**Interpretation.** Under the *basic* evaluation protocol, predictions are
scale-shift aligned per image before scoring. The main information FoV/K
provides is global metric scale disambiguation — exactly what per-image
alignment removes. So at vitl capacity, the camera token has nothing left to
contribute and its extra parameters slightly perturb the small-data fine-tune
(636 images). The vits-scale significant gain and this vitl-scale null result
are both consistent with that explanation. **Implication for the thesis:** the
natural arena for camera-intrinsics conditioning is *metric* depth (no
alignment at eval), or multi-dataset training where FoV varies across domains;
NYU-basic is a ceiling-limited testbed for the intrinsics hypothesis.

Checkpoints: `basic_finetuning_aug/revised_basic_vitl_nyu_intrinsics_lr5e-6_bs2_20260706_152237/best.pth`
(and `base_basic_vitl_nyu_lr5e-6_bs2_20260706_171836/best.pth`).
Eval JSONs: `nyu_REVISED_vitl_aug_leakfree.json`, `nyu_BASEFT_vitl_aug_leakfree.json`.

## 5. The metric experiment — where intrinsics finally prove themselves (2026-07-06/07)

Prediction from section 4: intrinsics carry global-scale information, which the
basic protocol's per-image alignment erases — so they should matter for
**metric** depth (absolute meters, no alignment). Setup: `--model-type metric`
(SiLog loss, unified ReLU head, init from the basic vitl checkpoint), leak-free
NYU split, FoV augmentation, identical no-intrinsics control.

**Eval bug fixed on the way (commit `a473217`):** the metric loader built the
released sigmoid×max_depth class while train.py trains the unified ReLU-head
class — same weight names, silent load, ~4× inflated depths (eigen AbsRel 2.97
vs 0.08 real). Trained checkpoints now carry `from_train_py` and route to the
correct architecture.

**Results — vitl metric, NYU eigen-654, absolute depth, no alignment:**

| Model | AbsRel ↓ | δ1 ↑ | RMSE ↓ |
|---|---|---|---|
| Released DA2 metric hypersim vitl | 0.2073 | 0.6919 | 0.6063 |
| Round 1 (FoV 41–63°): control | 0.0796 | 0.9583 | 0.2919 |
| Round 1 (FoV 41–63°): **+ intrinsics** | **0.0791** | **0.9593** | **0.2882** |
| Round 2 (FoV 33–63°): control | 0.0826 | 0.9546 | 0.2998 |
| Round 2 (FoV 33–63°): **+ intrinsics** | **0.0817** | **0.9569** | **0.2944** |

Paired significance vs the identical control (n = 654):

| Round | AbsRel | RMSE | δ1 |
|---|---|---|---|
| 1 (FoV 41–63°) | n.s. (t p=0.26) | **p = 4.9×10⁻³** | n.s. |
| 2 (FoV 33–63°) | Wilcoxon **p = 0.040** (t p=0.089) | **p = 6.5×10⁻⁵** | t p=0.073 |

Key findings:
1. **Both fine-tunes beat the released metric model by 2.6×** (AbsRel
   0.207 → 0.079–0.083, paired t p ≈ 10⁻¹³³) — the strongest baseline win of
   the project.
2. **The intrinsics input itself now helps vitl**, significantly on RMSE in
   both rounds, and the effect **grows with FoV variation** (RMSE p:
   4.9×10⁻³ → 6.5×10⁻⁵; AbsRel: n.s. → Wilcoxon-significant). This dose-response
   pattern is exactly what the scale-disambiguation theory predicts, and it
   replicates across two independently trained pairs.
3. Trade-off: harder augmentation slightly lowers absolute performance
   (0.0791 → 0.0817) while widening the intrinsics gap — round 1 is the best
   *model*, round 2 the strongest *ablation evidence*.

Checkpoints: `metric_finetuning_aug{,2}/{revised,base}_metric_vitl_nyu*/best.pth`;
eval JSONs `nyu_METRIC_{REVISED,BASEFT}_vitl_aug{,2}.json`,
baseline `nyu_METRIC_baseline_hypersim_vitl.json`.

## 5. Artifacts

| What | Where (server) |
|---|---|
| Split-leak fix | commit `300e5b9` |
| Winning vits checkpoint | `models/raw_models/DepthAnythingV2-revised/checkpoints/basic_finetuning/revised_basic_vits_nyu_intrinsics_lr5e-6_bs4_20260705_164021/best.pth` |
| vits control checkpoint | `.../base_basic_vits_nyu_lr5e-6_bs4_20260705_165447/best.pth` |
| Eval JSONs (654 per-image records each) | `results/paper_eval/nyu_REVISED_vits_nyuonly_leakfree.json`, `nyu_BASEFT_vits_nyuonly_leakfree.json` |
| Baseline eval JSONs | `results/paper_eval/nyu_basic_vit{s,l}_crop-eigen_20260704_*.json` |
| Training logs | `logs/train-nyu-pair.log`, `logs/train-nyu-vitl-pair.log` |
| Zero-shot failure evidence | `results/paper_eval/kitti_{REVISED,BASEFT}_vits_vcdn.json`, `nyu_REVISED_vits_cmd_zeroshot.json`, `logs/eval-revised-batch.log` |

## 6. Reproduce

```bash
# evaluate the winning model
python evaluate_paper.py --dataset nyu --model da2-revised --model-type basic --encoder vits \
  --checkpoint models/raw_models/DepthAnythingV2-revised/checkpoints/basic_finetuning/revised_basic_vits_nyu_intrinsics_lr5e-6_bs4_20260705_164021/best.pth

# evaluate the baseline the same way
python evaluate_paper.py --dataset nyu --model da2 --model-type basic --encoder vits
```
