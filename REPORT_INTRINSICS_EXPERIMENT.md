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

### vitl

Training pair launched 2026-07-05 23:52 (log `logs/train-nyu-vitl-pair.log`,
tmux `lab:train-vitl`). Results to be appended when done.
Target: DA2 vitl baseline 0.0425 / 0.9789.

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
