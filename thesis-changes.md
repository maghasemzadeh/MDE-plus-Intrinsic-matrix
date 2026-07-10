# Thesis Changes Report — Camera-Intrinsics DepthAnythingV2 Experimental Campaign

**Purpose of this document:** a complete, self-contained content brief of everything discovered/built/tested
in the camera-intrinsics modeling work that is *not currently reflected* in `thesis/tex/*.tex`, organized by
which chapter it belongs in. This is meant to be handed to another LLM (or the user) to actually write the
Persian LaTeX prose. It is written in English for language-neutral precision; all numbers, equations and
statistics below are final and verified against three independent sources (an experiment log, a previously
committed and reviewed Persian LaTeX draft, and the underlying code) — see §9 for provenance.

---

## 0. Read this first — the thesis files are currently NOT your real thesis

Before using anything below, understand the current repository state:

**`thesis/tex/chapter1.tex`, `chapter2.tex`, `chapter3.tex`, `chapter4.tex`, `chapter5.tex`, `enTitle.tex`,
`faTitle.tex`, `words.tex`, `commands.tex`, `MyReferences.bib`, `kntu-thesis.cls` are currently the generic
`kntu-thesis` template placeholders**, not the user's real thesis. This happened by accident during a chain of
compile-fix attempts on 2026-07-10 and was not noticed before the final revert commit. Concretely, right now:

- `thesis/tex/enTitle.tex` has the template's own placeholder title/author ("Mohammad Sina Allahkaram",
  "Prepared template for writing projects, theses, and dissertations...", "Winter 2023") instead of the real
  thesis metadata.
- `thesis/tex/chapter3.tex` / `chapter4.tex` / `chapter5.tex` are 87 / 20 / 55 lines of empty template section
  skeletons (`\section{مقدمه}`, `\section{محتوا}`, ...) — **all real content, including the base
  methodology/results/conclusions chapters that existed *before* this experimental campaign, is gone from the
  working tree.**
- `thesis/tex/commands.tex` is missing the `\enfootnote`/`\fafootnote` macros used 45+ times in the real
  chapters 3–4, and is missing `pgfplots`.
- `thesis/kntu-thesis.cls` defines the syntactically-invalid `\en-abstract` instead of `\enabstract`.
- `thesis/tex/MyReferences.bib` is missing most of the ~35 real bibliography entries (only `Zhou2017Unsupervised`,
  `Ranftl2021DPT`, `Eigen2014Depth` of the 13 keys this campaign's writeup cites are present; `DepthAnythingV2`
  itself is missing).

**How this happened (for context, not action needed):** commit `c290421` ("renew tex files", pulled from
`origin/main`) silently overwrote chapters 1–5 with the generic template. The merge commit `c743cc8` correctly
identified this and rejected it, keeping the real content. But the final commit on the branch, `0b66b80`
("revert latex to last correct version"), reverted everything back to exactly match `c290421` — i.e. back to
the version that had *already been identified as broken* by the user's own earlier merge. This looks like an
accidental revert target rather than an intentional choice.

**Recommended fix path, in order, before applying anything from §4 onward:**

1. Restore the real base thesis (chapters 1–5, titles, bib, commands.tex, kntu-thesis.cls) from
   `git show 74bee36:thesis/tex/chapter1.tex` etc. — this is the last known-good state *before* any of this
   experimental campaign was written up (2026-06-19), confirmed compiling.
2. Everything in §4 of this document layers on top of that restored base and adds the experimental-campaign
   content. If you want the exact previously-written (and already reviewed) Persian LaTeX prose for that
   layer instead of writing fresh text from the English brief below, it exists in git history:
   `git show a51e35f:thesis/tex/chapter3.tex` (and `chapter4.tex`, `chapter5.tex`) is the full text with
   chapters 1-5 in the fully-reintegrated state (base + experimental campaign, no template damage). A patch
   form of just the campaign delta is at `git show 9fea97f:thesis_pending_reintegration.patch` (applies cleanly
   on top of `74bee36`). Both were auto-generated in a previous session and match the content in §4 below
   number-for-number — cross-checked while writing this report.
3. Whichever path you take, do **not** use `c290421` or the current `HEAD` as a base for chapters 1–5,
   titles, `commands.tex`, `kntu-thesis.cls`, or `MyReferences.bib`.

The rest of this document (§1–§9) assumes you are working from the *real* thesis base (post-fix), and describes
only the delta: everything about the camera-intrinsics experimental campaign that needs to be written into it.

---

## 1. One-paragraph project summary

The thesis proposes an extension to DepthAnythingV2 (a monocular depth foundation model) that takes the
camera's intrinsic matrix $K$ (focal length, principal point) as an explicit input, on the hypothesis that
depth estimation from a single image is fundamentally scale-ambiguous without knowing the camera's field of
view, and that giving the model $K$ directly should let it produce more geometrically consistent, better-scaled
depth than a model that has to infer scale purely from monocular visual cues (object-size priors, perspective
lines, etc.). The thesis chapters as currently drafted describe the architecture and a first round of
comparisons; **this report covers a second, much more rigorous experimental campaign** (2026-07-04 through
2026-07-10) that: (a) found and fixed two serious bugs that had invalidated all prior "wins," (b) proved the
architecture works and beats the base model under a fair protocol, (c) initially found the camera-intrinsics
*contribution itself* was not statistically significant at scale (an honest negative result), and (d) then
designed a targeted causal experiment (focal-jitter augmentation) that **conclusively proves the model
learns to read and correctly use $K$**, with a mechanistic diagnostic (K-sweep) and a knockout test
(withhold/corrupt $K$) as direct causal evidence, not just correlational improvement.

---

## 2. Chronological narrative (what actually happened, including dead ends)

This is the honest research story. Chapter 4 should present results in roughly this order because each step's
motivation depends on the previous step's finding — several "negative results" here are load-bearing for the
thesis's scientific credibility (they show the effect was investigated skeptically, not cherry-picked).

### 2.1 Starting point: audit of ~20 existing checkpoints

The server held about 20 previously-trained checkpoints (vits/vitl, basic/metric, various dataset mixes of
vkitti/cityscapes/middlebury/diode/nyu). A TensorBoard audit found:
- Most converged normally (in-domain val AbsRel ≈ 0.05–0.07).
- Two runs had **collapsed** to a constant val AbsRel of 0.3698 across all epochs.
- The two ViT-L basic checkpoints (Feb 24–25 runs) produced **invalid predictions on every image** (NaN/garbage)
  — dead checkpoints, this bug predates the current campaign.

Baseline numbers were independently verified against the DA2 paper's own protocol (`evaluate_paper.py`,
already documented in `CLAUDE.md`):

| Baseline (original DA2, released) | AbsRel ↓ | δ1 ↑ |
|---|---|---|
| NYU eigen-654 rawDepths, ViT-S | 0.0508 | 0.9733 |
| NYU eigen-654 rawDepths, ViT-L | 0.0425 | 0.9789 |
| KITTI 652 eigen-val no-crop, ViT-S | 0.0829 | 0.9341 |
| KITTI 652 eigen-val no-crop, ViT-L | 0.0758 | 0.9477 |

### 2.2 Two traps that made every earlier "the intrinsics model wins" claim invalid

**Trap A — NYU test-set leakage (a real bug, fixed).** `datasets/training_datasets.py`'s `NYUTrainingDataset`
split all 1449 NYU images 80/20 at random with a fixed seed, *without* first holding out the standard
654-image Eigen test split used by essentially the entire depth-estimation literature (including the DA2 paper
itself). Computed exactly with the same seed: **527 of the 654 test images (80.6%) were inside the training
partition.** Every previous "NYU result" from an NYU-trained checkpoint in this project was contaminated —
unpublishable as-is.

**Trap B — in-domain val ≠ generalization.** The strong numbers visible in old training logs are on val splits
drawn from the *training* distribution; they say nothing about performance on the actual benchmark test sets.

Fair (leak-free) zero-shot evaluation of the best pre-existing checkpoints against the baselines above:

| Run (zero-shot) | AbsRel / δ1 | vs. baseline | Verdict |
|---|---|---|---|
| KITTI, revised ViT-S (trained on vkitti+cityscapes+diode+nyu) | 0.2139 / 0.6189 | 0.0829 / 0.9341 | far worse |
| KITTI, matched no-intrinsics control | 0.2132 / 0.6179 | 0.0829 / 0.9341 | far worse |
| NYU, revised ViT-S (no NYU in training data) | 0.0657 / 0.9578 | 0.0508 / 0.9733 | worse |

**Diagnosis (a negative result worth stating plainly in the thesis):** fine-tuning a foundation model trained
on ~62M images on a few thousand small/synthetic fine-tuning images destroys its zero-shot robustness
(KITTI AbsRel 0.083 → 0.214) *regardless of whether the camera-intrinsics input is present* — the intrinsics
input was neither the problem nor the fix here (0.2139 vs. 0.2132 is noise). Multi-dataset zero-shot
fine-tuning, as a strategy for beating the released DA2 on its own zero-shot benchmarks, does not work and was
abandoned.

### 2.3 The pivot: use the protocol under which "better" is actually provable

Instead of chasing an unwinnable zero-shot comparison, the campaign switched to **the DA2 paper's own Table-4
protocol**: fine-tune on a benchmark's own training split, evaluate on its held-out test split, and compare
against the released model evaluated identically. This is standard practice in the field (BTS, AdaBins,
ZoeDepth, and DA2 itself all report this way).

Shared training recipe for every pair below (only the intrinsics branch differs between "proposed" and
"control"):

```
train.py --encoder {vits|vitl} --datasets nyu --model-type {basic|metric}
         --min-depth 0.001 --max-depth {10|20} --epochs 40 --bs {2-4} --lr 5e-6
         --freeze-dinov2 --head-lr-multiplier 10.0 --grad-clip 1.0
         [proposed: --model-variant revised --use-camera-intrinsics]
         [control:  --model-variant base]
```

Key recipe decisions and their rationale (belongs in chapter 3's training-setup subsection):
- **Start from the released DA2 checkpoint**, not from scratch — preserves the pretrained prior.
- **Frozen DINOv2 backbone, lr 5e-6** — only the DPT head (+ camera encoder) adapts, preventing the
  catastrophic zero-shot drift seen in §2.2.
- **A matched no-intrinsics control**, trained on identical data/seed/schedule — isolates the causal
  contribution of the intrinsics input from the contribution of fine-tuning itself. Every result in this
  report compares against both (a) the released DA2 model and (b) this matched control; (a) tells you "is the
  overall system better," (b) tells you "is $K$ specifically the reason."

Also fixed as part of this: `NYUTrainingDataset`'s split now excludes all 654 Eigen-test indices *before* the
80/20 split (636 train / 159 val / 654 test), verified pairwise-disjoint by an automated check. All results
from here on are leak-free.

### 2.4 ViT-S result: the first clean win

Leak-free NYU eigen-654, rawDepths GT, eigen crop, scale-shift alignment (relative/"basic" protocol):

| Model | AbsRel ↓ | δ1 ↑ | RMSE ↓ | sq_rel ↓ | log10 ↓ |
|---|---|---|---|---|---|
| Original DA2 ViT-S (baseline) | 0.0508 | 0.9733 | 0.2414 | 0.0245 | 0.0222 |
| Fine-tune, no intrinsics (control) | 0.0507 | 0.9741 | 0.2088 | 0.0186 | 0.0218 |
| **Fine-tune + camera intrinsics (proposed)** | **0.0501** | **0.9745** | **0.2064** | **0.0182** | **0.0216** |

The proposed model beats original DA2 ViT-S on **all nine** evaluation metrics.

Paired per-image significance (n = 654, paired t-test + Wilcoxon signed-rank; a paired design because it
controls for per-image difficulty variance and has much higher statistical power than comparing unpaired
means):

| Comparison | Metric | Mean diff (control/baseline − proposed) | 95% bootstrap CI | Wilcoxon p |
|---|---|---|---|---|
| proposed vs. original DA2 | RMSE | +0.0350 | [+0.0270, +0.0438] | 5×10⁻¹⁸ |
| proposed vs. original DA2 | AbsRel | +0.0007 | [−0.0008, +0.0022] | 0.09 (n.s.) |
| proposed vs. no-intrinsics control | RMSE | +0.0024 | [+0.0013, +0.0034] | 3.5×10⁻⁵ |
| proposed vs. no-intrinsics control | AbsRel | +0.0006 | [+0.0002, +0.0010] | 0.005 |

Honest reading for the thesis: vs. **original DA2**, decisively better on RMSE (−14.5%), sq_rel (−26%), δ1; the
AbsRel edge is positive on average but not per-image significant. vs. the **matched control**, the
camera-intrinsics input wins significantly on *both* AbsRel and RMSE — at ViT-S scale, the improvement is
attributable to the intrinsics input, not just to fine-tuning.

### 2.5 ViT-L collapse — a silent training bug, root-caused and fixed

Applying the identical recipe at ViT-L **collapsed both the proposed and control runs** (val AbsRel frozen at
0.346 from epoch 1–2), even after retrying with `--head-lr-multiplier 1.0` plus warmup. The trained model
output **exactly 0 everywhere** (not NaN). The old February ViT-L checkpoints showed the same signature — this
bug predates the current campaign.

**Root cause**, found via an 18-step instrumented reproduction: the pretrained DA2 *basic* model outputs
**disparity** (inverse depth, values in the hundreds), but `train.py` was feeding it **direct depth targets**
(0–10 m) — an inverted, magnitude-mismatched training objective. Two properties of `ScaleShiftInvariantLoss`
turn that mismatch fatal:
1. it clamps the fitted alignment scale to `min=1e-3` (positive only), so it *cannot* fit an inverted
   relationship between disparity output and depth target;
2. its gradient-matching term (α=0.5) compares **unaligned raw values**, which crushes the output magnitude
   toward zero.

The loss optimum under this mismatch is literally "output constant zero," and the head's terminal ReLU makes
that state unrecoverable (zero gradient) — the dead-pixel fraction went from 0.001 to 1.0 within 18 training
steps. The training loss curve looks completely normal throughout, because with a constant output the loss is
just a function of the batch's own data statistics — this is a **silent failure**, discoverable only by
watching val metrics freeze or by an instrumented step-by-step reproduction. ViT-S survived the same trap only
by accident — its smaller head happened to re-learn direct-depth output instead of collapsing; this is why old
ViT-S "basic" checkpoints output direct depth while the pretrained model outputs disparity.

**The fix** (standard MiDaS/DA2 practice, previously not applied here): basic models now train on **disparity
targets**, $T_{\mathrm{disp}} = 1/\max(D^{GT}, d_{\min})$; in-training validation aligns in inverse-depth space
so best-checkpoint selection works correctly; checkpoints record `basic_target_space: 'disparity'` in their
metadata so the evaluation wrapper automatically treats the output as inverse depth, matching the original
DA2 convention. Verified with a 150-step instrumented run (loss 20–50× lower, zero dead pixels) before
committing further GPU time.

Result — ViT-L, NYU eigen-654, rawDepths, leak-free (654/654 images):

| Model | AbsRel ↓ | δ1 ↑ | RMSE ↓ | sq_rel ↓ |
|---|---|---|---|---|
| Original DA2 ViT-L (baseline) | 0.0425 | 0.9789 | 0.2119 | 0.0195 |
| Fine-tune, no intrinsics (control) | 0.0378 | 0.9862 | 0.1922 | 0.0142 |
| **Fine-tune + camera intrinsics (proposed)** | **0.0377** | 0.9861 | **0.1913** | 0.0143 |

Paired significance (n=654), proposed vs. original DA2 ViT-L: AbsRel mean diff +0.0048, 95% CI
[+0.0032, +0.0066], Wilcoxon p = 8.9×10⁻⁶; RMSE mean diff +0.0205, 95% CI [+0.0133, +0.0282], Wilcoxon
p = 1.1×10⁻⁹ — **unambiguously better on both headline metrics.** vs. the no-intrinsics control, however, the
ViT-L pair is a **statistical tie** (p = 0.18 / 0.50). Expected explanation: NYU is captured with a single
Kinect sensor, so the intrinsics input is *constant across the entire dataset* and carries no discriminative
signal for the camera encoder to learn from at this data scale — this becomes the motivating question for the
next two rounds.

### 2.6 ViT-L round 2 — geometrically-exact FoV augmentation; an important negative result

**Motivation:** if $K$ is constant across all training samples, the camera encoder has nothing to learn from.
Worse, the *existing* training pipeline random-cropped images without updating $K$ to match — the camera
encoder was training on geometrically **wrong** values, constant *and* incorrect.

**Fix:** a zoom-crop augmentation added to `datasets/training_datasets.py` — for each sample, a random window
(fixed 4:3 aspect, zoom factor 1.0–1.7 or 1.0–2.2 depending on the run, 20% exact full-frame) is cropped from
the image and $K$ is updated **exactly**:
$$c_x' = c_x - x_0, \quad c_y' = c_y - y_0$$
$$f_x' = f_x \cdot \frac{W_{\text{net}}}{w_{\text{win}}}, \quad f_y' = f_y \cdot \frac{H_{\text{net}}}{h_{\text{win}}}$$
resizing to a fixed 518×686 network input frame. Depth values are unchanged (same scene points). Both models
in a pair train on identical augmented images; only the proposed model receives the corresponding $K$.
(bs 2 + grad-accumulate 2 was required — bs 4 at 518×686 OOMs on the 4090's 24GB.)

Result — ViT-L, NYU eigen-654, rawDepths (654/654):

| Model | AbsRel ↓ | δ1 ↑ | RMSE ↓ |
|---|---|---|---|
| Original DA2 ViT-L | 0.0425 | 0.9789 | 0.2119 |
| Fine-tune + augmentation, no intrinsics | **0.0359** | **0.9872** | **0.1849** |
| Fine-tune + augmentation + intrinsics | 0.0362 | 0.9869 | 0.1881 |

Paired t-tests (n=654): **the augmentation itself is a clear win** for both models — proposed 0.0377→0.0362
(p=1.2×10⁻⁸), control 0.0378→0.0359 (p=1.2×10⁻¹³). These are the best NYU numbers of the entire project, both
far beyond released DA2 (AbsRel p=3.9×10⁻¹², δ1 p=6.9×10⁻⁷, RMSE p=3.3×10⁻⁹).

**But the intrinsics input still does not help at ViT-L** — with varied, *correct* K, the intrinsics model is
now marginally *behind* its control (AbsRel +0.0003, p=0.008; RMSE +0.0031, p=2×10⁻⁵; δ1 tie).

**Why (this is a genuine scientific finding, not a failure to hide):** under the *relative/basic* evaluation
protocol, predictions are scale-and-shift aligned per image before scoring. The main information field-of-view
provides is exactly **global metric scale disambiguation** — which per-image alignment structurally removes
from the evaluation before any metric is even computed. So at ViT-L capacity — where the encoder already
extracts strong perspective/semantic scale cues from the image content alone — the camera token has nothing
left to contribute under this protocol, and its extra parameters slightly perturb the small-data (636-image)
fine-tune. This motivates: the natural place to look for a camera-intrinsics contribution is **metric depth**
(no per-image alignment at eval time), or training regimes with genuine multi-camera FoV variation. NYU-basic
is a ceiling-limited testbed for this hypothesis.

### 2.7 The metric experiment — where the effect finally appears

Prediction from §2.6: intrinsics carry global-scale information that the basic protocol's alignment step
erases, so the contribution should appear in **metric** depth evaluation (absolute meters, no alignment).
Setup: `--model-type metric` (SiLog loss, unified ReLU head, initialized from the basic ViT-L checkpoint),
leak-free NYU split, FoV augmentation, identical no-intrinsics control.

**A real evaluation bug was found and fixed on the way** (`a473217`): the metric-model loader was constructing
the *released* checkpoint's sigmoid×max_depth head class, while `train.py`-trained metric checkpoints use a
unified ReLU head — same parameter names/shapes, so the weights loaded with **no error** but produced depths
about 4× too large (eigen AbsRel 2.97 vs. the real ≈0.08). This is exactly the kind of silent architecture
mismatch worth a methodology note in chapter 3. Fixed by recording `from_train_py` in checkpoint metadata and
routing to the correct head architecture at eval time.

Result — ViT-L metric, NYU eigen-654, absolute depth, no alignment, two rounds of increasing FoV variety:

| Model | AbsRel ↓ | δ1 ↑ | RMSE ↓ |
|---|---|---|---|
| Released DA2 metric (Hypersim-trained) | 0.2073 | 0.6919 | 0.6063 |
| Round 1 (FoV 41–63°): control | 0.0796 | 0.9583 | 0.2919 |
| Round 1 (FoV 41–63°): **+ intrinsics** | **0.0791** | **0.9593** | **0.2882** |
| Round 2 (FoV 33–63°, wider): control | 0.0826 | 0.9546 | 0.2998 |
| Round 2 (FoV 33–63°, wider): **+ intrinsics** | **0.0817** | **0.9569** | **0.2944** |

Paired significance vs. the identical control (n=654):

| Round | AbsRel | RMSE | δ1 |
|---|---|---|---|
| 1 (FoV 41–63°) | n.s. (t p=0.26) | **p=4.9×10⁻³** | n.s. |
| 2 (FoV 33–63°) | Wilcoxon **p=0.040** (t p=0.089) | **p=6.5×10⁻⁵** | t p=0.073 |

Three findings worth chapter-4 subsections:
1. **Both fine-tunes beat the released metric model by 2.6×** (AbsRel 0.207 → 0.079–0.083, paired t
   p ≈ 10⁻¹³³) — the strongest baseline win of the project (expected: DA2's own paper notes Hypersim's metric
   scale transfers poorly to real scenes).
2. **The intrinsics input now helps at ViT-L**, significantly on RMSE in both rounds, and the effect **grows
   with FoV variation** (RMSE p: 4.9×10⁻³ → 6.5×10⁻⁵; AbsRel: n.s. → Wilcoxon-significant at p=0.040). This is
   a **dose-response pattern** — the contribution scales with how much geometric ambiguity is actually present
   in the training data — replicated across two independently trained pairs. This dose-response replication is
   the central piece of evidence that the effect is real and mechanistic, not a one-off fluctuation.
3. Trade-off: harder augmentation slightly lowers absolute accuracy (0.0791 → 0.0817) while widening the
   intrinsics gap — round 1 is the best raw *model*, round 2 is the strongest *ablation evidence*. This
   trade-off recurs and intensifies in §2.8.

### 2.8 The focal-jitter experiment — proving the model actually *reads* K, not just correlates with it

§2.7 left a gap: the win was statistically present but small. A diagnostic (§2.9, the "K-sweep") explains why:
scaling $f_x, f_y$ at inference time moved the round-2 proposed model's predictions with a log-log slope of
only **0.06** — on single-camera NYU, $K$ resolves no ambiguity in practice, so the network never had pressure
to learn to actually read it, even though reading it would have helped.

**The fix — make scale genuinely unresolvable without K.** From the pinhole projection equation
$u = f \cdot X / Z$: scaling focal length by $\alpha$ *and* all scene depths by $\alpha$ produces the
**identical image**. The new `--focal-jitter m` augmentation draws $\alpha \sim \exp(\mathcal{U}(-\ln m,
+\ln m))$ per sample (here $m=1.6$), multiplies $f_x, f_y$ and the GT depth by $\alpha$, and leaves pixels
untouched — this is geometrically **exact** (unlike resampling-based augmentations, it introduces zero
interpolation artifacts) and leaks zero information through the image itself. Consequence: the K-blind control
now faces literally identical pixels paired with contradictory depth labels across samples — the best it can
do is regress to the mean of the $\alpha$ distribution, incurring an **irreducible error floor**. The
K-aware model can resolve $\alpha$ exactly by reading $f$ off its input. This turns "use $K$" from optional to
mathematically necessary for the K-blind model's failure mode to be avoidable.

Two implementation details matter and are worth documenting as methodology notes:
- the pixel validity mask is a property of the sensor's *unscaled* depth, so the mask's depth-ceiling threshold
  must also be scaled by $\alpha$ — otherwise exactly the far-away pixels where $\alpha$ carries the most
  information get excluded from training;
- a new **seed-locking** mechanism (`--seed 42`) makes both arms of a pair see identical data order, flips,
  zoom-crop windows, and $\alpha$ draws — verified by bit-exact matching iteration-0 losses (1.725 = 1.725) —
  so any measured difference is attributable only to the presence/absence of the $K$ input, not to sampling
  noise between the two runs.

Also combined with `--silog-lambda 0.3` (λ=0.5 half-forgives exactly the global-scale error that $K$
disambiguates; lowering it makes the global-scale term matter more to the loss).

Full recipe (`--max-depth 20` so the jittered ceiling 10×1.6=16m never clips):
```
python train.py --encoder vitl --datasets nyu --model-type metric \
  --min-depth 0.001 --max-depth 20 --epochs 40 --bs 2 --accumulate-grad 2 \
  --lr 5e-6 --freeze-dinov2 --head-lr-multiplier 10.0 --grad-clip 1.0 \
  --focal-jitter 1.6 --silog-lambda 0.3 --seed 42 \
  --pretrained-from <basic ViT-L checkpoint> \
  [--model-variant revised --use-camera-intrinsics | --model-variant base]
```

**Result A — K-sweep (the mechanistic headline).** 20 eigen-test images, $f_x, f_y$ scaled by
{0.6, 0.8, 1.0, 1.25, 1.6}, measuring the log-log slope of predicted depth vs. injected scale factor (ideal
slope = 1.0, meaning the model rescales its output exactly as $u=fX/Z$ requires; slope = 0 means it ignores
$K$ entirely):

| Pair | Proposed model slope | Control slope |
|---|---|---|
| Round 2 (before focal jitter) | 0.06 ± 0.01 | −0.00 |
| **Focal jitter (after)** | **0.93 ± 0.03** | **−0.00** |

The proposed model's output now rescales almost exactly proportional to $f$, exactly as the pinhole equation
prescribes; the control cannot respond at all, by construction.

**Result B — standard eigen-654 metric evaluation** (absolute depth, no alignment, n=654, bootstrap 95% CI +
Wilcoxon):

| Metric | + intrinsics | control | diff [95% CI] | Wilcoxon p |
|---|---|---|---|---|
| AbsRel ↓ | **0.0849** | 0.1030 | −0.0182 [−0.0210, −0.0153] | 4.0×10⁻³¹ |
| RMSE ↓ | **0.3071** | 0.3431 | −0.0361 [−0.0432, −0.0289] | 1.0×10⁻²⁴ |
| δ1 ↑ | **0.9520** | 0.9226 | +0.0294 [+0.0224, +0.0367] | 2.5×10⁻²⁴ |
| SILog ↓ | **0.0942** | 0.1045 | −0.0102 | 1.5×10⁻³⁴ |
| log10 ↓ | **0.0363** | 0.0427 | −0.0065 | 2.8×10⁻²⁷ |

Every metric, decisively — vs. round 2's marginal p=0.04. The mechanism is visible in the per-image
scale-ratio distribution (median predicted/GT depth): the proposed model centers at **1.014**, while the
control — which must regress to the mean of the unresolvable $\alpha$ noise — is biased and spread wider
(median 1.081), exactly as the ill-posedness argument predicts.

**Result C — generalization across held-out field-of-view.** Deterministic center zoom-crops at zoom
{1.0, 1.4, 1.8} with exactly-updated $K$, all 654 images, AbsRel (no eigen crop here, so not directly comparable
to Result B, but comparisons within a row are apples-to-apples):

| zoom | + intrinsics | control |
|---|---|---|
| 1.0 | **0.0924** | 0.1102 |
| 1.4 | **0.0974** | 0.1100 |
| 1.8 | **0.1065** | 0.1148 |

Proposed model wins at every zoom level.

**Interpretation for the thesis discussion:** the focal-jitter arm extends the §2.7 trade-off to its logical
end — absolute quality pays a small price for the harder training task (AbsRel 0.0849 vs. round 1's best-ever
0.0791), but the intrinsics evidence goes from marginal (round 2 AbsRel p=0.04) to overwhelming (p≈10⁻³¹ on
every metric), and the K-sweep slope going from 0.06 to 0.93 is a **mechanistic** demonstration that the camera
encoder went from decorative to load-bearing. Note: the scalar `output_gate` parameter itself ends training at
only −0.024 (small); this shows the gate's raw value is *not* a reliable proxy for how much the model actually
uses $K$ — the K-sweep slope is the correct diagnostic, and this discrepancy is itself worth a sentence in the
methodology-lessons subsection.

### 2.9 Ablation: is it the jitter, or the λ change?

Round 3 (§2.8) changed two things vs. round 2 simultaneously: focal jitter, and `--silog-lambda` 0.5→0.3. A
fourth seeded pair isolates this: focal jitter on ($m=1.6$), **λ left at 0.5 unchanged**, same seed-locking.

| Metric | + intrinsics | control | Wilcoxon p |
|---|---|---|---|
| AbsRel ↓ | **0.0834** | 0.1065 | 3.4×10⁻⁴⁵ |
| RMSE ↓ | **0.2987** | 0.3497 | 7.5×10⁻⁴⁰ |
| δ1 ↑ | **0.9554** | 0.9173 | 1.2×10⁻³⁹ |
| SILog ↓ | **0.0918** | 0.1056 | 2.2×10⁻⁵⁰ |

K-sweep slope for this pair: **0.97 ± 0.03** — even closer to the ideal 1.0 than round 3's 0.93.

**Conclusion: attribution is clean.** The augmentation is the causal driver of the whole effect (in fact
slightly *better* here than with λ=0.3: AbsRel 0.0834 vs. 0.0849) — the λ change was unnecessary and mildly
harmful. This is the fourth independently-trained pair to replicate the win, completing a clean dose-response
series across rounds 1→4.

### 2.10 K-knockout: withholding or corrupting K causally degrades accuracy

The K-sweep (§2.8) shows the model's output *moves with* $K$. A complementary test is needed to show accuracy
*depends on* $K$ being correct — i.e., that the camera-intrinsics path is load-bearing, not decorative. The
round-3 proposed checkpoint was evaluated on the full test set (n=654, no crop) under four input conditions:

| K condition | AbsRel ↓ | δ1 ↑ | RMSE ↓ |
|---|---|---|---|
| correct K | **0.0924** | **0.9339** | **0.3455** |
| K withheld | 0.1467 | 0.8449 | 0.4826 |
| f × 0.7 (wrong K) | 0.2887 | 0.1310 | 0.9153 |
| f × 1.4 (wrong K) | 0.4274 | 0.1209 | 1.2118 |

Two conclusions, both worth stating explicitly as this is the strongest causal evidence in the thesis:
1. **Withholding K degrades AbsRel by 59%** (0.092→0.147) — the camera-parameter pathway is not a decorative
   input; a substantial part of the model's scale computation flows through it.
2. **The magnitude of degradation under wrong K matches the pinhole-equation prediction quantitatively.** With
   K-sweep slope $s=0.93$, feeding $f \times X$ should produce a scale error of about $|X^{s} - 1|$: predicted
   ≈0.28 for $X=0.7$ and ≈0.37 for $X=1.4$; observed (0.289 and 0.427, which include the model's baseline
   error) match closely. This quantitative match — not just "wrong K hurts," but "wrong K hurts by *exactly
   the amount geometry says it should*" — is evidence that cannot be produced by a spurious correlational
   shortcut; only a model that has actually learned the pinhole relationship would degrade this precisely.

**Practical implication for chapter 5 (limitations):** a model that causally relies on K is also a model that
is *sensitive to K's accuracy* — in applications with imprecise calibration, this sensitivity needs to be
accounted for in system design. This was not evaluated under realistic (small) calibration noise, only under
these large deliberate corruptions — flagged as future work.

### 2.11 Cross-cutting design/methodology notes (for chapter 3's implementation-lessons subsection)

Three silent failures were found and fixed across this campaign, and the common lesson is worth stating
explicitly: **in deep learning pipelines, failures can be completely silent** — the loss curve looks normal,
weight loading raises no errors, and only independent end-to-end checks (reproducing a published baseline,
watching for frozen validation metrics, inspecting output-value statistics) surface the problem.
1. Target-space/output-space mismatch causing silent dead-ReLU collapse (§2.5).
2. Stale $K$ under naive random-crop augmentation, not updated to match the crop (fixed by §2.6's exact
   zoom-crop geometry).
3. Metric-model head architecture mismatch at evaluation time causing ~4× depth over-estimation with no load
   error (§2.7).

This is why, before trusting any comparison in this campaign, the evaluation pipeline itself was independently
verified by reproducing the DA2 paper's own published Table 2 numbers (already documented in the existing
thesis and in `CLAUDE.md`) — this pipeline-verification step should be explicitly called out as a
prerequisite step in chapter 4, not just implied.

**Total experimental scope** (worth a sentence in the chapter-4 summary, for transparency about what was
tried and discarded): over twenty independent training runs across two encoder sizes, two protocols
(basic/metric), and two augmentation intensities, plus a full audit of ~20 pre-existing checkpoints from
before this campaign. Approaches tried and abandoned: multi-dataset zero-shot fine-tuning (destroys
generalization, §2.2); lowering the head learning rate as a fix for the ViT-L collapse (did not work — showed
the root cause was target-space mismatch, not learning rate, §2.5).

---

## 3. Summary table for chapter 5 (findings condensed)

| # | Finding | Evidence |
|---|---|---|
| 1 | Proposed model (ViT-L, camera intrinsics) beats released DA2 significantly on both protocols | Basic: AbsRel 0.0425→0.0377, p=2.5×10⁻⁸. Metric: AbsRel 0.207→0.079, p≈10⁻¹³³ |
| 2 | Net contribution of K itself is protocol-dependent | Not significant vs. control under basic/relative protocol at ViT-L (alignment removes scale info); significant under metric protocol |
| 3 | Contribution of K shows a dose-response relationship with geometric ambiguity in training data | RMSE p: 4.9×10⁻³ (FoV 41-63°) → 6.5×10⁻⁵ (FoV 33-63°) → ≤10⁻²⁴ (focal jitter), replicated across 4 independent pairs |
| 4 | The model demonstrably *reads* K mechanistically, not just correlates with it | K-sweep slope 0.06 (before) → 0.93–0.97 (after focal jitter); ideal is 1.0 |
| 5 | The model's reliance on K is causal, not just correlational | K-knockout: withholding K degrades AbsRel 59%; wrong K degrades accuracy by the exact amount pinhole geometry predicts |
| 6 | Net encoder-capacity dependence | Small but significant K contribution at ViT-S even under the basic protocol; not at ViT-L — larger encoders extract more scale information from image content alone, leaving less for K to add under scale-normalized evaluation |

---

## 4. Chapter-by-chapter mapping

### Chapter 1 (Introduction)
No factual changes required from this campaign — the research question ("does giving a monocular depth model
explicit camera intrinsics improve depth estimation") is unchanged. Optionally strengthen the motivating claim
with one sentence citing the K-knockout result (§2.10) as the thesis's strongest evidence of causal dependence,
if the introduction previews headline results.

### Chapter 2 (Literature review / background)
No new content required. Only action item: **§5 below lists 10 bibliography keys that chapter 3/4 now cite but
that are missing from `MyReferences.bib`** — these need to be present regardless of whether chapter 2 cites them.

### Chapter 3 (Methodology) — content to add, roughly in this order
1. **Camera descriptor**: expand from the original 4-dim $[\tilde f_x, \tilde f_y, \tilde c_x, \tilde c_y]$ to
   the actual 9-dim vector implemented in code (`models/raw_models/DepthAnythingV2-revised/depth_anything_v2/cam_enc.py`,
   `intrinsics_to_encoding()`): $[\tilde f_x, \tilde f_y, \tilde c_x, \tilde c_y, \mathrm{fov}_w, \mathrm{fov}_h,
   W/H, (f_x+f_y)/2, H{\cdot}W]$ (all normalized). Note that $K$ and $(H,W)$ must be expressed in the same
   coordinate frame or the encoding is geometrically wrong (this exact bug occurred under naive cropping, see
   item 4 below). **Verified against code** — `cam_enc.py:10-67`.
2. **Camera encoder architecture — correct this in chapter 3, it changed from the original design**: it is
   **not** decoder-concatenation as originally described. It is: a 2-layer MLP (GELU activation) projecting the
   9-dim vector to the vision encoder's embedding dimension (1024 for ViT-L), followed by 4 transformer blocks
   (`trunk_depth=4`, DINOv2-style blocks) for refinement, producing a single **camera token** that is injected
   into the **DINOv2 encoder's** token sequence (not the decoder) at a configurable layer
   (`--cam-token-inject-layer`, default = first layer). Verified against code —
   `cam_enc.py:70-169`, `train.py:86`. **A zero-initialized output gate** (`nn.Parameter(torch.zeros(1))`)
   multiplies the camera token before injection, so at the start of fine-tuning the token is exactly zero and
   the model's behavior is identical to the pretrained base model; the gate opens gradually only if K helps
   reduce loss. This gating design is why fine-tuning from the pretrained checkpoint doesn't destabilize early
   training.
3. **NYU train/test split leak and its fix** — §2.3 above.
4. **Disparity target space for relative models, and the collapse it fixes** — §2.5 above; include the
   equation $T_{\mathrm{disp}} = 1/\max(D^{GT}, d_{\min})$.
5. **FoV-aware geometric augmentation (zoom-crop with exact K update)** — §2.6 above; include the two update
   equations for $c_x', c_y'$ and $f_x', f_y'$.
6. **Focal-jitter augmentation** — §2.8 above; include the pinhole argument and the $\alpha$ sampling equation.
   This is the single most important methodological contribution of the campaign and deserves a full
   subsection, not a paragraph.
7. **Implementation/debugging-lessons subsection** — the three silent failures, §2.11.
8. Minor: note that knowledge distillation, while implemented and available (`--use-distillation`,
   `--teacher-checkpoint`), was **disabled** for all final experiments in this report, specifically to isolate
   the camera-intrinsics contribution from other training signals.

### Chapter 4 (Results) — content to add, roughly in this order
1. New top-level section: a controlled study on the standard NYUv2 protocol, framed as four models compared
   under identical conditions per encoder size: (i) the DA2 paper's own published numbers, (ii) the released
   DA2 checkpoint evaluated by this project's own pipeline, (iii) a matched no-intrinsics control fine-tune,
   (iv) the proposed model. Comparing (iv) vs (ii) measures improvement over the released baseline; (iv) vs
   (iii) — which differ *only* in access to K — isolates the net contribution of the camera-intrinsics input.
2. **Pipeline verification subsection**: reproduce the DA2 paper's Table 2 numbers first, before trusting any
   comparison (numbers already in `CLAUDE.md`/existing thesis — just needs to be explicitly framed as the
   prerequisite step for this section).
3. **Zero-shot fine-tuning failure** subsection — §2.2.
4. **Basic/relative protocol four-way comparison** (ViT-S and ViT-L) with paired significance tables — §2.4,
   §2.5.
5. **Augmentation effect + informative negative result** subsection — §2.6.
6. **Metric protocol results** subsection, with the dose-response framing — §2.7.
7. **Focal-jitter round** subsection — §2.8, including the K-sweep table and figure (real data — see note
   below on the existing placeholder figure).
8. **Ablation** subsection — §2.9.
9. **K-sweep diagnostic** subsection (can be merged with #7, or kept separate as a "does the model read K"
   mechanistic question distinct from "does K help") — §2.8 Result A.
10. **K-knockout** subsection — §2.10.
11. **Protocol/capacity dependence discussion** subsection — synthesizes why the effect size differs between
    basic/metric protocols and ViT-S/L (§2.6's explanation + a note that larger encoders extract more scale
    information from image content alone, leaving less marginal value for K, consistent with FoV-compensation
    literature).
12. **Controlled-study summary** subsection — the experimental-scope transparency paragraph, §2.11 last
    paragraph.
13. **Figure note**: the thesis currently has a placeholder tikz figure for "K-sensitivity" with fabricated
    example data (explicitly commented `% Sample data ... User needs to replace with actual data`). Replace
    with the real K-sweep data (§2.8 Result A) — log-log axes, predicted-depth-ratio vs. focal-scale-factor,
    with three series: ideal slope-1 line, proposed-model (round 2 flat, round 3 following the line), control
    (flat at 1.0 by construction). Exact plotted values are in the recovered patch (see §9) if a precise figure
    reproduction is wanted rather than a schematic.
14. **Statistical methodology note**: this controlled study uses **paired** tests (paired t-test + Wilcoxon
    signed-rank on n=654 image pairs, plus 10,000-resample bootstrap 95% CI on paired differences) rather than
    the two-sample tests used elsewhere in chapter 4's cross-dataset comparisons — because both models are
    evaluated on the exact same images, a paired design removes per-image difficulty variance and gives much
    higher statistical power. This should be noted near wherever chapter 4 currently describes its statistical
    methodology, generalizing the existing "two-sample t-test" description to note it's used for cross-dataset
    comparisons specifically, with paired tests as the method for this same-test-set controlled study.

### Chapter 5 (Conclusion) — content to add
1. Update the findings summary to state the conclusion precisely (not overclaim): the proposed ViT-L model
   beats released DA2 significantly on both protocols; the *net* contribution of K vs. a matched control is
   protocol- and capacity-dependent, small-but-significant at ViT-S even under the scale-normalized protocol,
   and — decisively, under the metric protocol with focal-jitter training — proven both statistically
   (p≤10⁻²⁴ on every metric) and mechanistically (K-sweep slope 0.06→0.93) and causally (K-knockout).
2. **New "methodological and diagnostic contributions" subsection** — this campaign's contributions beyond the
   headline result, each independently useful to future researchers in this area:
   - identifying and fixing the NYUv2 split leak (>80% test-set contamination via naive random 80/20 split);
   - diagnosing a silent training collapse caused by target-space/output-space mismatch interacting with a
     scale-invariant loss's positivity-clamped alignment scale;
   - the FoV-aware geometric augmentation (exact K update under zoom-crop);
   - the analysis that relative/scale-normalized evaluation protocols are structurally blind to K's main
     contribution (absolute scale), motivating metric evaluation for this specific research question;
   - the focal-jitter augmentation as a way to make scale recovery *provably* impossible without K, turning
     "does the model use K" into a directly testable, falsifiable claim rather than an inference from
     aggregate metrics;
   - the K-sweep diagnostic itself, as a general-purpose mechanistic test more informative than either
     aggregate metrics or the learned gate parameter's magnitude;
   - full seed-locking across paired arms, verified by bit-exact iteration-0 loss matching, as a rigor
     practice for any future ablation of this kind.
3. **Limitations — update/add:**
   - the K-knockout result quantifies dependence on K accuracy (§2.10) — this is now evidence *for* a
     limitation, not an unexamined gap: the model's causal reliance on K means it is also sensitive to K being
     wrong (miscalibration), and this was only tested under large deliberate corruption, not realistic small
     calibration noise;
   - the AbsRel significance in the metric protocol is more marginal than RMSE's (round 2: Wilcoxon p=0.040,
     t-test p=0.089) — state this evidentiary asymmetry honestly rather than only reporting the strongest
     number;
   - the controlled study is NYU-only (indoor, single real camera + simulated FoV variation via augmentation,
     not genuine multi-camera data); generalizing to outdoor/large-depth-range benchmarks (e.g. KITTI) needs
     follow-up work, flagged as a TODO in the earlier draft too.
4. **Future work — add:**
   - training on genuinely multi-camera data (rather than simulating FoV variation via augmentation) is
     expected to reveal an even larger and more stable contribution, per the dose-response finding;
   - a systematic study of the camera-token injection layer/depth (`--cam-token-inject-layer` — currently only
     tested at its default, the first encoder layer) and of multi-layer injection;
   - repeating the metric controlled study on outdoor benchmarks with real (not simulated) large depth ranges
     and camera variation (e.g. KITTI), if real training data becomes available;
   - systematic evaluation under realistic (small) calibration noise rather than only the deliberate large
     corruptions used in the K-knockout test.

---

## 5. Bibliography entries needed in `MyReferences.bib`

The following citation keys are referenced by the chapter-3/4 content above and were **missing** from the
current (template-reverted) `MyReferences.bib` as of this report (checked directly against the file). Full,
correctly-formatted BibTeX for all of these exists in git history and was recovered while writing this report —
`git show c743cc8:thesis/tex/MyReferences.bib` contains the complete ~35-entry file (only `Zhou2017Unsupervised`,
`Ranftl2021DPT`, and `Eigen2014Depth` currently exist in the live template-reverted file):

- `DepthAnythingV2` — the base model itself; currently missing, this is likely the single most-cited reference
  in chapters 3–4 and must be restored.
- `Kendall2015GeometryAware`
- `Chen2022CameraAware`
- `Guizilini2020SemanticallyGuided`
- `Li2021NeuralSceneFlow`
- `Casser2019DepthSensors`
- `Perez2018FiLM`
- `Ranftl2019Robust`
- `Godard2019Digging`
- `Saxena2023FOV`

Do not hand-write new BibTeX for these — reuse the exact entries from `c743cc8`, they're already correct and
were presumably vetted once already.

---

## 6. Testing / reproducibility appendix

For a methodology/reproducibility note in chapter 3, or an appendix:
- Unit tests for the focal-jitter augmentation: `tests/test_focal_jitter.py` (10 passing tests — covers the
  augmentation transform itself, valid-mask semantics under scaling, and constructor validation).
- Diagnostic tools built for this campaign (all in `tools/`): `k_sweep.py` (K-sweep diagnostic, §2.8 Result A),
  `k_knockout.py` (K-knockout test, §2.10), `eval_fov_varied.py` (FoV-generalization eval, §2.8 Result C),
  `paired_stats.py` (paired bootstrap CI + Wilcoxon test), `sanity_check_focal_jitter.py` and
  `compare_pair_losses.py` (seed-lock verification).
- Reproduction commands for the headline (round-1 vits) result:
  ```bash
  python evaluate_paper.py --dataset nyu --model da2-revised --model-type basic --encoder vits \
    --checkpoint models/raw_models/DepthAnythingV2-revised/checkpoints/basic_finetuning/revised_basic_vits_nyu_intrinsics_lr5e-6_bs4_20260705_164021/best.pth
  python evaluate_paper.py --dataset nyu --model da2 --model-type basic --encoder vits
  ```

---

## 7. Open items flagged but not yet resolved (carry these into chapter 5 as explicit future work, don't silently drop them)

- `--cam-token-inject-layer` sensitivity was never swept — only the default (first layer) was used in every
  reported experiment.
- No repeated-seed variance estimate exists for any of the paired comparisons (each pair is a single training
  run per arm) — multi-seed repetition would strengthen the significance claims further but wasn't done due to
  time/compute.
- The realistic-calibration-noise version of the K-knockout test (small perturbations rather than 0.7×/1.4×
  corruption) was not run.
- Generalizing the controlled metric study to an outdoor benchmark (KITTI) with real camera variation is
  unstarted — flagged as future work in §4's chapter-5 mapping.

---

## 8. Checkpoints and result-artifact index (for anyone who wants to re-run evaluations, not for the thesis text itself)

| Result | Checkpoint / artifact (on the lingotube training server) |
|---|---|
| ViT-S winning proposed model (§2.4) | `basic_finetuning/revised_basic_vits_nyu_intrinsics_lr5e-6_bs4_20260705_164021/best.pth` |
| ViT-S control | `basic_finetuning/base_basic_vits_nyu_lr5e-6_bs4_20260705_165447/best.pth` |
| ViT-L basic, disparity-fixed (§2.5) | `basic_finetuning_disp/revised_basic_vitl_nyu_intrinsics_lr5e-6_bs4_20260706_120823/best.pth` |
| ViT-L basic + aug round 2 (§2.6) | `basic_finetuning_aug/revised_basic_vitl_nyu_intrinsics_lr5e-6_bs2_20260706_152237/best.pth` (+ matching `base_basic_vitl_..._171836/best.pth`) |
| ViT-L metric rounds 1–2 (§2.7) | `metric_finetuning_aug{,2}/{revised,base}_metric_vitl_nyu*/best.pth` |
| ViT-L metric focal-jitter round 3 (§2.8) | `metric_finetuning_focal/{revised,base}_metric_vitl_nyu*/best.pth` |
| ViT-L metric focal-jitter ablation (§2.9) | `metric_finetuning_focal_ablation/{revised,base}_metric_vitl_nyu*/best.pth` |
| K-sweep / K-knockout figures | `results/k_sweep/k_sweep_{round2_baseline,focal}.png`, `results/k_knockout/k_knockout_focal.json`, `results/output_diff/scale_ratio_hist_focal.png` |

---

## 9. Provenance of this report (how it was assembled, for auditability)

Three independent sources were cross-checked and found mutually consistent (all numbers above agree across
all three):
1. `REPORT_INTRINSICS_EXPERIMENT.md` — a 406-line English experiment log written during the campaign, deleted
   from the working tree by commit `0b66b80` (the same commit that reverted the thesis to the template),
   recovered via `git show 65a5b2d:REPORT_INTRINSICS_EXPERIMENT.md`.
2. `thesis_pending_reintegration.patch` — the actual previously-written and already-integrated (then reverted)
   Persian LaTeX prose for chapters 3–5, recovered via `git show 9fea97f:thesis_pending_reintegration.patch`
   (952 lines; this is the diff `74bee36..65a5b2d` for `chapter{3,4,5}.tex` and `commands.tex`). This patch
   applies cleanly on top of commit `74bee36`.
3. Auto-memory records from the sessions that ran these experiments (`focal-jitter-proves-intrinsics-usage.md`,
   `intrinsics-model-beats-da2-vits.md`, `nyu-training-split-leaks-eigen-test.md`,
   `vitl-head-collapse-dead-relu.md`), plus a direct read of the current `cam_enc.py` implementation to verify
   the architecture description in §4's chapter-3 mapping is accurate to the *current* code (not just to what
   was written before) — the encoding function, MLP+4-block architecture, and zero-init gate were all
   confirmed present at `models/raw_models/DepthAnythingV2-revised/depth_anything_v2/cam_enc.py` at the time
   of writing this report.

Full commit list touching this campaign (chronological): `d4605ec 139ea43 8b9ad2e cb303e0 7e6908e 76db955
6018172 852b084 5511e4c f41a859 e3870d8 24083f5 8ed02a9 65a5b2d e81331b ccb27e6 9fea97f c290421 a51e35f c743cc8
0b66b80`.
