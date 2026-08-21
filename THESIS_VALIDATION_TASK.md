# Task brief: finish the thesis number validation on the GPU server

You are picking up a numerical audit of a Persian LaTeX thesis (`thesis/tex/chapter1..5.tex`)
about camera intrinsics in monocular depth estimation. An offline audit is already done. What
remains needs the **lingotube** GPU server, which the previous session could not reach.

**Your job:** run the checks below, then report which findings are confirmed, refuted, or
inconclusive. **Do not edit any `thesis/tex/*.tex` file** — the author decides every change.

---

## 1. Server access

| Item | Value |
|---|---|
| Host | `lingotube` — `93.118.171.194`, user `ubuntu`, key auth |
| Remote repo | `/home/ubuntu/projects/MDE-plus-Intrinsic-matrix` |
| Python env | `source venv/bin/activate` (system python3 has no torch — always activate) |
| GPU | RTX 4090, 24 GB |
| tmux | session `lab` (create with `tmux new-session -d -s lab` if missing) |

If `ssh lingotube` fails, add to `~/.ssh/config`:

```
Host lingotube
    HostName 93.118.171.194
    User ubuntu
    IdentityFile ~/.ssh/id_rsa
```

Known state as of 2026-08-21: the host answers ping (~107 ms) but TCP 22, 2222, 22022 and 2200
were all filtered from the author's laptop. If it is still unreachable from your machine, stop
and say so — do not fabricate results.

### Sync first

`tools/audit_section_4_4_units.py` (needed for Check A) may not be on the server yet:

```bash
cd /home/mohamadamin.ghasemzade/MDE-plus-Intrinsic-matrix
git add tools/audit_section_4_4_units.py THESIS_VALIDATION_TASK.md
git commit -m "Add section 4.4 units audit + validation task brief"
git push origin main
ssh lingotube "cd ~/projects/MDE-plus-Intrinsic-matrix && git pull --rebase"
```

Check capacity before launching anything:

```bash
ssh lingotube 'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader; \
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

ViT-L inference needs ~6–9 GB. Two evals can coexist; anything alongside training cannot.
Run long jobs in tmux with a log, and make sure the completion marker reaches the log:

```bash
ssh lingotube "tmux new-window -t lab -n audit"
ssh lingotube "tmux send-keys -t lab:audit 'cd ~/projects/MDE-plus-Intrinsic-matrix && \
  source venv/bin/activate && mkdir -p logs && \
  { <COMMAND>; echo EXIT_CODE=\$?; } 2>&1 | tee logs/audit.log' Enter"
```

Poll with `ssh lingotube "tail -40 ~/projects/MDE-plus-Intrinsic-matrix/logs/audit.log"`.
Kill the window when done: `ssh lingotube "tmux kill-window -t lab:audit"`.

---

## 2. Background: the four findings

Full detail is in the audit artifact; here is what you need to test them.

**Finding 1 (critical).** Section 4.4's Table 4.2 (`tab:da2_revised_comparison`) reports the
proposed model beating the DA2 baseline by 55–74% on SiLog. Three repo facts suggest this is a
units artefact, not a result:

- `models/depth_anything_v2.py:201` — the released DA2 basic model returns
  `outputs_inverse_depth() == True` (it outputs **disparity**).
- `models/depth_anything_v2_revised.py:501` — the §4.4 checkpoints predate the disparity-target
  fix, carry no `basic_target_space` field, and return `False` (they output **direct depth**).
- `tools/validate_section_4_4_1.py:225` (`article_silog`) — the January 2026 protocol
  median-scales in **depth** space and never inverts. Its own docstring says §4.4.1 "predates
  the repository's February 2026 correction from median depth scaling to affine inverse-depth
  alignment."

So the baseline column may have been scored in the wrong units. Offline simulation showed a
*perfect* prediction scores SiLog 0.0000 as depth but 1.73–2.26 as disparity under this
protocol — and the thesis baseline column sits at 0.833–1.835.

> **Important:** `validate_section_4_4_1.py` reproduces the *historical* protocol faithfully, so
> it will **pass** and does **not** test this. Use Check A instead.

**Finding 2 (moderate).** The round-3 proposed model is reported as AbsRel **0.0849** in
`tab:nyu_metric_focal` but **0.0924** in `tab:k_knockout` and `fig:fov_generalization`. Suspected
cause: `tools/k_knockout.py:61` and `tools/eval_fov_varied.py:83` build
`NYUDataset(DatasetConfig(split='test'))` and never set `depth_field`, which defaults to
`'depths'` (inpainted) at `datasets/nyu.py:79` instead of `'rawDepths'`, and neither applies the
Eigen crop. Check B tests this with existing flags, no code changes.

**Finding 3 (moderate).** Table 4.2 mixes encoders: per `tools/validate_section_4_4_2.py:31–47`,
DrivingStereo is a ViT-S pair while NYUv2/DIODE/CityScapes are ViT-L. Check A prints the encoder
per row and confirms this — no separate run needed.

**Finding 4 (minor).** Stated FoV ranges (41–63°, 33–63°) should be ~40–63° and ~31–63°.
Arithmetic only, already settled offline. No GPU work.

---

## 3. Checks to run, in priority order

### Check A — does Table 4.2's improvement survive a correct alignment? (highest value)

```bash
python tools/audit_section_4_4_units.py --max-items 200
```

This evaluates both checkpoints on the same images with the same valid-pixel mask under two
alignments that differ in exactly one thing: the historical median-depth scaling (no inversion)
versus `src.metrics.align_non_metric_predictions` (affine, inverting when the model reports
inverse-depth output). It prints both improvement columns side by side and writes
`results/section_4_4_units_audit.json`.

`--max-items 200` gives a fast verdict; drop it for the full evaluation. Sample size does not
change the conclusion — the effect under test applies identically to every image.

**How to read it:**

| Observation | Conclusion |
|---|---|
| `historical` column lands near the thesis values (1.284 / 0.833 / 1.835 / 1.399) | The run reproduces §4.4; the comparison is like-for-like. |
| baseline prints `outputs_inverse_depth=True`, proposed prints `False` | **Finding 1 confirmed** — the two columns of Table 4.2 were scored in different units. |
| `corrected` improvement collapses toward 0 or goes negative while `historical` stays at 55–74% | **Finding 1 confirmed** — the reported gain is the units mismatch. |
| `corrected` improvement stays large and positive | **Finding 1 refuted** — report this clearly, it is good news for the thesis. |

If a dataset root is missing, run the subset that exists:
`--datasets NYUv2 DIODE` etc. Report which rows you could not cover.

### Check B — explain the 0.0849 vs 0.0924 gap

Four runs on the round-3 proposed checkpoint. Only the crop and GT field change:

```bash
CKPT=models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_focal/revised_metric_vitl_nyu_intrinsics_lr5e-6_bs2_20260709_214151/best.pth

for CROP in eigen none; do for FIELD in rawDepths depths; do
  python evaluate_paper.py --dataset nyu --model da2-revised --model-type metric \
    --encoder vitl --checkpoint "$CKPT" --crop $CROP --nyu-depth-field $FIELD \
    --clip-pred false --output-json results/finding2_${CROP}_${FIELD}.json
done; done
```

Expected if Finding 2 is right: `eigen`+`rawDepths` ≈ **0.0849** (the thesis table) and
`none`+`depths` ≈ **0.0924** (the knockout table). The other two isolate how much each factor
contributes. Report all four AbsRel values.

### Check C — regenerate the controlled study (confirms the §4.5 tables)

These recompute from checkpoints and self-check against the thesis values:

```bash
python tools/validate_section_4_5_6.py --regenerate
python tools/validate_section_4_5_7.py --regenerate --regenerate-diagnostics
```

They raise `AssertionError` naming the exact metric if a mean does not round to the printed
value. `4_5_7 --regenerate-diagnostics` also regenerates the K-sweep (expect slope
**0.93 ± 0.03**) and the FoV evaluation.

`validate_table_4_9.py` has **no** `--regenerate` — it only audits JSON that already exists:

```bash
ls results/table_4_9_validation/    # must contain all six files below
python tools/validate_table_4_9.py
```

Required: `nyu_basic_vits_crop-eigen_20260704_230541.json`,
`nyu_BASEFT_vits_nyuonly_leakfree.json`, `nyu_REVISED_vits_nyuonly_leakfree.json`,
`nyu_da2_basic_vitl_crop-eigen_20260709_140905.json`,
`nyu_BASEFT_vitl_disp_leakfree.json`, `nyu_REVISED_vitl_disp_leakfree.json`.
If they are absent on the server, say so and skip it rather than trying to reconstruct them —
they come from the relative-protocol (`--model-type basic`) runs, not the metric ones above.

### Check D — reproduce §4.4 as printed (optional, slow)

```bash
python tools/validate_section_4_4_1.py
python tools/validate_section_4_4_2.py
```

These parse the tables straight out of `chapter4.tex` and re-run inference on every image. They
confirm the printed numbers are *reproducible* under the historical protocol — which is a
separate question from whether that protocol was correct (Check A). Run only if time allows.

---

## 4. Reference: thesis values these should match

`tab:da2_revised_comparison` (§4.4.2), SiLog:

| Dataset | Encoder | DA2 baseline | Proposed | Improvement |
|---|---|---|---|---|
| NYUv2 | ViT-L | 1.284 | 0.329 | 74.38% |
| DIODE | ViT-L | 0.833 | 0.377 | 54.78% |
| CityScapes | ViT-L | 1.835 | 0.662 | 63.90% |
| DrivingStereo | **ViT-S** | 1.399 | 0.414 | 70.42% |

`tab:nyu_metric_results` (§4.5.6), NYU eigen-654, metric, no alignment:

| Model | Round | AbsRel | δ1 | RMSE |
|---|---|---|---|---|
| DA2 metric (Hypersim) | — | 0.2073 | 0.6919 | 0.6063 |
| Control | 1 | 0.0796 | 0.9583 | 0.2919 |
| Proposed | 1 | 0.0791 | 0.9593 | 0.2882 |
| Control | 2 | 0.0826 | 0.9546 | 0.2998 |
| Proposed | 2 | 0.0817 | 0.9569 | 0.2944 |

`tab:nyu_metric_focal` (§4.5.7, round 3):

| Metric | Proposed | Control | Diff [95% CI] | Wilcoxon p |
|---|---|---|---|---|
| AbsRel | 0.0849 | 0.1030 | −0.0182 [−0.0210, −0.0153] | 4.0×10⁻³¹ |
| RMSE | 0.3071 | 0.3431 | −0.0361 [−0.0432, −0.0289] | 1.0×10⁻²⁴ |
| δ1 | 0.9520 | 0.9226 | +0.0294 [+0.0224, +0.0367] | 2.5×10⁻²⁴ |
| SILog | 0.0942 | 0.1045 | −0.0102 [−0.0118, −0.0087] | 1.5×10⁻³⁴ |
| log10 | 0.0363 | 0.0427 | −0.0065 [−0.0076, −0.0053] | 2.8×10⁻²⁷ |

`tab:k_knockout` (round-3 proposed, n=654, no alignment):

| K condition | AbsRel | δ1 | RMSE |
|---|---|---|---|
| correct K | 0.0924 | 0.9339 | 0.3455 |
| K withheld | 0.1467 | 0.8449 | 0.4826 |
| f × 0.7 | 0.2887 | 0.1310 | 0.9153 |
| f × 1.4 | 0.4274 | 0.1209 | 1.2118 |

K-sweep slopes: round 2 = 0.06 ± 0.01, round 3 = 0.93 ± 0.03, ablation = 0.97 ± 0.03.

---

## 5. What to report back

1. **Finding 1 — confirmed or refuted?** Give the `outputs_inverse_depth` flags and both
   improvement columns from Check A. This is the one that decides whether §4.4 stays in the
   thesis, so state it plainly either way.
2. **Finding 2** — the four AbsRel values from Check B and whether they explain the 0.0849 /
   0.0924 gap.
3. **Check C** — pass/fail per validator, with the exact assertion text for any failure, and
   whether `validate_table_4_9.py` had its six input JSON files available.
4. Anything that could not run (missing dataset root, missing checkpoint, OOM) and why.
5. Any *new* discrepancy you notice between a printed thesis number and a regenerated one.

Report the numbers as they came out. If a check contradicts the audit's expectation, say so —
a refuted finding is a useful result, and the author would rather hear it than have it smoothed
over. Do not modify the thesis.
