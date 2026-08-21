# Handoff — finish the thesis figures that need the GPU server

**For a Claude session running on a machine that can reach `lingotube`.**
Read this file end to end before acting. It is self-contained: assume no memory of
the conversation that produced it.

---

## 1. Situation

A full visualisation pass on this Persian (RTL, XeLaTeX) thesis is **already
complete** — 21 figures and 29 tables, up from 6 and 16. Section 12 lists exactly
what was added so you do not redo it.

Three items could not be finished because they need GPU inference or the
TensorBoard run logs, and the machine that did the work had **all outbound TCP
blocked** (verified: `1.1.1.1:443`, `8.8.8.8:53` and `93.118.171.194:22` all time
out; ICMP passes, which is misleading). Your job is those three items:

| # | Deliverable | Needs |
|---|---|---|
| A | Pre-registered qualitative depth-map figure | GPU inference, both arms |
| B | Per-image scale-ratio distribution figure | same run as A (free) |
| C | Training-curve figure for the silent collapse | TensorBoard logs under `runs/` |

A fourth item, the dataset sample strip, needs no server — see §9.

Everything needed is already written: two runnable scripts under `tools/`, and
ready-to-`\input` LaTeX under `thesis/figures/pending/`. You are wiring them up and
running them, not designing them.

**Also do this, and report it:** nothing in the visualisation pass has ever been
compiled — the origin machine had no `xelatex`. Step 10 builds the document. If the
build fails, fixing it takes priority over A–C.

---

## 2. Environment

| Item | Value |
|---|---|
| Server | `lingotube` → `ubuntu@93.118.171.194`, key auth |
| Remote repo | `/home/ubuntu/projects/MDE-plus-Intrinsic-matrix` |
| Remote venv | `source venv/bin/activate` — **required**; system python has no torch |
| GPU | RTX 4090, 24 GB. This task is ViT-L *inference* only, ≈6–9 GB. No training. |
| tmux | session `lab` (`tmux new-session -d -s lab` if absent). Not required — these jobs are short. |
| Runtime | Minutes, not hours. Run in the foreground and read the output. |

If the bare alias does not resolve, add to `~/.ssh/config`:

```
Host lingotube
    HostName 93.118.171.194
    User ubuntu
    IdentityFile ~/.ssh/id_rsa
```

There is a `gpu` skill in this repo that documents the same server; invoke it if you
want its conventions, but this file is sufficient.

---

## 3. Guardrails — read before editing any `.tex`

1. **Reported numbers are frozen.** Never alter a result value while editing. Verify
   with `python3 tools/check_thesis_numbers.py` after any edit. Additions are
   expected and fine; a **removed or changed** result number means you broke
   something — stop and investigate.
2. **Do not invent data.** Every number in a caption must come from a script's
   actual output or from elsewhere in the thesis. If a run fails, say so; do not
   estimate a plausible value.
3. **Do not `\input` a pending file before its images exist** — the build breaks on
   a missing `\includegraphics` target. That is why they are not wired in yet.
4. **§4.6 must be rewritten if and only if step 6 succeeds.** It currently argues the
   qualitative figure was *deliberately excluded*. Adding the figure without editing
   that text makes the thesis contradict itself. Exact replacement in §8.
5. The thesis is RTL Persian. Keep `\lr{}` around Latin runs and use `\fadecimal{}`
   for Persian decimals in body text (but **not** inside `tikzpicture` — use plain
   `$0.93$` there, which renders Persian digits via the document's digit font).

---

## 4. Verify access

```bash
ssh lingotube 'hostname && nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader'
```

If `nvidia-smi` reports it cannot talk to the driver, the kernel module is missing
and GPU work is impossible; report that and skip to C and step 10.

---

## 5. Transfer the two scripts

They are **untracked** in the local repo, so a `git pull` on the server will not
bring them.

```bash
scp tools/make_qualitative_figure.py tools/tensorboard_to_thesis.py \
    lingotube:~/projects/MDE-plus-Intrinsic-matrix/tools/
```

Optionally commit and push them instead, so the next `git pull` on the server does
not clobber them — **ask the user before pushing.**

---

## 6. Discovery — identify checkpoints and runs

```bash
ssh lingotube 'cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate && \
  echo "== GPU ==" && nvidia-smi --query-gpu=memory.free --format=csv,noheader && \
  echo "== RUNS ==" && (test -d runs \
      && python tools/tensorboard_to_thesis.py --runs runs/ --list 2>&1 | head -60 \
      || echo "no runs/ directory") && \
  echo "== CHECKPOINTS ==" && find models -name "best.pth" -printf "%T+  %p\n" | sort -r | head -30'
```

**Selection criterion.** You need the **round-3 pair** (focal jitter, `m=1.6`): two
seed-locked sibling runs that differ only in whether the model receives `K`.

- proposed arm: trained with `--use-camera-intrinsics --model-variant revised`
- control arm: same command with `--model-variant base`

They are normally siblings under one parent directory with `revised` / `base` in the
names, and have adjacent timestamps. If several candidates exist, prefer the pair
whose training command included `--focal-jitter 1.6`; check with:

```bash
ssh lingotube 'cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate && \
  python -c "
import torch,sys
for p in sys.argv[1:]:
    c=torch.load(p,map_location=\"cpu\")
    print(p, c.get(\"config\", \"no config block\"))
" <CANDIDATE_1>/best.pth <CANDIDATE_2>/best.pth'
```

If you cannot identify the pair unambiguously, **ask the user** rather than guessing
— the whole claim rests on the two arms being the seed-locked twins.

---

## 7. Deliverables A and B — the pre-registered qualitative figure

The protocol is enforced in two phases and **the order is the point**: the sample is
fixed from a seed and hashed before any model loads, so it cannot be changed after
the outputs are seen. Do not reorder, and do not hand-edit the manifest.

### 7a. Lock (no GPU, no model, no depth map observed)

```bash
ssh lingotube 'cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate && \
  python tools/make_qualitative_figure.py lock \
      --num-images 6 --seed 20260821 \
      --manifest results/qualitative/manifest.json'
```

**Record the printed SHA-256 digest and the seed** — they go into the figure caption
and are the auditable part of the claim.

### 7b. Render

```bash
ssh lingotube 'cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate && \
  python tools/make_qualitative_figure.py render \
      --manifest results/qualitative/manifest.json \
      --checkpoint proposed=<ROUND3_REVISED>/best.pth \
      --checkpoint control=<ROUND3_BASE>/best.pth \
      --encoder vitl --model-type metric \
      --output-dir results/qualitative'
```

**Mandatory sanity check.** The script prints `uses_camera_intrinsics = True/False`
as each checkpoint loads. It must be `True` for `proposed` and `False` for
`control`. If both print the same value you have the wrong pair — stop, return to §6.

Produces:

| File | Contents |
|---|---|
| `results/qualitative/qualitative_depth_maps.pdf` | 6 rows × (RGB, GT, control, proposed); one shared colormap and depth range per row |
| `results/qualitative/scale_ratio_distribution.pdf` | `median(pred)/median(GT)` histogram, both arms |
| `results/qualitative/qualitative_report.json` | manifest, checkpoints, scale-ratio stats |

The histogram is the more valuable of the two: the thesis currently quotes only two
medians (`1.014` proposed vs `1.081` control); this shows the full distribution and
the control arm regressing to the mean of the `α` distribution, exactly as §3.7.8
predicts.

**Cross-check.** The medians in `qualitative_report.json` should land near `1.014`
and `1.081`. If they differ materially, you likely loaded a different training
round — verify before wiring the figure in, and tell the user.

---

## 8. Wire A and B into the thesis

### 8a. Fill the caption placeholders

In `thesis/figures/pending/qualitative_figures.tex`:

- `NNNNNNNN` → the seed from 7a (`20260821` unless changed)
- `XXXXXXXXXXXX` → first twelve characters of the SHA-256 digest from 7a

### 8b. Copy the PDFs where the LaTeX expects them

```bash
rsync -avz lingotube:~/projects/MDE-plus-Intrinsic-matrix/results/qualitative/ results/qualitative/
cp results/qualitative/qualitative_depth_maps.pdf \
   results/qualitative/scale_ratio_distribution.pdf thesis/figures/
```

### 8c. Rewrite §4.6 — required

In `thesis/tex/chapter4.tex`, section `\label{sec:qualitative_results}`, replace the
**third (last) paragraph**, which currently begins
`برای بازتولید کیفی در کار آینده باید شناسهٔ تصاویر پیش از مشاهده قفل شود`
and ends `...به شواهد کمی و آزمون‌های علّی گزارش‌شده محدود می‌ماند.`

Leave the first two paragraphs untouched — they remain accurate history.

Replacement:

```latex
پروتکل پیش‌ثبت‌شده‌ای که پیش‌تر به‌عنوان کار آینده معرفی شده بود، اکنون اجرا
شده است: شناسهٔ تصاویر نمونه تنها از روی یک بذر تصادفی و \emph{پیش از هرگونه
استنتاج} انتخاب، و چکیدهٔ رمزنگاشتی آن پیش از رندر ثبت شد؛ سپس برای هر نمونه،
تصویر ورودی، عمق مرجع، خروجی \controlmodel{} و خروجی \proposedmodel{} با یک
نگاشت رنگ و بازهٔ عمق یکسان ذخیره گردید. نتیجه در
شکل~\ref{fig:qualitative_locked} آمده است. از آنجا که انتخاب نمونه پیش از
مشاهدهٔ خروجی‌ها قفل شده است، این شکل برخلاف یک نمایش گزینشی، قابل ممیزی است.
افزون بر آن، شکل~\ref{fig:scale_ratio_distribution} توزیع کامل نسبت مقیاس هر
تصویر را نشان می‌دهد و سازوکار عددی برتری گزارش‌شده در
جدول~\ref{tab:nyu_metric_focal} را آشکار می‌سازد.

\input{./figures/pending/qualitative_figures}
```

---

## 9. Deliverable C — training curves

Only if §6 found a `runs/` directory. This turns §3.7.4 / §3.9.8 from prose into a
picture: the collapsed ViT-L run whose **loss curve looks entirely normal** while its
validation metric sits frozen — the thesis's clearest methodological lesson.

```bash
ssh lingotube 'cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate && \
  python tools/tensorboard_to_thesis.py --runs runs/ \
      --run collapsed=<RUN_TRAINED_ON_DEPTH_TARGET> \
      --run fixed=<RUN_TRAINED_ON_DISPARITY_TARGET> \
      --tag <TRAIN_LOSS_TAG> --tag <VAL_METRIC_TAG> \
      --output-dir thesis/figures/curves'
```

Use the exact run names and tag strings printed by §6. Emits one `.csv` and one
`.tex` per tag, already in the thesis palette. Pull them back with `rsync`, write a
caption into each `.tex` (the script leaves `\caption{}` empty deliberately), and
`\input` it in §3.7.4 or §4.4.4. Commit the CSVs — they are small text and let the
figure be rebuilt later without the server.

The collapse signature to look for and describe in the caption: training loss
smooth and decreasing, validation metric flat at ~`0.346` from the first epoch.

---

## 10. Dataset sample strip — no server needed

`thesis/figures/pending/dataset_samples.tex` is written and waiting on four RGB
frames in `thesis/figures/samples/`: `nyu.jpg`, `kitti.jpg`, `cityscapes.jpg`,
`middlebury.jpg`. The user was asked for these and may have supplied them; check.
Source them **from the datasets, not from papers**, and note each licence. Then add
at the end of §4.3.1 (`\subsection{دیتاست‌های مورد استفاده}`, chapter4.tex):

```latex
\input{./figures/pending/dataset_samples}
```

---

## 11. Build and validate — do this regardless of A–C

```bash
cd thesis && ./build.sh 2>&1 | tail -60
```

Needs `latexmk` + `xelatex`. Then:

```bash
python3 tools/check_thesis_numbers.py
```

Also worth re-running the structural check (environments balanced, no duplicate
labels, every `\ref` resolving, table column counts, `\includegraphics` targets
present) — the previous session used an ad-hoc script for this; regenerate it if
useful.

**Known risks in the newly added material**, in likelihood order:

1. **Overfull hboxes** in wide `tabularx` tables — most are already wrapped in
   `\begin{adjustbox}{max width=\textwidth}`; add it where a table overflows.
2. **Float pressure.** ~24 floats were added; chapter 4 alone has 8 figures and 18
   tables. Float parameters were already relaxed in `commands.tex` and 13 strict
   `[h]` placements changed to `[htbp]`. If `Too many unprocessed floats` still
   appears, insert `\clearpage` before the offending section.
3. **pgfplots specifics.** The K-knockout figure uses `\addplot[ybar]` at plot level
   (not axis level) so an `only marks` series can align with the bars — documented
   and intended, but if bars misrender that is the place to look.
4. `\cite` inside a `tikzpicture` node — used once, in the chapter 2 timeline.

Report the build log to the user either way.

---

## 12. Already done — do not redo

- **Ch.1** (was 0 figures / 0 tables): scale-ambiguity figure; thesis roadmap;
  RQ → hypothesis → evidence table.
- **Ch.2** (was 0/0): approach taxonomy; milestone timeline; 14-row prior-work
  comparison; research-gap table; three-supervision-paradigms figure; four-strategies-for-`K`
  figure; new closing section §2.6.
- **Ch.3**: pinhole and architecture diagrams rewritten; new figures for zoom-crop
  augmentation, focal jitter, and conditioning-mechanism comparison; new tables for
  the 9-D descriptor vector, datasets, hyperparameters, losses, and the three silent bugs.
- **Ch.4**: dose–response figure + summary table; effect-size forest plot; real
  log–log K-sweep replacing a hand-drawn sketch; K-knockout figure against its
  geometric prediction; FoV generalisation; evaluation-protocol figure; phase
  diagram; hypothesis-verdict table.
- **Ch.5**: rewritten contribution figure; contributions table; limitations →
  future-work table.
- **Shared system** in `commands.tex`: proposed always blue (`thProposed`), control
  always red (`thControl`), ideal always green (`thIdeal`), published baseline grey
  (`thBase`); `thesisplot` pgfplots style; `thblock`/`thdata`/`tharrow` tikz styles.
- Nine mechanical typo fixes in the original prose.

All figures are vector TikZ/pgfplots built only from numbers already in the thesis.
No published figure was reproduced — original schematics were drawn instead, which
avoids permission and attribution problems.

---

## 13. Definition of done

- [ ] `thesis/build.sh` completes and `thesis/main.pdf` is regenerated
- [ ] `tools/check_thesis_numbers.py` shows no removed or changed result number
- [ ] A and B rendered, wired in, and §4.6 rewritten to match (or explicitly reported
      as blocked, with §4.6 left untouched)
- [ ] C exported and wired in, or reported as unavailable because `runs/` is absent
- [ ] Seed and SHA-256 digest recorded in the figure caption
- [ ] `uses_camera_intrinsics` sanity check passed and stated
- [ ] Build log and `qualitative_report.json` reported to the user
