---
name: gpu
description: >
  Run scripts and GPU work on the remote lingotube server (RTX 4090, 24GB VRAM) over SSH.
  Use this skill for ANYTHING involving the server or GPU: training a model, running any
  evaluation or inference, running compare_models / compare_dataset_results / train.py or
  any other Python script that needs a GPU, executing any command or script remotely,
  copying code/files/checkpoints/datasets to or from the server (scp/rsync), checking on a
  running or finished remote job, retrieving logs or results, or checking server status and
  capacity (GPU VRAM, RAM, running processes, whether jobs can run simultaneously). Trigger
  it even on vague phrasing like "train", "run this", "test the model", "run on GPU",
  "is the GPU free?", "what's running on the server?", "copy this over", "check the
  results", "how's the job going?", or "/gpu". The local machine has NO usable GPU — any
  script that imports torch/CUDA or touches model weights must run on lingotube via this
  skill, never locally. When in doubt whether a task belongs on the server, use this skill
  to check server status first.
---

# GPU jobs on lingotube

Manages the full lifecycle of remote GPU work on **lingotube** via SSH: sync code, check
capacity, launch jobs in tmux, poll until done, retrieve and analyze results.

## Server facts

| Item | Value |
|------|-------|
| SSH alias | `lingotube` (in `~/.ssh/config`: 93.118.171.194, user `ubuntu`, key auth, no password) |
| GPU | NVIDIA RTX 4090, 24 GB VRAM |
| RAM | 62 GB |
| Remote repo | `/home/ubuntu/projects/MDE-plus-Intrinsic-matrix` |
| Python env | `source /home/ubuntu/projects/MDE-plus-Intrinsic-matrix/venv/bin/activate` — system python3 has no torch; **always activate the venv** |
| tmux session | `lab` (create if missing: `tmux new-session -d -s lab`) |
| Local repo | `/home/mohamadamin.ghasemzade/MDE-plus-Intrinsic-matrix` |

All `ssh`/`scp`/`rsync` commands should use the plain alias, e.g. `ssh lingotube "..."`.

---

## Step 0 — Check server capacity (always do this first)

Before launching anything, check whether the GPU and RAM have room — this decides whether a
new job can run now, run **concurrently** with an existing one, or must wait:

```bash
ssh lingotube 'nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader; free -g | sed -n 2p; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

Decision guide (RTX 4090 = 24 576 MiB total):

- **GPU idle** (no compute apps, <1 GB used) → launch freely.
- **Job already running** → estimate the new job's VRAM need and launch concurrently only if
  it fits in the *free* VRAM with ~2 GB headroom. Rough needs for this project:
  vitl training with bs 4 ≈ 20+ GB (never concurrent); vitl inference/eval ≈ 6–9 GB;
  vits/vitb eval ≈ 2–5 GB. Two evals can usually coexist; training + anything usually cannot.
- **Not enough room** → tell the user what's running and either queue (wait for the current
  job by polling, then launch) or ask which job has priority.
- Also check RAM: if available RAM < ~8 GB, don't add another dataloader-heavy job.

**Known issue**: if `nvidia-smi` reports "couldn't communicate with the NVIDIA driver", the
kernel module isn't loaded (this happens after server kernel updates — the DKMS module for
the new kernel is missing). GPU work is impossible until fixed. Tell the user to run
interactively (needs sudo password):
`! ssh -t lingotube 'sudo apt install -y nvidia-dkms-595 && sudo dkms autoinstall && sudo modprobe nvidia && nvidia-smi'`
(check the installed driver series first with `dpkg -l | grep nvidia-driver` — it was 595 as of 2026-07)
(or reboot into a kernel that has the module).

---

## Step 1 — Understand the task

Derive the exact command from the conversation before touching the server:

- Extract dataset, encoder, flags, checkpoint paths, etc. `CLAUDE.md` in the project root is
  the canonical flag reference.
- If anything is ambiguous (e.g., which checkpoint), ask the user before proceeding.
- State the full command you plan to run so the user can catch mistakes.

---

## Step 2 — Sync code to the server

Two ways; prefer **git** for real runs (reproducible, keeps histories aligned), use
**scp/rsync** for quick experiments or files that shouldn't be committed.

### Git sync (default)

```bash
cd /home/mohamadamin.ghasemzade/MDE-plus-Intrinsic-matrix
git status --short
git add -u        # never add .env, credentials, or large data files
git commit -m "wip: sync before lingotube run"   # skip if nothing staged
git push origin main
ssh lingotube "cd ~/projects/MDE-plus-Intrinsic-matrix && git pull --rebase"
```

If the pull hits conflicts (the server may have local edits), report them; resolve on the
server or `git stash` there — never force-reset the server without asking.

### Direct copy (uncommitted/quick changes)

```bash
# Single file
scp /home/mohamadamin.ghasemzade/MDE-plus-Intrinsic-matrix/train.py lingotube:~/projects/MDE-plus-Intrinsic-matrix/train.py

# A directory, excluding junk
rsync -avz --exclude '__pycache__' --exclude '.git' \
    /home/mohamadamin.ghasemzade/MDE-plus-Intrinsic-matrix/models/ \
    lingotube:~/projects/MDE-plus-Intrinsic-matrix/models/
```

Warn the user that direct-copied changes will be clobbered by the next `git pull` on the
server unless committed.

---

## Step 3 — Launch the job in tmux

Run inside tmux so the job survives SSH disconnects; tee to a log so it can be polled
without attaching. Use a unique window name per job so concurrent runs don't collide.

```bash
JOB_NAME="train-$(date +%Y%m%d-%H%M)"     # or eval-..., compare-...
LOG_FILE="~/projects/MDE-plus-Intrinsic-matrix/logs/${JOB_NAME}.log"

ssh lingotube "tmux has-session -t lab 2>/dev/null || tmux new-session -d -s lab; tmux new-window -t lab -n $JOB_NAME"
ssh lingotube "tmux send-keys -t lab:$JOB_NAME 'cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate && mkdir -p logs && <YOUR_COMMAND> 2>&1 | tee $LOG_FILE; echo EXIT_CODE=\$?' Enter"
```

Remember `JOB_NAME` and `LOG_FILE` for monitoring. The trailing `echo EXIT_CODE=$?` makes
completion and success/failure detectable from the log alone.

---

## Step 4 — Monitor until done

Poll the log periodically. The job is finished when the log contains `EXIT_CODE=` (0 =
success). Between polls, report progress to the user: current epoch/step, latest loss or
metric values, errors.

```bash
ssh lingotube "tail -40 $LOG_FILE"
ssh lingotube "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"   # still running?
```

Polling cadence:
- Short jobs (eval, <30 min): check every ~4 minutes with `ScheduleWakeup` (`delaySeconds=240`, stays cache-warm).
- Long training runs (hours): poll every 20–30 min (`delaySeconds=1200–1800`); the per-epoch
  log lines change slowly, so tighter polling is wasted.
- On OOM/CUDA errors in the log: stop polling, report immediately, suggest lower `--bs` or
  `--accumulate-grad`.

If the user just asked to *launch* a long run, offer the choice: keep monitoring in this
session, or stop here and let them ask "check the gpu job" later (the log file persists).

### Shell & tmux hygiene (do this every job)

- **One self-terminating poller per job.** To wait for completion, run a single
  background shell whose loop *exits* as soon as the job is done, e.g.
  `until ssh lingotube 'grep -qE "EXIT_CODE=|Traceback" <LOG>'; do sleep 300; done`
  (Bash `run_in_background: true`). Don't stack several pollers on one job and
  don't use unbounded `tail -f`-style watchers that outlive the job.
- **The completion marker must actually reach the log.** In
  `<CMD> | tee $LOG; echo EXIT_CODE=$?` the echo goes to the tmux pane, NOT the
  log — a poller grepping the log for it will spin forever. Either put the
  marker inside the piped command (`{ <CMD>; echo EXIT_CODE=$?; } 2>&1 | tee $LOG`)
  or append it explicitly (`echo EXIT_CODE=$? | tee -a $LOG`). Before arming a
  poller, double-check its pattern is something the log will really contain.
- **One tmux window per job, killed when done.** After results/logs are
  retrieved, remove the job's window:
  `ssh lingotube "tmux kill-window -t lab:$JOB_NAME"`.
  If stale windows have accumulated (crashed/old sessions), list them with
  `ssh lingotube 'tmux list-windows -t lab -F "#{window_index} #{window_name}"'`
  and kill every window that has no live job (keep window 0).

---

## Step 5 — Retrieve results

```bash
# Log file
mkdir -p logs && scp lingotube:$LOG_FILE ./logs/

# Training checkpoints — check size first, skip if >2 GB and ask the user
ssh lingotube "du -sh ~/projects/MDE-plus-Intrinsic-matrix/<save-path>"
rsync -avz lingotube:~/projects/MDE-plus-Intrinsic-matrix/<save-path>/ ./<save-path>/

# Evaluation outputs (metrics.json, arrays.npz, PNGs)
rsync -avz lingotube:~/projects/MDE-plus-Intrinsic-matrix/output/ ./output/

# Tensorboard events, if any
rsync -avz lingotube:~/projects/MDE-plus-Intrinsic-matrix/runs/ ./runs/ 2>/dev/null || true
```

---

## Step 6 — Analyze and report

1. Parse the log for final metrics (AbsRel, RMSE, SILog, δ1/δ2/δ3) and loss curves.
2. Read retrieved `metrics.json` files and summarize.
3. Compare against baseline/prior runs when available; consider invoking the
   `mde-results-evaluator` agent for comparison runs.
4. Report: final numbers, improved-or-not vs baseline, anomalies (divergence, plateau,
   errors), and a concrete suggestion for the next experiment.

---

## Error handling

| Situation | Action |
|-----------|--------|
| SSH connection refused/timeout | Retry once after 30 s; then tell the user (server may be down or IP changed — check `~/.ssh/config`) |
| `Host key verification failed` | Server IP/keys changed; rerun with `-o StrictHostKeyChecking=accept-new` after confirming the IP with the user |
| `nvidia-smi` can't reach driver | See Known issue in Step 0 — needs sudo fix, GPU unusable until then |
| `No module named torch` | venv wasn't activated — every remote python command needs `source venv/bin/activate` |
| git push rejected | Pull/rebase locally, re-push |
| tmux session `lab` missing | `ssh lingotube "tmux new-session -d -s lab"` |
| OOM / CUDA error | Report; suggest smaller `--bs` + `--accumulate-grad`, or wait for the other job to finish |
| Non-zero EXIT_CODE | Pull last 100 log lines, include in the report |
| Checkpoint >2 GB | Don't rsync automatically; ask the user |

---

## Quick reference: common project commands

```bash
# Training — intrinsics model on VKITTI
python train.py --encoder vitl --datasets vkitti --max-depth 80.0 \
    --epochs 40 --bs 4 --lr 0.000005 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
    --save-path models/raw_models/DepthAnythingV2-revised/checkpoints/revised \
    --use-camera-intrinsics --freeze-dinov2 \
    --head-lr-multiplier 10.0 --grad-clip 1.0 --accumulate-grad 4

# Training — baseline (no intrinsics)
python train.py --encoder vitl --datasets vkitti --max-depth 80.0 \
    --epochs 40 \
    --pretrained-from models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth \
    --save-path models/raw_models/DepthAnythingV2/checkpoints/baseline \
    --freeze-dinov2

# Compare two models (--max-depth >20 selects the vkitti/outdoor checkpoint)
python compare_models.py --dataset cityscapes \
    --model1 da2 --model2 da2-revised --encoder vitl --max-depth 80.0 --max-items 2000

# Cross-dataset evaluation
python compare_dataset_results.py \
    --dataset cityscapes,drivingstereo --model-name da2-revised \
    --encoder vitl --model-type metric --max-depth 120 --max-items 2000
```
