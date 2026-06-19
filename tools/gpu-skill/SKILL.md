---
name: gpu
description: >
  Offload GPU-intensive model training and evaluation to the lingotube server (RTX 4090).
  Use this skill whenever the user wants to train a model, run an evaluation, run compare_models,
  run compare_dataset_results, or do any GPU-heavy inference — even if they just say "train",
  "run on GPU", "evaluate the model", "test on cityscapes", or ask to check results from a
  previous run. This skill handles the full cycle: commit & push local changes, SSH to lingotube,
  pull, launch training in tmux, monitor progress, retrieve logs and output artifacts, analyze
  results, and send a desktop notification when done. Always use this skill instead of running
  GPU workloads locally.
---

# GPU Training on lingotube

This skill manages the full lifecycle of GPU training and evaluation on the remote **lingotube**
server (RTX 4090) via SSH. Use it any time you need to run `train.py`, `compare_models.py`,
`compare_dataset_results.py`, or any GPU-heavy script.

## Server facts

| Item | Value |
|------|-------|
| SSH alias | `lingotube` (configured in `~/.ssh/config`, key auth, no password) |
| Remote repo | `/home/ubuntu/projects/MDE-plus-Intrinsic-matrix` |
| tmux session | `lab` (already running; attach to window 0 or create a new named window) |
| GPU | NVIDIA RTX 4090 |

---

## Workflow overview

1. **Understand the task** — derive the exact command to run from the conversation history
2. **Commit & push** local changes so the server gets the latest code
3. **SSH: pull** the latest commit on the server
4. **Launch** the training/eval script in the tmux session in the background
5. **Monitor** every few minutes until the process completes
6. **Retrieve** logs, metrics, and output artifacts (copy to local machine if small enough)
7. **Analyze** results and advise on next steps
8. **Notify** the user via a desktop notification

---

## Step 1 — Understand the task

Before touching git, figure out exactly which command to run:

- Read the conversation history to extract the dataset, encoder, flags, checkpoint path, etc.
- If the user said "train da2-revised on cityscapes with vitl", construct the full `python train.py …` invocation.
- If the user said "compare da2 vs da2-revised on cityscapes", construct the full `python compare_models.py …` invocation.
- If anything is ambiguous (e.g., which checkpoint to use), ask the user before proceeding.
- Write out the full command you plan to run and confirm it looks right.

Refer to `CLAUDE.md` in the project root for the canonical flag reference.

---

## Step 2 — Commit and push local changes

```bash
# Check for uncommitted changes
git -C /Users/amin/Desktop/Uni/Arshad/Project/Projects\ Source\ Code/MDE\ plus\ Intrinsic\ matrix status --short

# Stage all tracked changes (never add .env or credentials)
git -C /Users/amin/Desktop/Uni/Arshad/Project/Projects\ Source\ Code/MDE\ plus\ Intrinsic\ matrix add -u

# Commit if there are staged changes
git -C /Users/amin/Desktop/Uni/Arshad/Project/Projects\ Source\ Code/MDE\ plus\ Intrinsic\ matrix \
    commit -m "wip: sync before lingotube training run"

# Push
git -C /Users/amin/Desktop/Uni/Arshad/Project/Projects\ Source\ Code/MDE\ plus\ Intrinsic\ matrix \
    push origin main
```

If there are no changes, skip the commit step but still push to make sure remote is up to date.

---

## Step 3 — Pull on the server

```bash
ssh lingotube "cd /home/ubuntu/projects/MDE-plus-Intrinsic-matrix && git pull --rebase"
```

Check the output — if there are merge conflicts, resolve them locally and re-push before continuing.

---

## Step 4 — Launch the job in tmux

Create a **new tmux window** named after the job so multiple runs don't collide:

```bash
# Create a timestamp-named window (e.g. train-20260619-1430)
JOB_NAME="train-$(date +%Y%m%d-%H%M)"

ssh lingotube "tmux new-window -t lab -n $JOB_NAME"

# Run the job inside that window, with stdout+stderr tee'd to a log file
LOG_FILE="/home/ubuntu/projects/MDE-plus-Intrinsic-matrix/logs/${JOB_NAME}.log"
CMD="cd /home/ubuntu/projects/MDE-plus-Intrinsic-matrix && mkdir -p logs && <YOUR_COMMAND> 2>&1 | tee ${LOG_FILE}"
ssh lingotube "tmux send-keys -t lab:${JOB_NAME} '${CMD}' Enter"
```

Replace `<YOUR_COMMAND>` with the full Python command derived in Step 1.

Store `JOB_NAME` and `LOG_FILE` for use in later steps.

**Important**: always tee to a log file. This lets you poll the log without attaching to tmux interactively.

---

## Step 5 — Monitor progress

Poll the log file every **5 minutes** until the job ends. A job is finished when:
- The log contains `Training complete` or `Best model saved` (training)
- The log contains a final metrics table or `Evaluation complete` (evaluation)
- The Python process exits (check with `pgrep` or the tmux window exits)

```bash
# Poll: tail the last 40 lines of the log
ssh lingotube "tail -40 /home/ubuntu/projects/MDE-plus-Intrinsic-matrix/logs/${JOB_NAME}.log"

# Check if the process is still running
ssh lingotube "pgrep -f 'python train.py' | wc -l"
# Returns 0 if finished
```

Tell the user the current status each time you poll: current epoch / step, latest loss values, ETA if visible.

For very long runs (>2 hours), use `ScheduleWakeup` with `delaySeconds=270` between polls so you stay cache-warm without burning tokens.

---

## Step 6 — Retrieve results

Once the job finishes, collect:

### Training runs

```bash
# Copy the checkpoints directory (may be large — skip if >2GB total)
REMOTE_CKPT="/home/ubuntu/projects/MDE-plus-Intrinsic-matrix/<save-path>"
rsync -avz --progress lingotube:${REMOTE_CKPT}/ <local-save-path>/

# Copy the log file
scp lingotube:/home/ubuntu/projects/MDE-plus-Intrinsic-matrix/logs/${JOB_NAME}.log ./logs/

# Copy tensorboard event files if they exist
REMOTE_TB="/home/ubuntu/projects/MDE-plus-Intrinsic-matrix/runs/"
rsync -avz lingotube:${REMOTE_TB} ./runs/ 2>/dev/null || true
```

### Evaluation runs

```bash
# Copy the output directory (numpy arrays, PNG visualizations, metrics JSON)
REMOTE_OUT="/home/ubuntu/projects/MDE-plus-Intrinsic-matrix/output/"
rsync -avz lingotube:${REMOTE_OUT} ./output/
```

### Generate visual summaries

If output images are available, read a few representative ones and describe what you see:

```bash
# List output PNGs (take at most 5 for visual inspection)
ssh lingotube "find /home/ubuntu/projects/MDE-plus-Intrinsic-matrix/output -name '*.png' | head -5"
```

If tensorboard logs were retrieved, start tensorboard locally and tell the user the port:

```bash
tensorboard --logdir ./runs --port 6006 &
echo "TensorBoard running at http://localhost:6006"
```

---

## Step 7 — Analyze results

After collecting the data:

1. **Parse the log** — extract final metrics (AbsRel, RMSE, SILog, δ1, δ2, δ3) and training curves (loss per epoch).
2. **Read any `metrics.json` files** from the output directory and summarize them.
3. **Compare with prior runs** if available in `./output/` or the conversation history.
4. **Generate a concise report** with:
   - Final metric values
   - Whether the run improved over baseline
   - Notable observations (divergence, plateau, unexpected errors in the log)
   - Concrete suggestions for the next experiment (learning rate, freeze settings, dataset mix, etc.)

---

## Step 8 — Send a desktop notification

When the job completes (or errors out), send a macOS notification:

```bash
# Success
osascript -e 'display notification "Training finished on lingotube. Check Claude for results." with title "GPU Job Done" sound name "Glass"'

# On error
osascript -e 'display notification "Training FAILED on lingotube. Check Claude for error log." with title "GPU Job Error" sound name "Basso"'
```

---

## Error handling

| Situation | Action |
|-----------|--------|
| SSH connection refused | Retry once after 30s; if still failing, tell the user |
| git push rejected (diverged) | Pull and rebase locally, then re-push |
| tmux session `lab` not found | Create it: `ssh lingotube "tmux new-session -d -s lab"` |
| OOM / CUDA error in log | Report immediately, suggest reducing `--bs` or enabling `--accumulate-grad` |
| Process exits with non-zero code | Copy the last 100 lines of the log and include in analysis |
| Checkpoint file >2GB | Skip rsync; tell user to retrieve manually or inspect on server |

---

## Logging

Keep a local session log for each job at `./logs/<JOB_NAME>-session.md`:

```markdown
# Job: <JOB_NAME>
- Command: `<full command>`
- Started: <timestamp>
- Finished: <timestamp>
- Final metrics: <key numbers>
- Artifacts retrieved: <list>
- Analysis: <summary>
```

---

## Quick reference: common commands

```bash
# Training — intrinsics model on VKITTI
python train.py --encoder vitl --dataset vkitti --max-depth 80.0 \
    --epochs 40 --bs 4 --lr 0.000005 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vitl.pth \
    --save-path models/raw_models/DepthAnythingV2-revised/checkpoints/revised \
    --use-camera-intrinsics --freeze-dinov2 \
    --head-lr-multiplier 10.0 --grad-clip 1.0 --accumulate-grad 4

# Training — baseline (no intrinsics)
python train.py --encoder vitl --dataset vkitti --max-depth 80.0 \
    --epochs 40 \
    --pretrained-from models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth \
    --save-path models/raw_models/DepthAnythingV2/checkpoints/baseline \
    --freeze-dinov2

# Evaluate: compare two models
python compare_models.py --dataset cityscapes \
    --model1 da2 --model2 da2-revised --encoder vitl --max-items 2000

# Cross-dataset evaluation
python compare_dataset_results.py \
    --dataset cityscapes,drivingstereo --model-name da2-revised \
    --encoder vitl --model-type metric --max-depth 120 --max-items 2000
```
