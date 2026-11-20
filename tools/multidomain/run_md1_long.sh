#!/usr/bin/env bash
# Convergence check of the final pair: the md_v1 recipe (md1_P / md1_C, seed 42)
# trained for EPOCHS epochs instead of 12; the polynomial lr schedule stretches with it.
# Then evaluates both selected checkpoints on the eight thesis test sets.
#
# Run on lingotube from the repo root inside tmux; logs to logs/md1-long$EPOCHS.log,
# last line EXIT_CODE=<n>. Stops the llm-qwen container for the GPU and always restarts it.
# Usage: EPOCHS=40 tools/multidomain/run_md1_long.sh
set -u
cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate
EPOCHS=${EPOCHS:-40}
mkdir -p logs
LOG=logs/md1-long$EPOCHS.log
D=nyu,kitti,cityscapes,diode_outdoor,sunrgbd_kv1,sunrgbd_kv2,sunrgbd_realsense,sunrgbd_xtion
CK_DIR=models/raw_models/DepthAnythingV2-revised/checkpoints
R=results/multidomain
EV="python tools/multidomain/eval_multidomain.py"
FAIL=0

restart_llm() { docker start llm-qwen >/dev/null 2>&1 && echo "llm-qwen restarted"; }
trap restart_llm EXIT

run() {
  local label=$1; shift
  echo "=== STEP $label: $(date '+%m-%d %H:%M')"
  "$@"
  local rc=$?
  echo "=== STEP_DONE $label rc=$rc"
  [ $rc -eq 0 ] || FAIL=$((FAIL + 1))
  return $rc
}

{
  echo "md1-long EPOCHS=$EPOCHS   commit: $(git rev-parse --short HEAD) (+ uncommitted working tree)   $(date)"
  docker stop llm-qwen >/dev/null 2>&1 && echo "llm-qwen stopped"
  BASE="--encoder vitl --model-type metric --epochs $EPOCHS --bs 4 --lr 0.000005 --head-lr-multiplier 10.0 --grad-clip 1.0 \
    --max-depth 250 --min-depth 0.001 --silog-lambda 0.5 --seed 42 --amp --freeze-dinov2 --val-per-domain 50 \
    --multidomain nyu:1500,kitti:1500,vkitti:1500,cityscapes:1500 --kitti-rgb jpg --zoom-max 2.0 --focal-jitter 1.6 \
    --num-workers 8 --pretrained-from $CK_DIR/depth_anything_v2_vitl.pth"
  for arm in P C; do
    if [ $arm = P ]; then VAR="--model-variant revised --use-camera-intrinsics"; else VAR="--model-variant base"; fi
    SAVE=experiments/md_long$EPOCHS/$arm
    OUT=md1L${EPOCHS}_$arm
    if ! ls $SAVE/*/best.pth >/dev/null 2>&1; then
      run "train-$OUT" python train.py $BASE $VAR --save-path $SAVE || continue
    fi
    CK=$(ls -t $SAVE/*/best.pth | head -1)
    run "$OUT-eval" $EV --checkpoint "$CK" --datasets $D --out-dir $R/$OUT
    rm -f "$(dirname "$CK")/latest.pth"
  done
  echo "FAILED_STEPS=$FAIL"
  echo "EXIT_CODE=$FAIL"
} 2>&1 | tee -a $LOG
