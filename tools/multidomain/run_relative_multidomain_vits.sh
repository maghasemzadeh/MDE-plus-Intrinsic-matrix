#!/usr/bin/env bash
# ViT-S version of run_relative_multidomain.sh (same data, recipe, seed and test sets as
# the ViT-L relative pair) plus the released ViT-S baselines that the ViT-S tables need:
#   PR / CR                  relative (basic) camera-aware model and its matched control
#   mdrels_released_*        released relative DA2 ViT-S, relative protocol
#   released_{hypersim,vkitti}_vits   released metric DA2 ViT-S, absolute protocol
# (the absolute ViT-S pair already exists: experiments/md_v2s/{P1,C}, results md2s_{P1,C}).
# The two trainings run at the same time. The llm-qwen container is stopped for the run
# and started again on exit. Log: logs/mdrels.log, last line EXIT_CODE=<n>.
set -u
cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate
mkdir -p logs
D=nyu,kitti,cityscapes,diode_indoor,diode_outdoor,middlebury,sunrgbd_kv1,sunrgbd_kv2,sunrgbd_realsense,sunrgbd_xtion
CK_DIR=models/raw_models/DepthAnythingV2-revised/checkpoints
REL_DIR=models/raw_models/DepthAnythingV2/checkpoints
R=results/multidomain
EV="python tools/multidomain/eval_multidomain.py"
FAIL=0

run() {
  local label=$1; shift
  local out=${*: -1}
  if [ -e "$out/summary.json" ]; then echo "=== STEP_SKIP $label (done)"; return 0; fi
  echo "=== STEP $label: $(date '+%m-%d %H:%M')"
  "$@"
  local rc=$?
  echo "=== STEP_DONE $label rc=$rc"
  [ $rc -eq 0 ] || FAIL=$((FAIL + 1))
  return $rc
}

{
  echo "mdrels  $(date)"
  VK=$REL_DIR/depth_anything_v2_metric_vkitti_vits.pth
  [ -s $VK ] || curl -sL -o $VK https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth
  ls -l $VK
  docker stop llm-qwen
  trap 'docker start llm-qwen' EXIT

  BASE="--encoder vits --model-type basic --epochs 12 --bs 4 --lr 0.000005 --head-lr-multiplier 10.0 --grad-clip 1.0 \
    --max-depth 80 --min-depth 0.001 --seed 42 --amp --freeze-dinov2 --val-per-domain 50 \
    --multidomain nyu:1500,kitti:1500,vkitti:1500,cityscapes:1500 --kitti-rgb png --zoom-max 2.0 \
    --num-workers 8 --pretrained-from $CK_DIR/depth_anything_v2_vits.pth"
  for arm in PR CR; do
    if [ $arm = PR ]; then VAR="--model-variant revised --use-camera-intrinsics"; else VAR="--model-variant base"; fi
    SAVE=experiments/md_rel_s/$arm
    if ! ls $SAVE/*/best.pth >/dev/null 2>&1; then
      echo "=== TRAIN_START $arm $(date '+%m-%d %H:%M')"
      python train.py $BASE $VAR --save-path $SAVE > logs/mdrels-train-$arm.log 2>&1 &
    fi
  done

  # released ViT-S baselines while the two trainings run
  for z in 1.0 1.5; do
    run "released-basic-zoom$z" $EV --released basic --encoder vits --datasets $D --center-zoom $z --out-dir $R/mdrels_released_zoom$z
  done
  for m in hypersim vkitti; do
    run "released-$m" $EV --released $m --encoder vits --datasets $D --out-dir $R/released_${m}_vits
  done

  wait
  echo "=== TRAIN_DONE $(date '+%m-%d %H:%M')"
  for arm in PR CR; do
    tail -n 3 logs/mdrels-train-$arm.log | tr '\r' '\n' | tail -n 3
    CK=$(ls -t experiments/md_rel_s/$arm/*/best.pth 2>/dev/null | head -1)
    if [ -z "$CK" ]; then echo "=== NO_CHECKPOINT $arm"; FAIL=$((FAIL + 1)); continue; fi
    for z in 1.0 1.5; do
      run "$arm-zoom$z" $EV --checkpoint "$CK" --datasets $D --center-zoom $z --out-dir $R/mdrels_${arm}_zoom$z
    done
    if [ $arm = PR ]; then
      run "PR-noK" $EV --checkpoint "$CK" --datasets $D --withhold-k --out-dir $R/mdrels_PR_noK
    fi
  done
  echo "EXIT_CODE=$FAIL"
} 2>&1 | tee -a logs/mdrels.log
