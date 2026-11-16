#!/usr/bin/env bash
# Relative-depth (basic, scale-and-shift-invariant loss on inverse depth) version of
# the four-dataset multi-domain pair: does the camera-aware model beat its matched
# control under the relative protocol once the training cameras really differ?
#
#   PR  camera-aware (K fed), zoom-crop up to 2x
#   CR  control: same data, augmentation, seed and batches, no K
# No focal jitter: scaling depth leaves the scale-invariant loss unchanged.
# Both arms and the released relative DA2 ViT-L are then evaluated on the ten test
# sets with per-image scale+shift alignment in inverse depth (standard and centre-zoom
# 1.5), and PR once more with K withheld.
#
# Run on lingotube from the repo root inside tmux; the two trainings run one after the
# other so they fit beside another job. Log: logs/mdrel.log, last line EXIT_CODE=<n>.
set -u
cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate
mkdir -p logs
LOG=logs/mdrel.log
D=nyu,kitti,cityscapes,diode_indoor,diode_outdoor,middlebury,sunrgbd_kv1,sunrgbd_kv2,sunrgbd_realsense,sunrgbd_xtion
CK_DIR=models/raw_models/DepthAnythingV2-revised/checkpoints
R=results/multidomain
EV="python tools/multidomain/eval_multidomain.py"
FAIL=0

run() {  # an evaluation whose summary.json already exists is skipped (safe to re-launch)
  local label=$1; shift
  local out=${*: -1}
  if [ "$1" = python ] && [ "$2" = tools/multidomain/eval_multidomain.py ] && [ -e "$out/summary.json" ]; then
    echo "=== STEP_SKIP $label (done)"; return 0
  fi
  echo "=== STEP $label: $(date '+%m-%d %H:%M')"
  "$@"
  local rc=$?
  echo "=== STEP_DONE $label rc=$rc"
  [ $rc -eq 0 ] || FAIL=$((FAIL + 1))
  return $rc
}

{
  echo "mdrel   $(date)"
  BASE="--encoder vitl --model-type basic --epochs 12 --bs 4 --lr 0.000005 --head-lr-multiplier 10.0 --grad-clip 1.0 \
    --max-depth 80 --min-depth 0.001 --seed 42 --amp --freeze-dinov2 --val-per-domain 50 \
    --multidomain nyu:1500,kitti:1500,vkitti:1500,cityscapes:1500 --kitti-rgb png --zoom-max 2.0 \
    --num-workers 8 --pretrained-from $CK_DIR/depth_anything_v2_vitl.pth"
  for z in 1.0 1.5; do
    run "base-zoom$z" $EV --released basic --datasets $D --center-zoom $z --out-dir $R/mdrel_released_zoom$z
  done
  for arm in PR CR; do
    if [ $arm = PR ]; then VAR="--model-variant revised --use-camera-intrinsics"; else VAR="--model-variant base"; fi
    SAVE=experiments/md_rel/$arm
    if ! ls $SAVE/*/best.pth >/dev/null 2>&1; then
      run "train-$arm" python train.py $BASE $VAR --save-path $SAVE || continue
    fi
    CK=$(ls -t $SAVE/*/best.pth | head -1)
    for z in 1.0 1.5; do
      run "$arm-zoom$z" $EV --checkpoint "$CK" --datasets $D --center-zoom $z --out-dir $R/mdrel_${arm}_zoom$z
    done
    if [ $arm = PR ]; then
      run "PR-noK" $EV --checkpoint "$CK" --datasets $D --withhold-k --out-dir $R/mdrel_PR_noK
    fi
  done
  echo "FAILED_STEPS=$FAIL"
  echo "EXIT_CODE=$FAIL"
} 2>&1 | tee -a $LOG
