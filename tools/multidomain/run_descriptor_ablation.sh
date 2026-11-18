#!/usr/bin/env bash
# Camera-descriptor ablation (ViT-S, v1 multi-domain metric recipe = md_v2s/P1, seed 42).
# Same data, schedule and seed for every arm; only the camera-descriptor features change.
# Existing reference arms (not retrained): md2s_P1 (9-D, original run), md2s_C (no K).
#   full9   the reported 9-D descriptor (re-run: run-to-run noise floor)
#   fov2    FoV only (the intrinsics part of Depth-Anything-3's pose encoding)
#   k4      normalised K only (fx/W, fy/H, cx/W, cy/H)
#   focal2  focal only (fx/W, fy/H)
#   noshape 9-D minus aspect ratio and image area (7)
#   nofov   9-D minus the two FoV angles (7)
#   plus11  9-D plus log fx/W, log fy/H (11)
# llm-qwen is stopped for the run and started again on exit. Log: logs/descabl.log
set -u
cd ~/projects/MDE-plus-Intrinsic-matrix && source venv/bin/activate
mkdir -p logs
D=nyu,kitti,cityscapes,diode_outdoor,sunrgbd_kv1,sunrgbd_kv2,sunrgbd_realsense,sunrgbd_xtion
R=results/multidomain
EV="python tools/multidomain/eval_multidomain.py"
FAIL=0
declare -A FS=(
  [full9]=fx,fy,cx,cy,fovh,fovw,aspect,favg,area
  [fov2]=fovh,fovw
  [k4]=fx,fy,cx,cy
  [focal2]=fx,fy
  [noshape]=fx,fy,cx,cy,fovh,fovw,favg
  [nofov]=fx,fy,cx,cy,aspect,favg,area
  [plus11]=fx,fy,cx,cy,fovh,fovw,aspect,favg,area,logfx,logfy
)
ARMS="full9 fov2 k4 focal2 noshape nofov plus11"

{
  echo "descabl $(date)"
  docker stop llm-qwen 2>/dev/null
  trap 'docker start llm-qwen' EXIT
  BASE="--encoder vits --model-type metric --model-variant revised --use-camera-intrinsics \
    --epochs 12 --bs 4 --lr 0.000005 --head-lr-multiplier 10.0 --grad-clip 1.0 --max-depth 250 --min-depth 0.001 \
    --silog-lambda 0.5 --seed 42 --amp --num-workers 4 --freeze-dinov2 --val-per-domain 50 \
    --multidomain nyu:1500,kitti:1500,vkitti:1500,cityscapes:1500 --kitti-rgb jpg \
    --zoom-max 2.0 --focal-jitter 1.6 \
    --pretrained-from models/raw_models/DepthAnythingV2-revised/checkpoints/depth_anything_v2_vits.pth"
  echo "BASE: $BASE"
  for a in $ARMS; do
    SAVE=experiments/descabl/$a
    if ! ls $SAVE/*/best.pth >/dev/null 2>&1; then
      echo "=== TRAIN_START $a (${FS[$a]}) $(date '+%m-%d %H:%M')"
      python train.py $BASE --cam-features ${FS[$a]} --save-path $SAVE > logs/descabl-train-$a.log 2>&1 &
      sleep 20
    fi
  done
  wait
  echo "=== TRAIN_DONE $(date '+%m-%d %H:%M')"

  # reference P1 (original 9-D run): K-sweep points only (its standard eval exists: md2s_P1)
  P1=$(ls experiments/md_v2s/P1/*/best.pth | head -1)
  ev() {  # label out-dir args...
    local label=$1 out=$2; shift 2
    if [ -e "$out/summary.json" ]; then echo "=== SKIP $label"; return; fi
    $EV "$@" --out-dir "$out" > logs/descabl-eval-$label.log 2>&1 || { echo "=== EVAL_FAIL $label"; FAIL=$((FAIL+1)); }
    echo "=== EVAL_DONE $label $(date '+%H:%M')"
  }
  for s in 0.7 1.4; do
    ev P1-k$s $R/descabl_P1_k$s --checkpoint "$P1" --datasets nyu,kitti --max-items 200 --k-scale $s --workers 3 &
  done
  for a in $ARMS; do
    CK=$(ls -t experiments/descabl/$a/*/best.pth 2>/dev/null | head -1)
    if [ -z "$CK" ]; then echo "=== NO_CHECKPOINT $a"; tail -5 logs/descabl-train-$a.log; FAIL=$((FAIL+1)); continue; fi
    (
      ev $a $R/descabl_$a --checkpoint "$CK" --datasets $D --workers 3
      for s in 0.7 1.4; do
        ev $a-k$s $R/descabl_${a}_k$s --checkpoint "$CK" --datasets nyu,kitti --max-items 200 --k-scale $s --workers 3
      done
    ) &
  done
  wait
  echo "EXIT_CODE=$FAIL"
} 2>&1 | tee -a logs/descabl.log
