#!/usr/bin/env bash
# GPU-queue gate for the slot after v2x_all: swap in L7 only if L5 helps.
# L5 helps = the long-truck PointPillars (pp_lumpi_long) beats the full-density one (pp_lumpi) on the 50k-point real
# val, same inference (per-class NMS, flip TTA, no class copies): Truck AP >= +0.02 and mAP >= -0.003.
#   yes -> train L7, the LUMPI CenterPoint-Voxel on the long-truck data (cpv_lumpi_long; batch 2 if batch 4 fails in
#          its first 30 min); if L7 fails in its first 30 min at both batch sizes, train cp_lumpi_d50 so the slot is
#          not lost
#   no  -> train cp_lumpi_d50 as planned (L7 dropped)
#   scripts/gate_l7.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/gate_l7.log
EV=kit_mc/eval_pred_val50k_val.json
TEST="TEST_FRAMES=data/kit/lumpi/detection_test_frames"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

decision=$(python3 - "output/lumpi/pp_lumpi/$EV" "output/lumpi/pp_lumpi_long/$EV" <<'EOF'
import json, sys
try:
    b, l = (json.load(open(p))["detection"]["0.5"] for p in sys.argv[1:3])
except (OSError, KeyError, ValueError) as e:
    print(f"no (eval missing: {e})"); sys.exit()
bt, lt = b["per_class"]["Truck"]["ap40"], l["per_class"]["Truck"]["ap40"]
ok = lt >= bt + 0.02 and l["mAP40"] >= b["mAP40"] - 0.003
print(f"{'yes' if ok else 'no'} Truck {bt:.4f} -> {lt:.4f}, mAP {b['mAP40']:.4f} -> {l['mAP40']:.4f}")
EOF
)
say "L5 helps? $decision"

cp_d50() {
  say "training cp_lumpi_d50"
  env $TEST scripts/run_track.sh lumpi configs/pcdet/cp_lumpi_d50.yaml && scripts/kit_predict.sh lumpi \
      configs/pcdet/cp_lumpi_d50.yaml output/pcdet/cp_lumpi_d50/default/ckpt/checkpoint_epoch_20.pth kit
  rc=$?; say "cp_lumpi_d50 rc=$rc"; return $rc
}

l7() {  # $1 = config name; returns 0 on success, 2 on an early failure (< 30 min), 1 on a late failure
  local t0=$(date +%s)
  say "training $1"
  if env $TEST scripts/run_track.sh lumpi "configs/pcdet/$1.yaml" && scripts/kit_predict.sh lumpi \
      "configs/pcdet/$1.yaml" "output/pcdet/$1/default/ckpt/checkpoint_epoch_15.pth" kit; then
    say "$1 done"; return 0
  fi
  if [ $(( $(date +%s) - t0 )) -lt 1800 ]; then
    say "$1 failed within 30 min: $(grep -m1 -iE 'error|out of memory' "logs/train_$1.log")"; return 2
  fi
  say "$1 failed after 30 min (see logs/train_$1.log)"; return 1
}

case "$decision" in
  yes*)
    l7 cpv_lumpi_long; rc=$?
    [ $rc = 2 ] && { l7 cpv_lumpi_long_b2; rc=$?; }
    [ $rc = 2 ] && { cp_d50; rc=$?; }
    exit $rc ;;
  *)
    cp_d50 ;;
esac
