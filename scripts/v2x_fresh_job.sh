#!/usr/bin/env bash
# V2X fresh set: inference only, scoring only, on data/pcdet/v2x_fresh_50k (328 frames from the
# V2X-Real train split, scripts/v2x_fresh_build.py). Its own GPU slot.
# 1. Each of the 7 members re-predicts 10 OS1-64 val frames with the same settings; they must equal the saved val predictions, else stop.
# 2. Members on fresh, same settings and 4-view TTA as pred_val50k (kit_predict.sh).
# 3. CPU: second-stage inputs x7 / v18 / x6bf on fresh (scripts/v2x_fresh_inputs.py; first rebuilds the val_i1 inputs exactly).
# 4. Second stages the shelf files need: X6 on x7, X6b on v18 and x6bf, negatives on x6bf. Plus, only for a candidate that passed val:
#    step 2's (logs/v2x_step2_choice.json, cand true), phase 1's PV-RCNN++ (logs/v2x_phase1_choice.json, cand true),
#    Part-A2's (output/v2x/ens/parta2_v2x_test.json, written only when it passes val).
# 5. CPU: scripts/v2x_fresh_read.py > logs/v2x_fresh_read.log.
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/docker_run.sh
F=data/pcdet/v2x_fresh_50k
MC="--multi_class_nms --max_dets 100"
MEM=("pp_v2x kit_mc 30 $MC" "cp_v2x kit 30" "pp_v2x_ped kit_mc 30 $MC" "pp_v2x_all kit_mc 30 $MC" "pp_v2x_walk kit_mc 30 $MC" "vrcnn_v2x kit 20" "vrcnn_v2x_b kit 20")

pred() {  # name tag epoch [infer args...] frames ids out
  local n=$1 t=$2 e=$3
  shift 3
  local out=${!#} ids=${@: -2:1} fr=${@: -3:1}
  local args=("${@:1:$#-3}")
  $R python -m utl.infer --cfg configs/pcdet/$n.yaml --ckpt output/pcdet/$n/default/ckpt/checkpoint_epoch_$e.pth \
      --frames "$fr" --ids "$ids" --out "$out" --tta "${args[@]}"
}

ref() {  # model epoch input out [x6_refine args...]
  local m=$1 e=$2 in=$3 out=$4
  shift 4
  $R python scripts/x6_refine.py --cfg configs/pcdet/$m.yaml --ckpt output/pcdet/$m/default/ckpt/checkpoint_epoch_$e.pth \
      --frames $F/points --pred output/v2x/ens/${in}_fresh.json --out "$out" "$@"
}

cand() {  # a choice file whose candidate gate passed on val
  [ -f "$1" ] && python3 - "$1" <<'EOF'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))["cand"] else 1)
EOF
}

DIRS=()
for m in "${MEM[@]}"; do
  set -- $m
  pred "$@" data/pcdet/v2x_real_50k/points data/pcdet/v2x_real_50k/ImageSets/repro10.txt output/v2x/$1/$2/pred_repro10.json
  DIRS+=("output/v2x/$1/$2")
done
UT_NOGPU=1 $R python scripts/v2x_fresh_repro.py "${DIRS[@]}"
for m in "${MEM[@]}"; do
  set -- $m
  pred "$@" $F/points $F/ImageSets/val.txt output/v2x/$1/$2/pred_fresh50k.json
done
UT_NOGPU=1 OMP_NUM_THREADS=6 $R python scripts/v2x_fresh_inputs.py

ref vrcnn_v2x 20 x7 output/v2x/vrcnn_v2x/x6c/x7_fresh_ref.json
ref vrcnn_v2x_b 20 v18 output/v2x/vrcnn_v2x_b/x6bc/v18_fresh_ref.json
ref vrcnn_v2x_b 20 x6bf output/v2x/vrcnn_v2x_b/x6bc/x6bf_fresh_ref.json
ref vrcnn_v2x_neg 20 x6bf output/v2x/vrcnn_v2x_neg/s2/x6bf_fresh_ref.json --only_class pedestrian
# Combining-rule study: PV-RCNN++'s second stage on the fresh pedestrian boxes, same settings as on val_i1
mkdir -p output/v2x/plan_s2
ref pv_v2x 20 x6bf output/v2x/plan_s2/pv_pedestrian_fresh_ref.json --only_class pedestrian
# (Per-class combination study: vehicle and truck failed on val_i1, so their fresh second stages are not run.)

S2=output/v2x/plan_s2
if cand logs/v2x_step2_choice.json; then
  ref vrcnn_v2x_b 20 x7 $S2/x6b_vehicle_fresh_ref.json --only_class vehicle
  ref vrcnn_v2x_neg 20 x7 $S2/neg_vehicle_fresh_ref.json --only_class vehicle
  ref vrcnn_v2x_neg 20 v18 $S2/neg_truck_fresh_ref.json --only_class truck
  for e in 15 17; do ref vrcnn_v2x_neg $e x6bf $S2/neg${e}_pedestrian_fresh_ref.json --only_class pedestrian; done
  ref vrcnn_v2x_b_all 20 x6bf output/v2x/vrcnn_v2x_b/x6bc/x6bh_fresh_ref.json
fi
if cand logs/v2x_phase1_choice.json; then
  ref pv_v2x 20 x7 $S2/pv_vehicle_fresh_ref.json --only_class vehicle
  ref pv_v2x 20 v18 $S2/pv_truck_fresh_ref.json --only_class truck
  ref pv_v2x 20 x6bf $S2/pv_pedestrian_fresh_ref.json --only_class pedestrian
fi
if [ -f output/v2x/ens/parta2_v2x_test.json ]; then
  O=output/v2x/parta2_v2x/s2
  ref parta2_v2x 20 x7 $O/vehicle_fresh_ref.json --only_class vehicle
  ref parta2_v2x 20 v18 $O/truck_fresh_ref.json --only_class truck
  ref parta2_v2x 20 x6bf $O/pedestrian_fresh_ref.json --only_class pedestrian
fi
UT_NOGPU=1 OMP_NUM_THREADS=6 $R python scripts/v2x_fresh_read.py > logs/v2x_fresh_read.log 2>&1
# Combining-rule study: R0 and the rule fixed on val_i1 (logs/ensemble_rules_v2x.json), one fresh reading each
UT_NOGPU=1 OMP_NUM_THREADS=6 $R python scripts/ensemble_rules.py v2x fresh > logs/ensemble_rules_v2x_fresh.log 2>&1
# Per-class combination study: one fresh reading of each V2X class winner that passed val_i1 (logs/v2x_class_combos.json)
UT_NOGPU=1 OMP_NUM_THREADS=6 $R python scripts/v2x_class_combos.py fresh > logs/v2x_class_combos_fresh.log 2>&1
# Member route (built Nov 8): for each new model whose V2X route read val_i1 in its own job (FSHNet V2X runs before this job), the fresh lanes,
# second stages and reading (scripts/route_v2x.sh <model> fresh); a model whose job comes after this one does its fresh part itself.
for d in output/v2x/*/s2/val_done; do
  [ -f "$d" ] || continue
  m=$(basename "$(dirname "$(dirname "$d")")")
  scripts/route_v2x.sh "$m" fresh > logs/route_v2x_${m}_fresh.log 2>&1 || echo "member route fresh step for $m failed rc=$?"
done
