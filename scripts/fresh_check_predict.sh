#!/usr/bin/env bash
# Fresh check: every LUMPI member model on the 320 labelled real frames never
# used to score detection (ids 10, 30, ... 6390; used only for the realism density fit; none within 20 frames of a forbidden
# frame), thinned to 50k as lumpi_real_50k -> data/pcdet/lumpi_real_fresh50k. Scoring only: they enter no training or fit.
# Same inference as val (--tta; per-class NMS for PointPillars). Writes output/lumpi/<model>/<kit_mc|kit|rot/kit_mc>/pred_fresh50k.json.
set -uo pipefail
cd "$(dirname "$0")/.."
R=scripts/docker_run.sh
F=data/pcdet/lumpi_real_fresh50k/points
run() {  # model epoch tag extra-args
  local m=$1 e=$2 t=$3; shift 3
  local c=output/pcdet/$m/default/ckpt/checkpoint_epoch_$e.pth
  [ -f "$c" ] || { echo "skip $m (no checkpoint)"; return; }
  mkdir -p "output/lumpi/$m/$t"
  $R python -m utl.infer --cfg "configs/pcdet/$m.yaml" --ckpt "$c" --frames "$F" --out "output/lumpi/$m/$t/pred_fresh50k.json" --tta "$@" \
      || echo "FAILED $m $t"
}
NMS="--multi_class_nms --max_dets 100"
for m in pp_lumpi_long pp_lumpi_d50 pp_lumpi_long_all pp_lumpi_walk pp_lumpi_walk_all pp_lumpi_br pp_lumpi; do run $m 20 kit_mc $NMS; done
run cpv_lumpi_long 15 kit
run vrcnn_lumpi_fix 15 kit
run vrcnn_lumpi_b 15 kit
for m in pp_lumpi_br pp_lumpi_walk_all; do run $m 20 rot/kit_mc $NMS --rot 22.5 -22.5; done
