#!/usr/bin/env bash
# LUMPI PointPillars on all synthetic frames (one-off test), on the base the L7 gate chose:
# the long-truck config if logs/gate_l7.log says L5 helps, else the full-density one. Same inference as that base, so
# the upload is the best LUMPI build on that base with only its PointPillars swapped for this one.
#   scripts/lumpi_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if grep -q "L5 helps? yes" logs/gate_l7.log 2>/dev/null; then CFG=pp_lumpi_long_all; else CFG=pp_lumpi_all; fi
echo "$(date '+%F %T') lumpi_all: training $CFG" | tee -a logs/gate_l7.log
TEST_FRAMES=data/kit/lumpi/detection_test_frames scripts/run_track.sh lumpi "configs/pcdet/$CFG.yaml"
scripts/kit_predict.sh lumpi "configs/pcdet/$CFG.yaml" "output/pcdet/$CFG/default/ckpt/checkpoint_epoch_20.pth" kit_mc \
    --multi_class_nms --max_dets 100
