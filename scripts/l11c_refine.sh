#!/usr/bin/env bash
# L11c GPU step: the fixed L11's second stage refines our final v29 boxes on the 50k val and the test frames (as X6c on V2X,
# scripts/x6_refine.py). Inputs output/lumpi/ens_x/lumpi_v29_{val,test}.json; outputs output/lumpi/vrcnn_lumpi_fix/l11c/.
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/docker_run.sh
CFG=configs/pcdet/vrcnn_lumpi_fix.yaml
CKPT=output/pcdet/vrcnn_lumpi_fix/default/ckpt/checkpoint_epoch_15.pth
O=output/lumpi/vrcnn_lumpi_fix/l11c
mkdir -p $O
$R python scripts/x6_refine.py --cfg $CFG --ckpt $CKPT --frames data/pcdet/lumpi_real_50k/points \
    --pred output/lumpi/ens_x/lumpi_v29_val.json --out $O/v29_val_ref.json
$R python scripts/x6_refine.py --cfg $CFG --ckpt $CKPT --frames data/kit/lumpi/detection_test_frames \
    --pred output/lumpi/ens_x/lumpi_v29_test.json --out $O/v29_test_ref.json
