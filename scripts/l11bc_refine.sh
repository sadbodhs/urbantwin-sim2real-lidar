#!/usr/bin/env bash
# L11b's second stage refines the final v29 boxes on the 50k val and the test frames, as scripts/l11c_refine.sh does with L11's.
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/docker_run.sh
CFG=configs/pcdet/vrcnn_lumpi_b.yaml
CKPT=output/pcdet/vrcnn_lumpi_b/default/ckpt/checkpoint_epoch_15.pth
O=output/lumpi/vrcnn_lumpi_b/l11bc
mkdir -p $O
$R python scripts/x6_refine.py --cfg $CFG --ckpt $CKPT --frames data/pcdet/lumpi_real_50k/points \
    --pred output/lumpi/ens_x/lumpi_v29_val.json --out $O/v29_val_ref.json
$R python scripts/x6_refine.py --cfg $CFG --ckpt $CKPT --frames data/kit/lumpi/detection_test_frames \
    --pred output/lumpi/ens_x/lumpi_v29_test.json --out $O/v29_test_ref.json
# L11's second stage (lumpi_v32's refiner) on the boxes with L11b added for Bicycle, when they are there
CFG=configs/pcdet/vrcnn_lumpi_fix.yaml
CKPT=output/pcdet/vrcnn_lumpi_fix/default/ckpt/checkpoint_epoch_15.pth
O=output/lumpi/vrcnn_lumpi_fix/l11c
if [ -f output/lumpi/ens_x/zLb4_chain_test.json ]; then
  $R python scripts/x6_refine.py --cfg $CFG --ckpt $CKPT --frames data/pcdet/lumpi_real_50k/points \
      --pred output/lumpi/ens_x/zLb4_chain_val.json --out $O/zLb4_val_ref.json
  $R python scripts/x6_refine.py --cfg $CFG --ckpt $CKPT --frames data/kit/lumpi/detection_test_frames \
      --pred output/lumpi/ens_x/zLb4_chain_test.json --out $O/zLb4_test_ref.json
fi
