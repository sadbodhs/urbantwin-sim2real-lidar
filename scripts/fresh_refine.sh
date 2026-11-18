#!/usr/bin/env bash
# One-time fresh check, GPU step: L11's and L11b's second stages on the v29 boxes of the 320 fresh frames, as scripts/l11c_refine.sh and l11bc_refine.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/docker_run.sh
F=data/pcdet/lumpi_real_fresh50k/points
P=output/lumpi/ens_x/lumpi_v29_fresh.json
mkdir -p output/lumpi/vrcnn_lumpi_fix/l11c output/lumpi/vrcnn_lumpi_b/l11bc
$R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_lumpi_fix.yaml --ckpt output/pcdet/vrcnn_lumpi_fix/default/ckpt/checkpoint_epoch_15.pth --frames $F --pred $P --out output/lumpi/vrcnn_lumpi_fix/l11c/v29_fresh_ref.json
$R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_lumpi_b.yaml --ckpt output/pcdet/vrcnn_lumpi_b/default/ckpt/checkpoint_epoch_15.pth --frames $F --pred $P --out output/lumpi/vrcnn_lumpi_b/l11bc/v29_fresh_ref.json
# Person refine extended to v29 boxes scoring 0.01-0.05: L11's second stage with --min_score 0.01 on val, test and fresh; separate output files
C="--cfg configs/pcdet/vrcnn_lumpi_fix.yaml --ckpt output/pcdet/vrcnn_lumpi_fix/default/ckpt/checkpoint_epoch_15.pth --min_score 0.01"
O=output/lumpi/vrcnn_lumpi_fix/l11c
$R python scripts/x6_refine.py $C --frames data/pcdet/lumpi_real_50k/points --pred output/lumpi/ens_x/lumpi_v29_val.json --out $O/v29_val_ref_min01.json
$R python scripts/x6_refine.py $C --frames data/kit/lumpi/detection_test_frames --pred output/lumpi/ens_x/lumpi_v29_test.json --out $O/v29_test_ref_min01.json
$R python scripts/x6_refine.py $C --frames $F --pred $P --out $O/v29_fresh_ref_min01.json
