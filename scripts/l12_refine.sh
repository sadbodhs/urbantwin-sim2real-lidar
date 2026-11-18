#!/usr/bin/env bash
# L12 routes, as L11b: its predictions on the fresh frames, and its second stage on the v29 boxes down to score 0.01 on val, test and fresh. Separate output files.
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/docker_run.sh
F=data/pcdet/lumpi_real_fresh50k/points
CF=configs/pcdet/vrcnn_lumpi_c.yaml
CK=output/pcdet/vrcnn_lumpi_c/default/ckpt/checkpoint_epoch_15.pth
O=output/lumpi/vrcnn_lumpi_c/l12c
mkdir -p $O output/lumpi/vrcnn_lumpi_c/kit
$R python -m utl.infer --cfg $CF --ckpt $CK --frames $F --out output/lumpi/vrcnn_lumpi_c/kit/pred_fresh50k.json --tta
C="--cfg $CF --ckpt $CK --min_score 0.01"
E=output/lumpi/ens_x
$R python scripts/x6_refine.py $C --frames data/pcdet/lumpi_real_50k/points --pred $E/lumpi_v29_val.json --out $O/v29_val_ref_min01.json
$R python scripts/x6_refine.py $C --frames data/kit/lumpi/detection_test_frames --pred $E/lumpi_v29_test.json --out $O/v29_test_ref_min01.json
$R python scripts/x6_refine.py $C --frames $F --pred $E/lumpi_v29_fresh.json --out $O/v29_fresh_ref_min01.json
