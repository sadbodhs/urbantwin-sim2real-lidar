#!/usr/bin/env bash
# X6b's second stage (relabel-fixed, per-class) refines the final v2x_v18 boxes on the OS1-64 val and the test frames, as scripts/x6c_refine.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/docker_run.sh
CFG=configs/pcdet/vrcnn_v2x_b.yaml
CKPT=output/pcdet/vrcnn_v2x_b/default/ckpt/checkpoint_epoch_20.pth
O=output/v2x/vrcnn_v2x_b/x6bc
mkdir -p $O
$R python scripts/x6_refine.py --cfg $CFG --ckpt $CKPT --frames data/pcdet/v2x_real_50k/points --pred output/v2x/ens/v18_val_i1.json --out $O/v18_val_i1_ref.json
$R python scripts/x6_refine.py --cfg $CFG --ckpt $CKPT --frames data/kit/v2x/detection_test_frames --pred output/v2x/ens/v18_test.json --out $O/v18_test_ref.json
