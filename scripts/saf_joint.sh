#!/usr/bin/env bash
# SAFDNet on top of v2x_v29: joint second stage, read by the same rules as v29.
# No training.
#   1. CPU: joint fusion lanes (scripts/route_v2x_lanes_joint.py): v25 chain + FSHNet V2X + SAFDNet V2X, with its checks (STOP on a miss).
#   2. GPU: the same second stages as v29 (as scripts/route_v2x.sh, 4 views): vehicle X6, pedestrian X6b and the negatives model, epoch 20,
#      on val_i1, test and fresh. Outputs output/v2x/fsh_saf_v2x/s2/, then s2/done. Mac reading: scripts/r6r7_read_v2x.py saf.
#   scripts/saf_joint.sh
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/docker_run.sh
I=output/v2x/fsh_saf_v2x/s2_in
O=output/v2x/fsh_saf_v2x/s2
mkdir -p $I $O
unset UT_IMAGE
echo "saf_joint: lanes $(date)"
LP=()
for s in val_i1 test fresh; do UT_NOGPU=1 OMP_NUM_THREADS=6 $R python scripts/route_v2x_lanes_joint.py $s > logs/saf_joint_lanes_$s.log 2>&1 & LP+=($!); done
for p in "${LP[@]}"; do wait $p; done
for s in val_i1 test fresh; do grep -v Warning logs/saf_joint_lanes_$s.log | tail -4; done
frames() { case $1 in val_i1) echo data/pcdet/v2x_real_50k/points ;; test) echo data/kit/v2x/detection_test_frames ;; fresh) echo data/pcdet/v2x_fresh_50k/points ;; esac; }
for s in val_i1 test fresh; do
  F=$(frames $s)
  $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x.yaml --ckpt output/pcdet/vrcnn_v2x/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/vehicle_$s.json --out $O/vehicle_x6_${s}_ref.json --only_class vehicle
  $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_b.yaml --ckpt output/pcdet/vrcnn_v2x_b/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/pedestrian_$s.json --out $O/pedestrian_x6b_${s}_ref.json --only_class pedestrian
  $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_neg.yaml --ckpt output/pcdet/vrcnn_v2x_neg/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/pedestrian_$s.json --out $O/pedestrian_neg_${s}_ref.json --only_class pedestrian
  echo "saf_joint $s done $(date)"
done
touch $O/done
echo "saf_joint done $(date)"
