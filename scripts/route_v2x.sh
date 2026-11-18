#!/usr/bin/env bash
# Member route on V2X (built Nov 8 for the new models), run by scripts/newmodel_jobs.sh at the end of a
# FSHNet / SAFDNet / VoxelNeXt V2X job, everything on the 3090 inside that job's slot (the V2X sets are small: 146 val_i1, 50 test, 328 fresh frames).
#   1. The model's own predictions on the V2X fresh set (same 4-view TTA as its kit predictions; its image).
#   2. CPU: fusion lanes (scripts/route_v2x_lanes.py) on val_i1 and test, and on fresh once the members' fresh predictions exist.
#   3. v2x_v25's second stages on the fused boxes, one class each: vehicle X6, truck X6b, pedestrian X6b and the negatives model (epoch 20).
#   4. CPU: scripts/route_read_v2x.py val (and fresh when ready) > logs/route_v2x_<model>_read.log
# When the fresh members do not exist yet (FSHNet V2X runs before the v2x_fresh job), scripts/v2x_fresh_job.sh runs this script again later
# with "fresh" for the fresh lanes, second stages and reading.
#   scripts/route_v2x.sh fshnet_v2x          (in the job)
#   scripts/route_v2x.sh fshnet_v2x fresh    (from the v2x_fresh job)
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1; MODE=${2:-job}
R=scripts/docker_run.sh
CFG=configs/pcdet/$M.yaml
EP=$(grep -E "^\s*NUM_EPOCHS:" "$CFG" | awk "{print \$2}")
CKPT=output/pcdet/$M/default/ckpt/checkpoint_epoch_$EP.pth
FR=data/pcdet/v2x_fresh_50k
I=output/v2x/$M/s2_in
O=output/v2x/$M/s2
mkdir -p $I $O
LOG=logs/route_v2x_${M}_read.log
if [ "$MODE" = job ]; then
  echo "route: fresh predictions of $M $(date)"
  $R python -m utl.infer --cfg "$CFG" --ckpt "$CKPT" --frames $FR/points --ids $FR/ImageSets/val.txt --out output/v2x/$M/kit/pred_fresh50k.json --tta ${TTA_X:-}   # TTA_X: F3 flags for two-stage models
  SPLITS="val_i1 test"
  [ -s output/v2x/ens/x7_fresh.json ] && [ -s output/v2x/vrcnn_v2x_neg/s2/x6bf_fresh_ref.json ] && SPLITS="val_i1 test fresh"
else
  SPLITS=fresh
fi
unset UT_IMAGE   # lanes and refiners: original image
echo "route: lanes on $SPLITS $(date)"
# One process per split in parallel (to use more cores; the splits are
# independent; outputs checked byte-identical to the one-process run on val_i1 + test, Nov 9; each split keeps the one-process run's 6 threads);
# any failing split stops the route as before (set -e on wait).
LP=()
for s in $SPLITS; do UT_NOGPU=1 OMP_NUM_THREADS=6 $R python scripts/route_v2x_lanes.py "$M" $s > logs/route_v2x_lanes_${M}_$s.log 2>&1 & LP+=($!); done
for p in "${LP[@]}"; do wait $p; done
for s in $SPLITS; do grep -v Warning logs/route_v2x_lanes_${M}_$s.log | tail -6; done
frames() { case $1 in val_i1) echo data/pcdet/v2x_real_50k/points ;; test) echo data/kit/v2x/detection_test_frames ;; fresh) echo $FR/points ;; esac; }
for s in $SPLITS; do
  F=$(frames $s)
  $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x.yaml --ckpt output/pcdet/vrcnn_v2x/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/vehicle_$s.json --out $O/vehicle_x6_${s}_ref.json --only_class vehicle
  $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_b.yaml --ckpt output/pcdet/vrcnn_v2x_b/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/truck_$s.json --out $O/truck_x6b_${s}_ref.json --only_class truck
  $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_b.yaml --ckpt output/pcdet/vrcnn_v2x_b/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/pedestrian_$s.json --out $O/pedestrian_x6b_${s}_ref.json --only_class pedestrian
  $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_neg.yaml --ckpt output/pcdet/vrcnn_v2x_neg/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/pedestrian_$s.json --out $O/pedestrian_neg_${s}_ref.json --only_class pedestrian
done
echo "route: second stages done on $SPLITS $(date)"
if [ "$MODE" = job ]; then
  UT_NOGPU=1 OMP_NUM_THREADS=6 $R python scripts/route_read_v2x.py val "$M" > $LOG 2>&1
  touch $O/val_done
fi
case " $SPLITS " in *" fresh "*) UT_NOGPU=1 OMP_NUM_THREADS=6 $R python scripts/route_read_v2x.py fresh "$M" >> $LOG 2>&1 ;; esac
grep -v Warning $LOG | tail -25
echo "route: read $(date)"
