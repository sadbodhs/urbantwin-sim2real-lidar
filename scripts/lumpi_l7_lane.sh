#!/usr/bin/env bash
# One lumpi_v8 candidate, run inside the CPU image (one container per candidate, so they run in parallel):
# fusion of the long-truck PointPillars (1.0), the density-robust PointPillars (0.5) and the voxel CenterPoint L7
# (weight W, without its Person boxes when NP=1), then the lumpi_v7 post-processing (class copies, Person column test,
# truck length refine, Bicycle/Person size priors). Writes val and test predictions and the val scores.
#   scripts/mac_job.sh l7_cpv0.5 bash scripts/lumpi_l7_lane.sh 0.5 0
set -euo pipefail
W=$1 NP=$2
NAME=cpv$([ "$NP" = 1 ] && echo np || true)$W
VAL=data/pcdet/lumpi_real_50k
TEST=data/kit/lumpi/detection_test_frames
OUT=output/lumpi/ens_l7
mkdir -p "$OUT"
COPIES="Motorcycle>Bicycle Unknown>Car Van>Car Van>Truck:tail Bus>Truck:tail"
for SET in val test; do
  F=val50k; FR=$VAL/points
  [ $SET = test ] && { F=test; FR=$TEST; }
  C=output/lumpi/cpv_lumpi_long/kit/pred_$F.json
  if [ "$NP" = 1 ]; then
    python -c "import json,sys; d=json.load(open(sys.argv[1])); json.dump({f:[x for x in v if x['class']!='Person'] for f,v in d.items()}, open(sys.argv[2],'w'))" \
        "$C" "$OUT/${NAME}_cpv_$SET.json"
    C=$OUT/${NAME}_cpv_$SET.json
  fi
  P=$OUT/${NAME}_$SET
  python -m utl.ensemble --preds output/lumpi/pp_lumpi_long/kit_mc/pred_$F.json output/lumpi/pp_lumpi_d50/kit_mc/pred_$F.json \
      "$C" --weights 1.0 0.5 "$W" --out "$P.json" > /dev/null
  python -m utl.postproc --pred "$P.json" --out "${P}_B.json" --copy $COPIES > /dev/null
  python -m utl.column_test --pred "${P}_B.json" --frames "$FR" --out "${P}_col.json" --classes Person --k 3 --factor 0.1 > /dev/null
  python -m utl.length_refine --pred "${P}_col.json" --frames "$FR" --out "${P}_len.json" --gap 0.75 > /dev/null
  python -m utl.size_prior --pred "${P}_len.json" --out "${P}_v7.json" --mode max \
      --prior Bicycle:1.78,0.84,1.78 Person:0.73,0.66,1.76 > /dev/null
done
for S in val val_h1 val_h2; do
  python -m utl.evaluate --track lumpi --pred "$OUT/${NAME}_val_v7.json" --root "$VAL" --split val \
      --ids "$VAL/ImageSets/$S.txt" --out "$OUT/eval_${NAME}_$S.json" 2>/dev/null | grep -A9 "IoU 0.5" > "$OUT/eval_${NAME}_$S.txt"
  echo "$NAME $S: $(head -1 "$OUT/eval_${NAME}_$S.txt" | cut -c1-24) | $(grep -E "Person|Bicycle|Truck|Car " "$OUT/eval_${NAME}_$S.txt" | awk '{printf "%s %s  ", $1, $3}')"
done
