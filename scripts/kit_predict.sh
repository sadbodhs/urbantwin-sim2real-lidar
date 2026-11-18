#!/usr/bin/env bash
# Predict the official test frames of the starting kit and score the same settings on the 50k-point validation copy.
#   scripts/kit_predict.sh lumpi configs/pcdet/pp_lumpi.yaml <ckpt> <tag> [infer args...]
# LUMPI predictions also get the class copies (utl.postproc, arm B) as *_B.json.
set -euo pipefail
TRACK=$1; CFG=$2; CKPT=$3; TAG=$4; shift 4
NAME=$(basename "$CFG" .yaml)
R="$(dirname "$0")/docker_run.sh"
cd "$(dirname "$0")/.."
OUT=output/$TRACK/$NAME/$TAG
mkdir -p "$OUT"
VAL=data/pcdet/${TRACK}_real_50k
$R python -m utl.infer --cfg "$CFG" --ckpt "$CKPT" --frames "$VAL/points" --ids "$VAL/ImageSets/val.txt" \
    --out "$OUT/pred_val50k.json" --tta "$@"
$R python -m utl.infer --cfg "$CFG" --ckpt "$CKPT" --frames "data/kit/$TRACK/detection_test_frames" \
    --out "$OUT/pred_test.json" --tta "$@"
SETS="pred_val50k"
if [ "$TRACK" = lumpi ]; then
  COPIES="Motorcycle>Bicycle Unknown>Car Van>Car Van>Truck:tail Bus>Truck:tail"
  for P in pred_val50k pred_test; do
    UT_NOGPU=1 $R python -m utl.postproc --pred "$OUT/$P.json" --out "$OUT/${P}_B.json" --copy $COPIES
  done
  SETS="pred_val50k pred_val50k_B"
fi
for P in $SETS; do
  for S in val val_h1 val_h2 val_i1 val_i2; do
    [ -f "$VAL/ImageSets/$S.txt" ] || continue
    UT_NOGPU=1 $R python -m utl.evaluate --track "$TRACK" --pred "$OUT/$P.json" --root "$VAL" --split val \
        --ids "$VAL/ImageSets/$S.txt" --out "$OUT/eval_${P}_$S.json" 2>/dev/null | grep -A9 "IoU 0.5" > "$OUT/eval_${P}_$S.txt"
    echo "$TRACK $NAME $TAG $P $S: $(head -1 "$OUT/eval_${P}_$S.txt")"
  done
done
