#!/usr/bin/env bash
# Train one config, then predict + score on the real validation frames and predict the test frames.
#   scripts/run_track.sh lumpi configs/pcdet/pp_lumpi.yaml [epoch]
#   scripts/run_track.sh v2x   configs/pcdet/pp_v2x.yaml   [epoch]
# Env: SKIP_TRAIN=1 (reuse checkpoint), UT_GPUS, WORKERS (dataloader workers, default 7), TEST_FRAMES (dir).
set -euo pipefail
TRACK=$1; CFG=$2
NAME=$(basename "$CFG" .yaml)
R="$(dirname "$0")/docker_run.sh"
cd "$(dirname "$0")/.."
EPOCHS=$(grep -E "^\s*NUM_EPOCHS:" "$CFG" | awk '{print $2}')
EPOCH=${3:-$EPOCHS}
CKPT=output/pcdet/$NAME/default/ckpt/checkpoint_epoch_$EPOCH.pth
OUT=output/$TRACK/$NAME
mkdir -p "$OUT" logs

if [ "${SKIP_TRAIN:-0}" != 1 ]; then
  # OpenPCDet evaluates the last checkpoint on the synthetic val split after training
  $R python -m utl.train --cfg_file "$CFG" --workers "${WORKERS:-7}" --num_epochs_to_eval 1 --fix_random_seed \
      > "logs/train_$NAME.log" 2>&1
fi

case "$TRACK" in
  lumpi) VAL_ROOT=data/pcdet/lumpi_real; VAL_SPLIT=val; TEST=${TEST_FRAMES:-data/lumpi_test_m4}; REAL=output/lumpi/realism_v1 ;;
  v2x)   VAL_ROOT=data/pcdet/v2x_real;   VAL_SPLIT=val; TEST=${TEST_FRAMES:-}; REAL=output/v2x/realism_v1_zeros_keep ;;
esac

for TTA in "" "--tta"; do
  SUF=${TTA:+_tta}
  $R python -m utl.infer --cfg "$CFG" --ckpt "$CKPT" --frames "$VAL_ROOT/points" --ids "$VAL_ROOT/ImageSets/$VAL_SPLIT.txt" \
      --out "$OUT/pred_val$SUF.json" $TTA > "logs/infer_${NAME}_val$SUF.log" 2>&1
  UT_NOGPU=1 $R python -m utl.evaluate --track "$TRACK" --pred "$OUT/pred_val$SUF.json" --root "$VAL_ROOT" \
      --split "$VAL_SPLIT" --realism "$REAL/realism_report.json" --out "$OUT/eval_val$SUF.json" 2>/dev/null \
      | tee "$OUT/eval_val$SUF.txt"
  if [ -n "$TEST" ]; then
    $R python -m utl.infer --cfg "$CFG" --ckpt "$CKPT" --frames "$TEST" --out "$OUT/pred_test$SUF.json" $TTA \
        > "logs/infer_${NAME}_test$SUF.log" 2>&1
  fi
done
