#!/usr/bin/env bash
# pv_v2x_n kit predictions on the 3090: the Nov 7 pv_v2x checkpoint (PV-RCNN++ V2X, configs/pcdet/pv_v2x.yaml,
# queue line v2x_pv = scripts/run_track.sh v2x, seed 666 via --fix_random_seed, epoch 20) re-predicted with the current utl/infer.py as the V2X
# fusion member pv_v2x_n, exactly as the voxelnext_v2x / vmamba_v2x kit inputs (the tail of scripts/voxelnext_v2x_train.sh, 81a1c3c2):
#   val50k + kit test = scripts/kit_predict.sh v2x's two utl.infer calls (same --frames / --ids, --tta, no extra args) and its val50k scoring;
#   then fresh50k --tta; utl.infer's default --score_thresh 0.01, NO --max_dets (the yaml's own NMS settings apply).
#   kit_predict.sh is NOT called: it names its output dir after the config (output/v2x/pv_v2x/kit = the Nov 7 kit files, never written or read).
#   -> output/v2x/pv_v2x_n/kit/pred_{val50k,test,fresh50k}.json (+ kit_predict's eval_pred_val50k_<S>.{json,txt}); their md5s in
#      logs/pv_v2x_n_pred.md5 (exactly those 3 paths, md5sum format), then logs/pv_v2x_n.done.
# Image urbantwin-lidar:latest (UT_IMAGE unset), as the Nov 7 training and predictions (queue line v2x_pv sets no UT_IMAGE; :scatter and :mamba were
#   built later) and scripts/v2x_opt2_pv.sh (Nov 13, same checkpoint, UT_IMAGE unset).
# Checks first (stop = nothing predicted): config md5; checkpoint md5 (= scripts/v2x_opt2_pv.sh's record); utl/infer.py md5; image present;
#   no pv_v2x_n kit file, md5 record or done file yet (no overwrite). Small GPU job (670 frames x 4 TTA views).
# Exit: 0 done; 2 a check failed; 3 a prediction or the val50k scoring failed. Test: DRY=1 runs every check, then exits 0 without predicting.
#   CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 scripts/pv_v2x_n_pred.sh
set -uo pipefail
cd "$(dirname "$0")/.."
N=pv_v2x_n; unset UT_IMAGE; IMG=urbantwin-lidar:latest; R=scripts/docker_run.sh; DRY=${DRY:-0}
CFG=configs/pcdet/pv_v2x.yaml; CK=output/pcdet/pv_v2x/default/ckpt/checkpoint_epoch_20.pth
KIT=output/v2x/$N/kit; LOG=logs/$N.log; PL=logs/${N}_pred.log; NOTE=logs/gpu_queue_notes.txt
VAL=data/pcdet/v2x_real_50k; FR=data/pcdet/v2x_fresh_50k
[ "$DRY" = 1 ] && LOG=/dev/null
say() { echo "$(date '+%F %T %Z') [$N] $*" | tee -a $LOG; }
note() { [ "$DRY" = 1 ] || echo "$(date) $N: $*" >> $NOTE; say "$*"; }

md5sum -c --quiet - <<MD5 || { note "input md5 differs (config / checkpoint / utl/infer.py): stop"; exit 2; }
91729977cd1b8c5c1ede1d7dcf7dd402  $CFG
14eb528da96316c0b3f4a437222558bf  $CK
ec938383225c181b3d9f68f162df8ac6  utl/infer.py
MD5
docker image inspect $IMG > /dev/null 2>&1 || { note "image $IMG missing: stop"; exit 2; }
ex=$(ls $KIT/* logs/$N.done logs/${N}_pred.md5 2>/dev/null | tr '\n' ' ')
[ -z "$ex" ] || { note "pv_v2x_n files exist ($ex): stop, no overwrite"; exit 2; }
if [ "$DRY" = 1 ]; then say "DRY=1: every check passed; nothing predicted"; exit 0; fi
mkdir -p $KIT
note "checks passed; kit predictions val50k / test / fresh50k --tta -> $KIT"
# = scripts/kit_predict.sh v2x $CFG $CK kit, with the output dir $KIT
$R python -m utl.infer --cfg "$CFG" --ckpt "$CK" --frames "$VAL/points" --ids "$VAL/ImageSets/val.txt" \
    --out "$KIT/pred_val50k.json" --tta >> $PL 2>&1 || { note "val50k prediction failed: stop"; exit 3; }
$R python -m utl.infer --cfg "$CFG" --ckpt "$CK" --frames "data/kit/v2x/detection_test_frames" \
    --out "$KIT/pred_test.json" --tta >> $PL 2>&1 || { note "kit test prediction failed: stop"; exit 3; }
for S in val val_h1 val_h2 val_i1 val_i2; do
  [ -f "$VAL/ImageSets/$S.txt" ] || continue
  UT_NOGPU=1 $R python -m utl.evaluate --track v2x --pred "$KIT/pred_val50k.json" --root "$VAL" --split val \
      --ids "$VAL/ImageSets/$S.txt" --out "$KIT/eval_pred_val50k_$S.json" 2>/dev/null | grep -A9 "IoU 0.5" > "$KIT/eval_pred_val50k_$S.txt" \
    || { note "val50k scoring ($S) failed: stop"; exit 3; }
  echo "v2x $N kit pred_val50k $S: $(head -1 "$KIT/eval_pred_val50k_$S.txt")" >> $PL
done
# = the fresh50k call of scripts/voxelnext_v2x_train.sh
$R python -m utl.infer --cfg $CFG --ckpt $CK --frames $FR/points --ids $FR/ImageSets/val.txt \
    --out $KIT/pred_fresh50k.json --tta >> $PL 2>&1 || { note "fresh50k failed: stop"; exit 3; }
md5sum $KIT/pred_val50k.json $KIT/pred_test.json $KIT/pred_fresh50k.json > logs/${N}_pred.md5
say "done: $(wc -l < logs/${N}_pred.md5) kit files; $(head -1 $KIT/eval_pred_val50k_val.txt 2>/dev/null)"; echo ok > logs/$N.done
note "finished kit predictions (rc 0)"
