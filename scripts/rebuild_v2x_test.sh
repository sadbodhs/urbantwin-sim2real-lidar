#!/usr/bin/env bash
# Test-split rebuild of v2x_v35d from the released checkpoints: 12 members on the 50 test frames, the member check,
# the lanes, the three second stages, the final file (P3 off and P3 from the rebuilt map) and the comparison with the
# submitted predictions.json. Each step logs to logs/rb_<step>.log; a finished step leaves logs/rb_<step>.done and is skipped on a rerun.
#   scripts/rebuild_v2x_test.sh
# Needs: output/pcdet/<model>/default/ckpt/ (the released checkpoints), data/kit/v2x/detection_test_frames,
# data/hilbert_lumpi (scripts/make_hilbert_lumpi.py), anchors/v2x_v27_predictions.json, sub/predictions.json,
# data/pcdet/v2x_real/ImageSets/rfit_i1.txt with its points (tools/rebuild_v2x_test.py p3-ids), and the images
# urbantwin-lidar:{latest,scatter,mamba} and urbantwin-realism:cpu.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/rb_pipeline.log; mkdir -p logs
say() { echo "$(date '+%F %T %Z') $*" | tee -a "$LOG"; }
run() { local tag=$1; shift; [ -e "logs/rb_$tag.done" ] && { say "skip $tag (done)"; return 0; }; say "start $tag"; "$@" > "logs/rb_$tag.log" 2>&1; local rc=$?; say "end $tag rc=$rc"; [ $rc -eq 0 ] || { say "STOP: $tag failed"; exit $rc; }; touch "logs/rb_$tag.done"; }
P=output/pcdet; T=data/kit/v2x/detection_test_frames; R=scripts/docker_run.sh; MC="--multi_class_nms --max_dets 100"
for m in pp_v2x pp_v2x_ped pp_v2x_all pp_v2x_walk; do
  run "infer_$m" $R python -m utl.infer --cfg configs/pcdet/$m.yaml --ckpt $P/$m/default/ckpt/checkpoint_epoch_30.pth --frames $T --out output/v2x/$m/kit_mc/pred_test.json --tta $MC
done
run infer_cp_v2x $R python -m utl.infer --cfg configs/pcdet/cp_v2x.yaml --ckpt $P/cp_v2x/default/ckpt/checkpoint_epoch_30.pth --frames $T --out output/v2x/cp_v2x/kit/pred_test.json --tta
for m in vrcnn_v2x vrcnn_v2x_b; do
  run "infer_$m" $R python -m utl.infer --cfg configs/pcdet/$m.yaml --ckpt $P/$m/default/ckpt/checkpoint_epoch_20.pth --frames $T --out output/v2x/$m/kit/pred_test.json --tta
done
run infer_pv_v2x_n $R python -m utl.infer --cfg configs/pcdet/pv_v2x.yaml --ckpt $P/pv_v2x/default/ckpt/checkpoint_epoch_20.pth --frames $T --out output/v2x/pv_v2x_n/kit/pred_test.json --tta
for m in fshnet_v2x safdnet_v2x dsvt_v2x; do
  run "infer_$m" env UT_IMAGE=urbantwin-lidar:scatter $R python -m utl.infer --cfg configs/pcdet/$m.yaml --ckpt $P/$m/default/ckpt/checkpoint_epoch_20.pth --frames $T --out output/v2x/$m/kit/pred_test.json --tta
done
run infer_vmamba_v2x env UT_IMAGE=urbantwin-lidar:mamba $R python -m utl.infer --cfg configs/pcdet/vmamba_v2x.yaml --ckpt $P/vmamba_v2x/default/ckpt/checkpoint_epoch_20.pth --frames $T --out output/v2x/vmamba_v2x/kit/pred_test.json --tta
mkdir -p output/v2x/sweep_in
cp output/v2x/pp_v2x/kit_mc/pred_test.json output/v2x/sweep_in/pp_test.json
cp output/v2x/cp_v2x/kit/pred_test.json output/v2x/sweep_in/cp_test.json
cp output/v2x/pp_v2x_ped/kit_mc/pred_test.json output/v2x/sweep_in/ped_test.json
cp output/v2x/pp_v2x_all/kit_mc/pred_test.json output/v2x/sweep_in/ppall_test.json
D="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/workspace -w /workspace urbantwin-realism:cpu python tools/rebuild_v2x_test.py"
run members_check $D members-check
run lanes env UT_NOGPU=1 OMP_NUM_THREADS=1 $R python tools/rebuild_v2x_test.py lanes --workers 6
Q=output/v2x/q7_dsvt_vmamba_pvn_v2x_j; J=output/v2x/dsvt_v2x_j; mkdir -p $Q/s2 $J/s2
run refine_x6_vehicle $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x.yaml --ckpt $P/vrcnn_v2x/default/ckpt/checkpoint_epoch_20.pth --frames $T --pred $Q/s2_in/vehicle_test.json --out $Q/s2/vehicle_x6_test_ref.json --only_class vehicle
run refine_x6b_ped $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_b.yaml --ckpt $P/vrcnn_v2x_b/default/ckpt/checkpoint_epoch_20.pth --frames $T --pred $J/s2_in/pedestrian_test.json --out $J/s2/pedestrian_x6b_test_ref.json --only_class pedestrian
run refine_neg_ped $R python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_neg.yaml --ckpt $P/vrcnn_v2x_neg/default/ckpt/checkpoint_epoch_20.pth --frames $T --pred $J/s2_in/pedestrian_test.json --out $J/s2/pedestrian_neg_test_ref.json --only_class pedestrian
run final_p3off $D final --v27 anchors/v2x_v27_predictions.json --p3 off --submitted sub/predictions.json
run final_p3ids $D final --v27 anchors/v2x_v27_predictions.json --p3 ids data/pcdet/v2x_real/ImageSets/rfit_i1.txt --submitted sub/predictions.json --force
run compare $D compare --submitted sub/predictions.json
say "ALL DONE"
