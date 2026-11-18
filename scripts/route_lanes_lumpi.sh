#!/usr/bin/env bash
# Member route, Mac side, for a new one-stage model on LUMPI (FSHNet, SAFDNet, or VoxelNeXt as FSHNet's fallback).
# The pv_lanes.sh route of PV-RCNN++ with the model's own kit predictions (last checkpoint, 4-view TTA): lanes = the v29 lane plus the new model at
# weight 1 for one class each (Person, Bicycle, Car, Truck, Van), on val, test and the fresh overlay, each through the same chain.
# First confirms the rebuilt base equals lumpi_v29 on val. Person and Bicycle chain files go to the 3090, where the job's hook
# (scripts/route_lumpi.sh) runs the second stages; Car, Truck and Van lanes are read without a second stage, as in lumpi_v51.
# UT_3090_ROOT: the repository path on the 3090 host (relative to its home directory, or absolute).
#   UT_3090_ROOT=<path> scripts/route_lanes_lumpi.sh fshnet_lumpi
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1
ONLY=""   # class specialists: one lane only
case $M in spec_lumpi_van) T=zXV; ONLY=v ;; spec_lumpi_van_rs) T=zXR; ONLY=v ;; spec_lumpi_car) T=zXC; ONLY=c ;; fshnet_lumpi_s2) T=zF2 ;; vrcnn_lumpi_c_s2) T=zQC ;; pv_lumpi_s2) T=zQP ;; vrcnn_lumpi_fix_s3) T=zQL ;; vrcnn_lumpi_rs_wa) T=zQW ;; vrcnn_lumpi_rs) T=zQR ;; vrcnn_lumpi_rs_b) T=zQB ;; vrcnn_lumpi_g) T=zQG ;; vrcnn_lumpi_gb) T=zQH ;; dsvt_lumpi) T=zDS ;; vmamba_lumpi) T=zVM ;; safdnet_lumpi_rs) T=zSR ;; vrcnn_lumpi_b_fix2_rs) T=zQ2 ;; fshnet_*) T=zFS ;; safdnet_*) T=zSD ;; voxelnext_*) T=zVX ;; *) echo "unknown model $M"; exit 2 ;; esac
W=fresh_ws; R=3090:${UT_3090_ROOT:?set UT_3090_ROOT}
# One route at a time: the runs share fresh_ws and the lane names' folders.
LOCK=$PWD/logs/route_lanes_lumpi.lock
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
rsync -a scripts utl $W/
B='["pp_lumpi_br",1,["Person","Car","Bicycle","Motorcycle","Bus","Van","Unknown"]],["pp_lumpi_walk",1,["Truck"]],["pp_lumpi_walk_all",1,["Person","Truck"]],["vrcnn_lumpi_fix",1,["Person"]]'
CP=${RCPU:+--cpus $RCPU -e OMP_NUM_THREADS=$RCPU}; WK=${RCPU:+--workers $RCPU}   # lumpi_lane_fast workers = RCPU when set; unset: its default, unchanged   # RCPU / SKIP_CTV (G3): unset = unchanged for every other model
D="docker run --rm $CP -v $PWD:/workspace -w /workspace -e PYTHONPATH=.pylib_sk152:. urbantwin-realism:cpu"
DF="docker run --rm $CP -v $PWD/$W:/workspace -v $PWD/.pylib_sk152:/workspace/.pylib_sk152 -w /workspace -e PYTHONPATH=.pylib_sk152:. urbantwin-realism:cpu"
E=output/lumpi/ens_x
mkdir -p output/lumpi/route_in $W/output/lumpi/route_in
date "+%T inputs: $M own kit predictions"
rsync -a $R/output/lumpi/$M/kit/pred_val50k.json output/lumpi/route_in/${M}_val.json
rsync -a $R/output/lumpi/$M/kit/pred_test.json output/lumpi/route_in/${M}_test.json
rsync -a $R/output/lumpi/$M/kit/pred_fresh50k.json $W/output/lumpi/route_in/${M}_val.json
if [ "$M" = vrcnn_lumpi_b_fix2_rs ]; then   # LUMPI #4: its test kit file staged into g3in/ for a later build, md5 vs the 3090
  mkdir -p output/lumpi/g3in
  cp -p output/lumpi/route_in/${M}_test.json output/lumpi/g3in/${M}_test.json
  a=$(ssh 3090 "md5sum $UT_3090_ROOT/output/lumpi/$M/kit/pred_test.json" | cut -d' ' -f1); b=$(md5 -q output/lumpi/g3in/${M}_test.json)
  echo "$(date '+%F %T %Z') $M test: 3090 $UT_3090_ROOT/output/lumpi/$M/kit/pred_test.json $a -> output/lumpi/g3in/${M}_test.json $b : $([ "$a" = "$b" ] && echo match || echo MISMATCH) (LUMPI #4, route_lanes_lumpi.sh)" >> logs/g3in_test_stage.log
  [ "$a" = "$b" ] || { echo "STOP g3in/${M}_test.json md5 differs from the 3090's kit/pred_test.json"; exit 3; }
  date "+%T g3in/${M}_test.json staged ($b)"
fi
I=output/lumpi/$M/s2_in
ssh 3090 "mkdir -p $UT_3090_ROOT/$I && rm -f $UT_3090_ROOT/$I/ready* $UT_3090_ROOT/$I/*.json"   # stale inputs and markers
date "+%T base check"
K=zPKchk_$M   # per-model base-check names
$D python scripts/lumpi_lane_fast.py $K "[$B]" --splits val $WK
$D python scripts/save_lumpi_chain.py $E/$K $E/${K}_chain_val.json val
$D python -c "
import json
a = json.load(open('$E/${K}_chain_val.json')); b = json.load(open('$E/lumpi_v29_val.json'))
same = all(sorted(json.dumps(d, sort_keys=True) for d in a[f]) == sorted(json.dumps(d, sort_keys=True) for d in b[f]) for f in b) and set(a) == set(b)
print('rebuilt base equals lumpi_v29 on val:', same, flush=True)
raise SystemExit(0 if same else 1)
"
# Block 1 slot 4: FSHNet seed 2 is read as one member with both FSHNet seeds averaged, i.e. the two seeds'
# kit predictions as two fusion members at weight 0.5 each (one member's weight), against FSHNet's references.
member() { if [ "$M" = fshnet_lumpi_s2 ]; then echo "[\"file:route_in/fshnet_lumpi\",0.5,[\"$1\"]],[\"file:route_in/fshnet_lumpi_s2\",0.5,[\"$1\"]]"; else echo "[\"file:route_in/$M\",1,[\"$1\"]]"; fi; }
lane() {  # lane class
  local n=$1 c=$2 S="[$B,$(member $2)]"
  date "+%T lane $n $c"
  $D python scripts/lumpi_lane_fast.py $n "$S" $WK
  $D python scripts/save_lumpi_chain.py $E/$n $E/${n}_chain_val.json val
  $D python scripts/save_lumpi_chain.py $E/$n $E/${n}_chain_test.json test
  $DF python scripts/lumpi_lane_fast.py $n "$S" --splits val $WK
  $DF python scripts/save_lumpi_chain.py $E/$n $E/${n}_chain_fresh.json val
}
send() {  # lane-letter name: the lane's three chain files, then its ready marker, so the GPU step can start on it at once
  rsync -a $E/${T}$1_chain_val.json $R/$I/$2_val.json
  rsync -a $E/${T}$1_chain_test.json $R/$I/$2_test.json
  rsync -a $W/$E/${T}$1_chain_fresh.json $R/$I/$2_fresh.json
  ssh 3090 "touch $UT_3090_ROOT/$I/ready_$2"
  date "+%T $2 second-stage inputs sent"
}
if [ -z "$ONLY" ]; then
case $T in
  zQ?) echo "Person lane skipped: block 1 refiner seed, Person gated by lumpi_member_read.py" ;;
  *) lane ${T}p Person; send p person ;;
esac
lane ${T}b Bicycle
send b bicycle
fi
ssh 3090 "touch $UT_3090_ROOT/$I/ready"
mkdir -p $E/route_$M
for k in c t v; do
  [ "${SKIP_CTV:-0}" = 1 ] && { echo "Car / Truck / Van route lanes skipped (SKIP_CTV=1: read by igm60 on v60 instead)"; break; }
  [ -z "$ONLY" ] || [ "$ONLY" = $k ] || continue
  case $k in c) lane ${T}c Car ;; t) lane ${T}t Truck ;; v) lane ${T}v Van ;; esac
  cp $W/$E/${T}${k}_chain_fresh.json $E/route_$M/
done
date "+%T base check on test and fresh"   # the rebuilt base (model weight 0) must equal lumpi_v29 on every split, as on V2X
$D python scripts/lumpi_lane_fast.py $K "[$B]" --splits test $WK
$D python scripts/save_lumpi_chain.py $E/$K $E/${K}_chain_test.json test
$DF python scripts/lumpi_lane_fast.py $K "[$B]" --splits val $WK
$DF python scripts/save_lumpi_chain.py $E/$K $E/${K}_chain_fresh.json val
cp $W/$E/${K}_chain_fresh.json $E/${K}_chain_fresh.json
$D python -c "
import json
ok = True
for s in ('test', 'fresh'):
    a = json.load(open('$E/${K}_chain_%s.json' % s)); b = json.load(open('$E/lumpi_v29_%s.json' % s))
    same = all(sorted(json.dumps(d, sort_keys=True) for d in a[f]) == sorted(json.dumps(d, sort_keys=True) for d in b[f]) for f in b) and set(a) == set(b)
    print('rebuilt base equals lumpi_v29 on', s + ':', same, flush=True)
    ok = ok and same
raise SystemExit(0 if ok else 1)
"
date "+%T lanes done"
