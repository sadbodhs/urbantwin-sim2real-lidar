#!/usr/bin/env bash
# Q7 V2X vehicle joint route, GPU step: v2x_v31's vehicle second stage (X6, vrcnn_v2x epoch 20, x6_refine
# defaults: 4 flip views, min_score 0.05, original image) on the joint vehicle lanes output/v2x/<lanes>/s2_in/vehicle_<s>.json written by
# scripts/q7_v2x_lanes.py -> output/v2x/<lanes>/s2/vehicle_x6_<s>_ref.json (s = val_i1, test, fresh: 3 refines). A copy of
# scripts/vmamba_v2x_refine.sh (da20b10c), vehicle only: same md5 checks (x6_refine.py, the X6 config and checkpoint), lane records
# logs/<lanes>_lanes.{done,json,md5} with mode "full", lane-off AND one-member checks passed, and the md5 list naming exactly the 3 lanes.
# Records logs/<lanes>_refine.{md5,done,log}. No overwrite. Small job. Reading on the Mac (scripts/q7_v2x_read.py).
#   scripts/q7_v2x_refine.sh q7_dsvt_vmamba_v2x_j
set -uo pipefail
cd "$(dirname "$0")/.."
unset UT_IMAGE
J=${1:?lanes name}; R=scripts/docker_run.sh; I=output/v2x/$J/s2_in; O=output/v2x/$J/s2; LOG=logs/${J}_refine.log
D=logs/${J}_lanes.done; JS=logs/${J}_lanes.json; M5=logs/${J}_lanes.md5
say() { echo "$(date '+%F %T %Z') [$J refine] $*" | tee -a $LOG; }
[[ $J =~ ^q7_[a-z0-9_]+_v2x_j$ ]] || { echo "lanes name $J is not q7_<members>_v2x_j: stop"; exit 2; }
while [ ! -s $D ] && [ "$(date +%Y%m%d%H%M)" -lt "${WAIT_UNTIL:-0}" ]; do sleep 300; done
[ -s $D ] && grep -q '"mode": "full"' $JS && grep -q '"lane_off_passed": true' $JS && grep -q '"one_member_passed": true' $JS && ! grep -q '"stopped"' $JS \
  || { say "no full lane run with passed lane-off and one-member checks ($D / $JS): stop"; exit 2; }
[ "$(grep -cE "  $I/vehicle_(val_i1|test|fresh)\.json$" $M5)" = 3 ] && [ "$(grep -c . $M5)" = 3 ] \
  || { say "$M5 does not list exactly the 3 vehicle lanes: stop"; exit 2; }
md5sum -c --quiet $M5 || { say "lane md5 differs from $M5: stop"; exit 2; }
md5sum -c --quiet - <<MD5 || { say "input md5 differs: stop"; exit 2; }
9d9bae46dfd7611b138081ec51666757  scripts/x6_refine.py
cc13b77f96eac9e999621a0dcac5f438  configs/pcdet/vrcnn_v2x.yaml
39ad88fe1bbca3c857464d7ccef9f6a5  output/pcdet/vrcnn_v2x/default/ckpt/checkpoint_epoch_20.pth
MD5
ls $O/*_ref.json > /dev/null 2>&1 && { say "$O already holds refines: stop (no overwrite)"; exit 2; }
mkdir -p $O
say "md5 checks passed; GPU $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader)"
fr() { case $1 in val_i1) echo data/pcdet/v2x_real_50k/points ;; test) echo data/kit/v2x/detection_test_frames ;; fresh) echo data/pcdet/v2x_fresh_50k/points ;; esac; }
ref() {  # $1 config stem, $2 class, $3 tag, $4 split
  $R python scripts/x6_refine.py --cfg configs/pcdet/$1.yaml --ckpt output/pcdet/$1/default/ckpt/checkpoint_epoch_20.pth \
      --frames $(fr $4) --pred $I/$2_$4.json --out $O/$2_$3_$4_ref.json --only_class $2 \
      >> logs/${J}_refine_x6.log 2>&1 || { say "refine failed ($4 $2 $3): stop"; exit 3; }
}
for s in val_i1 test fresh; do
  ref vrcnn_v2x vehicle x6 $s
  say "$s done"
done
md5sum $O/*_ref.json > logs/${J}_refine.md5
touch $O/done
say "done: $(wc -l < logs/${J}_refine.md5) files"; echo ok > logs/${J}_refine.done
