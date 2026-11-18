#!/usr/bin/env bash
# Build a Codabench zip and run the official kit's local scorer on it (CPU, Docker). The scorer runs in
# urbantwin-realism:cpu (docker/Dockerfile.realism) against data/kit_ref/<ref>; it checks the zip's format and gives the
# realism score (its detection score uses the local reference, not the hidden test labels).
#   scripts/package_check.sh lumpi output/lumpi/ens_l5/v6a_test_B_col3_len.json output/lumpi/realism_v2 lumpi_v6 [lumpi]
#   -> submissions/lumpi_v6/submission.zip and output/<track>/local_eval_lumpi_v6/local_eval.log
set -euo pipefail
cd "$(dirname "$0")/.."
TRACK=$1 PRED=$2 SYN=$3 VER=$4 REF=${5:-$1}
export UT_NOGPU=1
scripts/docker_run.sh python -m utl.package --track "$TRACK" --pred "$PRED" --synthetic "$SYN" \
    --out "submissions/$VER/submission.zip" --name "Sadbodh Sharma"
W=output/$TRACK/local_eval_$VER
mkdir -p "$W"
cp "submissions/$VER/submission.zip" "$W/"
cp -R "data/kit/$TRACK" "$W/kit"
UT_IMAGE=urbantwin-realism:cpu scripts/docker_run.sh bash -c \
    "cd $W && python -I kit/local_eval.py submission.zip /workspace/data/kit_ref/$REF" > "$W/local_eval.log" 2>&1 || true
echo "$VER: $(grep -o "valid_submission.: [0-9]" "$W/local_eval.log" | head -1) $(grep -o "score_realism.: [0-9.]*" "$W/local_eval.log" | head -1)"
