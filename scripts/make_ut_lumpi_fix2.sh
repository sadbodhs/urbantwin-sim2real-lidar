#!/usr/bin/env bash
# UT-LUMPI with rider_v2 riders (utl.synth_ped: legs to the pedals, upright torso, arms to the
# handlebars; box grown to bike + rider) -> data/pcdet/ut_lumpi_fix2, then infos, the GT database and the long-truck
# copies, as ut_lumpi_fix was made. CPU only (no GPU).   nice -n 10 scripts/make_ut_lumpi_fix2.sh [workers]
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-4}
export UT_NOGPU=1 OMP_NUM_THREADS=1
R=scripts/docker_run.sh
$R python -m utl.prepare ut_fix --src data/pcdet/ut_lumpi --out data/pcdet/ut_lumpi_fix2 --seed 0 --workers "$W" --rider v2
$R python -m utl.prepare infos --root data/pcdet/ut_lumpi_fix2 --track lumpi --gt_db --db_every 3   # as ut_lumpi_fix (75k Car samples)
$R python -m utl.stretch_db --root data/pcdet/ut_lumpi_fix2
echo "done: $(ls data/pcdet/ut_lumpi_fix2/points | wc -l) frames"
