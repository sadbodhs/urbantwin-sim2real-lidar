#!/usr/bin/env bash
# L11-RS data prep. CPU only (no GPU). data/pcdet/ut_lumpi_rs must be
# the full copy of ut_lumpi_fix's points, labels and ImageSets (cp -R data/pcdet/ut_lumpi_fix data/pcdet/ut_lumpi_rs). Then the resize (scripts/make_ut_lumpi_rs.py; extra options are
# passed through), infos, the GT database (db_every 3) and the long-truck copies, as ut_lumpi_fix was made. Marker PREP_DONE on success.
#   nohup scripts/make_ut_lumpi_rs.sh [make_ut_lumpi_rs.py options] > logs/l11rs_prep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export UT_NOGPU=1 OMP_NUM_THREADS=1
R=scripts/docker_run.sh
D=data/pcdet/ut_lumpi_rs
rm -f $D/PREP_DONE
echo "== resize $(date) options: ${*:-spec}"
$R python scripts/make_ut_lumpi_rs.py --workers 8 "$@"
echo "== infos and GT database $(date)"
$R python -m utl.prepare infos --root $D --track lumpi --gt_db --db_every 3
echo "== long-truck copies $(date)"
$R python -m utl.stretch_db --root $D
[ -s $D/infos_train.pkl ] && [ -s $D/dbinfos_train.pkl ] && [ -s $D/dbinfos_train_long.pkl ]
date '+%F %T' > $D/PREP_DONE
echo "== done $(date): $(ls $D/points | wc -l) frames"
