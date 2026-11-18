#!/usr/bin/env bash
# ut_lumpi_fix2_rs prep: the same recipe as ut_lumpi_fix2 with real-size vehicles, for the fix2-based
# models. CPU only (no GPU). Full copy of ut_lumpi_fix2's points, labels and ImageSets,
# then L11-RS (c)'s Car / Van / Truck sizes (scripts/make_ut_lumpi_fix2_rs.py), infos, the GT database (db_every 3) and the long-truck copies, as
# ut_lumpi_fix2 was made (scripts/make_ut_lumpi_fix2.sh). ut_lumpi_fix2 is only read. Marker PREP_DONE on success.
#   setsid nohup scripts/make_ut_lumpi_fix2_rs.sh > logs/fix2_rs_prep.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
export UT_NOGPU=1 OMP_NUM_THREADS=1
R=scripts/docker_run.sh
S=data/pcdet/ut_lumpi_fix2 D=data/pcdet/ut_lumpi_fix2_rs
[ -f data/pcdet/ut_lumpi_rs/PREP_DONE ]
[ ! -e $D ] || { echo "$D exists; stop"; exit 3; }
mkdir $D
echo "== copy $(date)"
cp -r $S/ImageSets $S/labels $S/points $D/
[ "$(ls $D/points | wc -l)" = "$(ls $S/points | wc -l)" ] && [ "$(ls $D/labels | wc -l)" = "$(ls $S/labels | wc -l)" ]
echo "== resize Car / Van / Truck $(date)"
$R python scripts/make_ut_lumpi_fix2_rs.py --workers 8
echo "== infos and GT database $(date)"
$R python -m utl.prepare infos --root $D --track lumpi --gt_db --db_every 3
echo "== long-truck copies $(date)"
$R python -m utl.stretch_db --root $D
[ -s $D/infos_train.pkl ] && [ -s $D/dbinfos_train.pkl ] && [ -s $D/dbinfos_train_long.pkl ]
date '+%F %T' > $D/PREP_DONE
echo "== done $(date): $(ls $D/points | wc -l) frames"
