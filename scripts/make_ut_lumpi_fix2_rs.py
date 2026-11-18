"""ut_lumpi_fix2_rs: ut_lumpi_fix2 with L11-RS (c)'s real-size Car, Van and Truck (same recipe, real-size vehicles;
for b1_l12_s2 / b1_pv_s2, which train on ut_lumpi_fix2).
Exactly L11-RS (c)'s vehicle sizes, not a re-fit: every Car / Van / Truck box of ut_lumpi_fix2 is matched to the same box of ut_lumpi_fix (same
frame, class, centre, size and yaw within 1e-3), and its target size is the size of that box (same index) in ut_lumpi_rs, which
scripts/make_ut_lumpi_rs.py wrote from rs_mapping.json (realism_fit labels of 3 m or longer for Car / Van / Truck, seed 20261109, per-object u by
(frame, box index in ut_lumpi_fix)). Box and inside points are rescaled about the bottom centre with utl.objfix.rescale, as make_ut_lumpi_rs.py.
Everything else is ut_lumpi_fix2 as it is: Bicycle (rider_v2 boxes; the 1.78 x 0.84 x 1.78 m floor and TWO_WHEEL_NEG are config settings), riders,
Person, Bus, Motorcycle, Unknown. Any vehicle box without exactly one match, or a match whose ut_lumpi_rs box is not the same object, stops the run.
data/pcdet/ut_lumpi_fix2_rs/{points,labels,ImageSets} must be the full copy of ut_lumpi_fix2's. Writes points, labels and fix2_rs_mapping.json.
    python scripts/make_ut_lumpi_fix2_rs.py [--workers 8]"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

F2, FIX, RS = Path("data/pcdet/ut_lumpi_fix2"), Path("data/pcdet/ut_lumpi_fix"), Path("data/pcdet/ut_lumpi_rs")
OUT = Path("data/pcdet/ut_lumpi_fix2_rs")
CLS = ("Car", "Van", "Truck")
TOL = 1e-3


def read(p):
    rows = [l.split() for l in p.read_text().splitlines() if l.strip()]
    return np.array([[float(x) for x in r[:7]] for r in rows], np.float32).reshape(-1, 7), [r[7] for r in rows]


def one(fid):
    b2, n2 = read(F2 / "labels" / f"{fid}.txt")
    bf, nf = read(FIX / "labels" / f"{fid}.txt")
    br, nr = read(RS / "labels" / f"{fid}.txt")
    assert nf == nr and len(bf) == len(br), f"{fid}: ut_lumpi_rs boxes do not line up with ut_lumpi_fix"
    tags, scales, rows = [], {}, []
    for k, (b, n) in enumerate(zip(b2, n2)):
        if n not in CLS:
            tags.append("-"); continue
        m = [j for j in range(len(bf)) if nf[j] == n and np.abs(bf[j] - b).max() < TOL]
        assert len(m) == 1, f"{fid} box {k} {n}: {len(m)} matches in ut_lumpi_fix"
        j = m[0]
        # the ut_lumpi_rs box is the same object: same x, y, yaw and bottom
        assert np.abs(br[j][[0, 1, 6]] - bf[j][[0, 1, 6]]).max() < TOL and abs((br[j][2] - br[j][5] / 2) - (bf[j][2] - bf[j][5] / 2)) < TOL, f"{fid} box {k}"
        t = br[j][3:6].astype(np.float64)
        tags.append(f"B{k}"); scales[f"B{k}"] = tuple(t / b[3:6])
        rows.append((n, list(map(float, b[3:6])), list(map(float, t))))
    from utl import objfix
    from utl.prepare import write_labels
    pts = np.load(F2 / "points" / f"{fid}.npy")
    boxes = b2
    if scales:
        pts, boxes = objfix.rescale(pts, b2, np.array(tags), scales)
    np.save(OUT / "points" / f"{fid}.npy", pts)
    write_labels(OUT / "labels" / f"{fid}.txt", boxes, n2)
    other = [k for k, n in enumerate(n2) if n not in CLS]
    same = bool(np.array_equal(boxes[other], b2[other])) if other else True
    return fid, rows, len(n2), same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ids = sorted(p.stem for p in (F2 / "labels").glob("*.txt"))
    assert ids == sorted(p.stem for p in (FIX / "labels").glob("*.txt")) == sorted(p.stem for p in (RS / "labels").glob("*.txt"))
    for d in ("points", "labels"):
        assert (OUT / d).is_dir() and len(list((OUT / d).iterdir())) == len(ids), f"{OUT / d} is not the full copy of ut_lumpi_fix2"
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, ids, chunksize=32))
    assert all(r[3] for r in res), "a non-vehicle box changed"
    log = {"source": str(F2), "sizes_from": str(RS), "matched_via": str(FIX), "classes": list(CLS), "frames": len(ids),
           "boxes_total": sum(r[2] for r in res), "rs_mapping": json.loads((RS / "rs_mapping.json").read_text()), "per_class": {}}
    Q = [0.1, 0.5, 0.9]
    for c in CLS:
        s = np.array([x[1] for _, v, _, _ in res for x in v if x[0] == c])
        t = np.array([x[2] for _, v, _, _ in res for x in v if x[0] == c])
        log["per_class"][c] = {"objects": len(t), "fix2_q10_50_90": np.round(np.quantile(s, Q, axis=0), 3).tolist(),
                               "out_q10_50_90": np.round(np.quantile(t, Q, axis=0), 3).tolist()}
        print(c, json.dumps(log["per_class"][c]), flush=True)
    (OUT / "fix2_rs_mapping.json").write_text(json.dumps(log, indent=1))
    print("wrote", OUT, "frames", len(ids), "boxes", log["boxes_total"], flush=True)


if __name__ == "__main__":
    main()
