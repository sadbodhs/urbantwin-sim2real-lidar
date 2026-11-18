"""X2 (V2X walker PointPillars, pp_v2x_walk) in the v12/v13 chain, OS1-64 val (val_i1, halves), IoU 0.5 and 0.7.
Members: pp, cp, rider model (ped), all-frames pp (ppall), walkers (walk).
  vehicle     v13 weights (pp 1, cp 1.5, ped 0.5, ppall 0.5) + walk {0, 0.5, 1, 1.5}; walk replacing pp  (-> height refine)
  pedestrian  v12 weights (pp 1, cp 1, ped 0.5) + walk {0, 0.5, 1, 1.5, 2}; walk replacing pp
              (-> sensor shift, column test, low-point penalty, recentre)
walk 0 must reproduce v13 vehicle 0.7455 and v12 pedestrian 0.1746.
    python scripts/x2_fuse.py [--workers 8]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from utl import column_test, height_refine, metrics
from utl.ensemble import ensemble

_S = {}


def chain(cls, w, members, pts, one_per_member=False):
    if cls == "vehicle":
        ens = ensemble(members, w, iou_thr=0.55, class_weights={"vehicle": w}, classes={"vehicle"}, one_per_member=one_per_member)
        return {f: height_refine.refine(v, pts[f], pct=99.9) for f, v in ens.items()}
    from utl.v2x_sweep import low_point_penalty, shift_toward_sensor
    from v2x_sweep_r5 import recentre
    ens = ensemble(members, w, iou_thr=0.55, class_weights={"pedestrian": w}, classes={"pedestrian"}, one_per_member=one_per_member)
    out = {}
    for f, dets in ens.items():
        d = shift_toward_sensor(dets, "pedestrian", 0.05)
        d = column_test.apply(d, pts[f], {"pedestrian"}, 3, 0.1)
        d = low_point_penalty(d, pts[f], 4)
        out[f] = recentre(d, pts[f], 0.25)
    return out


def run(job):
    cls, w = job
    pred = chain(cls, w, _S["members"], _S["pts"])
    res = {}
    for s, ids in _S["halves"].items():
        r = metrics.detection_scores({f: pred.get(f, []) for f in ids}, {f: _S["gt"][f] for f in ids}, [cls], thrs=(0.5, 0.7))
        res[s] = [round(r[0.5]["per_class"][cls]["ap40"], 4), round(r[0.7]["per_class"][cls]["ap40"], 4)]
    out = {"class": cls, "w(pp,cp,ped,ppall,walk)": w, **res}
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from utl.evaluate import load_gt
    ROOT = Path("data/pcdet/v2x_real")
    i1 = sorted(l.strip() for l in open(ROOT / "ImageSets/val_i1.txt") if l.strip())
    gt_all = load_gt(ROOT, "val", ["vehicle", "pedestrian", "truck"])
    I = Path("output/v2x/sweep_in")
    srcs = [I / f"{n}_val50k.json" for n in ("pp", "cp", "ped", "ppall")] + [Path("output/v2x/pp_v2x_walk/kit_mc/pred_val50k.json")]
    members = [{f: m.get(f, []) for f in i1} for m in (json.loads(p.read_text()) for p in srcs)]
    pts = {}
    for f in i1:
        p = np.load(Path("data/pcdet/v2x_real_50k/points") / f"{f}.npy")[:, :3]
        pts[f] = p[np.abs(p).sum(1) > 0]
    _S.update(gt={f: gt_all[f] for f in i1}, members=members, pts=pts,
              halves={"all": i1, "h1": i1[: len(i1) // 2], "h2": i1[len(i1) // 2:]})
    jobs = [("vehicle", [1.0, 1.5, 0.5, 0.5, x]) for x in (0.0, 0.5, 1.0, 1.5)] + [("vehicle", [0.0, 1.5, 0.5, 0.5, 1.0])]
    jobs += [("pedestrian", [1.0, 1.0, 0.5, 0.0, x]) for x in (0.0, 0.5, 1.0, 1.5, 2.0)] + [("pedestrian", [0.0, 1.0, 0.5, 0.0, 1.0])]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, jobs))
    Path("output/v2x/x2_fuse.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
