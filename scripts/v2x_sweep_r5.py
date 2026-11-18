"""V2X sweep round 5, on the v2x_v9 val predictions (output/v2x/diag/v9_val.json from scripts/diag_v2x_classes.py),
scored per roadside sensor. Label-free box fixes suggested by the class diagnostic:
  trucks   our boxes are about 6 m long against 8.3-9.6 m real: length refine (extend along the heading while the points
           continue, as on LUMPI) with gap 0.5 / 0.75 / 1.0 m, up to 16.5 m (EU limit); width at least 2.55 m (EU limit)
  peds     found but off by about 0.2 m: move the box centre a fraction `a` toward the centroid of the points in a 0.5 m
           cylinder around it (from 0.3 m above the box bottom to its top)
A setting is worth keeping only if it helps on both sensors.
    python scripts/v2x_sweep_r5.py --workers 3
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from utl import length_refine, metrics
from utl.evaluate import load_gt
from utl.tracks import TRACKS

CLS = TRACKS["v2x"]["classes"]
ROOT = Path("data/pcdet/v2x_real")
_S = {}


def recentre(dets, points, a, r=0.5):
    if not a:
        return dets
    out = []
    for d in dets:
        if d["class"] == "pedestrian":
            b = list(d["box"])
            bottom, top = b[2] - b[5] / 2, b[2] + b[5] / 2
            m = (np.hypot(points[:, 0] - b[0], points[:, 1] - b[1]) < r) & (points[:, 2] > bottom + 0.3) & (points[:, 2] < top)
            if m.sum() >= 3:
                c = points[m, :2].mean(0)
                b[0], b[1] = b[0] + a * (c[0] - b[0]), b[1] + a * (c[1] - b[1])
                d = {**d, "box": b}
        out.append(d)
    return out


def apply(pred, pts, gap=0.0, width=0.0, a=0.0):
    out = {}
    for f, dets in pred.items():
        if gap:
            dets = length_refine.refine(dets, pts[f], classes=("truck",), gap=gap, max_len=16.5)
        if width:
            dets = [{**d, "box": d["box"][:4] + [max(d["box"][4], width)] + d["box"][5:]} if d["class"] == "truck" else d
                    for d in dets]
        out[f] = recentre(dets, pts[f], a)
    return out


def run(cfg):
    pred = apply(_S["pred"], _S["pts"], **cfg)
    res = {}
    for s, g in _S["gts"].items():
        r = metrics.detection_scores({f: pred[f] for f in g}, g, CLS, thrs=(0.5,))[0.5]
        res[s] = {"mAP": round(r["mAP40"], 4), **{c: round(r["per_class"][c]["ap40"], 4) for c in CLS}}
    print(json.dumps({"cfg": cfg, **res}), flush=True)
    return {"cfg": cfg, "res": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="output/v2x/diag/v9_val.json")
    ap.add_argument("--frames", default="data/pcdet/v2x_real_50k/points")
    ap.add_argument("--out", default="output/v2x/v2x_sweep_r5.json")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    pred = json.loads(Path(a.pred).read_text())
    pts = {}
    for f in pred:
        p = np.load(Path(a.frames) / f"{f}.npy")[:, :3]
        pts[f] = p[np.abs(p).sum(1) > 0]
    gt = load_gt(ROOT, "val", CLS)
    gts = {s: {f: v for f, v in gt.items() if f in {l.strip() for l in open(ROOT / "ImageSets" / f"{s}.txt")}}
           for s in ("val_i1", "val_i2")}
    _S.update(pred=pred, pts=pts, gts=gts)
    grid = [{}, {"gap": 0.5}, {"gap": 0.75}, {"gap": 1.0}, {"width": 2.55}, {"gap": 0.75, "width": 2.55},
            {"a": 0.25}, {"a": 0.5}, {"a": 1.0}]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, grid))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
