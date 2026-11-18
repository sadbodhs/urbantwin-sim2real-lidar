"""V2X post-processing sweep on the 50k-point validation copy, scored separately per roadside sensor: model fusion
weight and IoU, height-refinement percentile, pedestrian shift toward the sensor, column test on/off, (round 3) a
third fused model, (round 4) rank-normalised fusion and a pedestrian point-count penalty. A setting is worth keeping
only if it helps on both sensors.

    python -m utl.v2x_sweep --pp pp_val50k.json --cp cp_val50k.json --frames data/pcdet/v2x_real_50k/points --out sweep.json
"""
import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import column_test, height_refine, metrics
from .ensemble import ensemble
from .evaluate import load_gt
from .tracks import TRACKS

CLS = TRACKS["v2x"]["classes"]
_S = {}


def shift_toward_sensor(dets, cls, d):
    if not d:
        return dets
    out = []
    for x in dets:
        if x["class"] == cls:
            b = list(x["box"])
            r = np.hypot(b[0], b[1])
            if r > 1e-3:
                b[0] -= d * b[0] / r
                b[1] -= d * b[1] / r
            x = {**x, "box": b}
        out.append(x)
    return out


def rank_maps(src, ref):
    """Per class, the sorted scores of a model (src) and of the reference model (ref) over all frames: a score is
    mapped to the reference score at the same quantile (label-free), so differently scaled models fuse fairly."""
    out = {}
    for c in CLS:
        a = np.sort([d["score"] for v in src.values() for d in v if d["class"] == c])
        b = np.sort([d["score"] for v in ref.values() for d in v if d["class"] == c])
        if len(a) and len(b):
            out[c] = (a, b)
    return out


def rank_normalise(pred, maps):
    out = {}
    for f, dets in pred.items():
        out[f] = []
        for d in dets:
            if d["class"] in maps:
                a, b = maps[d["class"]]
                q = np.searchsorted(a, d["score"], side="right") / len(a)
                d = {**d, "score": float(np.quantile(b, min(q, 1.0)))}
            out[f].append(d)
    return out


def low_point_penalty(dets, points, minpts, factor=0.3):
    """Pedestrian boxes with fewer than minpts points inside get their score x factor (label-free)."""
    if not minpts:
        return dets
    out = []
    for d in dets:
        if d["class"] == "pedestrian":
            b = d["box"]
            m = height_refine.inside_footprint(points, b, margin=0.0)
            n = int(((points[m, 2] > b[2] - b[5] / 2) & (points[m, 2] < b[2] + b[5] / 2)).sum())
            if n < minpts:
                d = {**d, "score": d["score"] * factor}
        out.append(d)
    return out


def pipeline(pp, cp, pts, w_cp, iou, pct, shift, col, w3=0.0, m3_cls="all", rank=False, truck_m3=False, minpts=0,
             m3=None, maps=None):
    """w3 > 0 adds a third model m3 to the fusion, for every class but truck (m3_cls 'all') or for pedestrians only.
    rank: scores of cp and m3 rank-normalised to pp's per class first (maps from rank_maps, fitted on these predictions
    unless given); truck_m3: trucks from pp fused with m3 instead of pp alone; minpts: low_point_penalty."""
    if rank:
        maps = maps or {"cp": rank_maps(cp, pp), "m3": rank_maps(m3, pp) if m3 is not None else {}}
        cp = rank_normalise(cp, maps["cp"])
        m3 = rank_normalise(m3, maps["m3"]) if m3 is not None else None
    ens = ensemble([pp, cp], [1.0, w_cp], iou_thr=iou)
    ens3 = ensemble([pp, cp, m3], [1.0, w_cp, w3], iou_thr=iou) if w3 else ens
    trucks = ensemble([pp, m3], [1.0, w3 or 0.5], iou_thr=iou) if truck_m3 else pp
    out = {}
    for f in pp:
        dets = [d for d in (ens3 if m3_cls == "all" else ens)[f] if d["class"] not in ("truck", "pedestrian")]
        dets += [d for d in ens3[f] if d["class"] == "pedestrian"] + [d for d in trucks[f] if d["class"] == "truck"]
        dets = height_refine.refine(dets, pts[f], pct=pct)
        dets = shift_toward_sensor(dets, "pedestrian", shift)
        if col:
            dets = column_test.apply(dets, pts[f], {"pedestrian"}, 3, 0.1)
        out[f] = sorted(low_point_penalty(dets, pts[f], minpts), key=lambda d: -d["score"])
    return out


def run(cfg):
    pred = pipeline(_S["pp"], _S["cp"], _S["pts"], *cfg, m3=_S.get("m3"))
    res = {}
    for s, g in _S["gts"].items():
        r = metrics.detection_scores({f: pred[f] for f in g}, g, CLS, thrs=(0.5,))[0.5]
        res[s] = {"mAP": r["mAP40"], **{c: r["per_class"][c]["ap40"] for c in CLS}}
    print(json.dumps({"cfg": cfg, **{s: round(v["mAP"], 4) for s, v in res.items()}}), flush=True)
    return {"cfg": cfg, "res": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp", required=True)
    ap.add_argument("--cp", required=True)
    ap.add_argument("--m3", help="third model's predictions (round 3)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--root", default="data/pcdet/v2x_real")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    pp, cp = json.loads(Path(a.pp).read_text()), json.loads(Path(a.cp).read_text())
    pts = {}
    for f in pp:
        p = np.load(Path(a.frames) / f"{f}.npy")[:, :3]
        pts[f] = p[np.abs(p).sum(1) > 0]
    gt = load_gt(a.root, "val", CLS)
    gts = {s: {f: v for f, v in gt.items() if f in {l.strip() for l in open(Path(a.root) / "ImageSets" / f"{s}.txt")}}
           for s in ("val_i1", "val_i2")}
    _S.update(pp=pp, cp=cp, pts=pts, gts=gts)
    if a.m3:
        _S["m3"] = json.loads(Path(a.m3).read_text())
    ap_round = a.round
    if ap_round == 1:
        grid = [(w, iou, 98, 0.0, True) for w, iou in itertools.product((0.5, 1.0), (0.5, 0.55, 0.6))]
        grid += [(1.0, 0.55, p, s, c) for p, s, c in itertools.product((95, 99.5), (0.0,), (True,))]
        grid += [(1.0, 0.55, 98, s, True) for s in (0.05, 0.1)]
        grid += [(1.0, 0.55, 98, 0.0, False)]
    elif ap_round == 2:  # combine the round-1 winners (taller boxes, pedestrian shift)
        grid = [(1.0, 0.55, p, s, True) for p, s in itertools.product((99.5, 99.9, 100), (0.03, 0.05, 0.07))]
    elif ap_round == 3:  # plan item E2: third model on the v2x_v7 settings
        grid = [(1.0, 0.55, 99.9, 0.05, True)]
        grid += [(1.0, 0.55, 99.9, 0.05, True, w, c) for w, c in itertools.product((0.5, 1.0), ("all", "pedestrian"))]
    else:  # round 4, on v2x_v8: rank-normalised fusion, third-model weight, trucks with m3, pedestrian point penalty
        v8 = (1.0, 0.55, 99.9, 0.05, True, 0.5, "all")
        grid = [v8, v8[:7] + (True,), v8[:7] + (False, True), v8[:7] + (False, False, 2), v8[:7] + (False, False, 4),
                v8[:5] + (0.25, "all"), v8[:5] + (0.75, "all"), v8[:5] + (1.0, "all", True)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, grid))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
