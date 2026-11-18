"""L9 and L10 (LUMPI, label-free post-processing on the lumpi_v8 chain output).
L9  person vs two-wheeler by point footprint: for each Bicycle / Motorcycle box (score >= 0.05), the returns inside the
    footprint grown by 0.2 m in the wheel band 0.1-0.7 m above the box bottom; with >= 5 of them and an extent along the
    heading (p95 - p5) below E (walking legs span < ~0.8 m, a bicycle's wheels 1.6-1.8 m), add a Person box at the
    centroid of the box's returns (nuScenes size 0.73 x 0.66 x 1.76, bottom kept) at the two-wheeler score x a, unless a
    Person box with at least that score lies within 0.5 m, and multiply the two-wheeler score by b.
L10 car vs van by measured height (public classes: vans 1.9-2.8 m, cars 1.4-1.7 m, SUVs up to ~1.8 m): h = 99.9th
    percentile of return z inside the footprint minus the box bottom (>= 20 returns). Car box with h >= 2.0 m: add a Van
    copy at the Car score (Car untouched). Van box with h < 1.75 m: Van score x 0.3. One fixed setting.
Val: 50k real val (all / h1 / h2), IoU 0.5 and 0.7; test: firing rates only.
    python scripts/lumpi_l9_l10.py --workers 6
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from utl import metrics
from utl.height_refine import inside_footprint
from utl.tracks import TRACKS

CLS = TRACKS["lumpi"]["classes"]
VAL = Path("data/pcdet/lumpi_real_50k")
TEST = Path("data/kit/lumpi/detection_test_frames")
_S = {}


def load_pts(split, f):
    if split == "val":
        return np.load(VAL / "points" / f"{f}.npy")[:, :3]
    return np.fromfile(TEST / f"{f}.bin", np.float32).reshape(-1, 3)


def l9_frame(dets, P, E, a, b):
    out, added, fired, n2w = [], [], 0, 0
    persons = [d for d in dets if d["class"] == "Person"]
    for d in dets:
        if d["class"] in ("Bicycle", "Motorcycle") and d["score"] >= 0.05:
            bx = d["box"]
            m = inside_footprint(P, bx, 0.2)
            bot = bx[2] - bx[5] / 2
            Q = P[m & (P[:, 2] > bot + 0.1) & (P[:, 2] < bx[2] + bx[5] / 2)]
            W = Q[(Q[:, 2] < bot + 0.7)]
            if d["score"] >= 0.3:
                n2w += 1
            if len(W) >= 5:
                u = (W[:, 0] - bx[0]) * np.cos(bx[6]) + (W[:, 1] - bx[1]) * np.sin(bx[6])
                if np.percentile(u, 95) - np.percentile(u, 5) < E:
                    if d["score"] >= 0.3:
                        fired += 1
                    s = d["score"] * a
                    cx, cy = Q[:, 0].mean(), Q[:, 1].mean()
                    if a > 0 and not any(p["score"] >= s and np.hypot(p["box"][0] - cx, p["box"][1] - cy) < 0.5 for p in persons):
                        added.append({"box": [cx, cy, bot + 0.88, 0.73, 0.66, 1.76, bx[6]], "score": s, "class": "Person"})
                    d = {**d, "score": d["score"] * b}
        out.append(d)
    return sorted(out + added, key=lambda d: -d["score"]), fired, n2w


def l10_frame(dets, P):
    out, added, n_car, f_car, n_van, f_van = [], [], 0, 0, 0, 0
    for d in dets:
        if d["class"] in ("Car", "Van"):
            bx = d["box"]
            m = inside_footprint(P, bx, 0.0)
            bot = bx[2] - bx[5] / 2
            z = P[m, 2]
            z = z[z > bot - 0.3]
            conf = d["score"] >= 0.3
            if len(z) >= 20:
                h = np.percentile(z, 99.9) - bot
                if d["class"] == "Car" and h >= 2.0:
                    added.append({**d, "class": "Van"})
                    f_car += conf
                elif d["class"] == "Van" and h < 1.75:
                    d = {**d, "score": d["score"] * 0.3}
                    f_van += conf
            n_car += conf and d["class"] == "Car"
            n_van += conf and d["class"] == "Van"
        out.append(d)
    return sorted(out + added, key=lambda d: -d["score"]), (f_car, n_car, f_van, n_van)


def run(job):
    kind, args, split = job
    pred, frames = _S[split], _S[split + "_frames"]
    out, stats = {}, np.zeros(4)
    for f in frames:
        P = _S["pts"][split][f]
        if kind == "base":
            out[f] = pred.get(f, [])
        elif kind == "l9":
            out[f], fi, n = l9_frame(pred.get(f, []), P, *args)
            stats[:2] += (fi, n)
        else:
            out[f], st = l10_frame(pred.get(f, []), P)
            stats += st
    res = {"kind": kind, "args": args, "split": split}
    if kind == "l9":
        res["fired"] = f"{int(stats[0])}/{int(stats[1])} confident two-wheelers"
    elif kind == "l10":
        res["fired"] = f"Car->Van copies {int(stats[0])}/{int(stats[1])}, Van x0.3 {int(stats[2])}/{int(stats[3])} (confident)"
    if split == "val":
        for s, ids in _S["halves"].items():
            g = {f: _S["gt"][f] for f in ids}
            r = metrics.detection_scores({f: out[f] for f in ids}, g, CLS, thrs=(0.5, 0.7))
            res[s] = {t: {"mAP": round(r[t]["mAP40"], 4), **{c: round(r[t]["per_class"][c]["ap40"] or 0, 4)
                                                           for c in ("Person", "Bicycle", "Motorcycle", "Car", "Van")}} for t in (0.5, 0.7)}
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val", default="output/lumpi/ens_l7/cpvnp1.0_val_v7.json")
    ap.add_argument("--test", default="output/lumpi/ens_l7/cpvnp1.0_test_v7.json")
    ap.add_argument("--l9_only", type=str, default="", help="JSON list of [E, a, b] to run instead of the full grid")
    ap.add_argument("--out", default="output/lumpi/l9_l10.json")
    a = ap.parse_args()
    ids = lambda n: [l.strip() for l in open(VAL / "ImageSets" / f"{n}.txt") if l.strip()]
    val = json.loads(Path(a.val).read_text())
    test = json.loads(Path(a.test).read_text())
    gt = {}
    for f in ids("val"):
        rows = [l.split() for l in (VAL / "labels" / f"{f}.txt").read_text().splitlines() if l.strip()]
        gt[f] = [([float(x) for x in r[:7]], r[7]) for r in rows]
    _S.update(val=val, test=test, val_frames=ids("val"), test_frames=sorted(test), gt=gt,
              halves={"all": ids("val"), "h1": ids("val_h1"), "h2": ids("val_h2")},
              pts={"val": {f: load_pts("val", f) for f in ids("val")}, "test": {f: load_pts("test", f) for f in test}})
    if a.l9_only:
        jobs = [("l9", tuple(g), sp) for g in json.loads(a.l9_only) for sp in ("val", "test")]
    else:
        jobs = [("base", (), "val"), ("l10", (), "val"), ("l10", (), "test")]
        jobs += [("l9", (E, aa, bb), "val") for E in (0.6, 0.8, 1.0) for aa in (0.5, 1.0) for bb in (0.3, 1.0)]
        jobs += [("l9", (0.8, 1.0, 1.0), "test")]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, jobs))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
