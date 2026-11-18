"""LUMPI static-object down-weight (label-free): the sensor is fixed, so a confident detection whose position recurs
(within RAD m) in >= FRAC of the other frames of its window is a parked object. On val no parked Bicycle detection sits
on a labelled bicycle (0 of 783), so LUMPI labels moving road users. Score x FACTOR for flagged boxes of the chosen
classes. Windows: val = its four 80-frame (160 s) runs, test = its 50 frames (one run, 006498-008100).
    python scripts/lumpi_static.py --study            # labelled share per class + val AP per class/factor, both halves
    python scripts/lumpi_static.py --pred P --split test --classes Bicycle Motorcycle --factor 0.1 --out O
"""
import argparse
import json
from pathlib import Path

import numpy as np

from utl.metrics import detection_scores

REAL = Path("data/pcdet/lumpi_real_50k")
CLASSES = ("Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van")


def ids_of(split):
    return [l.strip() for l in (REAL / "ImageSets" / f"{split}.txt").read_text().splitlines() if l.strip()]


def labels(ids):
    out = {}
    for i in ids:
        rows = [l.split() for l in (REAL / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
        out[i] = [(np.array(r[:7], float), r[7]) for r in rows]
    return out


def static_flags(pred, frames, cls, thr=0.3, frac=0.4, rad=0.5):
    """{frame: [bool per detection of pred[frame]]}: True for confident `cls` boxes at a recurring position."""
    pos = {f: np.array([d["box"][:2] for d in pred.get(f, []) if d["class"] == cls and d["score"] >= thr]).reshape(-1, 2)
           for f in frames}
    flags = {}
    for f in frames:
        fl, k = [], 0
        for d in pred.get(f, []):
            if d["class"] != cls or d["score"] < thr:
                fl.append(False)
                continue
            q = pos[f][k]
            k += 1
            hits = sum(1 for g, r in pos.items() if g != f and len(r) and (np.hypot(*(r - q).T) < rad).any())
            fl.append(hits >= frac * (len(frames) - 1))
        flags[f] = fl
    return flags


def apply(pred, windows, classes, factor, **kw):
    out = {f: [dict(d) for d in v] for f, v in pred.items()}
    for fr in windows:
        for c in classes:
            for f, fl in static_flags(pred, fr, c, **kw).items():
                for d, s in zip(out[f], fl):
                    if s:
                        d["score"] *= factor
    return out


def windows_of(split, frames):
    return [frames[i:i + 80] for i in range(0, len(frames), 80)] if split == "val" else [frames]


def ap(pred, gt, frames, classes):
    s = detection_scores({f: pred.get(f, []) for f in frames}, {f: gt[f] for f in frames}, list(classes),
                         thrs=(0.5, 0.7))
    return {t: {c: None if v["ap40"] is None else round(v["ap40"], 4) for c, v in s[t]["per_class"].items()} for t in s}


def study(pred_path):
    ids = ids_of("val")
    gt = labels(ids)
    pred = json.loads(Path(pred_path).read_text())
    wins = windows_of("val", ids)
    h = {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}
    for c in CLASSES:
        lab = tot = 0
        for fr in wins:
            for f, fl in static_flags(pred, fr, c).items():
                bb = np.array([b[:2] for b, k in gt[f] if k == c]).reshape(-1, 2)
                for d, s in zip(pred.get(f, []), fl):
                    if s:
                        tot += 1
                        lab += bool(len(bb) and (np.hypot(*(bb - d["box"][:2]).T) < 1.0).any())
        print(json.dumps({"class": c, "static_confident": tot, "on_labelled": lab}), flush=True)
    base = {k: ap(pred, gt, fr, CLASSES) for k, fr in h.items()}
    print(json.dumps({"base": base}), flush=True)
    for c in ("Bicycle", "Motorcycle", "Car", "Person", "Van", "Truck", "Bus"):
        for fac in (0.3, 0.1, 0.0):
            p = apply(pred, wins, [c], fac)
            r = {k: ap(p, gt, fr, [c]) for k, fr in h.items()}
            print(json.dumps({"class": c, "factor": fac,
                              "ap05": {k: (base[k][0.5][c], r[k][0.5][c]) for k in h},
                              "ap07": {k: (base[k][0.7][c], r[k][0.7][c]) for k in h}}), flush=True)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--study", action="store_true")
    ap_.add_argument("--pred", default="output/lumpi/ens_l8/l8all1.0_val_v9.json")
    ap_.add_argument("--split", default="val")
    ap_.add_argument("--classes", nargs="+", default=["Bicycle"])
    ap_.add_argument("--factor", type=float, default=0.1)
    ap_.add_argument("--out")
    a = ap_.parse_args()
    if a.study:
        return study(a.pred)
    pred = json.loads(Path(a.pred).read_text())
    frames = sorted(pred)
    out = apply(pred, windows_of(a.split, frames), a.classes, a.factor)
    Path(a.out).write_text(json.dumps(out))
    n = sum(d["score"] < o["score"] for f in out for d, o in zip(out[f], pred[f]))
    print(f"{a.split}: {n} boxes down-weighted in {len(frames)} frames ({', '.join(a.classes)} x {a.factor})")


if __name__ == "__main__":
    main()
