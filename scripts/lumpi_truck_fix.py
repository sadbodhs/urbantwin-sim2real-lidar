"""LUMPI big-vehicle box fix on the v14 chain (label-free, public priors): labelled real trucks are wider and taller
than ours (val labels median w 2.93 / h 3.43 m; our confident trucks w 2.32-2.46 / h 2.55-3.04). Variants: width at
least 2.55 m (EU legal width, as v2x_v10), point-based height (utl.height_refine, Truck/Bus only), both.
    python scripts/lumpi_truck_fix.py study
    python scripts/lumpi_truck_fix.py build VARIANT IN OUT SPLIT
"""
import json
import sys
from pathlib import Path

import numpy as np

from utl.height_refine import refine
from utl.metrics import detection_scores

REAL = Path("data/pcdet/lumpi_real_50k")
TEST = Path("data/kit/lumpi/detection_test_frames")
BIG = ("Truck", "Bus")


def pts(split, f):
    if split == "val":
        p = np.load(REAL / "points" / f"{f}.npy")[:, :3]
    else:
        p = np.fromfile(TEST / f"{f}.bin", np.float32).reshape(-1, 3)
    return p[np.abs(p).sum(1) > 0]


def fix(pred, split, variant, wmin=2.55):
    out = {}
    for f, dets in pred.items():
        dd = [dict(d, box=list(d["box"])) for d in dets]
        if "h" in variant:
            big = [d for d in dd if d["class"] in BIG]
            ref = iter(refine(big, pts(split, f)))
            dd = [next(ref) if d["class"] in BIG else d for d in dd]
        if "w" in variant:
            for d in dd:
                if d["class"] in BIG:
                    d["box"][4] = max(d["box"][4], wmin)
        out[f] = dd
    return out


def labels(ids):
    out = {}
    for i in ids:
        rows = [l.split() for l in (REAL / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
        out[i] = [(np.array(r[:7], float), r[7]) for r in rows]
    return out


def study():
    ids = [l.strip() for l in (REAL / "ImageSets/val.txt").read_text().splitlines() if l.strip()]
    gt = labels(ids)
    h = {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}
    pred = json.loads(Path("output/lumpi/ens_l8/l8all1.0_val_v9.json").read_text())
    for v in ("base", "w", "h", "wh"):
        p = pred if v == "base" else fix(pred, "val", v)
        r = {}
        for k, fr in h.items():
            s = detection_scores({f: p[f] for f in fr}, {f: gt[f] for f in fr}, list(BIG), thrs=(0.5, 0.7))
            r[k] = {t: {c: None if s[t]["per_class"][c]["ap40"] is None else round(s[t]["per_class"][c]["ap40"], 4)
                        for c in BIG} for t in s}
        print(json.dumps({"variant": v, **r}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "study":
        study()
    else:
        _, _, v, i, o, split = sys.argv
        Path(o).write_text(json.dumps(fix(json.loads(Path(i).read_text()), split, v)))
