"""V2X plan, steps 1 and 2. Base v2x_v25; settings fixed before any reading; val-informed (the size values come from OS1-64 val_i1 medians); no fresh set on V2X.
Step 1, fixed size corrections on the final boxes of a class:
  vehicle     height +0.19 m with the bottom kept, length -0.125 m, width +0.067 m.
  pedestrian  bottom +0.038 m with the height kept, length -0.044 m, width -0.019 m.
  truck       length +0.67 m, bottom +0.12 m with the height kept.
Step 2, equal-weight averaging of second stages on the same input boxes (mean of the pass-1 boxes that moved <= the class cap, doubled-angle heading; then the class's v25 variant):
  vehicle     X6 (current) + X6b + negatives model, on the X7 boxes; variant c3 on the averaged box (centre, bottom, height; score unchanged), cap 1 m; rule X8.
  truck       X6b (current) + negatives model, on the v18 trucks; c12 (score sqrt(score x mean iou1)), cap 1 m, width >= 2.55 m.
  pedestrian  X6b + negatives model (v25) + X6b all-frames + negatives epochs 15 and 17 (from the one-cycle schedule: about 37% and 15% of the peak rate, as epochs 11 and 13 of 15 on LUMPI), on the x6bf boxes; c12, cap 0.5 m; rules X1 and X3.
Class gate: at least +0.003 on both val_i1 halves, IoU 0.7 not down, against v25. Candidate gate for a whole file: val mAP at least +0.004 against v25.
Each step is read on v25 alone first; for a class where both pass, one more reading stacked (averaging, then the size correction); the stack is kept only if it beats each step alone on both halves, else the step with the higher val AP."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import v2x_v20_rules as R
from utl import metrics

ROOT = Path("data/pcdet/v2x_real")
E = Path("output/v2x/ens")
X6 = Path("output/v2x/vrcnn_v2x/x6c")
XB = Path("output/v2x/vrcnn_v2x_b/x6bc")
NG = Path("output/v2x/vrcnn_v2x_neg/s2")
S2 = Path("output/v2x/plan_s2")
CL = ["vehicle", "pedestrian", "truck"]
V25 = {"vehicle": 0.8177, "pedestrian": 0.3240, "truck": 0.6361}
CAP = {"vehicle": 1.0, "pedestrian": 0.5, "truck": 1.0}
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}
frames = sorted(l.strip() for l in open(ROOT / "ImageSets/val_i1.txt") if l.strip())
halves = {"all": frames, "h1": frames[: len(frames) // 2], "h2": frames[len(frames) // 2:]}
_gt = {}


def gt():
    if not _gt:
        for i in frames:
            rows = [l.split() for l in (ROOT / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
            _gt[i] = [([float(x) for x in r[:7]], r[7]) for r in rows]
    return _gt


def score(p, cls):
    g = gt()
    row = {}
    for h, ids in halves.items():
        s = metrics.detection_scores({f: [d for d in p.get(f, []) if d["class"] == cls] for f in ids},
                                     {f: [x for x in g[f] if x[1] == cls] for f in ids}, [cls], thrs=(0.5, 0.7))
        row[h] = [round(s[0.5]["per_class"][cls]["ap40"], 4), round(s[0.7]["per_class"][cls]["ap40"], 4)]
    return row


def class_gate(r, ref):
    return all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1]


def avg(ds, cls):
    """Equal-weight average of second stages on one input box; returns (box, mean iou) or None if any lacks a pass-1 box."""
    d0 = ds[0]
    if any("ref1" not in d for d in ds):
        return None
    iou = float(np.mean([max(d["iou1"], 0.0) for d in ds]))
    near = [d["ref1"] for d in ds if np.hypot(d["ref1"][0] - d0["box"][0], d["ref1"][1] - d0["box"][1]) <= CAP[cls]]
    if not near:
        return list(d0["box"]), iou
    a = np.array(near, float)
    box = a.mean(0).tolist()
    t = 0.5 * np.arctan2(np.sin(2 * a[:, 6]).mean(), np.cos(2 * a[:, 6]).mean())
    box[6] = float(t + np.round((a[0, 6] - t) / np.pi) * np.pi)
    return box, iou


def finish_one(ds, cls):
    d0 = ds[0]
    r = avg(ds, cls)
    if r is None:
        return keep(d0)
    ref, iou = r
    b = list(d0["box"])
    if cls == "vehicle":
        if np.hypot(ref[0] - b[0], ref[1] - b[1]) <= CAP[cls]:
            bot = ref[2] - ref[5] / 2
            b = [ref[0], ref[1], bot + ref[5] / 2, b[3], b[4], ref[5], b[6]]
        return {"box": b, "score": d0["score"], "class": cls}
    if np.hypot(ref[0] - b[0], ref[1] - b[1]) <= CAP[cls]:
        b = list(ref)
    if cls == "truck":
        b = b[:4] + [max(b[4], 2.55)] + b[5:]
    return {"box": b, "score": float(np.sqrt(d0["score"] * iou)), "class": cls}


def build(paths, cls, kind):
    src = [{f: [d for d in v if d["class"] == cls] for f, v in json.loads(Path(p).read_text()).items()} for p in paths]
    out = {}
    for f in src[0]:
        rows = [s[f] for s in src]
        assert all(len(r) == len(rows[0]) for r in rows), f
        out[f] = []
        for tup in zip(*rows):
            assert all(t["box"] == tup[0]["box"] for t in tup)
            out[f].append(finish_one(list(tup), cls))
    rules = {"vehicle": ["X8"], "pedestrian": ["X1", "X3"], "truck": []}[cls]
    return R.apply(out, R.features(kind, out, sorted(out)), rules) if rules else out


def v25_paths(cls, s):
    return {"vehicle": [X6 / f"x7_{s}_ref.json"], "truck": [XB / f"v18_{s}_ref.json"],
            "pedestrian": [XB / f"x6bf_{s}_ref.json", NG / f"x6bf_{s}_ref.json"]}[cls]


def s2_paths(cls, s):
    return {"vehicle": [X6 / f"x7_{s}_ref.json", S2 / f"x6b_vehicle_{s}_ref.json", S2 / f"neg_vehicle_{s}_ref.json"],
            "truck": [XB / f"v18_{s}_ref.json", S2 / f"neg_truck_{s}_ref.json"],
            "pedestrian": [XB / f"x6bf_{s}_ref.json", NG / f"x6bf_{s}_ref.json", XB / f"x6bh_{s}_ref.json",
                           S2 / f"neg15_pedestrian_{s}_ref.json", S2 / f"neg17_pedestrian_{s}_ref.json"]}[cls]


def size(p, cls):
    def fix(b):
        x, y, z, l, w, h, t = b[:7]
        bot = z - h / 2
        if cls == "vehicle":
            l, w, h = l - 0.125, w + 0.067, h + 0.19
        elif cls == "pedestrian":
            bot, l, w = bot + 0.038, l - 0.044, w - 0.019
        else:
            l, bot = l + 0.67, bot + 0.12
        return [x, y, bot + h / 2, l, w, h, t] + list(b[7:])
    return {f: [{**d, "box": fix(d["box"])} if d["class"] == cls else d for d in v] for f, v in p.items()}
