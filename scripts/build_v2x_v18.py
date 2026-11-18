"""Build v2x_v18 = v2x_v17 + V6 tall-object test on pedestrian boxes (scripts/ped_shape.py, a second pass after the V3
pole / V4 short tests: points in the footprint + 0.15 m spanning more than 2.3 m in height -> score x 0.3). Val: OS1-64
val_i1 pedestrians (v16's chain on the v14 val boxes: 0.2323 -> 0.2334, halves 0.1856 -> 0.1866, 0.2763 ->
0.2775). Test: output/v2x/ens/v17_test.json -> v18_test.json; only pedestrian scores change."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np

import ped_shape as PS
from utl import metrics

ROOT = Path("data/pcdet/v2x_real")
CLS, T = "pedestrian", 2.3


def pts(split, f):
    if split == "val":
        return np.load(Path("data/pcdet/v2x_real_50k/points") / f"{f}.npy")[:, :3]
    return np.fromfile(Path("data/kit/v2x/detection_test_frames") / f"{f}.bin", np.float32).reshape(-1, 3)


i1 = sorted(l.strip() for l in open(ROOT / "ImageSets/val_i1.txt") if l.strip())
halves = {"all": i1, "h1": i1[: len(i1) // 2], "h2": i1[len(i1) // 2:]}
gt = {}
for i in i1:
    rows = [l.split() for l in (ROOT / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
    gt[i] = [([float(v) for v in r[:7]], r[7]) for r in rows if r[7] == CLS]
pv = json.loads(Path("output/v2x/ens/v14_val_i1.json").read_text())
pv = {f: [d for d in pv.get(f, []) if d["class"] == CLS] for f in i1}
v16 = PS.apply(pv, lambda f: pts("val", f), CLS, pole=True, short=1.0)
v18 = PS.apply(v16, lambda f: pts("val", f), CLS, tall=T)
for name, p in (("v16/v17", v16), ("v18", v18)):
    row = {}
    for h, ids in halves.items():
        s = metrics.detection_scores({f: p[f] for f in ids}, {f: gt[f] for f in ids}, [CLS], thrs=(0.5, 0.7))
        row[h] = [round(s[0.5]["per_class"][CLS]["ap40"], 4), round(s[0.7]["per_class"][CLS]["ap40"], 4)]
    print(name, "pedestrian", json.dumps(row), flush=True)
t17 = json.loads(Path("output/v2x/ens/v17_test.json").read_text())
t18 = PS.apply(t17, lambda f: pts("test", f), CLS, tall=T)
for f in t17:
    for a, b in zip(t17[f], t18[f]):
        assert a["box"] == b["box"] and a["class"] == b["class"] and (a["score"] == b["score"] or a["class"] == CLS)
Path("output/v2x/ens/v18_test.json").write_text(json.dumps(t18))
n = sum(a["class"] == CLS and b["score"] < a["score"] for f in t17 for a, b in zip(t17[f], t18[f]))
nc = sum(a["class"] == CLS and a["score"] >= 0.3 and b["score"] < a["score"] for f in t17 for a, b in zip(t17[f], t18[f]))
print(f"v18 test written: {n} pedestrian boxes demoted ({nc} confident)", flush=True)
