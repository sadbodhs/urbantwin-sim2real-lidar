"""Person footprint on lumpi_v35: every Person box length + 0.044 m, width + 0.023 m; centre, height, heading and score unchanged; other classes exactly v35; no boxes added. Values are val medians of label minus box, so val-informed; fresh is the real check.
Gate against v35: Person >= +0.003 on both halves, IoU 0.7 not down, on val and then once on fresh. Fresh is read only if val matches the earlier reading and passes. Writes output/lumpi/ens_x/lumpi_v35_F_test.json if both pass.
    python scripts/lumpi_v35_footprint.py logs/fresh_ids.txt"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_static as S

O = Path("output/lumpi/ens_x")
L = Path("output/lumpi/vrcnn_lumpi_fix/l11c")
CLASSES = ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van"]
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}


def edit(p, dl, dw, dh):
    out = {}
    for f, v in p.items():
        out[f] = [({**d, "box": d["box"][:3] + [d["box"][3] + dl, d["box"][4] + dw, d["box"][5] + dh] + d["box"][6:]} if d["class"] == "Person" else d) for d in v]
    return out


height = lambda p: edit(p, 0.0, 0.0, 0.10)
foot = lambda p: edit(p, 0.044, 0.023, 0.0)


def c12(d):
    if "ref1" not in d:
        return keep(d)
    r = d["ref1"]
    b = list(r) if np.hypot(r[0] - d["box"][0], r[1] - d["box"][1]) <= 0.5 else list(d["box"])
    return {"box": b, "score": float(np.sqrt(d["score"] * max(d["iou1"], 0.0))), "class": d["class"]}


def row(p, gt, ids, windows=None):
    hv = {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}
    r = {}
    for h, fr in hv.items():
        a = S.ap(p, gt, fr, CLASSES)
        r[h] = [a[0.5]["Person"], a[0.7]["Person"]]
    if windows:
        r["windows"] = [S.ap(p, gt, w, CLASSES)[0.5]["Person"] for w in windows]
        r["w2_w3"] = S.ap(p, gt, windows[2] + windows[3], CLASSES)[0.5]["Person"]
    return r


ok = lambda r, ref: all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1]
ids = S.ids_of("val")
gt = S.labels(ids)
win = S.windows_of("val", sorted(ids))
v35 = height(json.loads((O / "lumpi_v33_plow_val.json").read_text()))
ref, rf = row(v35, gt, ids, win), row(foot(v35), gt, ids, win)
print("val v35", json.dumps(ref), flush=True)
print("val F", json.dumps(rf), "PASS" if ok(rf, ref) else "fail", flush=True)
same = rf["all"][0] == 0.4146 and rf["h1"][0] == 0.4623 and rf["h2"][0] == 0.3525 and rf["all"][1] == 0.0119 and rf["windows"] == [0.4907, 0.4324, 0.3316, 0.3754] and rf["w2_w3"] == 0.3525
print("matches the earlier reading:", same, flush=True)
if not same or not ok(rf, ref):
    print("STOP before fresh", flush=True)
    sys.exit(0)
del v35
fr = [l.strip() for l in open(sys.argv[1]) if l.strip()]
gf = S.labels(fr)
a = json.loads((L / "v29_fresh_ref.json").read_text())
m = json.loads((L / "v29_fresh_ref_min01.json").read_text())
p35 = height({f: [keep(d) for d in a[f] if d["class"] not in ("Person", "Bicycle")] + [c12(d) for d in m[f] if d["class"] == "Person"] for f in fr})
del a, m
rr, fp = row(p35, gf, fr), row(foot(p35), gf, fr)
pf = ok(fp, rr)
print("fresh v35", json.dumps(rr), flush=True)
print("fresh F", json.dumps(fp), "PASS" if pf else "fail", flush=True)
if pf:
    t = foot(json.loads((O / "lumpi_v34_H_test.json").read_text()))
    (O / "lumpi_v35_F_test.json").write_text(json.dumps(t))
    print("test written", sum(map(len, t.values())), flush=True)
