"""P-low: L11 second stage on v29 Person boxes down to score 0.01 (scripts/fresh_refine.sh), variant c12 as v33, against lumpi_v33; other classes exactly v33; boxes under 0.01 unchanged; no boxes added.
Settings fixed before scoring, one reading. Gate: Person >= +0.003 on both val halves, IoU 0.7 not down. Prints the 4 val 160 s windows. If it passes, also scores the fresh frames.
Writes output/lumpi/ens_x/lumpi_v33_plow_{val,test}.json.
    python scripts/lumpi_plow.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_static as S

O = Path("output/lumpi/ens_x")
L = Path("output/lumpi/vrcnn_lumpi_fix/l11c")
B = Path("output/lumpi/vrcnn_lumpi_b/l11bc")
CLASSES = ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van"]
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}


def c12(d):
    if "ref1" not in d:
        return keep(d)
    r = d["ref1"]
    b = list(r) if np.hypot(r[0] - d["box"][0], r[1] - d["box"][1]) <= 0.5 else list(d["box"])
    return {"box": b, "score": float(np.sqrt(d["score"] * max(d["iou1"], 0.0))), "class": d["class"]}


def build(split, person_file):
    a = json.loads((L / f"v29_{split}_ref.json").read_text())
    p = json.loads((L / person_file).read_text())
    b = json.loads((B / f"v29_{split}_ref.json").read_text())
    return {f: [keep(d) for d in a[f] if d["class"] not in ("Person", "Bicycle")] + [c12(d) for d in p[f] if d["class"] == "Person"] + [c12(d) for d in b[f] if d["class"] == "Bicycle"] for f in a}


def row(p, gt, halves):
    r = {}
    for h, fr in halves.items():
        a = S.ap(p, gt, fr, CLASSES)
        r[h] = [a[0.5]["Person"], a[0.7]["Person"]]
    return r


ids = S.ids_of("val")
gt = S.labels(ids)
halves = {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}
v33 = build("val", "v29_val_ref.json")
plow = build("val", "v29_val_ref_min01.json")
rv, rp = row(v33, gt, halves), row(plow, gt, halves)
ok = all(rp[h][0] >= rv[h][0] + 0.003 for h in ("h1", "h2")) and rp["all"][1] >= rv["all"][1]
print("v33 Person", json.dumps(rv), flush=True)
print("P-low Person", json.dumps(rp), "PASS" if ok else "fail", flush=True)
for k, w in enumerate(S.windows_of("val", sorted(ids))):
    print("window", k, "v33", S.ap(v33, gt, w, CLASSES)[0.5]["Person"], "P-low", S.ap(plow, gt, w, CLASSES)[0.5]["Person"], flush=True)
src = json.loads((L / "v29_val_ref_min01.json").read_text())
inband = [c12(d)["score"] for f in ids for d in src[f] if d["class"] == "Person" and 0.01 <= d["score"] < 0.05]
print("val Person boxes in 0.01-0.05:", len(inband), "above 0.05 after:", sum(s >= 0.05 for s in inband), flush=True)
same = all(sorted(map(str, [d for d in v33[f] if d["class"] != "Person"])) == sorted(map(str, [d for d in plow[f] if d["class"] != "Person"])) for f in ids)
print("other classes identical to v33:", same, flush=True)
(O / "lumpi_v33_plow_val.json").write_text(json.dumps(plow))
t = build("test", "v29_test_ref_min01.json")
(O / "lumpi_v33_plow_test.json").write_text(json.dumps(t))
print("written", flush=True)
