"""Three score rules on lumpi_v36, each score x 0.3; no boxes added or moved; other classes and realism frames unchanged. T1: Truck with h < 2.0 m. V2: Van with h < 2.0 m. P2: Person with l >= 0.85 m on v36 sizes. Val-informed: thresholds set after val size distributions; 3 of 7 rules read on val were kept.
Val first, which must match the earlier reading, then one fresh reading of each. Truck and Person gate: >= +0.003 on both halves, IoU 0.7 not down. Van gate (looser, because h1 holds only about 15 real Vans): up on both halves, windows 2 and 3 each up, IoU 0.7 not down. Writes output/lumpi/ens_x/lumpi_v37_test.json with the rules that pass fresh.
    python scripts/lumpi_v36_rules.py logs/fresh_ids.txt"""
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
RULES = {
    "T1": ("Truck", lambda d: d["box"][5] < 2.0),
    "V2": ("Van", lambda d: d["box"][5] < 2.0),
    "P2": ("Person", lambda d: d["box"][3] >= 0.85),
}
EXPECT = {
    "T1": ([0.4561, 0.2508, 0.5781, 0.0368], [0.1449, 0.3603, 0.5817, 0.5893]),
    "V2": ([0.1672, 0.0010, 0.2652, 0.0539], [0.0000, 0.0058, 0.3628, 0.1358]),
    "P2": ([0.4257, 0.4730, 0.3631, 0.0122], [0.4979, 0.4462, 0.3446, 0.3836]),
}


def size(p):
    out = {}
    for f, v in p.items():
        out[f] = [({**d, "box": d["box"][:3] + [d["box"][3] + 0.044, d["box"][4] + 0.023, d["box"][5] + 0.10] + d["box"][6:]} if d["class"] == "Person" else d) for d in v]
    return out


def apply(p, names):
    out = {}
    for f, v in p.items():
        dd = []
        for d in v:
            for n in names:
                c, t = RULES[n]
                if d["class"] == c and t(d):
                    d = {**d, "score": d["score"] * 0.3}
            dd.append(d)
        out[f] = dd
    return out


def c12(d):
    if "ref1" not in d:
        return keep(d)
    r = d["ref1"]
    b = list(r) if np.hypot(r[0] - d["box"][0], r[1] - d["box"][1]) <= 0.5 else list(d["box"])
    return {"box": b, "score": float(np.sqrt(d["score"] * max(d["iou1"], 0.0))), "class": d["class"]}


def row(p, gt, ids, c):
    hv = {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}
    r = {}
    for h, fr in hv.items():
        a = S.ap(p, gt, fr, CLASSES)
        r[h] = [a[0.5][c], a[0.7][c]]
    w = [sorted(ids)[i:i + 80] for i in range(0, len(ids), 80)]
    r["windows"] = [S.ap(p, gt, x, CLASSES)[0.5][c] for x in w]
    return r


def gate(n, r, ref):
    if n == "V2":
        return r["h1"][0] > ref["h1"][0] and r["h2"][0] > ref["h2"][0] and r["windows"][2] > ref["windows"][2] and r["windows"][3] > ref["windows"][3] and r["all"][1] >= ref["all"][1]
    return all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1]


ids = S.ids_of("val")
gt = S.labels(ids)
v36 = size(json.loads((O / "lumpi_v33_plow_val.json").read_text()))
same = True
for n, (c, _) in RULES.items():
    ref, r = row(v36, gt, ids, c), row(apply(v36, [n]), gt, ids, c)
    e, w = EXPECT[n]
    m = [r["all"][0], r["h1"][0], r["h2"][0], r["all"][1]] == e and r["windows"] == w
    same = same and m
    print("val", n, c, "v36", json.dumps(ref), flush=True)
    print("val", n, c, "new", json.dumps(r), "PASS" if gate(n, r, ref) else "fail", "matches" if m else "DOES NOT MATCH", flush=True)
if not same:
    print("STOP before fresh", flush=True)
    sys.exit(0)
del v36
fr = [l.strip() for l in open(sys.argv[1]) if l.strip()]
gf = S.labels(fr)
a = json.loads((L / "v29_fresh_ref.json").read_text())
m = json.loads((L / "v29_fresh_ref_min01.json").read_text())
p36 = size({f: [keep(d) for d in a[f] if d["class"] not in ("Person", "Bicycle")] + [c12(d) for d in m[f] if d["class"] == "Person"] for f in fr})
del a, m
kept = []
for n, (c, _) in RULES.items():
    ref, r = row(p36, gf, fr, c), row(apply(p36, [n]), gf, fr, c)
    g = gate(n, r, ref)
    print("fresh", n, c, "v36", json.dumps(ref), flush=True)
    print("fresh", n, c, "new", json.dumps(r), "PASS" if g else "fail", flush=True)
    if g:
        kept.append(n)
print("kept", kept, flush=True)
if kept:
    t = json.loads((O / "lumpi_v35_F_test.json").read_text())
    t2 = apply(t, kept)
    (O / "lumpi_v37_test.json").write_text(json.dumps(t2))
    for n in kept:
        c, f = RULES[n]
        print("test", n, "demoted", sum(d["class"] == c and f(d) for v in t.values() for d in v), "confident", sum(d["class"] == c and f(d) and d["score"] >= 0.3 for v in t.values() for d in v), flush=True)
