"""New-models phase 1 on LUMPI: PV-RCNN++ (pv_lumpi, epoch 15, same 4-view TTA) second stage on the lumpi_v29 Person boxes down to score 0.01,
averaged with equal weights with lumpi_v40's three (L11, L12, L11b second seed), plus test A's seed (vrcnn_lumpi_fix_s2) only if A passed its own gate against lumpi_v37 (it did: val 0.4349, fresh 0.4454, both halves and IoU 0.7 up),
then the v37 size change and length rule, with no box added. One route; one val reading, one fresh reading if val passes; no retry.
Gate against lumpi_v37 Person: at least +0.003 on both halves, IoU 0.7 not down. It must also beat lumpi_v41 Person on val (0.4333) and fresh (0.4454) to earn a file; fresh half 2 is reported against lumpi_v41's 0.3679.
The stacked files S1 and S2 (scripts/lumpi_stacks.py) are not changed by this. Writes output/lumpi/ens_x/lumpi_pv_s2_test.json: lumpi_v41 with only Person replaced.
    python scripts/lumpi_pv_s2_routes.py logs/fresh_ids.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np

import lumpi_static as S

O = Path("output/lumpi/ens_x")
L = Path("output/lumpi/vrcnn_lumpi_fix/l11c")
C = Path("output/lumpi/vrcnn_lumpi_c/l12c")
Z = Path("output/lumpi/vrcnn_lumpi_b_s2/s2c")
N = Path("output/lumpi/vrcnn_lumpi_fix_s2/s2")
PV = Path("output/lumpi/pv_lumpi/s2p")
# Test A's seed joins only if A passed its own gate against lumpi_v37; it passed
# (scripts/lumpi_fix_s2_routes.py, Nov 7). Beating lumpi_v41 is the file rule, not this gate.
A_PASSED = True
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}


def grow(d):
    b = d["box"]
    return {**d, "box": b[:3] + [b[3] + 0.044, b[4] + 0.023, b[5] + 0.10] + b[6:]}


def rule(d):
    return {**d, "score": d["score"] * 0.3} if d["box"][3] >= 0.85 else d


def avg(ds):
    d0 = ds[0]
    rr = [d for d in ds if "ref1" in d]
    if len(rr) < len(ds):
        return keep(d0)
    iou = float(np.mean([max(d["iou1"], 0.0) for d in rr]))
    near = [d["ref1"] for d in rr if np.hypot(d["ref1"][0] - d0["box"][0], d["ref1"][1] - d0["box"][1]) <= 0.5]
    box = list(d0["box"])
    if near:
        a = np.array(near, float)
        box = a.mean(0).tolist()
        t = 0.5 * np.arctan2(np.sin(2 * a[:, 6]).mean(), np.cos(2 * a[:, 6]).mean())
        box[6] = float(t + np.round((a[0, 6] - t) / np.pi) * np.pi)
    return {"box": box, "score": float(np.sqrt(d0["score"] * iou)), "class": d0["class"]}


def load(path):
    r = json.loads(Path(path).read_text())
    return {f: [d for d in v if d["class"] == "Person"] for f, v in r.items()}


def person(paths):
    src = [load(p) for p in paths]
    out = {}
    for f in src[0]:
        rows = [s[f] for s in src]
        assert all(len(r) == len(rows[0]) for r in rows)
        out[f] = []
        for tup in zip(*rows):
            assert all(t["box"] == tup[0]["box"] for t in tup)
            out[f].append(rule(grow(avg(list(tup)))))
    return out


def row(p, gt, ids):
    r = {}
    for h, x in {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}.items():
        a = S.ap(p, gt, x, ["Person"])
        r[h] = [a[0.5]["Person"], a[0.7]["Person"]]
    return r


gate = lambda r, ref: all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1]
ROUTES = {
    "PV": lambda s: [L / f"v29_{s}_ref_min01.json", C / f"v29_{s}_ref_min01.json", Z / f"v29_{s}_ref_min01.json", PV / f"v29_{s}_min01.json"]
    + ([N / f"v29_{s}_min01.json"] if A_PASSED else []),
}
print("test A passed its gate:", A_PASSED, "members", [p.parent.parent.name for p in ROUTES["PV"]("val")], flush=True)

ids = S.ids_of("val")
gt = S.labels(ids)
ref = row(person([L / "v29_val_ref_min01.json"]), gt, ids)
print("v37 Person val", json.dumps(ref), "expected 0.4257", flush=True)
if ref["all"][0] != 0.4257:
    print("STOP base does not match", flush=True)
    sys.exit(0)
good = []
for k, paths in ROUTES.items():
    r = row(person(paths("val")), gt, ids)
    g = gate(r, ref)
    print("val route", k, json.dumps(r), "PASS" if g else "fail", flush=True)
    if g:
        good.append((r["all"][0], k))
GJ = Path("logs/lumpi_pv_s2_gate.json")  # read by scripts/parta2_lumpi_read.py: PV joins Part-A2's members only if it passed its own gate
if not good:
    GJ.write_text(json.dumps({"val": False, "fresh": None, "passed": False}))
    print("kept none", flush=True)
    sys.exit(0)
va, k = max(good)
fr = [l.strip() for l in open(sys.argv[1]) if l.strip()]
gf = S.labels(fr)
ref = row(person([L / "v29_fresh_ref_min01.json"]), gf, fr)
print("fresh v37", json.dumps(ref), "expected 0.4348", flush=True)
if ref["all"][0] != 0.4348:
    print("STOP base does not match", flush=True)
    sys.exit(0)
r = row(person(ROUTES[k]("fresh")), gf, fr)
g = bool(gate(r, ref))
b = bool(va > 0.4333 and r["all"][0] > 0.4454)
print("fresh route", k, json.dumps(r), "fresh half 2 vs lumpi_v41 0.3679: %+.4f" % (r["h2"][0] - 0.3679), "gate", "PASS" if g else "fail", "beats lumpi_v41 on val and fresh", b, flush=True)
GJ.write_text(json.dumps({"val": True, "fresh": g, "passed": g}))
if g and b:
    t = json.loads((O / "lumpi_v41_test.json").read_text())
    x = person(ROUTES[k]("test"))
    (O / "lumpi_pv_s2_test.json").write_text(json.dumps({f: [d for d in t[f] if d["class"] != "Person"] + x[f] for f in t}))
    print("wrote lumpi_pv_s2_test.json route", k, flush=True)
