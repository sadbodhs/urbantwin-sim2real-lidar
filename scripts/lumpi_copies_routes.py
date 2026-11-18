"""Two-wheeler copies as Person on lumpi_v41. One rule, fixed before any reading: every Bicycle or Motorcycle box in the base file with score >= 0.3 also becomes a Person box, same box, score x 0.3, added after the Person chain with no size change and no length rule. All existing boxes stay; other classes and realism frames exactly lumpi_v41.
One val reading, one fresh reading if val passes, no retry. Gate, fixed before reading because half 1 has few of these labels: half 2 at least +0.003, half 1 down by no more than 0.001, IoU 0.7 not down, on val and then on fresh. Fresh frames sit 1 s from val frames and share tracks, so fresh is a weak check here. Val-informed: the idea came from val labels. Writes output/lumpi/ens_x/lumpi_copies_test.json if both pass.
    python scripts/lumpi_copies_routes.py logs/fresh_ids.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np

import lumpi_static as S

O = Path("output/lumpi/ens_x")
L = Path("output/lumpi/vrcnn_lumpi_fix/l11c")
B = Path("output/lumpi/vrcnn_lumpi_b/l11bc")
C = Path("output/lumpi/vrcnn_lumpi_c/l12c")
Z = Path("output/lumpi/vrcnn_lumpi_b_s2/s2c")
TW = ("Bicycle", "Motorcycle")
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}


def grow(b):
    return b[:3] + [b[3] + 0.044, b[4] + 0.023, b[5] + 0.10] + b[6:]


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


def only(path, cls):
    r = json.loads(Path(path).read_text())
    return {f: [d for d in v if d["class"] == cls] for f, v in r.items()}


def person(split):
    src = [only(d / f"v29_{split}_ref_min01.json", "Person") for d in (L, C, Z)]
    out = {}
    for f in src[0]:
        rows = [s[f] for s in src]
        assert all(len(r) == len(rows[0]) for r in rows)
        out[f] = []
        for tup in zip(*rows):
            assert all(t["box"] == tup[0]["box"] for t in tup)
            d = avg(list(tup))
            b = grow(d["box"])
            out[f].append({"box": b, "score": d["score"] * 0.3 if b[3] >= 0.85 else d["score"], "class": "Person"})
    return out


def wheels(split):
    bi = only(B / f"v29_{split}_ref.json", "Bicycle")
    mo = only(L / f"v29_{split}_ref.json", "Motorcycle")
    return {f: [avg([d]) if d["score"] >= 0.05 else keep(d) for d in bi[f]] + [keep(d) for d in mo[f]] for f in bi}


def copies(w):
    return {f: [{"box": list(d["box"]), "score": d["score"] * 0.3, "class": "Person"} for d in v if d["score"] >= 0.3] for f, v in w.items()}


def row(p, gt, ids):
    r = {}
    for h, x in {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}.items():
        a = S.ap(p, gt, x, ["Person"])
        r[h] = [a[0.5]["Person"], a[0.7]["Person"]]
    return r


def stretches(p, gt, ids):
    r = {}
    for k in sorted({int(i) // 800 for i in ids}):
        x = [i for i in ids if int(i) // 800 == k]
        r[f"{k * 800}-{k * 800 + 799}"] = S.ap(p, gt, x, ["Person"])[0.5]["Person"]
    return r


def gate(r, ref):
    return round(1e4 * (r["h2"][0] - ref["h2"][0])) >= 30 and r["h1"][0] >= ref["h1"][0] - 0.001 and r["all"][1] >= ref["all"][1]


def read(tag, ids, expected, wchk):
    gt = S.labels(ids)
    base = {f: v for f, v in person(tag).items() if f in set(ids)}
    w = wheels(tag)
    a = S.ap(w, gt, ids, TW)[0.5]
    print(tag, "two-wheelers", {c: a[c] for c in TW}, "expected", wchk, flush=True)
    ref = row(base, gt, ids)
    print(tag, "v41 Person", json.dumps(ref), "expected", expected, flush=True)
    if [ref["all"][0], ref["h1"][0], ref["h2"][0]] != expected or [a[c] for c in TW] != wchk:
        print("STOP base does not match", flush=True)
        sys.exit(0)
    x = copies(w)
    n = [len(x[f]) for f in ids]
    print(tag, "copies per frame", round(float(np.mean(n)), 2), "total", sum(n), "frames with any", sum(k > 0 for k in n), flush=True)
    new = {f: base[f] + x[f] for f in ids}
    r = row(new, gt, ids)
    g = gate(r, ref)
    print(tag, "stretch v41", json.dumps(stretches(base, gt, ids)), flush=True)
    print(tag, "stretch new", json.dumps(stretches(new, gt, ids)), flush=True)
    print(tag, "new Person", json.dumps(r), "PASS" if g else "fail", flush=True)
    return g


if __name__ == "__main__":
    if not read("val", S.ids_of("val"), [0.4333, 0.4786, 0.3739], [0.5404, 0.0051]):
        print("kept none", flush=True)
        sys.exit(0)
    fr = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    if not read("fresh", fr, [0.4454, 0.4987, 0.3679], [0.5386, 0.0026]):
        print("kept none", flush=True)
        sys.exit(0)
    t = json.loads((O / "lumpi_v41_test.json").read_text())
    x = {f: [d for d in t[f] if d["class"] in TW] for f in t}
    x = copies(x)
    (O / "lumpi_copies_test.json").write_text(json.dumps({f: t[f] + x[f] for f in t}))
    print("wrote lumpi_copies_test.json, copies on test", sum(map(len, x.values())), flush=True)
