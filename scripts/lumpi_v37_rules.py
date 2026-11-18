"""Eleven label-free score rules on lumpi_v37, settings fixed before any reading, each read once on val. A rule fires on a box scoring >= 0.05 that no v37 rule has lowered (Truck, Van h >= 2.0 m; Person l < 0.85 m); score x 0.3, once at most; no boxes added or moved; other classes and realism frames unchanged.
npts = points inside the box up to 1.5 m above its top; top = 99th percentile point height above the box bottom, on the same points; d = plan distance from the sensor.
T3 Truck l < 5.75 m (Car-size). T4 Truck top < 2.0 m with npts >= 20. T5 Truck npts < 100. V3 Van top < 1.85 m with npts >= 20 (labels: Car 1.65 m, Van 2.04 m). V4 Van l < 4.5 m. V5 Van npts < 15. P3 Person top >= 2.3 m. P4 Person npts x d^2 < 10000 (a quarter of the true-box median). P5 Person centre inside a Car, Van, Truck or Bus box scoring >= 0.5. C1 Car top >= 2.2 m with npts >= 20. C2 Car l >= 6.0 m.
Val-informed: thresholds set after the val feature tables of the weak-class study. Gate on val, then one reading on fresh for rules that pass: Truck, Person, Car >= +0.003 on both halves, IoU 0.7 not down; Van up on both halves, windows 2 and 3 each up, IoU 0.7 not down. Rules passing fresh are applied together, read on fresh once, and written to output/lumpi/ens_x/lumpi_v39_test.json; if the stack in a class falls below its best single rule on fresh, that class keeps the single rule.
    python scripts/lumpi_v37_rules.py logs/fresh_ids.txt"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_static as S

O = Path("output/lumpi/ens_x")
L = Path("output/lumpi/vrcnn_lumpi_fix/l11c")
B = Path("output/lumpi/vrcnn_lumpi_b/l11bc")
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}
CLS = ("Truck", "Van", "Person", "Car")
VEH = ("Car", "Van", "Truck", "Bus")
RULES = {
    "T3": ("Truck", lambda x: x["l"] < 5.75),
    "T4": ("Truck", lambda x: x["top"] < 2.0 and x["n"] >= 20),
    "T5": ("Truck", lambda x: x["n"] < 100),
    "V3": ("Van", lambda x: x["top"] < 1.85 and x["n"] >= 20),
    "V4": ("Van", lambda x: x["l"] < 4.5),
    "V5": ("Van", lambda x: x["n"] < 15),
    "P3": ("Person", lambda x: x["top"] >= 2.3),
    "P4": ("Person", lambda x: x["n"] * x["d"] ** 2 < 10000),
    "P5": ("Person", lambda x: x["inside"]),
    "C1": ("Car", lambda x: x["top"] >= 2.2 and x["n"] >= 20),
    "C2": ("Car", lambda x: x["l"] >= 6.0),
}


def grow(d):
    if d["class"] != "Person":
        return d
    b = d["box"]
    return {**d, "box": b[:3] + [b[3] + 0.044, b[4] + 0.023, b[5] + 0.10] + b[6:]}


def rule37(d):
    h, l = d["box"][5], d["box"][3]
    if (d["class"] in ("Truck", "Van") and h < 2.0) or (d["class"] == "Person" and l >= 0.85):
        return {**d, "score": d["score"] * 0.3}
    return d


def lowered(d):
    return (d["class"] in ("Truck", "Van") and d["box"][5] < 2.0) or (d["class"] == "Person" and d["box"][3] >= 0.85)


def c12(d):
    if "ref1" not in d:
        return keep(d)
    r = d["ref1"]
    b = list(r) if np.hypot(r[0] - d["box"][0], r[1] - d["box"][1]) <= 0.5 else list(d["box"])
    return {"box": b, "score": float(np.sqrt(d["score"] * max(d["iou1"], 0.0))), "class": d["class"]}


def points(kind, f):
    if kind == "test":
        return np.fromfile(f"data/kit/lumpi/detection_test_frames/{f}.bin", np.float32).reshape(-1, 3)
    root = "lumpi_real_50k" if kind == "val" else "lumpi_real_fresh50k"
    return np.load(f"data/pcdet/{root}/points/{f}.npy")[:, :3]


def stats(job):
    """Per box: index in the frame's list, npts, top, d, l, inside."""
    kind, f, idx, boxes, classes, vehicles = job
    p = points(kind, f)
    p = p[np.abs(p).sum(1) > 0]
    tree = cKDTree(p[:, :2])
    V = np.array(vehicles, float).reshape(-1, 7)
    out = []
    for i, b, c in zip(idx, boxes, classes):
        ix = tree.query_ball_point(b[:2], float(np.hypot(b[3], b[4])) / 2 + 0.01)
        n, top = 0, 0.0
        if ix:
            q = p[ix]
            dx = q[:, :2] - np.array(b[:2])
            co, si = np.cos(b[6]), np.sin(b[6])
            u = dx[:, 0] * co + dx[:, 1] * si
            v = -dx[:, 0] * si + dx[:, 1] * co
            m = (np.abs(u) <= b[3] / 2) & (np.abs(v) <= b[4] / 2) & (q[:, 2] >= b[2] - b[5] / 2) & (q[:, 2] <= b[2] + b[5] / 2 + 1.5)
            n = int(m.sum())
            if n:
                top = float(np.percentile(q[m, 2], 99) - (b[2] - b[5] / 2))
        inside = False
        if c == "Person" and len(V):
            dx = np.array(b[:2]) - V[:, :2]
            co, si = np.cos(V[:, 6]), np.sin(V[:, 6])
            u = dx[:, 0] * co + dx[:, 1] * si
            v = -dx[:, 0] * si + dx[:, 1] * co
            inside = bool(((np.abs(u) <= V[:, 3] / 2) & (np.abs(v) <= V[:, 4] / 2)).any())
        out.append((i, {"n": n, "top": top, "d": float(np.hypot(b[0], b[1])), "l": b[3], "inside": inside}))
    return f, out


def features(kind, p, fr):
    jobs = []
    for f in fr:
        idx = [i for i, d in enumerate(p[f]) if d["class"] in CLS and d["score"] >= 0.05 and not lowered(d)]
        jobs.append((kind, f, idx, [p[f][i]["box"] for i in idx], [p[f][i]["class"] for i in idx], [d["box"][:7] for d in p[f] if d["class"] in VEH and d["score"] >= 0.5]))
    with Pool(10) as pool:
        return dict(pool.imap_unordered(stats, jobs, chunksize=4))


def apply(p, feats, names):
    out = {}
    tests = [RULES[n] for n in names]
    for f, v in p.items():
        vv = list(v)
        for i, x in feats[f]:
            if any(vv[i]["class"] == c and t(x) for c, t in tests):
                vv[i] = {**vv[i], "score": vv[i]["score"] * 0.3}
        out[f] = vv
    return out


def row(p, gt, ids, c):
    only = {f: [d for d in p[f] if d["class"] == c] for f in ids}
    hv = {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}
    r = {}
    for h, x in hv.items():
        a = S.ap(only, gt, x, [c])
        r[h] = [a[0.5][c], a[0.7][c]]
    if c == "Van":
        w = [sorted(ids)[i:i + 80] for i in range(0, len(ids), 80)]
        r["windows"] = [S.ap(only, gt, x, [c])[0.5][c] for x in w]
    return r


def gate(c, r, ref):
    if c == "Van":
        return r["h1"][0] > ref["h1"][0] and r["h2"][0] > ref["h2"][0] and r["windows"][2] > ref["windows"][2] and r["windows"][3] > ref["windows"][3] and r["all"][1] >= ref["all"][1]
    return all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1]


def read(tag, p, feats, gt, ids, names):
    refs, passed = {}, []
    for n in names:
        c = RULES[n][0]
        if c not in refs:
            refs[c] = row(p, gt, ids, c)
            print(tag, c, "v37", json.dumps(refs[c]), flush=True)
        r = row(apply(p, feats, [n]), gt, ids, c)
        g = gate(c, r, refs[c])
        hit = sum(RULES[n][1](x) and p[f][i]["class"] == c for f in ids for i, x in feats[f])
        hi3 = sum(RULES[n][1](x) and p[f][i]["class"] == c and p[f][i]["score"] >= 0.3 for f in ids for i, x in feats[f])
        print(tag, n, c, json.dumps(r), "boxes", hit, "confident", hi3, "PASS" if g else "fail", flush=True)
        if g:
            passed.append(n)
    return passed, refs


if __name__ == "__main__":
    ids = S.ids_of("val")
    gt = S.labels(ids)
    p = {f: [rule37(grow(keep(d))) for d in v] for f, v in json.loads((O / "lumpi_v33_plow_val.json").read_text()).items()}
    p = {f: p[f] for f in ids}
    chk = [row(p, gt, ids, c)["all"][0] for c in ("Truck", "Van", "Person")]
    print("v37 val Truck Van Person", chk, "expected [0.4561, 0.1672, 0.4257]", flush=True)
    if chk != [0.4561, 0.1672, 0.4257]:
        print("STOP base does not match", flush=True)
        sys.exit(0)
    feats = features("val", p, ids)
    ok, _ = read("val", p, feats, gt, ids, list(RULES))
    print("val passed", ok, flush=True)
    if not ok:
        sys.exit(0)
    del p, feats, gt
    fr = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    gf = S.labels(fr)
    a = json.loads((L / "v29_fresh_ref.json").read_text())
    cur = {f: [keep(d) for d in a[f] if d["class"] not in ("Person", "Bicycle")] for f in fr}
    del a
    m = json.loads((L / "v29_fresh_ref_min01.json").read_text())
    for f in fr:
        cur[f] += [c12(d) for d in m[f] if d["class"] == "Person"]
    del m
    b = json.loads((B / "v29_fresh_ref.json").read_text())
    for f in fr:
        cur[f] += [c12(d) for d in b[f] if d["class"] == "Bicycle"]
    del b
    pf = {f: [rule37(grow(d)) for d in cur[f]] for f in fr}
    del cur
    ff = features("fresh", pf, fr)
    kept, refs = read("fresh", pf, ff, gf, fr, ok)
    print("kept", kept, flush=True)
    if not kept:
        sys.exit(0)
    final = []
    for c in sorted({RULES[n][0] for n in kept}):
        ns = [n for n in kept if RULES[n][0] == c]
        rows = {n: row(apply(pf, ff, [n]), gf, fr, c) for n in ns}
        best = max(ns, key=lambda n: rows[n]["all"][0])
        use = ns
        if len(ns) > 1:
            st = row(apply(pf, ff, ns), gf, fr, c)
            print("fresh stack", c, ns, json.dumps(st), flush=True)
            if st["all"][0] < rows[best]["all"][0]:
                use = [best]
        final += use
    print("final", final, flush=True)
    t = json.loads((O / "lumpi_v37_test.json").read_text())
    ft = features("test", t, sorted(t))
    (O / "lumpi_v39_rules_test.json").write_text(json.dumps(apply(t, ft, final)))   # own name: scripts/build_lumpi_v39.py writes lumpi_v39_test.json
    for n in final:
        c, tf = RULES[n]
        print("test", n, "boxes", sum(tf(x) and t[f][i]["class"] == c for f in t for i, x in ft[f]), "confident", sum(tf(x) and t[f][i]["class"] == c and t[f][i]["score"] >= 0.3 for f in t for i, x in ft[f]), flush=True)
