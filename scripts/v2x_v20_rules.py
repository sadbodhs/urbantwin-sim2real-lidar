"""Eight label-free score rules on v2x_v20, settings fixed before any reading, thresholds from the false-box table of scripts/v2x_v20_study.py (val-informed), each read once on OS1-64 val_i1. Fires on boxes scoring >= 0.05, score x 0.3 once at most; no boxes added or moved; realism frames unchanged.
npts = points inside the box up to 1.5 m above its top; top = 99th percentile point height above the box bottom; nd2 = npts x plan distance squared (points expected for a body of that size at that range).
X1 pedestrian nd2 < 5000. X2 pedestrian l >= 0.8 m. X3 pedestrian top < 1.2 m. X4 truck nd2 < 50000. X5 truck h < 2.1 m. X6 vehicle nd2 < 15000. X7 vehicle l < 3.5 m. X8 vehicle w < 1.7 m.
Gate: class >= +0.003 on both val_i1 halves, IoU 0.7 not down. No fresh check on V2X. Rules that pass are stacked within a class unless the stack falls below the best single rule. Candidate gate: mAP >= +0.004 on val_i1. Test side counts firings on predictions and points, no labels; a firing share far from val's is printed. Writes output/v2x/ens/v22_test.json.
    python scripts/v2x_v20_rules.py"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from utl import metrics

ROOT = Path("data/pcdet/v2x_real")
CL = ["vehicle", "pedestrian", "truck"]
RULES = {
    "X1": ("pedestrian", lambda x: x["nd2"] < 5000),
    "X2": ("pedestrian", lambda x: x["l"] >= 0.8),
    "X3": ("pedestrian", lambda x: x["top"] < 1.2),
    "X4": ("truck", lambda x: x["nd2"] < 50000),
    "X5": ("truck", lambda x: x["h"] < 2.1),
    "X6": ("vehicle", lambda x: x["nd2"] < 15000),
    "X7": ("vehicle", lambda x: x["l"] < 3.5),
    "X8": ("vehicle", lambda x: x["w"] < 1.7),
}


def stats(job):
    kind, f, idx, boxes = job
    if kind == "val":
        p = np.load(f"data/pcdet/v2x_real_50k/points/{f}.npy")[:, :3]
    elif kind == "fresh":
        p = np.load(f"data/pcdet/v2x_fresh_50k/points/{f}.npy")[:, :3]
    else:
        p = np.fromfile(f"data/kit/v2x/detection_test_frames/{f}.bin", np.float32).reshape(-1, 3)
    p = p[np.abs(p).sum(1) > 0]
    t = cKDTree(p[:, :2])
    out = []
    for i, b in zip(idx, boxes):
        ix = t.query_ball_point(b[:2], float(np.hypot(b[3], b[4])) / 2 + 0.01)
        n, top = 0, 0.0
        if ix:
            q = p[ix]
            dx = q[:, :2] - np.array(b[:2])
            co, si = np.cos(b[6]), np.sin(b[6])
            u, v = dx[:, 0] * co + dx[:, 1] * si, -dx[:, 0] * si + dx[:, 1] * co
            m = (np.abs(u) <= b[3] / 2) & (np.abs(v) <= b[4] / 2) & (q[:, 2] >= b[2] - b[5] / 2) & (q[:, 2] <= b[2] + b[5] / 2 + 1.5)
            n = int(m.sum())
            if n:
                top = float(np.percentile(q[m, 2], 99) - (b[2] - b[5] / 2))
        out.append((i, {"n": n, "top": top, "nd2": n * float(np.hypot(b[0], b[1])) ** 2, "l": b[3], "w": b[4], "h": b[5]}))
    return f, out


def features(kind, p, fr):
    jobs = []
    for f in fr:
        idx = [i for i, d in enumerate(p[f]) if d["score"] >= 0.05]
        jobs.append((kind, f, idx, [p[f][i]["box"] for i in idx]))
    with Pool(6) as pool:
        return dict(pool.imap_unordered(stats, jobs, chunksize=4))


def apply(p, feats, names):
    tests = [RULES[n] for n in names]
    out = {}
    for f, v in p.items():
        vv = list(v)
        for i, x in feats[f]:
            if any(vv[i]["class"] == c and t(x) for c, t in tests):
                vv[i] = {**vv[i], "score": vv[i]["score"] * 0.3}
        out[f] = vv
    return out


def count(p, feats, n):
    c, t = RULES[n]
    a = [(t(x), p[f][i]["score"]) for f in feats for i, x in feats[f] if p[f][i]["class"] == c]
    hit = sum(h for h, _ in a)
    conf = sum(s >= 0.3 for _, s in a)
    hc = sum(h and s >= 0.3 for h, s in a)
    return hit, hc, round(hc / max(conf, 1), 3)


if __name__ == "__main__":
    fr = sorted(l.strip() for l in open(ROOT / "ImageSets/val_i1.txt") if l.strip())
    halves = {"all": fr, "h1": fr[: len(fr) // 2], "h2": fr[len(fr) // 2:]}
    gt = {}
    for i in fr:
        rows = [l.split() for l in (ROOT / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
        gt[i] = [([float(x) for x in r[:7]], r[7]) for r in rows]
    P = json.loads(Path("output/v2x/ens/v20_val_i1.json").read_text())
    P = {f: P[f] for f in fr}

    def row(p, c):
        r = {}
        for h, ids in halves.items():
            s = metrics.detection_scores({f: [d for d in p[f] if d["class"] == c] for f in ids}, {f: [g for g in gt[f] if g[1] == c] for f in ids}, [c], thrs=(0.5, 0.7))
            r[h] = [round(s[0.5]["per_class"][c]["ap40"], 4), round(s[0.7]["per_class"][c]["ap40"], 4)]
        return r

    gate = lambda r, ref: all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1]
    refs = {c: row(P, c) for c in CL}
    print("v20", json.dumps(refs), flush=True)
    F = features("val", P, fr)
    T = json.loads(Path("output/v2x/ens/v20_test.json").read_text())
    FT = features("test", T, sorted(T))
    ok = []
    res = {}
    for n, (c, _) in RULES.items():
        r = row(apply(P, F, [n]), c)
        res[n] = r
        g = gate(r, refs[c])
        print(n, c, json.dumps(r), "val boxes,confident,share", count(P, F, n), "test", count(T, FT, n), "PASS" if g else "fail", flush=True)
        if g:
            ok.append(n)
    print("val passed", ok, flush=True)
    if not ok:
        sys.exit(0)
    final = []
    gain = 0.0
    for c in CL:
        ns = [n for n in ok if RULES[n][0] == c]
        if not ns:
            continue
        best = max(ns, key=lambda n: res[n]["all"][0])
        use, a = [best], res[best]["all"][0]
        if len(ns) > 1:
            st = row(apply(P, F, ns), c)
            print("stack", c, ns, json.dumps(st), flush=True)
            if st["all"][0] >= a:
                use, a = ns, st["all"][0]
        final += use
        gain += a - refs[c]["all"][0]
    print("final", final, "val mAP gain", round(gain / 3, 4), "needs 0.004", flush=True)
    if gain / 3 >= 0.004:
        Path("output/v2x/ens/v22_test.json").write_text(json.dumps(apply(T, FT, final)))
        print("wrote v22_test.json", flush=True)
