"""M3' step 2: learned rescorer for Person (LUMPI) and pedestrian (V2X) boxes, trained on synthetic data only.
Training data: the walker models' detections (pred_walk.json, from scripts/r3_infer.sh) on the walker-inserted
synthetic val frames of r3_make.py; target = the box is a true positive (official-style matching, IoU >= 0.5 against
the synthetic labels incl. the inserted walkers). Features are computed from the points around each box only (no
detector score): range, box size, bottom above local ground, point count (raw and range-normalised), height span,
top height above ground, pole column ratio, points above the box, isolation ring ratio, footprint extents of the
points, and the share of points in the lower / middle / upper third of the box.
Model: sklearn HistGradientBoosting (fixed settings), trained on 80% of the synthetic frames; AUC on the other 20%.
Applied to the current real val chains (LUMPI v20 Person, V2X v16 pedestrians): score x (a + (1 - a) p), a in
{0.3, 0.0}. Gate: AP up on both halves, IoU 0.7 not down. Also: AUC of p on real val (scoring only) and mean p
synthetic vs real (shift). Real labels are used for scoring only.   python /work/r3_train.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/pylib")
sys.path.insert(0, "/work")
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from utl.metrics import iou3d  # noqa: E402

W = Path("/work/r3")
NAMES = ["range", "l", "w", "h", "bottom_above_ground", "log_n", "log_n_r2", "span", "top_above_ground", "pole",
         "above", "ring", "ext_x", "ext_y", "low", "mid", "high"]
OUT = {}


def local(P, b):
    dx, dy = P[:, 0] - b[0], P[:, 1] - b[1]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    return c * dx - s * dy, s * dx + c * dy


def feats(box, P):
    b = np.asarray(box, float)
    lx, ly = local(P, b)
    z = P[:, 2]
    bot, top = b[2] - b[5] / 2, b[2] + b[5] / 2
    ax, ay = np.abs(lx), np.abs(ly)
    fp = (ax <= b[3] / 2 + 0.15) & (ay <= b[4] / 2 + 0.15)
    near = (ax <= b[3] / 2 + 1.5) & (ay <= b[4] / 2 + 1.5) & (z < bot + 1.0)
    ground = float(np.percentile(z[near], 5)) if near.sum() >= 5 else bot
    inside = fp & (z > bot) & (z < top)
    win = fp & (z > bot - 0.5) & (z < bot + 2.5)
    n_in = int(inside.sum())
    r = float(np.hypot(b[0], b[1]))
    zw = z[win]
    span = float(np.percentile(zw, 98) - np.percentile(zw, 2)) if len(zw) >= 5 else np.nan
    topg = float(np.percentile(zw, 98) - ground) if len(zw) >= 5 else np.nan
    col = (np.hypot(lx, ly) < 0.3) & (z > top + 0.1) & (z < top + 1.2)
    above = fp & (z > top + 0.1) & (z < top + 1.2)
    band = (z > bot + 0.5) & (z < bot + 1.5)
    ring = band & (ax <= b[3] / 2 + 1.0) & (ay <= b[4] / 2 + 1.0) & ~((ax <= b[3] / 2 + 0.3) & (ay <= b[4] / 2 + 0.3))
    inb = band & (ax <= b[3] / 2 + 0.15) & (ay <= b[4] / 2 + 0.15)
    if n_in >= 3:
        ex = float(np.percentile(lx[inside], 95) - np.percentile(lx[inside], 5))
        ey = float(np.percentile(ly[inside], 95) - np.percentile(ly[inside], 5))
        t = (z[inside] - bot) / max(b[5], 1e-3)
        low, mid, high = float((t < 1 / 3).mean()), float(((t >= 1 / 3) & (t < 2 / 3)).mean()), float((t >= 2 / 3).mean())
    else:
        ex = ey = low = mid = high = np.nan
    n = max(n_in, 1)
    return [r, b[3], b[4], b[5], bot - ground, np.log1p(n_in), np.log1p(n_in) + 2 * np.log(max(r, 1.0)), span, topg,
            col.sum() / n, above.sum() / n, ring.sum() / max(int(inb.sum()), 1), ex, ey, low, mid, high]


def match(dets, gts, thr=0.5):
    """Official-style per frame: by score, each box to its best-IoU unused label; returns tp flags in dets order."""
    tp = np.zeros(len(dets), int)
    if not dets or not gts:
        return tp
    order = np.argsort([-d["score"] for d in dets], kind="stable")
    I = iou3d(np.array([dets[i]["box"] for i in order]), np.array(gts))
    used = np.zeros(len(gts), bool)
    for r, i in enumerate(order):
        cand = np.where(~used, I[r], -1.0)
        j = int(cand.argmax())
        if cand[j] >= thr:
            used[j] = True
            tp[i] = 1
    return tp


def syn_frame(args):
    track, cls, fid, dets, gts = args
    P = np.load(W / track / "points" / f"{fid}.npy")
    dets = [d for d in dets if d["class"] == cls and d["score"] >= 0.05]
    t = match(dets, gts)
    return fid, [feats(d["box"], P) for d in dets], t.tolist()


def synthetic(track, cls):
    labels = json.loads((W / track / "labels.json").read_text())
    pred = json.loads((W / track / "pred_walk.json").read_text())
    tasks = [(track, cls, f, pred.get(f, []), [b for b, n in labels[f] if n == cls]) for f in sorted(labels)]
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(syn_frame, tasks, chunksize=8))
    return res


def fit(res, seed=0):
    rng = np.random.default_rng(seed)
    hold = set(rng.choice([f for f, _, _ in res], size=len(res) // 5, replace=False))
    Xtr = np.array([x for f, X, _ in res if f not in hold for x in X], float)
    ytr = np.array([y for f, _, Y in res if f not in hold for y in Y], int)
    Xho = np.array([x for f, X, _ in res if f in hold for x in X], float)
    yho = np.array([y for f, _, Y in res if f in hold for y in Y], int)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True,
                                       validation_fraction=0.15, random_state=0).fit(Xtr, ytr)
    p = m.predict_proba(Xho)[:, 1]
    info = {"train_boxes": int(len(ytr)), "train_tp_share": round(float(ytr.mean()), 3),
            "holdout_auc": round(float(roc_auc_score(yho, p)), 4), "mean_p_synthetic": round(float(p.mean()), 3)}
    return m, info


def rescore(pred, X, m, cls, a):
    out = {}
    for f, v in pred.items():
        xs = X[f]
        idx = [i for i, d in enumerate(v) if d["class"] == cls and d["score"] >= 0.05]
        p = m.predict_proba(np.array([xs[i] for i in idx], float))[:, 1] if idx else []
        pm = dict(zip(idx, p))
        out[f] = [dict(d, score=d["score"] * (a + (1 - a) * pm[i])) if i in pm else d for i, d in enumerate(v)]
    return out


def real_feats(pred, load, cls):
    X = {}
    for f, v in pred.items():
        P = load(f)
        X[f] = {i: feats(d["box"], P) for i, d in enumerate(v) if d["class"] == cls and d["score"] >= 0.05}
    return X


def real_auc(pred, X, gt, m, cls):
    ys, ps = [], []
    for f, v in pred.items():
        dets = [(i, d) for i, d in enumerate(v) if d["class"] == cls and d["score"] >= 0.3]
        if not dets:
            continue
        t = match([d for _, d in dets], [b for b, k in gt[f] if k == cls])
        p = m.predict_proba(np.array([X[f][i] for i, _ in dets], float))[:, 1]
        ys += list(t)
        ps += list(p)
    return {"real_auc_confident": round(float(roc_auc_score(ys, ps)), 4), "real_tp_share": round(float(np.mean(ys)), 3),
            "mean_p_real": round(float(np.mean(ps)), 3), "n": len(ys)}


def lumpi():
    import lumpi_next as N
    from lumpi_next import S, T
    res = synthetic("lumpi", "Person")
    m, info = fit(res)
    ids = S.ids_of("val")
    gt = S.labels(ids)
    halves = {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}
    p = N.chain("val")
    pole = N.measure(p, "val", "Person", N.pole_ratio)
    p = N.demote(p, {f: [x is not None and x >= 0.1 for x in v] for f, v in pole.items()})   # v20 Person
    X = real_feats(p, lambda f: T.pts("val", f), "Person")
    info.update(real_auc(p, X, gt, m, "Person"))
    rows = {"v20": {h: round(N.cls_ap(p, gt, fr, "Person"), 4) for h, fr in halves.items()}}
    rows["v20"]["at0.7"] = round(N.cls_ap(p, gt, ids, "Person", 0.7), 4)
    for a in (0.3, 0.0):
        q = rescore(p, X, m, "Person", a)
        rows[f"a={a}"] = {h: round(N.cls_ap(q, gt, fr, "Person"), 4) for h, fr in halves.items()}
        rows[f"a={a}"]["at0.7"] = round(N.cls_ap(q, gt, ids, "Person", 0.7), 4)
    info["Person_AP"] = rows
    OUT["lumpi"] = info
    print("LUMPI", json.dumps(info), flush=True)


def v2x():
    import v2x_more as VM
    import v2x_ped as V
    res = synthetic("v2x", "pedestrian")
    m, info = fit(res)
    i1 = sorted(l.strip() for l in open(V.ROOT / "ImageSets/val_i1.txt") if l.strip())
    halves = {"all": i1, "h1": i1[: len(i1) // 2], "h2": i1[len(i1) // 2:]}
    gt = {}
    for i in i1:
        rows = [l.split() for l in (V.ROOT / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
        gt[i] = [([float(v) for v in r[:7]], r[7]) for r in rows if r[7] == "pedestrian"]
    p = VM.v16_peds(VM.load("val", i1), "val")
    X = real_feats(p, lambda f: V.pts("val", f), "pedestrian")
    info.update(real_auc(p, X, gt, m, "pedestrian"))
    V.report("v16", p, gt, halves)
    for a in (0.3, 0.0):
        V.report(f"a={a}", rescore(p, X, m, "pedestrian", a), gt, halves)
    info["pedestrian_AP"] = dict(V.OUT)
    OUT["v2x"] = info
    print("V2X", json.dumps(info), flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["v2x", "lumpi"]:
        {"v2x": v2x, "lumpi": lumpi}[t]()
    Path("/work/r3_train.json").write_text(json.dumps(OUT, indent=1))
