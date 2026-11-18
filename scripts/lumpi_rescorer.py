"""LUMPI Person rescorer (plan item M3', lumpi_v22): a HistGradientBoosting classifier (scikit-learn 1.5.2) that gives
each Person box (score >= 0.05) a probability p of being a person from 17 features of the points around it (no
detector score): range, box size, bottom above local ground, point count (raw and range-normalised), height span, top
above ground, pole column, points above, isolation ring, footprint extents, lower / middle / upper thirds. Trained only
on synthetic data: the walker models' detections on 1,000 walker-inserted synthetic val frames per track
(scripts/rescorer/r3_make.py, r3_train.py, r3_apply.py; detections from scripts/r3_infer.sh). Synthetic hold-out AUC
0.996, real val AUC 0.67. Score x (A + (1 - A) p), A = 0.3 (fixed in advance).
Model file: output/lumpi/rescorer/lumpi_rescorer.pkl."""
import pickle
from pathlib import Path

import numpy as np

A = 0.3
MODEL = Path("output/lumpi/rescorer/lumpi_rescorer.pkl")


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


def apply(pred, load, cls="Person", a=A, model=None):
    """pred {frame: [det]}, load(frame) -> (N, 3) points. Returns (new prediction dict, {frame: {index: p}})."""
    m = model or pickle.loads(MODEL.read_bytes())
    out, probs = {}, {}
    for f, v in pred.items():
        idx = [i for i, d in enumerate(v) if d["class"] == cls and d["score"] >= 0.05]
        if not idx:
            out[f] = v
            continue
        P = load(f)
        p = m.predict_proba(np.array([feats(v[i]["box"], P) for i in idx], float))[:, 1]
        pm = dict(zip(idx, p.tolist()))
        probs[f] = pm
        out[f] = [dict(d, score=d["score"] * (a + (1 - a) * pm[i])) if i in pm else d for i, d in enumerate(v)]
    return out, probs
