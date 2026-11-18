"""Local re-implementation of the challenge scorer (the official local_eval.py is on Codabench, which is down).

Detection: per-class 3D AP over 7-DoF OpenPCDet boxes, IoU = oriented-BEV overlap x Z overlap, KITTI 40-point
(primary) and 11-point interpolation, no difficulty filter, classes without GT excluded from the mean.
Realism: both sides masked to the track range, uniformly subsampled to 10,000 points (seed 1234), then CD
(symmetric squared-L2 NN), MMD (Gaussian kernel, median-heuristic bandwidth), EMD (exact W1, network simplex)
and FPD (Frechet distance between fitted 3-D Gaussians). Lower is better; normalised as 1 - clip.
"""
import numpy as np
from scipy import linalg
from scipy.spatial import cKDTree

from .tracks import NORM


# ----------------------------------------------------------------------------------------------- detection
def _bev_corners(b):
    c, s = np.cos(b[:, 6]), np.sin(b[:, 6])
    dx, dy = b[:, 3] / 2, b[:, 4] / 2
    loc = np.stack([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]], 0)  # (4,2,N), counter-clockwise
    x = b[:, 0] + loc[:, 0] * c - loc[:, 1] * s
    y = b[:, 1] + loc[:, 0] * s + loc[:, 1] * c
    return np.stack([x, y], -1).transpose(1, 0, 2)  # (N,4,2)


def _clip(poly, a, b):
    """Sutherland-Hodgman: clip polygon (list of xy) by the half-plane left of edge a->b."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        cp = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        cq = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
        if cp >= 0:
            out.append(p)
        if (cp >= 0) != (cq >= 0):
            t = cp / (cp - cq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def iou3d(a, b):
    """(N,7) x (M,7) -> (N,M) 3D IoU of oriented boxes."""
    a, b = np.asarray(a, np.float64).reshape(-1, 7), np.asarray(b, np.float64).reshape(-1, 7)
    out = np.zeros((len(a), len(b)))
    if not len(a) or not len(b):
        return out
    za0, za1 = a[:, 2] - a[:, 5] / 2, a[:, 2] + a[:, 5] / 2
    zb0, zb1 = b[:, 2] - b[:, 5] / 2, b[:, 2] + b[:, 5] / 2
    zov = np.clip(np.minimum(za1[:, None], zb1[None]) - np.maximum(za0[:, None], zb0[None]), 0, None)
    # cheap circumcircle rejection before the polygon clip
    ra, rb = np.hypot(a[:, 3], a[:, 4]) / 2, np.hypot(b[:, 3], b[:, 4]) / 2
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    cand = np.argwhere((d < ra[:, None] + rb[None]) & (zov > 0))
    if not len(cand):
        return out
    ca, cb = _bev_corners(a), _bev_corners(b)
    va, vb = a[:, 3] * a[:, 4] * a[:, 5], b[:, 3] * b[:, 4] * b[:, 5]
    for i, j in cand:
        poly = [tuple(p) for p in ca[i]]
        for k in range(4):
            poly = _clip(poly, cb[j][k], cb[j][(k + 1) % 4])
            if not poly:
                break
        inter = _area(poly) * zov[i, j]
        out[i, j] = inter / max(va[i] + vb[j] - inter, 1e-9)
    return out


def _ap(rec, prec, n_points):
    if n_points == 40:
        ts = np.linspace(1 / 40, 1, 40)
    else:
        ts = np.linspace(0, 1, 11)
    ap = 0.0
    for t in ts:
        m = prec[rec >= t - 1e-9]
        ap += m.max() if len(m) else 0.0
    return ap / len(ts)


def class_ap(preds, gts, thr):
    """preds: {frame: (boxes (K,7), scores (K,))}; gts: {frame: boxes (G,7)} -> (ap40, ap11, n_gt)."""
    n_gt = sum(len(g) for g in gts.values())
    if n_gt == 0:
        return None, None, 0
    scores, tps = [], []
    for f, g in gts.items():
        boxes, sc = preds.get(f, (np.zeros((0, 7)), np.zeros(0)))
        if not len(sc):
            continue
        order = np.argsort(-sc)
        boxes, sc = boxes[order], sc[order]
        iou = iou3d(boxes, g) if len(g) else np.zeros((len(sc), 0))
        used = np.zeros(len(g), bool)
        tp = np.zeros(len(sc), bool)
        for i in range(len(sc)):
            if iou.shape[1]:
                cand = np.where(~used, iou[i], -1)
                j = int(cand.argmax())
                if cand[j] >= thr:
                    used[j] = True
                    tp[i] = True
        scores.append(sc)
        tps.append(tp)
    # predictions on frames that are not in gts are ignored (gts must list every evaluated frame, even empty ones)
    if not scores:
        return 0.0, 0.0, n_gt
    sc, tp = np.concatenate(scores), np.concatenate(tps)
    order = np.argsort(-sc, kind="stable")
    tp = tp[order]
    ctp = np.cumsum(tp)
    rec = ctp / n_gt
    prec = ctp / np.arange(1, len(tp) + 1)
    # monotone precision envelope
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return _ap(rec, prec, 40), _ap(rec, prec, 11), n_gt


def detection_scores(pred_json, gt_by_frame, classes, thrs=(0.5, 0.7)):
    """pred_json: submission-style {frame: [{box, score, class}]}; gt_by_frame: {frame: [(box7, class)]}."""
    lc = {c.lower(): c for c in classes}
    out = {}
    for thr in thrs:
        per = {}
        for c in classes:
            P, G = {}, {}
            for f in gt_by_frame:
                G[f] = np.array([b for b, k in gt_by_frame[f] if k.lower() == c.lower()], np.float64).reshape(-1, 7)
                ps = [p for p in pred_json.get(f, []) if lc.get(str(p["class"]).lower()) == c]
                P[f] = (np.array([p["box"] for p in ps], np.float64).reshape(-1, 7),
                        np.array([p["score"] for p in ps], np.float64))
            per[c] = class_ap(P, G, thr)
        valid = {c: v for c, v in per.items() if v[2] > 0}
        out[thr] = dict(
            mAP40=float(np.mean([v[0] for v in valid.values()])) if valid else 0.0,
            mAP11=float(np.mean([v[1] for v in valid.values()])) if valid else 0.0,
            per_class={c: dict(ap40=v[0], ap11=v[1], n_gt=v[2]) for c, v in per.items()},
        )
    return out


# ----------------------------------------------------------------------------------------------- realism
def mask_range(p, pc_range):
    x0, y0, z0, x1, y1, z1 = pc_range
    m = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1) & (p[:, 2] >= z0) & (p[:, 2] <= z1)
    return p[m]


def subsample(frames, pc_range, n=10000, seed=1234):
    p = np.concatenate([mask_range(np.asarray(f)[:, :3], pc_range) for f in frames]).astype(np.float64)
    rng = np.random.RandomState(seed)
    return p[rng.choice(len(p), n, replace=len(p) < n)]


def chamfer(x, y):
    dx = cKDTree(y).query(x)[0]
    dy = cKDTree(x).query(y)[0]
    return float((dx ** 2).mean() + (dy ** 2).mean())


def _sqd(a, b):
    return np.clip((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0, None)


def mmd(x, y, chunk=2000):
    """Squared MMD, unbiased (off-diagonal) estimator, RBF kernel exp(-gamma |a-b|^2) with
    gamma = 1 / (2 * median |a-b|^2) over a 1,000-point joint subsample (challenge Evaluation page)."""
    z = np.concatenate([x, y])
    rng = np.random.RandomState(0)
    s = z[rng.choice(len(z), min(len(z), 1000), replace=False)]
    d2 = _sqd(s, s)
    gamma = 1.0 / (2 * np.median(d2[np.triu_indices(len(s), 1)]))

    def ksum(a, b):
        tot = 0.0
        for i in range(0, len(a), chunk):
            tot += np.exp(-gamma * _sqd(a[i:i + chunk], b)).sum()
        return tot

    nx, ny = len(x), len(y)
    kxx = (ksum(x, x) - nx) / (nx * (nx - 1))       # the diagonal terms are exp(0) = 1
    kyy = (ksum(y, y) - ny) / (ny * (ny - 1))
    return float(kxx + kyy - 2 * ksum(x, y) / (nx * ny))


def emd(x, y):
    import ot
    M = ot.dist(x, y, metric="euclidean")
    w = np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y))
    # the challenge reports the square root of the exact W1 transport cost (Evaluation page)
    return float(np.sqrt(ot.emd2(w[0], w[1], M, numItermax=10_000_000)))


def fpd(x, y):
    m1, m2 = x.mean(0), y.mean(0)
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    cs = linalg.sqrtm(s1 @ s2).real
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def realism_scores(syn_frames, real_frames, pc_range, n=10000, with_emd=True, seed=1234):
    x, y = subsample(syn_frames, pc_range, n, seed), subsample(real_frames, pc_range, n, seed)
    raw = dict(CD=chamfer(x, y), MMD=mmd(x, y), FPD=fpd(x, y))
    if with_emd:
        raw["EMD"] = emd(x, y)
    norm = {k: 1 - float(np.clip((v - NORM[k][0]) / (NORM[k][1] - NORM[k][0]), 0, 1)) for k, v in raw.items()}
    return dict(raw=raw, norm=norm, score=float(np.mean(list(norm.values()))))


def kit_realism_scores(syn_frames, real_frames, pc_range, track="lumpi", ref_seed=None):
    """Realism with the official kit's own code: data/kit/<track>/scoring_program/metrics.py (needs POT), so the synthetic
    subsample and the MMD bandwidth are the server's. ref_seed None: exactly the server's compute_realism(seed=1234). With ref_seed, only the
    reference side varies: the synthetic subsample is still the server's first draw (default_rng(1234)), the real subsample and the MMD
    bandwidth subsample come from default_rng(ref_seed). Normalised with NORM, as the kit's config. Same dict shape as realism_scores."""
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location(f"kit_metrics_{track}", _P(f"data/kit/{track}/scoring_program/metrics.py"))
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    syn = [np.asarray(f)[:, :3] for f in syn_frames]
    real = [np.asarray(f)[:, :3] for f in real_frames]
    if ref_seed is None:
        raw = km.compute_realism(syn, real, tuple(pc_range), seed=1234)
    else:
        xs = km._subsample(km._concat_and_mask(syn, tuple(pc_range)), km.N_SUBSAMPLE, np.random.default_rng(1234))
        rng = np.random.default_rng(ref_seed)
        xr = km._subsample(km._concat_and_mask(real, tuple(pc_range)), km.N_SUBSAMPLE, rng)
        raw = dict(CD=km.chamfer_distance(xs, xr), MMD=km.compute_mmd_rbf(xs, xr, rng=rng)["MMD"], EMD=km.compute_emd(xs, xr), FPD=km.compute_fpd(xs, xr))
    raw = {k: float(raw[k]) for k in ("CD", "MMD", "EMD", "FPD")}
    norm = {k: 1 - float(np.clip((v - NORM[k][0]) / (NORM[k][1] - NORM[k][0]), 0, 1)) for k, v in raw.items()}
    return dict(raw=raw, norm=norm, score=float(np.mean(list(norm.values()))))


def combined(map50, realism):
    det = float(np.clip((map50 - NORM["AP"][0]) / (NORM["AP"][1] - NORM["AP"][0]), 0, 1))
    return dict(score_detection=det, score_realism=realism, score_combined=0.6 * det + 0.4 * realism)
