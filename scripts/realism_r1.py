"""Realism plan R1, alignment. Settings fixed before any reading.
Per track, CPU:
  0. Base check: the uploaded pool rebuilt from its recorded 50 source frames with utl.realism_emit's steps (its config, zero rows) must equal
     the uploaded .bin files (same shapes, points within 1e-6 m: float rounding), else stop.
  1. Static real map: allowed fit frames only (V2X rfit_i1, LUMPI realism_fit; every id through the forbidden-frame guard), no-return rows removed,
     masked to the track range; 0.2 m voxels occupied in at least half of the fit frames; map point = mean of the fit points in the voxel.
  2. Synthetic static scene the same way from the pool's 50 source frames. One rigid 6-DoF transform per track: robust point-to-plane ICP of the
     synthetic static scene to the real map (normals: PCA of the 20 nearest map points; Huber weights, 0.1 m; correspondences within 1.0 m;
     at most 50 Gauss-Newton iterations, stop when the update is below 1e-6; no scaling).
  3. The 50 source frames moved by that transform, then utl.realism_emit's calibration unchanged (its config, its zero rows).
  4. Score with utl.metrics.realism_scores (the official terms and normalisation) on the check frames (V2X rcheck_i1, LUMPI realism_check;
     scoring only), seeds 1234, 1, 2, 3, 4, for the uploaded pool and the new pool. EMD and the median transport distance (seed 1234) reported.
  5. Gate: realism +0.003 or more over the uploaded pool on every seed, and no other term down by more than 0.001 (seed mean). One try per track.
Writes output/<track>/realism_r1/{0001..0050.bin, r1_report.json}.
    python scripts/realism_r1.py v2x"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from utl import guard, metrics
from utl import realism_sweep as rs
from utl.realism import MAX_POINTS, _load, add_zeros, nonzero
from utl.tracks import TRACKS

CFG = {
    "v2x": dict(pool="output/v2x/realism_v2_i1", syn="data/pcdet/ut_v2x/points", real="data/pcdet/v2x_real/points",
                fit="data/pcdet/v2x_real/ImageSets/rfit_i1.txt", check="data/pcdet/v2x_real/ImageSets/rcheck_i1.txt"),
    "lumpi": dict(pool="output/lumpi/realism_v2", syn="data/pcdet/ut_lumpi/points", real="data/pcdet/lumpi_real/points",
                  fit="data/pcdet/lumpi_real/ImageSets/realism_fit.txt", check="data/pcdet/lumpi_real/ImageSets/realism_check.txt"),
}
# Realism Try A: UT_REALISM_FIT=b swaps only the fit
# frames (rfit_b = rfit_i1 without clip _19_0 + the 11 train clips with no forbidden frame) and the matching start pool; nothing else changes.
import os
if os.environ.get("UT_REALISM_FIT") == "b":
    CFG["v2x"].update(pool="output/v2x/realism_v2_b", real="data/pcdet/v2x_real_b/points", fit="data/pcdet/v2x_real_b/ImageSets/rfit_b.txt",
                      check="data/pcdet/v2x_real_b/ImageSets/rcheck_i1.txt")
SEEDS = [1234, 1, 2, 3, 4]
VOX = 0.2


def real_frames(track, c, key):
    ids = [l.strip() for l in open(c[key]) if l.strip()]
    for i in ids:
        guard.check(track, i)
    return [metrics.mask_range(_load(Path(c["real"]) / f"{i}.npy"), TRACKS[track]["pc_range"]) for i in ids]


def emit(frames, rep):
    """utl.realism_emit's steps after the pick: calibrate, then zero rows (rng 1)."""
    return add_zeros(rs.calibrate(frames, rs._S["fit"], rep["cfg"]["voxels"], rep["cfg"]["align"], rep["cfg"]["jitter"]),
                     rep["zero_frac"], np.random.default_rng(1))


def static_map(frames):
    """Voxels (0.2 m) occupied in at least half the frames; returns the mean point per kept voxel."""
    enc = lambda k: ((k[:, 0] + 2000) * 4000 + (k[:, 1] + 2000)) * 4000 + (k[:, 2] + 2000)
    keys, P, K = [], [], []
    for f in frames:
        f = nonzero(f).astype(np.float64)
        k = enc(np.floor(f / VOX).astype(np.int64))
        keys.append(np.unique(k))
        P.append(f); K.append(k)
    u, cnt = np.unique(np.concatenate(keys), return_counts=True)
    keep = u[cnt >= len(frames) / 2]
    P, K = np.concatenate(P), np.concatenate(K)
    m = np.isin(K, keep)
    uk, inv = np.unique(K[m], return_inverse=True)
    sums = np.zeros((len(uk), 3)); np.add.at(sums, inv.ravel(), P[m])
    return sums / np.bincount(inv.ravel())[:, None]


def normals(p, k=20):
    _, idx = cKDTree(p).query(p, k)
    q = p[idx] - p[idx].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", q, q))
    return v[:, :, 0]


def rot(a):
    cx, cy, cz, sx, sy, sz = *np.cos(a), *np.sin(a)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def icp(src, dst, nd, iters=50, maxd=1.0, huber=0.1):
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    log = []
    for it in range(iters):
        s = src @ R.T + t
        d, j = tree.query(s, distance_upper_bound=maxd)
        m = np.isfinite(d)
        q, n, sm = dst[j[m]], nd[j[m]], s[m]
        r = ((sm - q) * n).sum(1)
        w = np.where(np.abs(r) <= huber, 1.0, huber / np.abs(r))
        J = np.hstack([np.cross(sm, n), n])
        dx = np.linalg.solve((J * w[:, None]).T @ J + 1e-9 * np.eye(6), -(J * w[:, None]).T @ r)
        dR = rot(dx[:3])
        R, t = dR @ R, dR @ t + dx[3:]
        log.append(dict(it=it + 1, inliers=int(m.sum()), share=round(float(m.mean()), 4), med_abs_res=round(float(np.median(np.abs(r))), 4)))
        if np.abs(dx).max() < 1e-6:
            break
    return R, t, log


def per_seed(frames, check, pc):
    return [metrics.realism_scores(frames, check, pc, 10000, seed=s) for s in SEEDS]


def median_move(frames, check, pc):
    import ot
    x, y = metrics.subsample(frames, pc, 10000, 1234), metrics.subsample(check, pc, 10000, 1234)
    M = ot.dist(x, y, metric="euclidean")
    P = ot.emd(np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y)), M, numItermax=10_000_000)
    return float(np.median(M[np.arange(len(x)), P.argmax(1)]))


def main(track):
    c, pc = CFG[track], TRACKS[track]["pc_range"]
    rep = json.loads((Path(c["pool"]) / "realism_report.json").read_text())
    print(track, "pool", c["pool"], "cfg", rep["cfg"], "zeros", rep["zeros"], "zero_frac %.4f" % rep["zero_frac"], flush=True)
    fit_raw = real_frames(track, c, "fit")
    fit = [nonzero(f) for f in fit_raw]
    check = real_frames(track, c, "check")
    if rep["zeros"] == "drop":
        check = [nonzero(f) for f in check]
    rs._init(dict(pc=pc, fit=fit))
    raw = [_load(Path(c["syn"]) / n) for n in rep["synthetic_sources"]]
    base = emit([metrics.mask_range(f, pc) for f in raw], rep)
    up = [_load(Path(c["pool"]) / f"{i:04d}.bin") for i in range(1, 51)]
    diff = max(float(np.abs(a.astype(np.float64) - b).max()) if a.shape == b.shape else np.inf for a, b in zip(base, up))
    same = bool(diff < 1e-6)   # equal up to float rounding (first run: max 1.4e-14 m V2X, 9.3e-10 m LUMPI, all shapes equal)
    print("base check: rebuilt pool equals the uploaded .bin files (same shapes, max difference %.1e m):" % diff, same, flush=True)
    if not same:
        sys.exit("STOP the uploaded pool is not rebuilt exactly")
    rmap = static_map(fit)
    smap = static_map([metrics.mask_range(f, pc) for f in raw])
    print("static real map %d points (%d fit frames) | synthetic static scene %d points" % (len(rmap), len(fit), len(smap)), flush=True)
    R, t, log = icp(smap, rmap, normals(rmap))
    ang = np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(R[2, 0]), np.arctan2(R[1, 0], R[0, 0])])
    print("ICP", len(log), "iterations | first", log[0], "| last", log[-1], flush=True)
    print("transform: roll/pitch/yaw (deg) %.3f %.3f %.3f | translation (m) %.3f %.3f %.3f" % (*ang, *t), flush=True)
    def move(f):  # no-return rows, if any, stay (0, 0, 0) so the frame keeps its make-up
        nz = np.abs(f).sum(1) > 0
        g = f.copy(); g[nz] = (f[nz].astype(np.float64) @ R.T + t).astype(np.float32)
        return g
    moved = [metrics.mask_range(move(f), pc) for f in raw]
    new = emit(moved, rep)
    for f in new:
        assert len(f) <= MAX_POINTS + 5000 and np.isfinite(f).all()
    a, b = per_seed(up, check, pc), per_seed(new, check, pc)
    gain = [y["score"] - x["score"] for x, y in zip(a, b)]
    terms = {k: float(np.mean([y["norm"][k] - x["norm"][k] for x, y in zip(a, b)])) for k in a[0]["norm"]}
    for s, x, y in zip(SEEDS, a, b):
        print("seed", s, "uploaded %.4f -> R1 %.4f (%+.4f) | EMD %.3f -> %.3f" % (x["score"], y["score"], y["score"] - x["score"], x["raw"]["EMD"], y["raw"]["EMD"]), flush=True)
    mm = (median_move(up, check, pc), median_move(new, check, pc))
    g = bool(min(gain) >= 0.003 and all(v >= -0.001 for k, v in terms.items() if k != "EMD"))
    print("mean realism %.4f -> %.4f | term changes" % (np.mean([x["score"] for x in a]), np.mean([y["score"] for y in b])),
          {k: round(v, 4) for k, v in terms.items()}, "| median transport distance (seed 1234) %.3f -> %.3f m" % mm, "| gate", "PASS" if g else "fail", flush=True)
    out = Path(f"output/{track}/realism_r1")
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(new, 1):
        f.astype(np.float32).tofile(out / f"{i:04d}.bin")
    (out / "r1_report.json").write_text(json.dumps(dict(track=track, pool=c["pool"], R=R.tolist(), t=t.tolist(), angles_deg=ang.tolist(), icp=log,
        map_points=len(rmap), scene_points=len(smap), uploaded=a, r1=b, gain_per_seed=gain, term_change=terms, median_move=mm, gate=g,
        synthetic_sources=rep["synthetic_sources"], cfg=rep["cfg"], zero_frac=rep["zero_frac"]), indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
