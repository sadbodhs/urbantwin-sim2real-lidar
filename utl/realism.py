"""Make the 50 synthetic frames of a submission and score them locally.

Pipeline (all fitted on allowed real frames only, never on forbidden or near-forbidden frames):
  1. pick 50 synthetic frames evenly spread over the synthetic sequence;
  2. sensor-density calibration: per voxel, resample synthetic points so that the mean points-per-voxel profile
     matches the real sensor's (removes points the real sensors never see, thins over-dense regions);
  3. global moment alignment: affine map so the pooled synthetic xyz mean / covariance match the real ones.
Scoring uses a disjoint set of real frames ("realism check" frames) as a stand-in for the hidden reference.

V2X-Real stores no-return beams as (0, 0, 0) (about 30 % of an infrastructure frame), and the range mask keeps
them. Whether the hidden reference frames contain them is not known yet, so `--zeros` selects the hypothesis:
"drop" removes them from the real frames, "keep" leaves them and adds the same fraction of zero points to the
synthetic frames (the fraction is measured on the fit frames).

    python -m utl.realism --track lumpi --syn data/pcdet/ut_lumpi/points --real_fit fit_ids.txt \
        --real_check check_ids.txt --real_dir data/pcdet/lumpi_real/points --out output/lumpi/realism
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from . import guard, metrics
from .tracks import TRACKS

MAX_POINTS = 120_000  # 120k x 3 x 4 B = 1.44 MB < 1.5 MB cap


def _load(p):
    p = Path(p)
    a = np.load(p) if p.suffix == ".npy" else np.fromfile(p, np.float32).reshape(-1, 3)
    return a[:, :3].astype(np.float32)


class Grid:
    def __init__(self, pc_range, voxel):
        self.lo = np.asarray(pc_range[:3], np.float64)
        self.vs = np.asarray(voxel, np.float64)
        self.n = np.ceil((np.asarray(pc_range[3:]) - self.lo) / self.vs).astype(np.int64)

    def keys(self, p):
        ijk = np.floor((p - self.lo) / self.vs).astype(np.int64)
        ijk = np.clip(ijk, 0, self.n - 1)
        return (ijk[:, 0] * self.n[1] + ijk[:, 1]) * self.n[2] + ijk[:, 2]

    def density(self, frames):
        d = np.zeros(int(np.prod(self.n)), np.float64)
        for f in frames:
            np.add.at(d, self.keys(f), 1.0)
        return d / max(len(frames), 1)


def density_resample(frames, d_syn, d_real, grid, n_out, rng):
    out = []
    ratio = np.where(d_syn > 0, d_real / np.maximum(d_syn, 1e-12), 0.0)
    for f in frames:
        w = ratio[grid.keys(f)]
        keep = rng.random(len(f)) < np.clip(w, 0, 1)            # thin where synthetic is denser than real
        extra = np.clip(w - 1, 0, None)                           # duplicate (with jitter) where it is sparser
        rep = rng.poisson(extra)
        idx = np.concatenate([np.where(keep)[0], np.repeat(np.arange(len(f)), rep)])
        p = f[idx].copy()
        dup = len(np.where(keep)[0])
        p[dup:] += rng.normal(0, 0.03, (len(p) - dup, 3)).astype(np.float32)
        if len(p) > n_out:
            p = p[rng.choice(len(p), n_out, replace=False)]
        out.append(p)
    return out


def moment_align(frames, mu_r, cov_r):
    allp = np.concatenate(frames).astype(np.float64)
    mu_s, cov_s = allp.mean(0), np.cov(allp, rowvar=False)
    A = linalg.sqrtm(cov_r).real @ np.linalg.inv(linalg.sqrtm(cov_s).real)
    return [((f - mu_s) @ A.T + mu_r).astype(np.float32) for f in frames], A


def nonzero(f):
    return f[np.abs(f).sum(1) > 0]


def add_zeros(frames, frac, rng):
    if frac <= 0:
        return frames
    out = []
    for f in frames:
        n0 = int(round(len(f) * frac / (1 - frac)))
        f = f[rng.choice(len(f), min(len(f), MAX_POINTS - n0), replace=False)] if len(f) + n0 > MAX_POINTS else f
        out.append(np.concatenate([f, np.zeros((n0, 3), np.float32)]))
    return out


def build(track, syn_files, real_fit, voxel, seed=0, align=True, density=True, zero_frac=0.0):
    pc = TRACKS[track]["pc_range"]
    grid = Grid(pc, voxel)
    rng = np.random.default_rng(seed)
    syn = [metrics.mask_range(_load(p), pc) for p in syn_files]
    real = [nonzero(metrics.mask_range(f, pc)) for f in real_fit]
    frames = density_resample(syn, grid.density(syn), grid.density(real), grid, MAX_POINTS, rng) if density else syn
    A = np.eye(3)
    if align:
        rp = np.concatenate(real).astype(np.float64)
        frames, A = moment_align(frames, rp.mean(0), np.cov(rp, rowvar=False))
    return add_zeros(frames, zero_frac, rng), A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--syn", required=True, help="directory of synthetic frames (.npy/.bin)")
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--real_fit", required=True, help="ids of allowed real frames used to fit the calibration")
    ap.add_argument("--real_check", required=True, help="disjoint ids of allowed real frames used for scoring")
    ap.add_argument("--voxel", type=float, nargs=3, default=[2.0, 2.0, 0.5])
    ap.add_argument("--no_align", action="store_true")
    ap.add_argument("--no_density", action="store_true")
    ap.add_argument("--zeros", choices=["drop", "keep"], default="drop")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    syn_all = sorted(p for p in Path(a.syn).iterdir() if p.suffix in (".npy", ".bin"))
    syn_files = [syn_all[i] for i in np.linspace(0, len(syn_all) - 1, 50).round().astype(int)]

    def real(ids_file):
        ids = [l.strip() for l in open(ids_file) if l.strip()]
        for i in ids:
            guard.check(a.track, i)
        rd = Path(a.real_dir)
        return [_load(next(rd.glob(f"{i}.*"))) for i in ids]

    fit, check = real(a.real_fit), real(a.real_check)
    pc = TRACKS[a.track]["pc_range"]
    zero_frac = 0.0
    if a.zeros == "drop":
        check = [nonzero(f) for f in check]
    else:
        m = [metrics.mask_range(f, pc) for f in fit]
        zero_frac = float(np.mean([1 - len(nonzero(f)) / max(len(f), 1) for f in m]))
    raw = [metrics.mask_range(_load(p), pc) for p in syn_files]
    frames, A = build(a.track, syn_files, fit, a.voxel, align=not a.no_align, density=not a.no_density,
                      zero_frac=zero_frac)
    report = {"raw_synthetic": metrics.realism_scores(raw, check, pc, a.n),
              "calibrated": metrics.realism_scores(frames, check, pc, a.n),
              "real_fit_vs_check": metrics.realism_scores(
                  [nonzero(f) for f in fit[:50]] if a.zeros == "drop" else fit[:50], check, pc, a.n),
              "voxel": a.voxel, "affine": A.tolist(), "zeros": a.zeros, "zero_frac": zero_frac, "synthetic_sources": [p.name for p in syn_files]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames, 1):
        f.astype(np.float32).tofile(out / f"{i:04d}.bin")
    (out / "realism_report.json").write_text(json.dumps(report, indent=1))
    for k in ("raw_synthetic", "calibrated", "real_fit_vs_check"):
        r = report[k]
        print(f"{k:18s} score {r['score']:.4f}  " + "  ".join(f"{m} {v:.3f}" for m, v in r["raw"].items()))


if __name__ == "__main__":
    main()
