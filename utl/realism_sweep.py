"""Search calibration settings for the 50 synthetic realism frames and score each against held-out real frames.

Each setting is fitted on the real "fit" frames only and scored on the disjoint "check" frames (both allowed,
far from every forbidden frame). Scores are averaged over several subsample seeds, since the hidden reference is
one random 10k-point subsample. Settings:
  voxels   one or more voxel sizes; several are applied in sequence (coarse to fine), each pass resampling the
           synthetic points to the real points-per-voxel profile of that grid
  align    global affine moment alignment after resampling
  jitter   Gaussian noise (m) on every synthetic point, standing in for real range noise
  pick     "spread": 50 frames evenly over the candidates; "match": the 50 candidates whose calibrated voxel
           density is closest to the real fit profile (synthetic frames only are chosen; nothing real is copied)

    python -m utl.realism_sweep --track lumpi --syn data/pcdet/ut_lumpi/points --real_dir data/pcdet/lumpi_real/points \
        --real_fit data/pcdet/lumpi_real/ImageSets/realism_fit.txt \
        --real_check data/pcdet/lumpi_real/ImageSets/realism_check.txt --grid default --out output/lumpi/sweep.json
"""
import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import guard, metrics
from .realism import Grid, MAX_POINTS, _load, add_zeros, density_resample, moment_align, nonzero
from .tracks import TRACKS

GRIDS = {
    "default": dict(voxels=[[[6, 6, 1]], [[4, 4, 1]], [[8, 8, 1]], [[6, 6, 0.5]], [[6, 6, 2]],
                            [[8, 8, 2], [2, 2, 0.5]], [[6, 6, 1], [2, 2, 0.5]], [[6, 6, 1], [3, 3, 0.25]],
                            [[8, 8, 2], [4, 4, 1], [1, 1, 0.25]]],
                    align=[True], jitter=[0.0, 0.02], pick=["spread", "match"]),
    "quick": dict(voxels=[[[6, 6, 1]]], align=[True], jitter=[0.0], pick=["spread"]),
}
_S = {}


def _init(state):
    _S.update(state)


def calibrate(syn, real, voxels, align, jitter, seed=0):
    pc = _S["pc"]
    rng = np.random.default_rng(seed)
    frames = syn
    for v in voxels:
        g = Grid(pc, v)
        frames = density_resample(frames, g.density(frames), g.density(real), g, MAX_POINTS, rng)
    if jitter:
        frames = [f + rng.normal(0, jitter, f.shape).astype(np.float32) for f in frames]
    if align:
        rp = np.concatenate(real).astype(np.float64)
        frames, _ = moment_align(frames, rp.mean(0), np.cov(rp, rowvar=False))
    return frames


def pick_frames(cands, real, voxels, pick, n=50):
    if pick == "spread" or len(cands) <= n:
        return [cands[i] for i in np.linspace(0, len(cands) - 1, n).round().astype(int)]
    # distance of each candidate's own voxel-density profile (after the coarse pass) to the real fit profile
    g = Grid(_S["pc"], voxels[0])
    d_real = g.density(real)
    d_real /= d_real.sum()
    cost = []
    for f in cands:
        d = g.density([f])
        cost.append(np.abs(d / max(d.sum(), 1) - d_real).sum())
    return [cands[i] for i in np.argsort(cost)[:n]]


def score(frames, check, seeds):
    rs = [metrics.realism_scores(frames, check, _S["pc"], 10000, seed=s) for s in seeds]
    raw = {k: float(np.mean([r["raw"][k] for r in rs])) for k in rs[0]["raw"]}
    norm = {k: float(np.mean([r["norm"][k] for r in rs])) for k in rs[0]["norm"]}
    return dict(raw=raw, norm=norm, score=float(np.mean([r["score"] for r in rs])),
                score_sd=float(np.std([r["score"] for r in rs])))


def run(cfg):
    t = time.time()
    syn = pick_frames(_S["cands"], _S["fit"], cfg["voxels"], cfg["pick"])
    frames = add_zeros(calibrate(syn, _S["fit"], cfg["voxels"], cfg["align"], cfg["jitter"]), _S["zero_frac"],
                       np.random.default_rng(1))
    out = dict(cfg=cfg, check=score(frames, _S["check"], _S["seeds"]), seconds=round(time.time() - t, 1))
    print(json.dumps({k: (round(v["score"], 4) if k == "check" else v) for k, v in out.items()}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--syn", required=True, help="directory of candidate synthetic frames")
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--real_fit", required=True)
    ap.add_argument("--real_check", required=True)
    ap.add_argument("--zeros", choices=["drop", "keep"], default="drop")
    ap.add_argument("--grid", default="default", choices=list(GRIDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pc = TRACKS[a.track]["pc_range"]

    def real(ids_file):
        ids = [l.strip() for l in open(ids_file) if l.strip()]
        for i in ids:
            guard.check(a.track, i)
        return [metrics.mask_range(_load(Path(a.real_dir) / f"{i}.npy"), pc) for i in ids]

    fit, check = real(a.real_fit), real(a.real_check)
    zero_frac = 0.0
    if a.zeros == "keep":
        zero_frac = float(np.mean([1 - len(nonzero(f)) / max(len(f), 1) for f in fit]))
    else:
        check = [nonzero(f) for f in check]
    fit = [nonzero(f) for f in fit]
    cands = [metrics.mask_range(_load(p), pc) for p in sorted(Path(a.syn).glob("*.npy"))]
    print(f"{len(cands)} candidate synthetic frames, {len(fit)} fit / {len(check)} check real frames, "
          f"zero_frac {zero_frac:.3f}", flush=True)
    g = GRIDS[a.grid]
    cfgs = [dict(voxels=v, align=al, jitter=j, pick=p)
            for v, al, j, p in itertools.product(g["voxels"], g["align"], g["jitter"], g["pick"])]
    state = dict(pc=pc, cands=cands, fit=fit, check=check, zero_frac=zero_frac, seeds=a.seeds)
    ref = dict(real_fit_vs_check=None)
    _init(state)
    ref["real_fit_vs_check"] = score(add_zeros(fit[:50], zero_frac, np.random.default_rng(1)), check, a.seeds)
    print("real fit vs check:", round(ref["real_fit_vs_check"]["score"], 4), flush=True)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(state,)) as ex:
        res = list(ex.map(run, cfgs))
    res.sort(key=lambda r: -r["check"]["score"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(ref=ref, results=res), indent=1))
    for r in res[:5]:
        print(round(r["check"]["score"], 4), {k: round(v, 3) for k, v in r["check"]["raw"].items()}, r["cfg"])


if __name__ == "__main__":
    main()
