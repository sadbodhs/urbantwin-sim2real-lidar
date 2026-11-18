"""Write the 50 submission realism frames for one calibration setting chosen with utl.realism_sweep /
utl.realism_robust, plus a report scored on the held-out check frames.

    python -m utl.realism_emit --track lumpi --syn data/pcdet/ut_lumpi/points --real_dir data/pcdet/lumpi_real/points \
        --real_fit .../realism_fit.txt --real_check .../realism_check.txt \
        --cfg '{"voxels": [[6, 6, 2]], "align": true, "jitter": 0.0, "pick": "match"}' --out output/lumpi/realism_v2
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import guard, metrics, realism_sweep as rs
from .realism import MAX_POINTS, _load, add_zeros, nonzero
from .tracks import TRACKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--syn", required=True)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--real_fit", required=True)
    ap.add_argument("--real_check", nargs="+", required=True, help="name=ids_file, one or more")
    ap.add_argument("--zeros", choices=["drop", "keep"], default="drop")
    ap.add_argument("--cfg", required=True, help="JSON calibration setting")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pc = TRACKS[a.track]["pc_range"]
    cfg = json.loads(a.cfg)

    def load(ids_file):
        ids = [l.strip() for l in open(ids_file) if l.strip()]
        for i in ids:
            guard.check(a.track, i)
        return [metrics.mask_range(_load(Path(a.real_dir) / f"{i}.npy"), pc) for i in ids]

    fit_raw = load(a.real_fit)
    zero_frac = float(np.mean([1 - len(nonzero(f)) / max(len(f), 1) for f in fit_raw])) if a.zeros == "keep" else 0.0
    fit = [nonzero(f) for f in fit_raw]
    checks = {n: load(p) for n, p in (s.split("=", 1) for s in a.real_check)}
    if a.zeros == "drop":
        checks = {n: [nonzero(f) for f in fr] for n, fr in checks.items()}
    files = sorted(Path(a.syn).glob("*.npy"))
    cands = [metrics.mask_range(_load(p), pc) for p in files]
    rs._init(dict(pc=pc, cands=cands))
    picked = rs.pick_frames(list(range(len(cands))), None, None, "spread") if cfg["pick"] == "spread" else None
    if picked is None:
        # same selection as realism_sweep.pick_frames, keeping track of which files were chosen
        g = rs.Grid(pc, cfg["voxels"][0])
        d_real = g.density(fit)
        d_real /= d_real.sum()
        cost = [np.abs(g.density([f]) / max(len(f), 1) - d_real).sum() for f in cands]
        picked = list(np.argsort(cost)[:50])
    syn = [cands[i] for i in picked]
    frames = add_zeros(rs.calibrate(syn, fit, cfg["voxels"], cfg["align"], cfg["jitter"]), zero_frac,
                       np.random.default_rng(1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames, 1):
        assert len(f) <= MAX_POINTS + 5000 and np.isfinite(f).all()
        f.astype(np.float32).tofile(out / f"{i:04d}.bin")
    report = dict(cfg=cfg, zeros=a.zeros, zero_frac=zero_frac, synthetic_sources=[files[i].name for i in picked],
                  checks={n: rs.score(frames, chk, a.seeds) for n, chk in checks.items()},
                  max_bytes=max((out / f"{i:04d}.bin").stat().st_size for i in range(1, 51)))
    (out / "realism_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({n: round(r["score"], 4) for n, r in report["checks"].items()}), "max bytes", report["max_bytes"])


if __name__ == "__main__":
    main()
