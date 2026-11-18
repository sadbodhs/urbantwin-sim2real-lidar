"""Make validation frames look like the official test frames: cut to the track range, then keep a uniform random
subset of 50,000 points (the kit's LUMPI test frames are exactly that; V2X test frames also have 50,000 points with
the (0,0,0) no-returns kept). Labels and ImageSets are linked from the source set.

    python -m utl.thin --track lumpi --src data/pcdet/lumpi_real --out data/pcdet/lumpi_real_50k
"""
import argparse
import os
from pathlib import Path

import numpy as np

from . import metrics
from .tracks import TRACKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    for d in ("labels", "ImageSets"):
        if not (out / d).exists():
            os.symlink(os.path.relpath(src / d, out), out / d)
    pc = TRACKS[a.track]["pc_range"]
    ids = [l.strip() for l in open(src / "ImageSets" / "val.txt") if l.strip()]
    kept = []
    for k, i in enumerate(ids):
        p = np.load(src / "points" / f"{i}.npy")[:, :3]
        p = metrics.mask_range(p, pc)
        rng = np.random.default_rng([a.seed, k])
        if len(p) > a.n:
            p = p[rng.choice(len(p), a.n, replace=False)]
        np.save(out / "points" / f"{i}.npy", p.astype(np.float32))
        kept.append(len(p))
    print(f"{len(ids)} frames -> {out}, points per frame {min(kept)}-{max(kept)}")


if __name__ == "__main__":
    main()
