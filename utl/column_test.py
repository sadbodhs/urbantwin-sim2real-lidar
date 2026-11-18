"""Label-free 'column test' for person false positives: a real person has free space above the head, whereas poles,
sign posts and tree trunks continue upwards. A Person box with at least --k points inside its (slightly widened)
footprint between 0.3 m and 3 m above its top gets its score multiplied by --factor.

On LUMPI most confident Person false positives are static vertical structures far from any person.

    python -m utl.column_test --pred pred.json --frames data/pcdet/lumpi_real_50k/points --out out.json --k 10
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .height_refine import inside_footprint


def apply(dets, points, classes, k, factor, margin=0.15, lo=0.3, hi=3.0):
    out = []
    for d in dets:
        if d["class"] in classes:
            b = d["box"]
            top = b[2] + b[5] / 2
            zs = points[inside_footprint(points, b, margin), 2]
            if ((zs > top + lo) & (zs < top + hi)).sum() >= k:
                d = {**d, "score": d["score"] * factor}
        out.append(d)
    return sorted(out, key=lambda d: -d["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", default=["Person"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--factor", type=float, default=0.1)
    a = ap.parse_args()
    pred = json.loads(Path(a.pred).read_text())
    out = {}
    for fid, dets in pred.items():
        f = next(Path(a.frames).glob(f"{fid}.*"))
        pts = np.load(f) if f.suffix == ".npy" else np.fromfile(f, np.float32).reshape(-1, 3)
        out[fid] = apply(dets, pts[np.abs(pts[:, :3]).sum(1) > 0][:, :3], set(a.classes), a.k, a.factor)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
