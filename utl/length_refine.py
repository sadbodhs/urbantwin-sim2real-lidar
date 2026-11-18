"""Label-free box-length refinement for long vehicles: extend a predicted Truck box along its heading while the LiDAR
points continue in its width corridor (above the ground, no gap longer than --gap), up to --max_len.

The synthetic LUMPI trucks are at most 7.2 m long, so the detector boxes a 12-16 m articulated truck as a short
truck on its cab or trailer. Boxes are only extended, never shortened; the far end keeps its position.

    python -m utl.length_refine --pred pred.json --frames data/pcdet/lumpi_real_50k/points --out out.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

BIN = 0.25


def extend(box, points, gap=0.75, max_len=16.5, margin=0.1, lo=0.3, hi=4.2):
    x, y, z, l, w, h, yaw = box
    bottom = z - h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    d = points[:, :2] - (x, y)
    u, v = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    m = (np.abs(v) <= w / 2 + margin) & (np.abs(u) <= max_len) & (points[:, 2] > bottom + lo) & (points[:, 2] < bottom + hi)
    occ = set(np.floor(u[m] / BIN).astype(int).tolist())
    max_gap = int(round(gap / BIN))

    def reach(start, step):  # last occupied bin when walking from the box end outwards, allowing gaps <= gap
        k, last, empty = start, start, 0
        while abs(k * BIN) < max_len and empty <= max_gap:
            k += step
            if k in occ:
                last, empty = k, 0
            else:
                empty += 1
        return last

    front = max(l / 2, (reach(int(np.floor(l / 2 / BIN)), 1) + 1) * BIN)
    back = max(l / 2, -reach(int(np.floor(-l / 2 / BIN)), -1) * BIN)
    if front + back > max_len:  # keep the longer extension within the limit, symmetric trim
        excess = front + back - max_len
        front, back = max(l / 2, front - excess / 2), max(l / 2, back - excess / 2)
    new_l = front + back
    off = (front - back) / 2
    return [x + off * c, y + off * s, z, new_l, w, h, yaw]


def refine(dets, points, classes=("Truck",), **kw):
    return [{**d, "box": extend(list(d["box"]), points, **kw)} if d["class"] in classes else d for d in dets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", default=["Truck"])
    ap.add_argument("--gap", type=float, default=0.75)
    ap.add_argument("--max_len", type=float, default=16.5)
    a = ap.parse_args()
    pred = json.loads(Path(a.pred).read_text())
    out = {}
    for fid, dets in pred.items():
        f = next(Path(a.frames).glob(f"{fid}.*"))
        pts = np.load(f) if f.suffix == ".npy" else np.fromfile(f, np.float32).reshape(-1, 3)
        out[fid] = refine(dets, pts[np.abs(pts[:, :3]).sum(1) > 0][:, :3], tuple(a.classes), gap=a.gap, max_len=a.max_len)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
