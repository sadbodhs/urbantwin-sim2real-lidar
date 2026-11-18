"""Label-free box-height refinement: keep each predicted box's bottom and footprint, and set its top from the LiDAR
points inside the footprint (a high percentile of their heights), clipped to a plausible range per class.

On V2X-Real the detector's box bottoms are right (median error 0.03 m) but heights are not: vehicles come out
1.55 m tall against about 1.70 m, trucks 3.6 m against about 2.45 m. The roadside sensors look down on objects, so
roofs are well covered by points.

    python -m utl.height_refine --track v2x --pred pred.json --frames data/pcdet/v2x_real_50k/points --out out.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

# class: (min height, max height, percentile of point heights taken as the top) — generic physical ranges
RANGES = {"vehicle": (1.3, 2.3, 98), "truck": (1.8, 4.2, 98), "Car": (1.3, 2.3, 98), "Van": (1.7, 3.0, 98),
          "Truck": (1.8, 4.2, 98), "Bus": (2.5, 4.2, 98)}


def inside_footprint(points, box, margin=0.1):
    x, y, z, l, w, h, yaw = box
    d = points[:, :2] - (x, y)
    c, s = np.cos(yaw), np.sin(yaw)
    u, v = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    return (np.abs(u) <= l / 2 + margin) & (np.abs(v) <= w / 2 + margin)


def refine(dets, points, min_points=5, pct=None):
    out = []
    for d in dets:
        b = list(d["box"])
        r = RANGES.get(d["class"])
        if r is not None and pct is not None:
            r = (r[0], r[1], pct)
        if r is not None:
            bottom = b[2] - b[5] / 2
            m = inside_footprint(points, b)
            zs = points[m, 2]
            zs = zs[(zs > bottom + 0.2) & (zs < bottom + r[1] + 0.5)]
            if len(zs) >= min_points:
                h = float(np.clip(np.percentile(zs, r[2]) - bottom, r[0], r[1]))
                b[5], b[2] = h, bottom + h / 2
        out.append({**d, "box": b})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True, help="directory with the frames the predictions were made on")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pred = json.loads(Path(a.pred).read_text())
    out = {}
    for fid, dets in pred.items():
        f = next(Path(a.frames).glob(f"{fid}.*"))
        pts = np.load(f) if f.suffix == ".npy" else np.fromfile(f, np.float32).reshape(-1, 3)
        pts = pts[np.abs(pts[:, :3]).sum(1) > 0][:, :3]
        out[fid] = refine(dets, pts)
    Path(a.out).write_text(json.dumps(out))
    print(f"refined {sum(len(v) for v in out.values())} boxes in {len(out)} frames -> {a.out}")


if __name__ == "__main__":
    main()
