"""Long-truck samples for gt_sampling (plan item L5): stretch the synthetic Truck objects of the GT database along
their heading (and slightly upwards) to the lengths of real rigid and articulated trucks.

The synthetic LUMPI trucks are 4.4-7.2 m long, so a detector trained on them never outputs a longer truck box.
Target sizes come from the EU limits for road vehicles (Directive 96/53/EC: rigid trucks up to 12 m, articulated
vehicles up to 16.5 m, height up to 4 m), not from any real-data labels. Every Truck sample gets --copies stretched
copies with length ~ U(--min_len, --max_len) and height x U(1, --max_hscale) capped at 4 m; the bottom stays on the
ground. The copies go to <root>/gt_database_long/ and <root>/dbinfos_train_long.pkl, which a config lists after
dbinfos_train.pkl in DB_INFO_PATH (OpenPCDet merges the lists per class).

    python -m utl.stretch_db --root data/pcdet/ut_lumpi_fix
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def stretch(points, box, length, hscale, max_h=4.0):
    """points: (N, 3) relative to the box centre; returns stretched points and box (bottom kept)."""
    x, y, z, l, w, h, yaw = box
    h2 = min(h * hscale, max(h, max_h))
    c, s = np.cos(yaw), np.sin(yaw)
    u = points[:, 0] * c + points[:, 1] * s
    v = -points[:, 0] * s + points[:, 1] * c
    u = u * (length / l)
    zz = (points[:, 2] + h / 2) * (h2 / h) - h2 / 2
    out = np.stack([u * c - v * s, u * s + v * c, zz], 1).astype(np.float32)
    return out, np.array([x, y, z - h / 2 + h2 / 2, length, w, h2, yaw], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cls", default="Truck")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--min_len", type=float, default=7.5)
    ap.add_argument("--max_len", type=float, default=16.5)
    ap.add_argument("--max_hscale", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    db = pickle.loads((root / "dbinfos_train.pkl").read_bytes())
    rng = np.random.default_rng(a.seed)
    (root / "gt_database_long").mkdir(exist_ok=True)
    out = {c: [] for c in db}
    for d in db[a.cls]:
        pts = np.fromfile(root / d["path"], np.float32).reshape(-1, 3)
        for k in range(a.copies):
            p2, b2 = stretch(pts, np.asarray(d["box3d_lidar"], np.float64),
                             rng.uniform(a.min_len, a.max_len), rng.uniform(1.0, a.max_hscale))
            path = f"gt_database_long/{Path(d['path']).stem}_long{k}.bin"
            p2.tofile(root / path)
            out[a.cls].append({**d, "path": path, "box3d_lidar": b2})
    (root / "dbinfos_train_long.pkl").write_bytes(pickle.dumps(out))
    L = np.array([d["box3d_lidar"][3] for d in out[a.cls]])
    H = np.array([d["box3d_lidar"][5] for d in out[a.cls]])
    print(f"{len(out[a.cls])} long {a.cls} samples: length {L.min():.1f}-{L.max():.1f} m, "
          f"height median {np.median(H):.2f} m -> {root / 'dbinfos_train_long.pkl'}")


if __name__ == "__main__":
    main()
