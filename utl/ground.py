"""Ground-plane alignment of a frame to the training data's ground plane (points only, no labels).

The V2X-Real infrastructure sensors are tilted differently: in its own frame, the ground slopes -7.1 deg (front-back)
for the synthetic twin and the real OS1-64 sensor, but +3.3 deg for the real OS1-128 sensor, so far objects sit
metres above or below anything seen in training. Each frame's ground plane z = a*x + b*y + c is fitted to the lowest
point of 1 m cells at 5-35 m range; a rigid transform then maps it onto a reference plane (the synthetic one), the
detector runs on the aligned frame, and boxes are mapped back.
"""
import numpy as np


def fit_plane(points, cell=1.0, r0=5.0, r1=35.0):
    """-> (a, b, c) of z = a*x + b*y + c, robust to objects and walls."""
    p = points[np.abs(points[:, :3]).sum(1) > 0]
    r = np.hypot(p[:, 0], p[:, 1])
    p = p[(r > r0) & (r < r1)]
    ij = np.floor(p[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 100003 + ij[:, 1]
    order = np.lexsort((p[:, 2], key))
    key, p = key[order], p[order]
    _, first = np.unique(key, return_index=True)
    g = p[first, :3].astype(np.float64)
    coef = np.zeros(3)
    for _ in range(3):
        A = np.c_[g[:, 0], g[:, 1], np.ones(len(g))]
        coef, *_ = np.linalg.lstsq(A, g[:, 2], rcond=None)
        res = g[:, 2] - A @ coef
        g = g[np.abs(res) < max(0.3, 2.5 * np.median(np.abs(res)))]
    return coef


def _normal(abc):
    n = np.array([-abc[0], -abc[1], 1.0])
    return n / np.linalg.norm(n)


def _rot(u, v):
    """Smallest rotation taking unit vector u onto unit vector v (Rodrigues)."""
    w = np.cross(u, v)
    s, c = np.linalg.norm(w), float(u @ v)
    if s < 1e-12:
        return np.eye(3)
    k = w / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def alignment(frame_abc, ref_abc):
    """-> (R, t) with p_aligned = R @ p + t, mapping the frame's ground plane onto the reference plane."""
    nf, nr = _normal(frame_abc), _normal(ref_abc)
    R = _rot(nf, nr)
    q = R @ np.array([0.0, 0.0, frame_abc[2]])           # a point of the rotated frame plane
    z_at_origin = q[2] + (nr[0] * q[0] + nr[1] * q[1]) / nr[2]
    t = np.array([0.0, 0.0, ref_abc[2] - z_at_origin])
    return R, t


def apply_points(points, R, t):
    out = points.copy()
    nz = np.abs(points[:, :3]).sum(1) > 0                 # keep (0,0,0) no-return points where they are
    out[nz, :3] = points[nz, :3] @ R.T + t
    return out


def boxes_back(boxes, R, t):
    """Map 7-DoF boxes (x, y, z, l, w, h, yaw) from the aligned frame back to the sensor frame."""
    if not len(boxes):
        return boxes
    out = boxes.copy()
    out[:, :3] = (boxes[:, :3] - t) @ R                   # R^-1 = R^T
    head = np.stack([np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))], 1) @ R
    out[:, 6] = np.arctan2(head[:, 1], head[:, 0])
    return out


def main():
    """Reference ground plane of a synthetic training set: median plane over evenly spaced frames.

        python -m utl.ground --frames data/pcdet/ut_v2x_fix/points --out configs/ground_ref_v2x.json
    """
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(Path(a.frames).glob("*.npy"))
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(a.n, len(files))).round().astype(int)]
    abc = np.median([fit_plane(np.load(f)[:, :3]) for f in files], 0)
    Path(a.out).write_text(json.dumps({"abc": abc.tolist(), "frames": a.frames, "n": len(files)}, indent=1))
    print("reference plane z = %.4f x + %.4f y + %.3f" % tuple(abc))


if __name__ == "__main__":
    main()
