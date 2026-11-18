"""Compare a rebuilt predictions.json with the ranked one, box by box.

    python tools/compare_predictions.py --ref predictions.json --new rebuilt_predictions.json [--min_score 0.05]

Re-running the GPU steps does not give bit-identical floats (different GPUs and CUDA atomics), so the rebuild is
judged by how closely its boxes match: within each frame and class, every reference box at or above --min_score
is paired with the nearest unused rebuilt box whose centre lies within --radius metres, highest scores first.
Box format: {"box": [x, y, z, l, w, h, yaw], "score": s, "class": c}.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(p):
    b = Path(p).read_bytes()
    return json.loads(b), hashlib.md5(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--min_score", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=0.5)
    a = ap.parse_args()
    ref, mr = load(a.ref)
    new, mn = load(a.new)
    print(f"md5 ref {mr}\nmd5 new {mn}  {'IDENTICAL' if mr == mn else 'different bytes'}")
    if set(ref) != set(new):
        print(f"frame sets differ: {len(set(ref) - set(new))} missing, {len(set(new) - set(ref))} extra")
    stats = defaultdict(lambda: dict(ref=0, new=0, matched=0, dpos=[], dsize=[], dyaw=[], dscore=[]))
    for f in ref:
        r_by, n_by = defaultdict(list), defaultdict(list)
        for d in ref[f]:
            if d["score"] >= a.min_score:
                r_by[d["class"]].append(d)
        for d in new.get(f, []):
            if d["score"] >= a.min_score:
                n_by[d["class"]].append(d)
        for c in set(r_by) | set(n_by):
            s = stats[c]
            R = sorted(r_by[c], key=lambda d: -d["score"])
            N = n_by[c]
            s["ref"] += len(R)
            s["new"] += len(N)
            if not N:
                continue
            nb = np.array([d["box"] for d in N], float)
            used = np.zeros(len(N), bool)
            for d in R:
                b = np.asarray(d["box"], float)
                dist = np.linalg.norm(nb[:, :3] - b[:3], axis=1)
                dist[used] = np.inf
                j = int(np.argmin(dist))
                if dist[j] > a.radius:
                    continue
                used[j] = True
                s["matched"] += 1
                s["dpos"].append(dist[j])
                s["dsize"].append(np.abs(nb[j, 3:6] - b[3:6]).max())
                dy = (nb[j, 6] - b[6] + np.pi) % (2 * np.pi) - np.pi
                s["dyaw"].append(min(abs(dy), np.pi - abs(dy)))
                s["dscore"].append(abs(N[j]["score"] - d["score"]))
    print(f"\nboxes with score >= {a.min_score}; a match is the nearest same-class box within {a.radius} m")
    print(f"{'class':12s} {'ref':>8s} {'new':>8s} {'matched':>8s} {'match%':>7s} {'dpos p50/p99/max':>24s} "
          f"{'dsize p50/p99/max':>24s} {'dyaw max':>9s} {'dscore p50/p99/max':>24s} {'mean dscore':>11s}")
    for c in sorted(stats):
        s = stats[c]
        q = lambda k: (np.percentile(s[k], 50), np.percentile(s[k], 99), max(s[k])) if s[k] else (0.0, 0.0, 0.0)  # noqa: E731
        f3 = lambda t: "/".join(f"{v:.4f}" for v in t)  # noqa: E731
        mean = float(np.mean(s["dscore"])) if s["dscore"] else 0.0
        pct = 100.0 * s["matched"] / max(s["ref"], 1)
        print(f"{c:12s} {s['ref']:8d} {s['new']:8d} {s['matched']:8d} {pct:6.2f}% {f3(q('dpos')):>24s} {f3(q('dsize')):>24s} "
              f"{q('dyaw')[2]:9.4f} {f3(q('dscore')):>24s} {mean:11.5f}")


if __name__ == "__main__":
    main()
