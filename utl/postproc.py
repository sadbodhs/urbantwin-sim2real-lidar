"""Class-hypothesis copies for predictions.json (label-free post-processing, applied before utl.package).

Two synthetic classes are easily confused with real ones: real ridden bicycles (about 2.0 x 0.85 x 1.9 m with the
rider) look like the synthetic ridden motorcycles (2.11 x 0.77 x 1.72 m) far more than like the synthetic bicycle box
(1.68 x 0.60 m), and real vans/cars split between neighbouring vehicle classes. AP is computed per class, so emitting a
box under a second class adds a hypothesis to that class's ranked list without touching the source class.

  --copy SRC>DST          copy every SRC box to DST with the same score
  --copy SRC>DST:tail     copy with scores rescaled below the lowest DST score of that frame (ranked after all
                          native DST boxes, so the native ranking is unchanged); a frame without a native DST box uses the
                          lowest native DST score over all the frames given (was 1.0), and gets no tail
                          copies when no frame given has one (e.g. a per-frame call on such a frame)
  --drop CLS              remove CLS boxes after copying (their copies stay)
A copy that overlaps a native DST box (3D IoU >= --dedup_iou) is merged into it, keeping the higher score.

    python -m utl.postproc --pred in.json --out out.json --copy "Motorcycle>Bicycle" "Van>Truck:tail"
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics


def apply(frames, copies, drop=(), dedup_iou=0.5):
    out = {}
    gmin = {dst: min([d["score"] for v in frames.values() for d in v if d["class"] == dst], default=None)
            for _, dst, mode in copies if mode == "tail"}
    for fid, dets in frames.items():
        by_cls = defaultdict(list)
        for d in dets:
            by_cls[d["class"]].append(d)
        new = {c: list(v) for c, v in by_cls.items()}
        for src, dst, mode in copies:
            src_d = by_cls.get(src, [])
            if not src_d:
                continue
            native = by_cls.get(dst, [])
            scores = np.array([d["score"] for d in src_d])
            if mode == "tail":
                floor = min([d["score"] for d in native], default=gmin[dst])
                if floor is None:
                    continue
                scores = scores / max(scores.max(), 1e-9) * floor * 0.999
            cand = [{"box": d["box"], "score": float(s), "class": dst} for d, s in zip(src_d, scores)]
            if native and cand:
                iou = metrics.iou3d(np.array([c["box"] for c in cand]), np.array([n["box"] for n in native]))
                keep = []
                for i, c in enumerate(cand):
                    j = int(iou[i].argmax())
                    if iou[i, j] >= dedup_iou:
                        if c["score"] > new[dst][j]["score"] and mode != "tail":
                            new[dst][j] = {**new[dst][j], "score": c["score"]}
                    else:
                        keep.append(c)
                cand = keep
            new.setdefault(dst, []).extend(cand)
        for c in drop:
            new.pop(c, None)
        out[fid] = sorted((d for v in new.values() for d in v), key=lambda d: -d["score"])
    return out


def parse_copies(specs):
    res = []
    for s in specs:
        src, rest = s.split(">", 1)
        dst, _, mode = rest.partition(":")
        res.append((src, dst, mode or "same"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", nargs="*", default=[])
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--dedup_iou", type=float, default=0.5)
    a = ap.parse_args()
    frames = json.loads(Path(a.pred).read_text())
    out = apply(frames, parse_copies(a.copy), a.drop, a.dedup_iou)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}: {sum(len(v) for v in out.values())} boxes in {len(out)} frames")


if __name__ == "__main__":
    main()
