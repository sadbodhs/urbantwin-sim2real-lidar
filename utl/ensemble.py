"""Ensemble of several models' predictions.json: per frame and class, boxes of all models are fused with weighted
box fusion (utl.fusion), and a box found by every model keeps a higher score than one found by a single model.

    python -m utl.ensemble --preds a.json b.json --out ens.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .fusion import wbf


def ensemble(preds, weights, iou_thr=0.55, class_weights=None, classes=None, one_per_member=False):
    """class_weights {class: [weight per member]} overrides `weights` for that class (0 leaves the member out of it), and
    that class's scores are divided by the sum of its own weights; other classes keep `weights` and their sum, as
    before. classes: fuse only these classes (others are dropped).
    one_per_member (opt-in, fusion fix F1, Nov 8): a cluster's score counts only each member's best box in it (weighted);
    clusters and geometry unchanged (utl.fusion.wbf one_per_view). Off by default.
    All members must cover the same frames and len(weights) == len(preds) (fails loudly otherwise)."""
    out = {}
    class_weights = class_weights or {}
    assert len(weights) == len(preds), f"{len(weights)} weights for {len(preds)} members"
    assert all(len(w) == len(preds) for w in class_weights.values()), "class_weights: one weight per member"
    for k, p in enumerate(preds[1:], 1):
        assert set(p) == set(preds[0]), f"member {k} covers other frames than member 0"
    for fid in preds[0]:
        views = defaultdict(list)
        for k, (p, w) in enumerate(zip(preds, weights)):
            by_cls = defaultdict(list)
            for d in p.get(fid, []):
                if classes is None or d["class"] in classes:
                    by_cls[d["class"]].append(d)
            for c, ds in by_cls.items():
                wc = class_weights[c][k] if c in class_weights else w
                if wc:
                    views[c].append((np.array([d["box"] for d in ds], np.float64).reshape(-1, 7),
                                     np.array([d["score"] for d in ds], np.float64) * wc))
        dets = []
        for c, v in views.items():
            n = sum(class_weights[c]) if c in class_weights else sum(weights)
            b, s = wbf(v, iou_thr=iou_thr, n_views=n, one_per_view=one_per_member)
            dets += [{"box": [float(x) for x in bb], "score": float(ss), "class": c} for bb, ss in zip(b, s)]
        out[fid] = sorted(dets, key=lambda d: -d["score"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = [json.loads(Path(p).read_text()) for p in a.preds]
    out = ensemble(preds, a.weights or [1.0] * len(preds))
    Path(a.out).write_text(json.dumps(out))
    print(f"fused {len(preds)} models: {sum(len(v) for v in out.values())} boxes in {len(out)} frames -> {a.out}")


if __name__ == "__main__":
    main()
