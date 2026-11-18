"""Weighted box fusion of same-class boxes from several views or models (numpy only, used by utl.infer for flip TTA
and by utl.ensemble for model ensembles)."""
import numpy as np

from . import metrics


def wbf(views, iou_thr=0.55, n_views=1, one_per_view=False):
    """Fuse same-class boxes from several views: cluster greedily by 3D IoU, score-weighted mean geometry,
    heading taken from the top box, score = sum / n_views. n_views may be a function of the fused box
    (utl.infer --rot: the number of views that count for that box).
    one_per_view (opt-in, fusion fix F1/F2, Nov 8): the score sums only each view's best box in the cluster (a view's other
    boxes there are absorbed without adding to the score); the clusters and their geometry are exactly as without it. Off by default."""
    boxes = np.concatenate([v[0] for v in views]) if views else np.zeros((0, 7))
    scores = np.concatenate([v[1] for v in views]) if views else np.zeros(0)
    if not len(scores):
        return boxes, scores
    src = np.concatenate([np.full(len(v[1]), k) for k, v in enumerate(views)])
    order = np.argsort(-scores)
    boxes, scores, src = boxes[order], scores[order], src[order]
    iou = metrics.iou3d(boxes, boxes)
    done = np.zeros(len(scores), bool)
    out_b, out_s = [], []
    for i in range(len(scores)):
        if done[i]:
            continue
        m = (~done) & (iou[i] >= iou_thr)
        m[i] = True
        done |= m
        w = scores[m] / scores[m].sum()
        b = boxes[i].copy()
        b[:6] = (boxes[m, :6] * w[:, None]).sum(0)
        out_b.append(b)
        tot = scores[m].sum() if not one_per_view else sum(scores[m][src[m] == k].max() for k in np.unique(src[m]))
        out_s.append(min(tot / (n_views(b) if callable(n_views) else n_views), 1.0))
    return np.array(out_b), np.array(out_s)
