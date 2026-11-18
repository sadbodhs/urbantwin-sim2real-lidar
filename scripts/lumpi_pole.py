"""LUMPI Person pole test (P3, label-free): a Person box on a pole, sign or post has points continuing straight up above
its top; a pedestrian does not. For each Person box with score >= 0.05: inside = points in the footprint grown by
0.15 m, from the box bottom to its top; column = points within 0.3 m (BEV) of the box centre, 0.1 to 1.2 m above the
top. With >= 5 inside points and column / inside >= 0.1 -> score x 0.3. Settings fixed in advance
(val Person 0.2815 -> 0.2944, up on both halves)."""
import numpy as np

C, F, MARGIN, MIN_PTS, RAD = 0.1, 0.3, 0.15, 5, 0.3


def ratio(d, P):
    b = np.asarray(d["box"], float)
    dx, dy = P[:, 0] - b[0], P[:, 1] - b[1]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    lx, ly = c * dx - s * dy, s * dx + c * dy
    bot, top = b[2] - b[5] / 2, b[2] + b[5] / 2
    inside = (np.abs(lx) <= b[3] / 2 + MARGIN) & (np.abs(ly) <= b[4] / 2 + MARGIN) & (P[:, 2] > bot) & (P[:, 2] < top)
    col = (np.hypot(lx, ly) < RAD) & (P[:, 2] > top + 0.1) & (P[:, 2] < top + 1.2)
    n_in = int(inside.sum())
    return col.sum() / n_in if n_in >= MIN_PTS else None


def apply(pred, load):
    """pred {frame: [det]}, load(frame) -> (N, 3) points. Returns a new prediction dict."""
    out = {}
    for f, v in pred.items():
        P = None
        dd = []
        for d in v:
            if d["class"] == "Person" and d["score"] >= 0.05:
                P = load(f) if P is None else P
                r = ratio(d, P)
                if r is not None and r >= C:
                    d = dict(d, score=d["score"] * F)
            dd.append(d)
        out[f] = dd
    return out
