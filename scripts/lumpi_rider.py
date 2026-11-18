"""LUMPI Bicycle rider test (B1, label-free): LUMPI labels ridden bicycles (about 1.9 m tall); a riderless (parked) bike
is about 1.1 m. For each Bicycle box with score >= 0.05, take the points inside its footprint grown by 0.15 m, between
0.5 m below and 2.5 m above its bottom; with >= 5 points, span = p98(z) - p2(z). Span < 1.35 m -> score x 0.3.
Settings fixed in advance."""
import numpy as np

H, F, MARGIN, MIN_PTS = 1.35, 0.3, 0.15, 5


def span(d, P):
    b = np.asarray(d["box"], float)
    dx, dy = P[:, 0] - b[0], P[:, 1] - b[1]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    lx, ly = c * dx - s * dy, s * dx + c * dy
    bot = b[2] - b[5] / 2
    m = (np.abs(lx) <= b[3] / 2 + MARGIN) & (np.abs(ly) <= b[4] / 2 + MARGIN) & (P[:, 2] > bot - 0.5) & (P[:, 2] < bot + 2.5)
    z = P[m, 2]
    return float(np.percentile(z, 98) - np.percentile(z, 2)) if len(z) >= MIN_PTS else None


def apply(pred, load):
    """pred {frame: [det]}, load(frame) -> (N, 3) points. Returns a new prediction dict."""
    out = {}
    for f, v in pred.items():
        P = None
        dd = []
        for d in v:
            if d["class"] == "Bicycle" and d["score"] >= 0.05:
                P = load(f) if P is None else P
                s = span(d, P)
                if s is not None and s < H:
                    d = dict(d, score=d["score"] * F)
            dd.append(d)
        out[f] = dd
    return out
