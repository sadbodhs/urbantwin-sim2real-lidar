"""Label-free shape demotions for person boxes, shared by LUMPI (Person) and V2X (pedestrian); score x 0.3, boxes with
score >= 0.05, >= 5 points. Settings fixed in advance.
pole: lumpi_pole.ratio >= 0.1 (points continue straight up above the box top: a pole, sign or post).
short: lumpi_rider.span < H (height span p98 - p2 of the points in the footprint + 0.15 m, 0.5 m below to 2.5 m above
the bottom; a standing person spans 1.5-1.9 m). LUMPI P4: short at H 1.2. V2X: pole or short at H 1.0 (V3 + V4).
tall: the same span > T (a pole, sign or tree trunk reaching well above a person); V2X V6: T 2.3 m, a second pass after
V3 + V4."""
import lumpi_pole as P3
import lumpi_rider as R

F = 0.3


def apply(pred, load, cls, pole=False, short=None, tall=None):
    """pred {frame: [det]}, load(frame) -> (N, 3) points. Returns a new prediction dict."""
    out = {}
    for f, v in pred.items():
        P = None
        dd = []
        for d in v:
            if d["class"] == cls and d["score"] >= 0.05:
                P = load(f) if P is None else P
                hit = False
                if pole:
                    r = P3.ratio(d, P)
                    hit = r is not None and r >= P3.C
                if not hit and (short is not None or tall is not None):
                    s = R.span(d, P)
                    hit = s is not None and ((short is not None and s < short) or (tall is not None and s > tall))
                if hit:
                    d = dict(d, score=d["score"] * F)
            dd.append(d)
        out[f] = dd
    return out
