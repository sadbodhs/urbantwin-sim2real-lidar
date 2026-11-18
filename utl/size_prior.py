"""Label-free size prior at inference: grow the boxes of a class to at least a public reference size, keeping each box's
centre in the ground plane and its bottom. Default: Bicycle to the Waymo Open Dataset cyclist size 1.78 x 0.84 x 1.78 m
(the synthetic LUMPI bicycle box is 1.68 x 0.60 m, which caps the IoU with a real ridden bicycle).

    python -m utl.size_prior --pred in.json --out out.json [--prior Bicycle:1.78,0.84,1.78] [--mode max|set]
"""
import argparse
import json
from pathlib import Path


def apply(dets, priors, mode="max"):
    out = []
    for d in dets:
        p = priors.get(d["class"])
        if p:
            b = list(d["box"])
            new = [max(x, y) if mode == "max" else y for x, y in zip(b[3:6], p)]
            b[2] += (new[2] - b[5]) / 2  # keep the bottom
            b[3:6] = new
            d = {**d, "box": b}
        out.append(d)
    return out


def parse(specs):
    return {s.split(":")[0]: [float(v) for v in s.split(":")[1].split(",")] for s in specs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior", nargs="+", default=["Bicycle:1.78,0.84,1.78"])
    ap.add_argument("--mode", choices=["max", "set"], default="max")
    a = ap.parse_args()
    pri = parse(a.prior)
    pred = json.loads(Path(a.pred).read_text())
    Path(a.out).write_text(json.dumps({f: apply(v, pri, a.mode) for f, v in pred.items()}))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
