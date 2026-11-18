"""SAFDNet on top of v2x_v29: joint fusion lanes, read by the same rules as v29.
Same chains as scripts/route_v2x_lanes.py, with FSHNet V2X
at weight 1 (as in v29's lanes) and SAFDNet V2X added as one more member at weight 1:
  vehicle     x7 members and weights + FSHNet 1 + SAFDNet 1, vehicle chain (WBF 0.55, height refine).
  pedestrian  x6bf members and weights + FSHNet 1 + SAFDNet 1, pedestrian chain, pole, short and tall tests.
Truck is not built (not read). Checks first, per split and class (STOP on a miss):
  both models at weight 0 give the v25 input back (x7 / x6bf); SAFDNet at weight 0 gives FSHNet's lane input back (output/v2x/fshnet_v2x/s2_in).
Writes output/v2x/fsh_saf_v2x/s2_in/{vehicle,pedestrian}_<split>.json.
    UT_NOGPU=1 python scripts/route_v2x_lanes_joint.py val_i1 [test] [fresh]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ped_shape as PS
from route_v2x_lanes import E, FRESH_MEMBERS, WP, WV, keep, same
from x2_fuse import chain

A, B = "fshnet_v2x", "safdnet_v2x"
OUT = Path("output/v2x/fsh_saf_v2x/s2_in")


def run(split):
    sfx = {"val_i1": "val50k", "test": "test", "fresh": "fresh50k"}[split]
    ref = {"vehicle": json.loads((E / f"x7_{split}.json").read_text()), "pedestrian": json.loads((E / f"x6bf_{split}.json").read_text())}
    fsh_in = {c: json.loads(Path(f"output/v2x/{A}/s2_in/{c}_{split}.json").read_text()) for c in ref}
    frames = sorted(json.loads((E / f"v18_{split}.json").read_text()))
    pts = {}
    for f in frames:
        if split == "test":
            a = np.fromfile(Path("data/kit/v2x/detection_test_frames") / f"{f}.bin", np.float32).reshape(-1, 3)
        else:
            root = "v2x_real_50k" if split == "val_i1" else "v2x_fresh_50k"
            a = np.load(Path("data/pcdet") / root / "points" / f"{f}.npy")[:, :3]
        pts[f] = a[np.abs(a).sum(1) > 0]
    rd = lambda p: (lambda m: {f: m.get(f, []) for f in frames})(json.loads(Path(p).read_text()))
    if split == "fresh":
        base5 = [rd(f"output/v2x/{m}/pred_{sfx}.json") for m in FRESH_MEMBERS]
    else:
        I = Path("output/v2x/sweep_in")
        base5 = [rd(I / f"{n}_{sfx}.json") for n in ("pp", "cp", "ped", "ppall")] + [rd(f"output/v2x/pp_v2x_walk/kit_mc/pred_{sfx}.json")]
    x6, x6b = rd(f"output/v2x/vrcnn_v2x/kit/pred_{sfx}.json"), rd(f"output/v2x/vrcnn_v2x_b/kit/pred_{sfx}.json")
    na, nb = rd(f"output/v2x/{A}/kit/pred_{sfx}.json"), rd(f"output/v2x/{B}/kit/pred_{sfx}.json")

    def veh(wa, wb):
        return chain("vehicle", WV + [wa, wb], base5 + [x6, na, nb], pts)

    def ped(wa, wb):
        p = chain("pedestrian", WP + [wa, wb], base5 + [x6b, na, nb], pts)
        p = PS.apply(p, lambda f: pts[f], "pedestrian", pole=True, short=1.0)
        return PS.apply(p, lambda f: pts[f], "pedestrian", tall=2.3)

    OUT.mkdir(parents=True, exist_ok=True)
    for c, fn in {"vehicle": veh, "pedestrian": ped}.items():
        ok0 = same(fn(0.0, 0.0), ref[c], c, 1e-6)
        print(split, c, "both models at weight 0 give the v25 input back:", ok0, flush=True)
        ok1 = same(fn(1.0, 0.0), fsh_in[c], c, 1e-6)
        print(split, c, "SAFDNet at weight 0 gives FSHNet's lane input (v29) back:", ok1, flush=True)
        if not (ok0 and ok1):
            sys.exit(f"STOP {split} {c}: base does not match")
        p = fn(1.0, 1.0)
        out = {f: sorted([keep(d) for d in p.get(f, []) if d["class"] == c], key=lambda d: -d["score"]) for f in frames}
        (OUT / f"{c}_{split}.json").write_text(json.dumps(out))
        n1 = sum(d["score"] >= 0.3 for v in fsh_in[c].values() for d in v)
        n2 = sum(d["score"] >= 0.3 for v in out.values() for d in v)
        print(split, c, "written, confident boxes FSHNet lane input", n1, "-> with SAFDNet", n2, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
