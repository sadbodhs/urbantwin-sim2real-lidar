"""V2X member route, fusion lanes (built Nov 8). A new model (FSHNet / SAFDNet / VoxelNeXt) joins
v2x_v25's chain as one extra fusion member at weight 1, one class at a time, exactly as PV-RCNN++ did (scripts/pv_v2x_prepare.py):
  vehicle     x7 members and weights (pp 1, cp 1.5, rider 0.5, all-frames 0.5, walker 1.5, X6 3) + model 1, vehicle chain (WBF 0.55, height refine).
  truck       pp 1 and all-frames pp 1 + model 1, WBF IoU 0.4, height refine, width >= 2.55 m.
  pedestrian  x6bf members and weights (pp 1, cp 1, rider 0.5, all-frames 0, walker 2, X6b 2) + model 1, pedestrian chain, pole, short and tall tests.
Check first, per split: with the model at weight 0 each chain gives the v25 inputs back (x7, v18, x6bf), else stop.
Splits: val_i1 and test; fresh only once the members' fresh predictions exist (scripts/v2x_fresh_job.sh).
Writes output/v2x/<model>/s2_in/{vehicle,truck,pedestrian}_<split>.json.
    UT_NOGPU=1 python scripts/route_v2x_lanes.py fshnet_v2x val_i1 test [fresh]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ped_shape as PS
from utl import height_refine
from utl.ensemble import ensemble
from x2_fuse import chain

E = Path("output/v2x/ens")
WV = [1.0, 1.5, 0.5, 0.5, 1.5, 3.0]
WP = [1.0, 1.0, 0.5, 0.0, 2.0, 2.0]
FRESH_MEMBERS = ["pp_v2x/kit_mc", "cp_v2x/kit", "pp_v2x_ped/kit_mc", "pp_v2x_all/kit_mc", "pp_v2x_walk/kit_mc"]
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}
only = lambda m, c: {f: [d for d in v if d["class"] == c] for f, v in m.items()}


def width(v):
    return [{**d, "box": d["box"][:4] + [max(d["box"][4], 2.55)] + d["box"][5:]} for d in v]


def same(a, b, cls, tol):
    for f in b:
        A = sorted([d["score"]] + list(d["box"]) for d in a.get(f, []) if d["class"] == cls)
        B = sorted([d["score"]] + list(d["box"]) for d in b[f] if d["class"] == cls)
        if len(A) != len(B) or (A and np.abs(np.array(A) - np.array(B)).max() > tol):
            return False
    return True


def run(model, split):
    sfx = {"val_i1": "val50k", "test": "test", "fresh": "fresh50k"}[split]
    ref = {"vehicle": json.loads((E / f"x7_{split}.json").read_text()), "truck": json.loads((E / f"v18_{split}.json").read_text()),
           "pedestrian": json.loads((E / f"x6bf_{split}.json").read_text())}
    frames = sorted(ref["truck"])
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
    new = rd(f"output/v2x/{model}/kit/pred_{sfx}.json")

    def veh(w):
        return chain("vehicle", WV + [w], base5 + [x6, new], pts)

    def trk(w):
        ms, ws = [only(base5[0], "truck"), only(base5[3], "truck")], [1.0, 1.0]
        if w:
            ms, ws = ms + [only(new, "truck")], ws + [w]
        e = ensemble(ms, ws, iou_thr=0.4)
        return {f: width(height_refine.refine(e.get(f, []), pts[f], pct=99.9)) for f in frames}

    def ped(w):
        p = chain("pedestrian", WP + [w], base5 + [x6b, new], pts)
        p = PS.apply(p, lambda f: pts[f], "pedestrian", pole=True, short=1.0)
        return PS.apply(p, lambda f: pts[f], "pedestrian", tall=2.3)

    O = Path(f"output/v2x/{model}/s2_in")
    O.mkdir(parents=True, exist_ok=True)
    for c, fn in {"vehicle": veh, "truck": trk, "pedestrian": ped}.items():
        ok = same(fn(0.0), ref[c], c, 1e-6)
        print(split, c, "model weight 0 gives the v25 input back:", ok, flush=True)
        if not ok:
            sys.exit(f"STOP {split} {c}: base does not match")
        p = fn(1.0)
        out = {f: sorted([keep(d) for d in p.get(f, []) if d["class"] == c], key=lambda d: -d["score"]) for f in frames}
        (O / f"{c}_{split}.json").write_text(json.dumps(out))
        n0 = sum(d["class"] == c and d["score"] >= 0.3 for v in ref[c].values() for d in v)
        n1 = sum(d["score"] >= 0.3 for v in out.values() for d in v)
        print(split, c, "written, confident boxes v25 input", n0, "-> with the model", n1, flush=True)


if __name__ == "__main__":
    for s in sys.argv[2:]:
        run(sys.argv[1], s)
