"""Build v2x_v19 = v2x_v18 with its pedestrians and trucks rescored and refined by the X6b second stage, variant c12: pass-1 box if its centre moved <= 0.5 m
(pedestrian) or <= 1 m (truck), score sqrt(score x predicted IoU); trucks keep width >= 2.55 m (scripts/x6bc_refine.sh, scripts/x6b_eval.py). Vehicles exactly v18. No boxes added.
OS1-64 val_i1 vs v18: pedestrian 0.2334 -> 0.2862 (halves 0.1866 -> 0.2554, 0.2775 -> 0.3182), IoU 0.7 0.0003 -> 0.0022; truck 0.5875 -> 0.6361 (0.4808 -> 0.4859, 0.6961 -> 0.7369), IoU 0.7 0.3373 -> 0.3736.
Writes output/v2x/ens/v19_test.json and v19_val_i1.json."""
import json
from pathlib import Path

import numpy as np

CAP = {"pedestrian": 0.5, "truck": 1.0}
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}


def c12(d):
    if "ref1" not in d:
        out = keep(d)
    else:
        r = d["ref1"]
        b = list(r) if np.hypot(r[0] - d["box"][0], r[1] - d["box"][1]) <= CAP[d["class"]] else list(d["box"])
        out = {"box": b, "score": float(np.sqrt(d["score"] * max(d["iou1"], 0.0))), "class": d["class"]}
    if d["class"] == "truck":
        out["box"] = out["box"][:4] + [max(out["box"][4], 2.55)] + out["box"][5:]
    return out


for split, tag in (("test", "test"), ("val_i1", "val_i1")):
    v18 = json.loads(Path(f"output/v2x/ens/v18_{split}.json").read_text())
    ref = json.loads(Path(f"output/v2x/vrcnn_v2x_b/x6bc/v18_{split}_ref.json").read_text())
    assert set(v18) == set(ref)
    out = {f: sorted([keep(d) for d in v18[f] if d["class"] == "vehicle"] + [c12(d) for d in ref[f] if d["class"] != "vehicle"], key=lambda d: -d["score"]) for f in v18}
    Path(f"output/v2x/ens/v19_{tag}.json").write_text(json.dumps(out))
    print(split, "written", sum(map(len, out.values())), "boxes, v18 had", sum(map(len, v18.values())))
