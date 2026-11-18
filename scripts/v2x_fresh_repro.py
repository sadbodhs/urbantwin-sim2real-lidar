"""V2X fresh job, first check: each member re-predicted on the first 10 OS1-64 val frames (data/pcdet/v2x_real_50k/ImageSets/repro10.txt) with the
settings used for fresh must give its saved pred_val50k.json back on those frames (same boxes per class, boxes and scores within 1e-4). Exit 1 on any mismatch.
    UT_NOGPU=1 python scripts/v2x_fresh_repro.py output/v2x/pp_v2x/kit_mc ..."""
import json
import sys
from pathlib import Path

import numpy as np

ids = [l.strip() for l in open("data/pcdet/v2x_real_50k/ImageSets/repro10.txt") if l.strip()]
bad = 0
for d in sys.argv[1:]:
    a = json.loads((Path(d) / "pred_repro10.json").read_text())
    b = json.loads((Path(d) / "pred_val50k.json").read_text())
    ok = True
    for f in ids:
        P = lambda m: np.array(sorted([[("vehicle", "pedestrian", "truck").index(x["class"]), x["score"]] + list(x["box"]) for x in m.get(f, [])]))
        pa, pb = P(a), P(b)
        if pa.shape != pb.shape or (pa.size and np.abs(pa - pb).max() > 1e-4):
            ok = False
    print(d, "re-prediction equals the saved val predictions:", ok, flush=True)
    bad += not ok
sys.exit(1 if bad else 0)
