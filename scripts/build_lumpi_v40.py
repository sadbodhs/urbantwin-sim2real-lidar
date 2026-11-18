"""lumpi_v40 = lumpi_v37 with Person from three second stages averaged with equal weights (L11, L12, L11b second seed; down to score 0.01), then the same size change and length rule as v37. Other classes and realism frames unchanged; no boxes added.
Per box: score = sqrt(score x mean predicted IoU); box = mean of the pass-1 boxes whose centre moved <= 0.5 m, else the original box. Val vs v37: Person 0.4257 -> 0.4333 (0.4730 -> 0.4786, 0.3631 -> 0.3739), IoU 0.7 0.0122 -> 0.0145. Fresh: 0.4348 -> 0.4454 (0.4906 -> 0.4987, 0.3581 -> 0.3679), IoU 0.7 0.0126 -> 0.0117, so it fails the gate on fresh IoU 0.7. A single check; alternative package only, whatever the test score.
    python scripts/build_lumpi_v40.py"""
import json
from pathlib import Path

import numpy as np

O = Path("output/lumpi/ens_x")
PATHS = ["output/lumpi/vrcnn_lumpi_fix/l11c/v29_test_ref_min01.json", "output/lumpi/vrcnn_lumpi_c/l12c/v29_test_ref_min01.json", "output/lumpi/vrcnn_lumpi_b_s2/s2c/v29_test_ref_min01.json"]


def person(ds):
    d0 = ds[0]
    b, s = list(d0["box"]), d0["score"]
    if all("ref1" in d for d in ds):
        iou = float(np.mean([max(d["iou1"], 0.0) for d in ds]))
        near = [d["ref1"] for d in ds if np.hypot(d["ref1"][0] - b[0], d["ref1"][1] - b[1]) <= 0.5]
        if near:
            a = np.array(near, float)
            b = a.mean(0).tolist()
            t = 0.5 * np.arctan2(np.sin(2 * a[:, 6]).mean(), np.cos(2 * a[:, 6]).mean())
            b[6] = float(t + np.round((a[0, 6] - t) / np.pi) * np.pi)
        s = float(np.sqrt(d0["score"] * iou))
    b = b[:3] + [b[3] + 0.044, b[4] + 0.023, b[5] + 0.10] + b[6:]
    return {"box": b, "score": s * 0.3 if b[3] >= 0.85 else s, "class": "Person"}


t = json.loads((O / "lumpi_v37_test.json").read_text())
src = []
for p in PATHS:
    r = json.loads(Path(p).read_text())
    src.append({f: [d for d in r[f] if d["class"] == "Person"] for f in t})
    del r
out = {}
for f in t:
    rows = [s[f] for s in src]
    assert all(len(r) == len(rows[0]) for r in rows)
    for tup in zip(*rows):
        assert all(d["box"] == tup[0]["box"] for d in tup)
    out[f] = [d for d in t[f] if d["class"] != "Person"] + [person(list(tup)) for tup in zip(*rows)]
(O / "lumpi_v40_test.json").write_text(json.dumps(out))
for n, p in (("v37", t), ("v40", out)):
    q = [d for v in p.values() for d in v if d["class"] == "Person"]
    print(n, "Person", len(q), "confident", sum(d["score"] >= 0.3 for d in q), "all boxes", sum(len(v) for v in p.values()), flush=True)
