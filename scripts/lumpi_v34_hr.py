"""Two Person tests on lumpi_v34, settings fixed before any reading, one reading each on 50k val. Other classes, box counts and realism identical; no boxes added.
H: every Person box h + 0.10 m, centre z, l, w, heading and score unchanged (value informed by val: labels 1.90 m, boxes 1.74 m; list it as val-informed in the report).
R: a Person box with final score in [0.3, 0.5) whose centre has a Person box with v29 score >= 0.05 within 0.5 m in at least 60% of the frames of its window (own frame included) gets score x 0.3. Label-free.
Gate against v34: Person >= +0.003 on both val halves, IoU 0.7 not down. If both pass, one reading of H + R, kept only if it beats the better of the two. Writes the variants' test files.
    python scripts/lumpi_v34_hr.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_static as S

O = Path("output/lumpi/ens_x")
CLASSES = ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van"]


def test_h(p):
    return {f: [({**d, "box": d["box"][:5] + [d["box"][5] + 0.10] + d["box"][6:]} if d["class"] == "Person" else d) for d in v] for f, v in p.items()}


def test_r(p, v29, windows):
    out, n = {}, 0
    for w in windows:
        cen = {f: np.array([d["box"][:2] for d in v29[f] if d["class"] == "Person" and d["score"] >= 0.05], float).reshape(-1, 2) for f in w}
        for f in w:
            dd = []
            for d in p[f]:
                if d["class"] == "Person" and 0.3 <= d["score"] < 0.5:
                    xy = np.array(d["box"][:2], float)
                    k = sum(len(cen[g]) > 0 and np.min(np.hypot(*(cen[g] - xy).T)) <= 0.5 for g in w)
                    if k >= 0.6 * len(w):
                        d = {**d, "score": d["score"] * 0.3}
                        n += 1
                dd.append(d)
            out[f] = dd
    return out, n


ids = S.ids_of("val")
gt = S.labels(ids)
halves = {"all": ids, "h1": [i for i in ids if int(i) < 3200], "h2": [i for i in ids if int(i) >= 3200]}
win = S.windows_of("val", sorted(ids))


def row(p):
    r = {}
    for h, fr in halves.items():
        a = S.ap(p, gt, fr, CLASSES)
        r[h] = [a[0.5]["Person"], a[0.7]["Person"]]
    r["windows_0123"] = [S.ap(p, gt, w, CLASSES)[0.5]["Person"] for w in win]
    r["windows_2_3"] = S.ap(p, gt, win[2] + win[3], CLASSES)[0.5]["Person"]
    return r


def ok(r, ref):
    return all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1]


v34 = json.loads((O / "lumpi_v33_plow_val.json").read_text())
v29 = json.loads((O / "lumpi_v29_val.json").read_text())
t34 = json.loads((O / "lumpi_v33_plow_test.json").read_text())
t29 = json.loads((O / "lumpi_v29_test.json").read_text())
tw = [sorted(t34)]
ref = row(v34)
print("v34", json.dumps(ref), flush=True)
H = test_h(v34)
rh = row(H)
ph = ok(rh, ref)
print("H", json.dumps(rh), "PASS" if ph else "fail", flush=True)
R, n = test_r(v34, v29, win)
rr = row(R)
pr = ok(rr, ref)
print("R", "demoted", n, json.dumps(rr), "PASS" if pr else "fail", flush=True)
tR, tn = test_r(t34, t29, tw)
print("R test boxes demoted", tn, flush=True)
if ph:
    (O / "lumpi_v34_H_test.json").write_text(json.dumps(test_h(t34)))
if pr:
    (O / "lumpi_v34_R_test.json").write_text(json.dumps(tR))
if ph and pr:
    hr = test_h(R)
    rc = row(hr)
    best = max(rh["all"][0], rr["all"][0])
    print("H+R", json.dumps(rc), "beats the better" if rc["all"][0] > best else "does not beat the better", flush=True)
    if rc["all"][0] > best:
        (O / "lumpi_v34_HR_test.json").write_text(json.dumps(test_h(tR)))
