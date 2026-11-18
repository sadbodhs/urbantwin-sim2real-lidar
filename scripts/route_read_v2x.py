"""V2X member route reader (built Nov 8, settings fixed before any new-model reading). The new model's fused boxes per class
(scripts/route_v2x_lanes.py) go through v2x_v25's second stages (scripts/route_v2x.sh): vehicle X6, truck X6b, pedestrian X6b and the negatives model.
Rules, each like for like against the same rule on v2x_v25's own inputs and second stages (references logged first, refs mode):
  vehicle     R0 (v25: variant c3, cap 1 m, score kept; rule X8)
  truck       R0 (v25: c12, cap 1 m, score sqrt(s x mean q)), R4 (score s^0.29 x (mean q)^0.71); width >= 2.55 m
  pedestrian  R0 (v25: c12, cap 0.5 m), R4, R6 (scripts/ensemble_rules.py); rules X1 and X3
R0 without the model must rebuild v2x_v25 on OS1-64 val_i1 (vehicle 0.8177, pedestrian 0.3240, truck 0.6361), else stop.
Chosen rule per class: the largest val_i1 gain over its own reference (R0 on a tie). Val gate: >= +0.003 on both val_i1 halves, IoU 0.7 not down.
Fresh (328-frame V2X fresh set), classes passing val only, the chosen rule once: >= +0.003 overall and on fresh half 2, IoU 0.7 not down.
A file only for classes passing both: v2x_v25 with those classes rebuilt with the model -> output/v2x/ens/route_<model>_test.json.
The candidate gate (val mAP >= +0.004 over v2x_v25) is printed for information.
    python scripts/route_read_v2x.py refs
    python scripts/route_read_v2x.py val fshnet_v2x
    python scripts/route_read_v2x.py fresh fshnet_v2x"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ensemble_rules as ER
import v2x_plan as P
import v2x_v20_rules as VR
from utl import metrics

CL = ["vehicle", "pedestrian", "truck"]
RULES = {"vehicle": ["R0"], "truck": ["R0", "R4"], "pedestrian": ["R0", "R4", "R6"]}
REFS = Path("logs/route_refs_v2x.json")
# Block 1 slot 1 seed: pedestrian is gated only by scripts/v2x_neg_s3_read.py, so here it is
# printed for information (no gate, not in the file); the file base is v2x_neg_s3_test.json when its pedestrian passed, else v2x_v25.
INFO_PED = {"vrcnn_v2x_neg_s3"}
# Class specialists (the best model per class): only that class is read; file base v2x_v25.
ONLY = {"spec_v2x_truck": ["truck"]}


def new_paths(m, cls, s):
    O = Path(f"output/v2x/{m}/s2")
    return {"vehicle": [O / f"vehicle_x6_{s}_ref.json"], "truck": [O / f"truck_x6b_{s}_ref.json"],
            "pedestrian": [O / f"pedestrian_x6b_{s}_ref.json", O / f"pedestrian_neg_{s}_ref.json"]}[cls]


def build(paths, cls, kind, rule):
    if rule == "R0":
        return P.build(paths, cls, kind)
    src = ER.load(paths, cls)
    out = {}
    for f in src[0]:
        rows = [x[f] for x in src]
        assert all(len(r) == len(rows[0]) for r in rows), f
        out[f] = []
        for tup in zip(*rows):
            assert all(t["box"] == tup[0]["box"] for t in tup)
            d = ER.combine(list(tup), rule, gate=P.CAP[cls])
            if cls == "truck":
                d["box"] = d["box"][:4] + [max(d["box"][4], 2.55)] + d["box"][5:]
            out[f].append(d)
    return VR.apply(out, VR.features(kind, out, sorted(out)), ["X1", "X3"]) if cls == "pedestrian" else out


_fr = {}


def fresh_score(p, cls):
    if not _fr:
        fr = Path("data/pcdet/v2x_fresh")
        ids = [l.strip() for l in open(fr / "ImageSets/fresh.txt") if l.strip()]
        _fr["H"] = {"all": ids, "h1": ids[: len(ids) // 2], "h2": ids[len(ids) // 2:]}
        _fr["gt"] = {}
        for i in ids:
            rows = [l.split() for l in (fr / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
            _fr["gt"][i] = [([float(x) for x in r[:7]], r[7]) for r in rows]
    row = {}
    for h, x in _fr["H"].items():
        s = metrics.detection_scores({f: [d for d in p.get(f, []) if d["class"] == cls] for f in x},
                                     {f: [g for g in _fr["gt"][f] if g[1] == cls] for f in x}, [cls], thrs=(0.5, 0.7))
        row[h] = [round(s[t]["per_class"][cls]["ap40"], 4) for t in (0.5, 0.7)]
    return row


def refs():
    out = {}
    for c in CL:
        out[c] = {}
        for k in RULES[c]:
            out[c][k] = P.score(build(P.v25_paths(c, "val_i1"), c, "val", k), c)
            print("reference val_i1", c, k, "(v2x_v25 inputs and second stages)", json.dumps(out[c][k]), flush=True)
        if out[c]["R0"]["all"][0] != P.V25[c]:
            sys.exit(f"STOP R0 {c} does not rebuild v2x_v25 ({P.V25[c]})")
    REFS.write_text(json.dumps({"val_i1": out}, indent=1))
    print("wrote", REFS, flush=True)


def val(m):
    ref = json.loads(REFS.read_text())["val_i1"]
    print("model", m, "| rules", json.dumps(RULES), "| references from", REFS, flush=True)
    alone = json.loads(Path(f"output/v2x/{m}/kit/pred_val50k.json").read_text())
    for c in CL:
        print("model alone, no chain", c, json.dumps(P.score(alone, c)), flush=True)
    res = {"model": m, "val": {}, "passed": {}}
    for c in ONLY.get(m, CL):
        new = {}
        for k in RULES[c]:
            new[k] = P.score(build(new_paths(m, c, "val_i1"), c, "val", k), c)
            print("val_i1", c, k, json.dumps(new[k]), "vs same rule without the model %+.4f" % (new[k]["all"][0] - ref[c][k]["all"][0]), flush=True)
        k = max(RULES[c], key=lambda x: (new[x]["all"][0] - ref[c][x]["all"][0], -RULES[c].index(x)))
        g = ER.gate(new[k], ref[c][k])
        print("val_i1", c, "chosen", k, "gain %+.4f" % (new[k]["all"][0] - ref[c][k]["all"][0]), "| vs v2x_v25 %+.4f" % (new[k]["all"][0] - P.V25[c]),
              "| gate", "PASS" if g else "fail", flush=True)
        if k != "R0":   # v2x_v25 uses R0 for every class
            print("FLAG", c, "chosen rule", k, "differs from v2x_v25's rule R0", flush=True)
        if m in INFO_PED and c == "pedestrian":
            g = False
            print("val_i1 pedestrian: information only for this seed (gated by v2x_neg_s3_read.py)", flush=True)
        res["val"][c] = {"new": new, "chosen": k, "gate": g}
        if g:
            res["passed"][c] = k
    print("passed val_i1:", res["passed"], flush=True)
    Path(f"logs/route_v2x_{m}.json").write_text(json.dumps(res, indent=1))


def fresh(m):
    js = Path(f"logs/route_v2x_{m}.json")
    res = json.loads(js.read_text())
    if not res["passed"]:
        print("no class passed val_i1: no fresh reading", flush=True)
        return
    allref = json.loads(REFS.read_text())
    fref = allref.setdefault("fresh", {})
    for c, k in res["passed"].items():   # references first
        if k not in fref.get(c, {}):
            fref.setdefault(c, {})[k] = fresh_score(build(P.v25_paths(c, "fresh"), c, "fresh", k), c)
            print("reference fresh", c, k, json.dumps(fref[c][k]), flush=True)
    REFS.write_text(json.dumps(allref, indent=1))
    ok, res["fresh"] = {}, {}
    for c, k in res["passed"].items():
        a, b = fresh_score(build(new_paths(m, c, "fresh"), c, "fresh", k), c), fref[c][k]
        g = bool(round(1e4 * (a["all"][0] - b["all"][0])) >= 30 and round(1e4 * (a["h2"][0] - b["h2"][0])) >= 30 and a["all"][1] >= b["all"][1])
        print("fresh", c, k, json.dumps(a), "vs same rule without the model", json.dumps(b), "%+.4f" % (a["all"][0] - b["all"][0]),
              "| gate", "PASS" if g else "fail", flush=True)
        res["fresh"][c] = {"new": a, "ref": b, "gate": g}
        if g:
            ok[c] = k
    print("classes for the file:", ok, flush=True)
    res["file_classes"] = ok
    if ok:
        gain = sum(res["val"][c]["new"][k]["all"][0] - P.V25[c] for c, k in ok.items()) / 3
        res["cand_gate"] = bool(gain >= 0.004)
        print("val mAP vs v2x_v25 %+.4f | candidate gate (information)" % gain, "PASS" if res["cand_gate"] else "fail", flush=True)
        base = "v2x_neg_s3_test.json" if m in INFO_PED and (P.E / "v2x_neg_s3_test.json").exists() else "v25_test.json"
        print("file base", base, flush=True)
        t = json.loads((P.E / base).read_text())
        out = {f: list(t[f]) for f in t}
        for c, k in ok.items():
            x = build(new_paths(m, c, "test"), c, "test", k)
            assert set(x) == set(t), "class writer: the new class covers other frames than the base file"
            out = {f: [d for d in out[f] if d["class"] != c] + [ER.keep(d) for d in x.get(f, [])] for f in out}
        out = {f: sorted(v, key=lambda d: -d["score"]) for f, v in out.items()}
        (P.E / f"route_{m}_test.json").write_text(json.dumps(out))
        print(f"wrote route_{m}_test.json: {base} with", ", ".join(f"{c} by {k}" for c, k in ok.items()), "with the model", flush=True)
    else:
        print("no file", flush=True)
    js.write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))


if __name__ == "__main__":
    {"refs": lambda: refs(), "val": lambda: val(sys.argv[2]), "fresh": lambda: fresh(sys.argv[2])}[sys.argv[1]]()
