"""V2X fresh reading: whole files on the 328 fresh frames from the V2X-Real train split (data/pcdet/v2x_fresh), scoring only.
Files: v2x_v22, v2x_v25, v2x_v26 (v25 + truck length +0.67 m, bottom +0.12 m), and every V2X candidate that passed val:
  plan step 2 (logs/v2x_step2_choice.json), new-models phase 1 (logs/v2x_phase1_choice.json), Part-A2 (logs/parta2_v2x_choice.json), each only with cand true.
Each class is built by the same code as on val (scripts/v2x_plan.py build / size: same members, caps, variants and rules X8, X1, X3, rule features on the fresh points).
Self-check first on val_i1: v25 per class (0.8177, 0.3240, 0.6361) and v22 pedestrian (0.3171) rebuilt by this code, else stop.
Gate on fresh against v2x_v25, fixed before any reading: every changed class >= +0.003 overall and on fresh half 2
(the second half of the sorted list) with IoU 0.7 not down,
and fresh mAP >= v25 + 0.004 for a whole file. v2x_v26 stays a candidate only if its truck passes the class gate on fresh. Writes logs/v2x_fresh_read.json.
    UT_NOGPU=1 python scripts/v2x_fresh_read.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import v2x_plan as P
from utl import metrics

FR = Path("data/pcdet/v2x_fresh")
ids = [l.strip() for l in open(FR / "ImageSets/fresh.txt") if l.strip()]
H = {"all": ids, "h1": ids[: len(ids) // 2], "h2": ids[len(ids) // 2:]}
GT = {}
for i in ids:
    rows = [l.split() for l in (FR / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
    GT[i] = [([float(x) for x in r[:7]], r[7]) for r in rows]
PV = P.S2
PA = Path("output/v2x/parta2_v2x/s2")


def score(p, c):
    row = {}
    for h, x in H.items():
        s = metrics.detection_scores({f: [d for d in p.get(f, []) if d["class"] == c] for f in x},
                                     {f: [g for g in GT[f] if g[1] == c] for f in x}, [c], thrs=(0.5, 0.7))
        row[h] = [round(s[0.5]["per_class"][c]["ap40"], 4), round(s[0.7]["per_class"][c]["ap40"], 4)]
    return row


def v22_ped(s):
    return [P.XB / f"x6bf_{s}_ref.json"]


def spec_step2(k, c):
    return {"step2": (P.s2_paths, False), "stack": (P.s2_paths, True), "step1": (P.v25_paths, True)}[k]


def spec_phase1(k, c):
    return {"pv": (lambda c, s: P.v25_paths(c, s) + [PV / f"pv_{c}_{s}_ref.json"], False), "step2": (P.s2_paths, False),
            "stack": (lambda c, s: P.s2_paths(c, s) + [PV / f"pv_{c}_{s}_ref.json"], False)}[k]


def main():
    for c in P.CL:
        r = P.score(P.build(P.v25_paths(c, "val_i1"), c, "val"), c)["all"][0]
        print("self-check val_i1 v25", c, r, "expected", P.V25[c], flush=True)
        if r != P.V25[c]:
            sys.exit("STOP val self-check")
    r = P.score(P.build(v22_ped("val_i1"), "pedestrian", "val"), "pedestrian")["all"][0]
    print("self-check val_i1 v22 pedestrian", r, "expected 0.3171", flush=True)
    if r != 0.3171:
        sys.exit("STOP val self-check")
    print("fresh frames", len(ids), "halves", len(H["h1"]), len(H["h2"]), "labels", {c: sum(g[1] == c for v in GT.values() for g in v) for c in P.CL}, flush=True)
    base = {c: P.build(P.v25_paths(c, "fresh"), c, "fresh") for c in P.CL}
    v25 = {c: score(base[c], c) for c in P.CL}
    rows = {"v2x_v25": v25,
            "v2x_v22": {**v25, "pedestrian": score(P.build(v22_ped("fresh"), "pedestrian", "fresh"), "pedestrian")},
            "v2x_v26": {**v25, "truck": score(P.size(base["truck"], "truck"), "truck")}}
    for name, path, spec in (("step2", "logs/v2x_step2_choice.json", spec_step2), ("phase1", "logs/v2x_phase1_choice.json", spec_phase1)):
        if not Path(path).exists() or not json.loads(Path(path).read_text())["cand"]:
            print(name, "no candidate passed val; not read", flush=True)
            continue
        ch = json.loads(Path(path).read_text())["choice"]
        row = dict(v25)
        for c, k in ch.items():
            fn, sz = spec(k, c)
            p = P.build(fn(c, "fresh"), c, "fresh")
            row[c] = score(P.size(p, c) if sz else p, c)
        rows[name] = row
        print(name, "classes", ch, flush=True)
    pa = Path("logs/parta2_v2x_choice.json")
    if pa.exists() and json.loads(pa.read_text())["cand"]:
        ch = json.loads(pa.read_text())["classes"]
        row = dict(v25)
        for c, x in ch.items():
            p = P.build([Path(t.format(s="fresh")) for t in x["paths"]], c, "fresh")
            row[c] = score(P.size(p, c) if x.get("size") else p, c)
        rows["parta2"] = row
        print("parta2 classes", list(ch), flush=True)
    m25 = round(sum(v25[c]["all"][0] for c in P.CL) / 3, 4)
    out = {}
    for name, row in rows.items():
        m = round(sum(row[c]["all"][0] for c in P.CL) / 3, 4)
        changed = [c for c in P.CL if row[c] != v25[c]]
        cg = {c: P.class_gate(row[c], v25[c]) for c in changed}
        file_gate = name != "v2x_v25" and m >= m25 + 0.004 and all(cg.values())
        for c in P.CL:
            print(name, c, json.dumps(row[c]), "vs v25 %+.4f" % (row[c]["all"][0] - v25[c]["all"][0]),
                  ("class gate " + ("PASS" if cg[c] else "fail")) if c in cg else "", flush=True)
        print(name, "fresh mAP", m, "vs v25 %+.4f" % (m - m25), "file gate", "PASS" if file_gate else ("-" if name == "v2x_v25" else "fail"), flush=True)
        out[name] = {"rows": row, "mAP": m, "class_gate": cg, "file_gate": file_gate}
    if "v2x_v26" in out:
        print("v2x_v26 stays a candidate (truck class gate on fresh):", out["v2x_v26"]["class_gate"].get("truck", False), flush=True)
    Path("logs/v2x_fresh_read.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))   # numpy bools (crashed Nov 9 after printing)


if __name__ == "__main__":
    main()
