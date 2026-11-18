"""Member route reading on LUMPI for a new one-stage model (FSHNet, SAFDNet, or VoxelNeXt as FSHNet's fallback),
on the base file's own chain (lumpi_v51) with like-for-like rule references. Fixed before any reading.
Route: lanes = the v29 lane plus the model at weight 1 for one class (scripts/route_lanes_lumpi.sh), then lumpi_v51's chain for that class:
  Person   lumpi_v51's five second stages (L11, L12, L11b second seed, PV-RCNN++, test A) on the lane's Person boxes down to 0.01 (scripts/route_lumpi.sh),
           combined by rule R0, R4 or R6 (scripts/ensemble_rules.py), then the v37 size change and length rule;
  Bicycle  L11b's second stage on the lane's Bicycle boxes (>= 0.05), combined by R0 or R4 (one refiner);
  Car, Truck, Van  the lane's boxes with lumpi_v41's rules (v37 height rule; T5 / V5 point rules for Truck and Van), no second stage (R0 only).
Like-for-like: for each rule X, the reference is the same chain under X without the new model (the v29 boxes), logged first ("refs" mode).
Per class the rule with the largest val gain over its own reference is chosen (R0 on a tie); the class gate applies to that gain:
  Person, Car, Truck: +0.003 on both halves, IoU 0.7 not down; Bicycle: the same plus labels-only up; Van: the Nov 8 Van gate (van_gate). Car flagged even if it passes.
Fresh, for classes that pass val: the chosen rule with and without the model (reference logged), plus R0 with the model for information.
File: base by Person's rule (R0 or Person not passing: lumpi_v53; R4: lumpi_v53 with Person by R4, flagged; R6: lumpi_v52, flagged),
with every class that passes val and fresh replaced. Writes output/lumpi/ens_x/route_<model>_test.json.
    python scripts/route_read_lumpi.py refs logs/fresh_ids.txt              (before any new-model reading; logs/route_refs_lumpi.json)
    python scripts/route_read_lumpi.py read logs/fresh_ids.txt fshnet_lumpi"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ensemble_rules as E
import lumpi_copies_routes as W
import lumpi_static as S
import lumpi_v37_rules as R
import lumpi_van_len as VL
from utl.metrics import detection_scores

O = Path("output/lumpi/ens_x")
L, C, Z, B = W.L, W.C, W.Z, W.B
N, PVD = Path("output/lumpi/vrcnn_lumpi_fix_s2/s2"), Path("output/lumpi/pv_lumpi/s2p")
keep = W.keep
KEEP = 1.64 / 4.84
ORDER = ["Person", "Car", "Truck", "Bicycle", "Van"]
RULES = {"Person": ["R0", "R4", "R6"], "Bicycle": ["R0", "R4"], "Car": ["R0"], "Truck": ["R0"], "Van": ["R0"]}
BASE_RULE = {"Person": "R0", "Bicycle": "R4"}   # lumpi_v53's rules (other classes R0)
REFJ = Path("logs/route_refs_lumpi.json")
EXPECT = {("val", "Person", "R0"): 0.4366, ("val", "Person", "R4"): 0.4422, ("val", "Person", "R6"): 0.4427, ("fresh", "Person", "R0"): 0.4481,
          ("fresh", "Person", "R6"): 0.4544, ("val", "Bicycle", "R0"): 0.5404, ("fresh", "Bicycle", "R0"): 0.5386, ("val", "Car", "R0"): 0.8858,
          ("fresh", "Car", "R0"): 0.8880, ("val", "Truck", "R0"): 0.4632, ("fresh", "Truck", "R0"): 0.4440, ("val", "Van", "R0"): 0.1708, ("fresh", "Van", "R0"): 0.1673}
TAG = {"fshnet": "zFS", "safdnet": "zSD", "voxelnext": "zVX", "fshnet_lumpi_s2": "zF2",
       "vrcnn_lumpi_c_s2": "zQC", "pv_lumpi_s2": "zQP", "vrcnn_lumpi_fix_s3": "zQL",
       "spec_lumpi_van": "zXV", "spec_lumpi_car": "zXC", "dsvt_lumpi": "zDS", "vmamba_lumpi": "zVM"}  # zF2: both FSHNet seeds as one member; zQ*: block 1 refiner seeds, every class
# Block 1 refiner seeds, every class: Person is gated only by scripts/lumpi_member_read.py,
# so here its route lane is skipped (no Person reading); Motorcycle and Bus model-alone AP for information; file base = that seed's Person file
# (lumpi_<seed>_test.json) when its Person passed, else lumpi_v53, with only the passing classes replaced.
REFINER = {"vrcnn_lumpi_c_s2", "pv_lumpi_s2", "vrcnn_lumpi_fix_s3"}
# Class specialists (a model trained for a single class): only that class's lane and reading;
# file base lumpi_v53 (its Bicycle by R4), only that class replaced; Car flagged even if it passes.
ONLY = {"spec_lumpi_van": ["Van"], "spec_lumpi_car": ["Car"]}
dumps = lambda o: json.dumps(o, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def thin(p, gt, ids):
    t = []
    for s in range(5):
        rng = np.random.default_rng(s)
        g = {f: [(x, k) for x, k in gt[f] if k != "Bicycle" or rng.random() <= KEEP] for f in ids}
        t.append(detection_scores({f: [d for d in p[f] if d["class"] == "Bicycle"] for f in ids}, g, ["Bicycle"], thrs=(0.5,))[0.5]["per_class"]["Bicycle"]["ap40"])
    return round(float(np.mean(t)), 4)


def person(paths, rule):
    p = E.fuse(E.load(paths, "Person"), rule)
    out = {}
    for f, v in p.items():
        out[f] = []
        for d in v:
            b = W.grow(d["box"])
            out[f].append({"box": b, "score": d["score"] * 0.3 if b[3] >= 0.85 else d["score"], "class": "Person"})
    return out


class Chain:
    """lumpi_v51's chain for one split, from the v29 boxes (model None) or from the model's lanes."""

    def __init__(self, split, ids, model=None):
        self.s, self.ids, self.m = split, ids, model
        self.lref = json.loads((L / f"v29_{split}_ref.json").read_text())
        if model:
            self.t = TAG.get(model) or TAG[model.split("_")[0]]
            self.s2 = Path(f"output/lumpi/{model}/s2")

    def person_paths(self):
        s = self.s
        if not self.m:
            return [L / f"v29_{s}_ref_min01.json", C / f"v29_{s}_ref_min01.json", Z / f"v29_{s}_ref_min01.json", PVD / f"v29_{s}_min01.json", N / f"v29_{s}_min01.json"]
        return [self.s2 / f"person_{r}_{s}_ref.json" for r in ("l11", "l12", "l11b_s2", "pv", "testa")]

    def lane(self, c):
        k = {"Car": "c", "Truck": "t", "Van": "v"}[c]
        p = O / f"route_{self.m}" / f"{self.t}{k}_chain_fresh.json" if self.s == "fresh" else O / f"{self.t}{k}_chain_{self.s}.json"
        return W.only(p, c)

    def pre(self, c, own):
        if own and self.m:
            x = self.lane(c)
            return {f: [R.rule37(keep(d)) for d in x.get(f, [])] for f in self.ids}
        return {f: [R.rule37(keep(d)) for d in self.lref[f] if d["class"] == c] for f in self.ids}

    def cls(self, c, rule="R0"):
        ids = self.ids
        if c == "Person":
            p = person(self.person_paths(), rule)
            return {f: p.get(f, []) for f in ids}
        if c == "Car":
            return self.pre(c, True)
        if c in ("Truck", "Van"):
            o = "Van" if c == "Truck" else "Truck"
            tv = {f: a + b for (f, a), b in zip(self.pre(c, True).items(), self.pre(o, False).values())}
            tv = R.apply(tv, R.features(self.s, tv, ids), ["T5", "V5"])
            return {f: [d for d in tv[f] if d["class"] == c] for f in ids}
        bi = W.only(self.s2 / f"bicycle_{self.s}_ref.json" if self.m else B / f"v29_{self.s}_ref.json", "Bicycle")
        return {f: [E.combine([d], rule) if d["score"] >= 0.05 else keep(d) for d in bi.get(f, [])] for f in ids}


def van_gate(r, ref):
    """The Van gate fixed on Nov 8 (for that day's Van rule and any later Van reading): windows 2 and 3 each
    up, half 2 >= +0.003, IoU 0.7 not down, half 1 down by no more than 0.001; ten-thousandths arithmetic. Fix Nov 9: the reader
    still had the older strict "half 1 up" form."""
    d = lambda a, b: round(1e4 * (a - b))
    return bool(r["windows"][2] > ref["windows"][2] and r["windows"][3] > ref["windows"][3] and d(r["h2"][0], ref["h2"][0]) >= 30
                and r["all"][1] >= ref["all"][1] and d(r["h1"][0], ref["h1"][0]) >= -10)


def gate(c, r, ref, t=None):
    if c == "Van":
        return van_gate(r, ref)
    g = bool(R.gate(c, r, ref))
    return bool(g and t[1] > t[0]) if c == "Bicycle" else g


def refs(fr):
    out = {}
    for split, ids in (("val", S.ids_of("val")), ("fresh", fr)):
        gt = S.labels(ids)
        ch = Chain(split, ids)
        for c in ORDER:
            for rule in RULES[c]:
                p = ch.cls(c, rule)
                r = R.row(p, gt, ids, c)
                if c == "Bicycle":
                    r["labels_only"] = thin(p, gt, ids)
                out.setdefault(split, {}).setdefault(c, {})[rule] = r
                e = EXPECT.get((split, c, rule))
                print("reference (no new model)", split, c, rule, json.dumps(r), "expected", e if e is not None else "new, logged now", flush=True)
                if e is not None and r["all"][0] != e:
                    sys.exit("STOP reference does not rebuild the logged reading")
    REFJ.write_text(dumps(out))
    print("wrote", REFJ, flush=True)


def read(fr, model):
    ref = json.loads(REFJ.read_text())
    print("model", model, "| Person second stages: L11, L12, L11b second seed, PV-RCNN++, test A | rules", json.dumps(RULES), "| references from", REFJ, flush=True)
    res = {"model": model, "val": {}, "fresh": {}}
    ids = S.ids_of("val")
    gt = S.labels(ids)
    ch = Chain("val", ids, model)
    order = ONLY.get(model) or [c for c in ORDER if not (model in REFINER and c == "Person")]
    if model in REFINER:
        print("Person: gated by lumpi_member_read.py only; its route lane is skipped for this seed", flush=True)
    raw = json.loads(Path(f"output/lumpi/route_in/{model}_val.json").read_text())
    for c in ORDER + (["Motorcycle", "Bus"] if model in REFINER else []):
        print("model alone, no chain", c, json.dumps(R.row({f: [keep(d) for d in raw.get(f, []) if d["class"] == c] for f in ids}, gt, ids, c)), flush=True)
    del raw
    for c in order:
        rows = {}
        for rule in RULES[c]:
            p = ch.cls(c, rule)
            r = R.row(p, gt, ids, c)
            if c == "Bicycle":
                r["labels_only"] = thin(p, gt, ids)
            rows[rule] = r
            print("val", c, rule, json.dumps(r), "vs same rule without the model %+.4f" % (r["all"][0] - ref["val"][c][rule]["all"][0]), flush=True)
        k = max(RULES[c], key=lambda x: (rows[x]["all"][0] - ref["val"][c][x]["all"][0], -RULES[c].index(x)))
        rr = ref["val"][c][k]
        g = gate(c, rows[k], rr, (rr.get("labels_only"), rows[k].get("labels_only")) if c == "Bicycle" else None)
        print("val", c, "chosen", k, "gain %+.4f" % (rows[k]["all"][0] - rr["all"][0]), "| vs lumpi_v51 (R0) %+.4f" % (rows[k]["all"][0] - ref["val"][c]["R0"]["all"][0]),
              "| gate", "PASS" if g else "fail", "(Car flagged even if it passes)" if c == "Car" else "", flush=True)
        if k != BASE_RULE.get(c, "R0"):   # the gate is like for like; say when the class would change rule against the base file
            print("FLAG", c, "chosen rule", k, "differs from the base file's rule", BASE_RULE.get(c, "R0"), "(lumpi_v53)", flush=True)
        res["val"][c] = {"rows": rows, "chosen": k, "gate": g}
    del gt
    passed = [c for c in order if res["val"][c]["gate"]]
    print("passed val:", {c: res["val"][c]["chosen"] for c in passed}, flush=True)
    if passed:
        gf = S.labels(fr)
        ch = Chain("fresh", fr, model)
        for c in passed:
            k = res["val"][c]["chosen"]
            rows = {}
            for rule in dict.fromkeys([k, "R0"]):
                p = ch.cls(c, rule)
                rows[rule] = R.row(p, gf, fr, c)
                if c == "Bicycle":
                    rows[rule]["labels_only"] = thin(p, gf, fr)
            rr = ref["fresh"][c][k]
            g = gate(c, rows[k], rr, (rr.get("labels_only"), rows[k].get("labels_only")) if c == "Bicycle" else None)
            print("fresh", c, k, json.dumps(rows[k]), "vs same rule without the model", json.dumps(rr), "| gate", "PASS" if g else "fail",
                  "| R0 with the model (information)", json.dumps(rows["R0"]), "vs lumpi_v51 %+.4f" % (rows["R0"]["all"][0] - ref["fresh"][c]["R0"]["all"][0]), flush=True)
            res["fresh"][c] = {"rows": rows, "gate": g}
    final = {c: res["val"][c]["chosen"] for c in passed if res["fresh"][c]["gate"]}
    res["file_classes"] = final
    print("classes for the file:", final, "(Car flagged)" if "Car" in final else "", flush=True)
    Path(f"logs/route_{model}_read.json").write_text(dumps(res))
    if not final:
        print("no file", flush=True)
        return
    pr = final.get("Person", "R0")
    # Base = the best unflagged file, lumpi_v53 (v51 + Bicycle R4), as fixed in advance; Person R6 -> lumpi_v52 (Bicycle R0), flagged.
    # (Fix Nov 8: this was lumpi_v51 with Bicycle R0 for models outside REFINER / ONLY.)
    base_file, bike = ("lumpi_r6_test.json", "R0") if pr == "R6" else ("lumpi_bicycle_r4_test.json", "R4")
    own = False
    if model in REFINER or model in ONLY:
        mj = Path(f"logs/lumpi_member_{model}.json")
        own = (O / f"lumpi_{model}_test.json").exists()
        if own:   # this seed's own member file (lumpi_v53 / lumpi_v52 with its Person)
            base_file = f"lumpi_{model}_test.json"
            bike = "R0" if json.loads(mj.read_text())["chosen"] == "R6" else "R4"   # v52-based (R6) keeps v51's Bicycle; v53-based has R4
    ref_file = "lumpi_r6_test.json" if bike == "R0" else "lumpi_bicycle_r4_test.json"   # lumpi_v52 or lumpi_v53
    print("file base", base_file, "| its Bicycle rule", bike, "| reference file for the classes not replaced", ref_file, flush=True)
    t = json.loads((O / base_file).read_text())
    tid = sorted(t)
    canon = lambda v: sorted(json.dumps([round(x, 5) for x in d["box"]] + [round(d["score"], 6), d["class"]]) for d in v)
    b0, ch = Chain("test", tid), Chain("test", tid, model)
    new = {}
    for c, k in final.items():
        ok = all(canon(b0.cls(c, pr if c == "Person" else bike if c == "Bicycle" else "R0").get(f, [])) == canon([d for d in t[f] if d["class"] == c]) for f in tid) if (c != "Person" or pr != "R4") else True
        print("test", c, "base chain rebuilt equals", base_file, ":", ok, flush=True)
        if not ok:
            sys.exit("STOP test base does not match")
        new[c] = ch.cls(c, k)
    out = {f: sorted([keep(d) for d in t[f] if d["class"] not in new] + [keep(d) for c in new for d in new[c].get(f, [])], key=lambda d: -d["score"]) for f in tid}
    rf = json.loads((O / ref_file).read_text())
    cls = sorted({d["class"] for v in list(rf.values()) + list(out.values()) for d in v} - set(new) - ({"Person"} if own else set()))
    ok = sorted(rf) == tid and all(canon([keep(d) for d in out[f] if d["class"] == c]) == canon([keep(d) for d in rf[f] if d["class"] == c]) for f in tid for c in cls)
    print("test: classes not replaced", cls, "equal", ref_file, ":", ok, flush=True)
    if not ok:
        sys.exit("STOP the classes not replaced do not equal the reference file")
    (O / f"route_{model}_test.json").write_text(json.dumps(out))
    print(f"wrote route_{model}_test.json: base {base_file} with", final, flush=True)


if __name__ == "__main__":
    fr = [l.strip() for l in open(sys.argv[2]) if l.strip()]
    refs(fr) if sys.argv[1] == "refs" else read(fr, sys.argv[3])
