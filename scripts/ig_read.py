"""Inside-gate reading for the new LUMPI members (base lumpi_v59).
Members: vrcnn_lumpi_fix_s3 (L11 seed 3), vrcnn_lumpi_c_s2 (L12 seed 2), vrcnn_lumpi_rs_wa (averaged L11-RS); each read add-one per class with G3's
machinery (scripts/g3_search.py, lumpi_v54's fusion and rules unchanged, only the member list changes):
  Car: v59's Car lane (the v29 lane + vrcnn_lumpi_rs at weight 1, = G3's "Car: add vrcnn_lumpi_rs" = lumpi_v58's Car) + the member at weight 1 for
       Car; save_lumpi_chain; rule37.
  Van: the v29 lane + the member at weight 1 for Van; then the route reader's Van steps (route_read_lumpi.Chain: rule37 on the lane's Van and the v29
       Truck, rules T5 / V5), as every member route read Van.
  Person: lumpi_v59's Person chain (R0 of lumpi_v51's five second stages, grow + length rule, P3) with the member's Person second stage (min 0.01, on
       the v29 boxes) as a sixth member, plus v59's two-wheeler copies (Bicycle / Motorcycle unchanged).
Gate (the new LUMPI rule): the class's AP@0.5 on the 120 inside frames +0.01 or more (Car 134, Van 8, Person 64 inside ids); val (320, halves pooled)
not down more than 0.01 (all three have >= 50 val labels). Before, AP@0.7: information. Bicycle (would need the member's Bicycle second stage on
near-test, a GPU step) and Truck / Bus (label quirks): not read here.
Self-checks (STOP): each base reproduces v59 on inside (Car = G3's v58 Car, Person = M3's copies reading, Van = v54's).
    python scripts/ig_read.py near   (in neartest_ws)  -> logs/ig_near.json
    python scripts/ig_read.py val    (main workspace)  -> logs/ig_val.json, then the gate table"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import g3_search as G3
import lumpi_copies_routes as W
import lumpi_static as S
import lumpi_v37_rules as R
import route_read_lumpi as RL
import size_check_lumpi as SZ

MODELS = ["vrcnn_lumpi_fix_s3", "vrcnn_lumpi_c_s2", "vrcnn_lumpi_rs_wa"]
TAGS = {"vrcnn_lumpi_fix_s3": "L3", "vrcnn_lumpi_c_s2": "C2", "vrcnn_lumpi_rs_wa": "RW"}
RS = ["file:g3in/vrcnn_lumpi_rs", 1, ["Car"]]
rd = lambda p: [l.strip() for l in open(p) if l.strip()]
PTS = {"near": "data/pcdet/lumpi_real_50k/points", "val": "data/pcdet/lumpi_real_50k/points"}


def van(chain, lref, ids):
    tv = {f: [R.rule37(RL.keep(d)) for d in chain.get(f, []) if d["class"] == "Van"] + [R.rule37(RL.keep(d)) for d in lref[f] if d["class"] == "Truck"]
          for f in ids}
    tv = R.apply(tv, R.features("val", tv, ids), ["T5", "V5"])
    return {f: [d for d in tv[f] if d["class"] == "Van"] for f in ids}


def build(mode, ids):
    load = lambda f: np.load(f"{PTS[mode]}/{f}.npy")[:, :3]
    keys = SZ.PP.static_keys("lumpi")
    P = SZ.build("val", ids)
    cp = W.copies({f: [d for c in W.TW for d in P[c].get(f, [])] for f in ids})
    lref = json.loads((RL.L / "v29_val_ref.json").read_text())
    out = {}
    out["Car: v59"] = ("Car", G3.lane_class(G3.run_lane("IGc00", G3.B_SPEC + [RS], {}, "val"), "Car", ids))
    out["Van: v59"] = ("Van", van(lref, lref, ids))
    base_p = G3.person(G3.P_REF("val"), ids, load, keys)
    out["Person: v59"] = ("Person", {f: base_p[f] + cp[f] for f in ids})
    for m in MODELS:
        t = TAGS[m]
        if not Path(f"output/lumpi/g3in/{m}_val.json").exists():
            print(f"no inputs for {m}: skipped", flush=True)
            continue
        out[f"Car: add {m}"] = ("Car", G3.lane_class(G3.run_lane(f"IGc{t}", G3.B_SPEC + [RS, [f"file:g3in/{m}", 1, ["Car"]]], {}, "val"), "Car", ids))
        out[f"Van: add {m}"] = ("Van", van(G3.run_lane(f"IGv{t}", G3.B_SPEC + [[f"file:g3in/{m}", 1, ["Van"]]], {}, "val"), lref, ids))
        pp = Path(f"output/lumpi/{m}/g3/v29_val_person_min01.json")
        if pp.exists():
            p = G3.person(G3.P_REF("val") + [pp], ids, load, keys)
            out[f"Person: add {m}"] = ("Person", {f: p[f] + cp[f] for f in ids})
        print("built", m, flush=True)
    return out


def near():
    ids = S.ids_of("val")
    assert len(ids) == 266
    H = {h: rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in ("before", "inside")}
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    cand = build("near", ids)
    v54 = SZ.build("val", ids)
    chk = {"Car": json.loads(Path("logs/g3_near.json").read_text())["v58_car"]["inside"]["v58"][0],
           "Person": json.loads(Path("logs/m3_neartest_copies.json").read_text())["copies_on"]["inside"][0],
           "Van": SZ.ap_from(*SZ.flags({f: v54["Van"][f] for f in H["inside"]}, {f: gt[f] for f in H["inside"]}, "Van"))[0]}
    for c, want in chk.items():
        got = G3.ap(cand[f"{c}: v59"][1], gt, H["inside"], c)
        print(f"self-check inside {c}: v59 base {got:.6f} vs {want:.6f}: {abs(got - want) < 1e-9}", flush=True)
        if abs(got - want) >= 1e-9:
            sys.exit(f"STOP the {c} base does not reproduce v59")
    res = {}
    for name, (c, p) in cand.items():
        res[name] = {"class": c, **{h: SZ.ap_from(*SZ.flags({f: p.get(f, []) for f in fr}, {f: gt[f] for f in fr}, c)) for h, fr in H.items()}}
    for name, r in res.items():
        b = res[f"{r['class']}: v59"]
        r["d_inside"], r["d_before"] = r["inside"][0] - b["inside"][0], r["before"][0] - b["before"][0]
        print(f"{name:32s} inside {r['inside'][0]:.4f} ({r['d_inside']:+.4f}) | before {r['before'][0]:.4f} ({r['d_before']:+.4f}) | AP@0.7 inside {r['inside'][1]:.4f}",
              flush=True)
    Path("logs/ig_near.json").write_text(json.dumps(res, indent=1))


def val():
    ids = S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    near_res = json.loads(Path("logs/ig_near.json").read_text())
    cand = build("val", ids)
    exp = {"Car": 0.896, "Person": 0.48}
    for c, want in exp.items():
        got = round(G3.ap(cand[f"{c}: v59"][1], gt, ids, c), 4)
        print(f"self-check val {c}: v59 base {got} vs {want}: {got == want}", flush=True)
        if got != want:
            sys.exit(f"STOP the val {c} base does not reproduce v59")
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())["inside"]
    res = {}
    for name, (c, p) in cand.items():
        if name.endswith(": v59"):
            continue
        base = G3.ap(cand[f"{c}: v59"][1], gt, ids, c)
        a = G3.ap(p, gt, ids, c)
        n = near_res.get(name)
        ok = bool(n and cnt.get(c, {}).get("distinct_ids", 0) >= 5 and n["d_inside"] >= 0.01 and a - base >= -0.01)
        res[name] = dict(val=a, val_base=base, d_val=a - base, d_inside=n and n["d_inside"], d_before=n and n["d_before"], passed=ok)
        print(f"{name:32s} inside {n['d_inside'] if n else float('nan'):+.4f} | val {a:.4f} ({a - base:+.4f}) | before {n['d_before'] if n else float('nan'):+.4f} | "
              f"{'PASS' if ok else 'fail'}", flush=True)
    Path("logs/ig_val.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"near": near, "val": val}[sys.argv[1]]()
