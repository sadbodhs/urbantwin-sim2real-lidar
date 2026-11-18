"""G3 near-test member search for LUMPI, Mac part: member-list permutations around the best trained models.
One try; settings fixed before reading.
Candidates per class (at most 20, all logged); lumpi_v54's fusion and rules unchanged, only the member list changes:
  Car (v29 lane member list; chain = save_lumpi_chain then rule37, as lumpi_v54's Car): reference, drop-one (each of the 4 base members and
      pp_lumpi_br), add-one (each pool model at weight 1 for Car).
  Motorcycle (lane; selectable at 3 distinct objects per half; no val condition, val has 6 labels): reference + the 7 non-empty subsets of {fshnet, safdnet, voxelnext}.
  Person (lumpi_v54's R0 fusion of five second stages, grow, length rule, then P3): reference, drop-one (5), add-one (L11b, L11-RS Person second
      stages at min 0.01).
  Bicycle (information: 2 distinct objects before): reference (L11b's second stage, R4, P3), add-one (+ L11's, + L11-RS's second stage, same boxes).
  Van, Truck, Bus: information only (v54's numbers from G1).
Gate (fixed in advance): a class is selectable only with >= 5 distinct object ids in each near-test half (logs/neartest_id_counts.json: Person and
Car; Motorcycle at >= 3, set from object counts only). A candidate passes at +0.01 AP@0.5 or more over v54's on both halves (before 76, inside 120 frames); where val has >= 50 labels of the class
(Person, Car) it must also not fall more than 0.005 below v54 on val (mode "val", main workspace). The best passing candidate on both halves together
(196 frames) is chosen; else v54 stays. If G1 read val-like (3+ of Car / Person / Bicycle / Truck), everything is information only.
Build note (set before any G3 number was read; G1 read Car val-like): a passing Car candidate is
kept out of the main-line build (v54's Car stays there) and built as a separate twin; one of the two is picked at upload.
    python scripts/g3_search.py near     (inside neartest_ws, where the near-test frames stand in for "val")
    python scripts/g3_search.py val      (main workspace: the near-test passes, read on the real val split)"""
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ensemble_rules as E
import lumpi_member_read as M
import lumpi_static as S
import lumpi_v37_rules as R
import m1_late_person as M1
import route_read_lumpi as RL
import size_check_lumpi as SZ

O = Path("output/lumpi/ens_x")
POOL = ["pp_lumpi_walk", "pp_lumpi_walk_all", "vrcnn_lumpi_fix", "file:g3in/vrcnn_lumpi_c", "file:g3in/vrcnn_lumpi_b_s2", "file:g3in/pv_lumpi",
        "file:g3in/voxelnext_lumpi", "file:g3in/fshnet_lumpi", "file:g3in/safdnet_lumpi", "file:g3in/vrcnn_lumpi_rs"]
MOTO = ["file:g3in/fshnet_lumpi", "file:g3in/safdnet_lumpi", "file:g3in/voxelnext_lumpi"]
BASE_NAMES = ["pp_lumpi_long", "pp_lumpi_d50", "cpv_lumpi_long (no Person)", "pp_lumpi_long_all"]
BASE_W = [1.0, 0.5, 1.0, 0.5]
B_SPEC = [["pp_lumpi_br", 1, ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Van", "Unknown"]], ["pp_lumpi_walk", 1, ["Truck"]],
          ["pp_lumpi_walk_all", 1, ["Person", "Truck"]], ["vrcnn_lumpi_fix", 1, ["Person"]]]
P_REF = lambda s: [RL.L / f"v29_{s}_ref_min01.json", RL.C / f"v29_{s}_ref_min01.json", RL.Z / f"v29_{s}_ref_min01.json",
                   RL.PVD / f"v29_{s}_min01.json", RL.N / f"v29_{s}_min01.json"]
P_NAMES = ["L11", "L12", "L11b seed 2", "PV-RCNN++", "L11 seed 2"]
P_ADD = {"L11b": lambda s: Path(f"output/lumpi/vrcnn_lumpi_b/g3/v29_{s}_person_min01.json"),
         "L11-RS": lambda s: Path(f"output/lumpi/vrcnn_lumpi_rs/g3/v29_{s}_person_min01.json")}
B_ADD = {"L11": lambda s: RL.L / f"v29_{s}_ref.json", "L11-RS": lambda s: Path(f"output/lumpi/vrcnn_lumpi_rs/g3/v29_{s}_ref.json")}
SELECTABLE_MIN = {"Motorcycle": 3}   # Motorcycle at 3 (set from object counts only, before any score); others 5
VAL_LABELS_MIN = 50


def lane_candidates():
    """name -> (class, spec, baseovr)"""
    c = {"Car: v54": ("Car", B_SPEC, {})}
    for i, n in enumerate(BASE_NAMES):
        w = list(BASE_W)
        w[i] = 0.0
        c[f"Car: drop {n}"] = ("Car", B_SPEC, {"Car": w})
    c["Car: drop pp_lumpi_br"] = ("Car", [[m, w, [x for x in cl if x != "Car"]] for m, w, cl in B_SPEC], {})
    for m in POOL:
        c[f"Car: add {m.replace('file:g3in/', '')}"] = ("Car", B_SPEC + [[m, 1, ["Car"]]], {})
    c["Motorcycle: v54"] = ("Motorcycle", B_SPEC, {})
    for k in (1, 2, 3):
        for sub in combinations(MOTO, k):
            c["Motorcycle: add " + " + ".join(m.replace("file:g3in/", "") for m in sub)] = ("Motorcycle", B_SPEC + [[m, 1, ["Motorcycle"]] for m in sub], {})
    return c


def run_lane(tag, spec, ovr, split):
    name = f"zG3{tag}"
    out = O / f"{name}_chain_{split}.json"
    if not out.exists():
        subprocess.run(["python", "scripts/lumpi_lane_fast.py", name, json.dumps(spec), "--baseovr", json.dumps(ovr), "--splits", split, "--workers", "6"],
                       check=True)
        subprocess.run(["python", "scripts/save_lumpi_chain.py", str(O / name), str(out), split], check=True)
    return json.loads(out.read_text())


def lane_class(chain, c, ids):
    if c == "Car":
        return {f: [R.rule37(SZ.keep(d)) for d in chain.get(f, []) if d["class"] == c] for f in ids}
    return {f: [SZ.keep(d) for d in chain.get(f, []) if d["class"] == c] for f in ids}


def person(paths, ids, load, keys):
    p = M1.person_chain(paths, True)
    return SZ.PP.filt({f: p.get(f, []) for f in ids}, "Person", load, keys)


def bicycle(paths, ids, load, keys):
    src = E.load(paths, "Bicycle")
    out = {}
    for f in ids:
        rows = [x.get(f, []) for x in src]
        assert all(len(r) == len(rows[0]) for r in rows), f
        out[f] = []
        for tup in zip(*rows):
            assert all(t["box"] == tup[0]["box"] for t in tup), f
            out[f].append(E.combine(list(tup), "R4") if tup[0]["score"] >= 0.05 else SZ.keep(tup[0]))
    return SZ.PP.filt(out, "Bicycle", load, keys)


def ap(pred, gt, frames, c):
    n, fl = SZ.flags({f: pred.get(f, []) for f in frames}, {f: gt[f] for f in frames}, c)
    return SZ.ap_from(n, fl)[0]


def candidates(split, ids, keys, only=None):
    """name -> (class, {frame: boxes}) for split; only = set of names to build (val mode)."""
    load = lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3]
    out = {}
    for i, (name, (c, spec, ovr)) in enumerate(lane_candidates().items()):
        if only is None or name in only:
            out[name] = (c, lane_class(run_lane(f"{i:02d}", spec, ovr, split), c, ids))
            print("built", name, flush=True)
    pc = {"Person: v54": P_REF(split)}
    for k in range(5):
        pc[f"Person: drop {P_NAMES[k]}"] = [p for j, p in enumerate(P_REF(split)) if j != k]
    for n, f in P_ADD.items():
        pc[f"Person: add {n}"] = P_REF(split) + [f(split)]
    for name, paths in pc.items():
        if only is None or name in only:
            out[name] = ("Person", person(paths, ids, load, keys))
            print("built", name, flush=True)
    if only is None:
        bc = {"Bicycle: v54": [RL.B / f"v29_{split}_ref.json"]}
        for n, f in B_ADD.items():
            bc[f"Bicycle: add {n}"] = [RL.B / f"v29_{split}_ref.json", f(split)]
        for name, paths in bc.items():
            out[name] = ("Bicycle", bicycle(paths, ids, load, keys))
            print("built", name, flush=True)
    return out


def near():
    ids = S.ids_of("val")
    seg = lambda s: [l.strip() for l in open(f"data/pcdet/lumpi_real_50k/ImageSets/{s}.txt") if l.strip()]
    H = {"before": seg("before"), "inside": seg("inside")}
    H["both"] = sorted(H["before"] + H["inside"])
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    sel = {c: all(cnt.get(h, {}).get(c, {}).get("distinct_ids", 0) >= SELECTABLE_MIN.get(c, 5) for h in ("before", "inside")) for c in ("Person", "Car", "Bicycle", "Motorcycle", "Van")}
    g1 = json.loads(Path("logs/neartest_read.json").read_text())
    vallike = sum(v == "val-like" for v in g1["test_like"].values()) >= 3
    print("selectable (>= 5 distinct ids in each half):", sel, "| G1 val-like:", vallike, flush=True)
    keys = SZ.PP.static_keys("lumpi")
    cand = candidates("val", ids, keys)
    # self-check: the reference candidates equal G1's v54 classes
    v54 = SZ.build("val", ids)
    for c in ("Car", "Motorcycle", "Person", "Bicycle"):
        ref = cand[f"{c}: v54"][1]
        same = all(SZ.flags({f: ref[f]}, {f: gt[f]}, c)[1][0.5] == SZ.flags({f: v54[c][f]}, {f: gt[f]}, c)[1][0.5] for f in H["both"])
        print(f"self-check: {c} reference candidate scores as lumpi_v54's {c}: {same}", flush=True)
        if not same:
            sys.exit(f"STOP the {c} reference does not reproduce lumpi_v54")
    res = {"selectable": sel, "g1_val_like": vallike, "cands": {}, "pass": {}}
    for name, (c, p) in cand.items():
        res["cands"][name] = {"class": c, **{h: ap(p, gt, fr, c) for h, fr in H.items()}}
    for name, r in res["cands"].items():
        c = r["class"]
        ref = res["cands"][f"{c}: v54"]
        d = {h: (None if r[h] is None or ref[h] is None else r[h] - ref[h]) for h in H}
        ok = bool(sel.get(c) and not vallike and name != f"{c}: v54" and d["before"] is not None and d["before"] >= 0.01 and d["inside"] >= 0.01)
        r.update(diff=d, near_pass=ok)
        print(f"{name:45s} before {r['before']} inside {r['inside']} both {r['both']} | vs v54 {d['before'] if d['before'] is None else round(d['before'], 4)} / "
              f"{d['inside'] if d['inside'] is None else round(d['inside'], 4)} | {'PASS (near-test)' if ok else ''}", flush=True)
    # information only (no gate, no G3 choice): lumpi_v58's Car lane. v58's Car is L11-RS (c)'s route
    # lane = the v29 lane + vrcnn_lumpi_rs at weight 1 for Car (its 4-view TTA with one vote per view, as its kit files), save_lumpi_chain, rule37:
    # exactly the candidate "Car: add vrcnn_lumpi_rs" here (G3 runs vrcnn_lumpi_rs with the same flags on the near-test frames).
    c58 = cand["Car: add vrcnn_lumpi_rs"][1]
    c54 = cand["Car: v54"][1]
    res["v58_car"] = {}
    for h in ("before", "inside"):
        fr = H[h]
        a = SZ.ap_from(*SZ.flags({f: c58.get(f, []) for f in fr}, {f: gt[f] for f in fr}, "Car"))
        b = SZ.ap_from(*SZ.flags({f: c54.get(f, []) for f in fr}, {f: gt[f] for f in fr}, "Car"))
        res["v58_car"][h] = dict(v58=a, v54=b, distinct_ids=cnt.get(h, {}).get("Car", {}).get("distinct_ids"))
        print(f"information: lumpi_v58 Car vs lumpi_v54 Car, {h} ({len(fr)} frames, {res['v58_car'][h]['distinct_ids']} distinct Car ids): "
              f"AP@0.5 {b[0]:.4f} -> {a[0]:.4f} ({a[0] - b[0]:+.4f}), AP@0.7 {b[1]:.4f} -> {a[1]:.4f} ({a[1] - b[1]:+.4f})", flush=True)
    Path("logs/g3_near.json").write_text(json.dumps(res, indent=1, default=str))


def val():
    res = json.loads(Path("logs/g3_near.json").read_text())
    passed = [n for n, r in res["cands"].items() if r["near_pass"]]
    if not passed:
        print("no near-test pass: lumpi_v54 stays", flush=True)
        res["chosen"] = {}
        Path("logs/g3_final.json").write_text(json.dumps(res, indent=1, default=str))
        return
    ids = S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    keys = SZ.PP.static_keys("lumpi")
    cls = {res["cands"][n]["class"] for n in passed}
    cand = candidates("val", ids, keys, only=set(passed) | {f"{c}: v54" for c in cls})
    chosen = {}
    for c in cls:
        nval = sum(1 for f in ids for g in gt[f] if g["class"] == c)
        ref = ap(cand[f"{c}: v54"][1], gt, ids, c)
        ok = []
        for n in passed:
            if res["cands"][n]["class"] != c:
                continue
            a = ap(cand[n][1], gt, ids, c)
            v_ok = nval < VAL_LABELS_MIN or a >= ref - 0.005
            res["cands"][n].update(val=a, val_ref=ref, val_ok=v_ok)
            print(f"val {n}: {a:.4f} vs v54 {ref:.4f} ({a - ref:+.4f}; val labels {nval}) | {'ok' if v_ok else 'falls > 0.005'}", flush=True)
            if v_ok:
                ok.append(n)
        if ok:
            chosen[c] = max(ok, key=lambda n: res["cands"][n]["both"])
            print(f"chosen for {c}: {chosen[c]} (both halves {res['cands'][chosen[c]]['both']:.4f})", flush=True)
    res["chosen"] = chosen
    Path("logs/g3_final.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"near": near, "val": val}[sys.argv[1]]()
