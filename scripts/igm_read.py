"""Inside-gate reading for the four new LUMPI members on lumpi_v59 (settings
fixed before any of them is read). One member per run, each class read add-one against v59, with the machinery of scripts/ig_read.py (Car, Van,
Person) and scripts/ig_bike_read.py (Bicycle), unchanged:
  Member forms:  fshnet_lumpi_s2  zF2 form (route_lanes_lumpi.sh): fshnet_lumpi at 0.5 + fshnet_lumpi_s2 at 0.5 as one lane member;
                 dsvt_lumpi, vmamba_lumpi  a lane member at weight 1;
                 pv_lumpi_s2  (refiner) Person: its own Person second stage (min 0.01, on the v29 lane) as a sixth member of v59's chain, as ig_read;
                 Car / Van / Bicycle: its 4-view kit predictions as a lane member at weight 1.
  Car:     v59's Car lane (the v29 lane + vrcnn_lumpi_rs at weight 1) + the member for Car; save_lumpi_chain; rule37 (ig_read).
  Van:     the v29 lane + the member for Van; then rule37 on the lane's Van and the v29 Truck, rules T5 / V5 (ig_read).
  Person:  one-stage members: lumpi_v51's five second stages (L11, L12, L11b seed 2, PV-RCNN++, test A; min 0.01) on the member's Person lane
           (the v29 lane + the member for Person), R0, grow + length rule, P3 (f2nt_read's chain), plus v59's two-wheeler copies;
           pv_lumpi_s2: as ig_read (sixth member).  Files output/lumpi/<m>/igm/person_<r>_val_ref.json (near: scripts/igm_refine.sh on the
           3090; val: the member route's s2 files), pv: output/lumpi/<m>/igm/v29_val_person_min01.json.
  Bicycle: L11b's second stage on the member's Bicycle lane (the v29 lane + the member for Bicycle), R4 at >= 0.05, P3 (ig_bike_read);
           Person recomputed with the candidate's copies (v59's Person + copies from the new Bicycle and v59's Motorcycle).
           File output/lumpi/<m>/igm/bicycle_val_ref.json (near: igm_refine.sh; val: the member route's s2/bicycle_val_ref.json).
Gate (ig_read's rule, unchanged): inside AP@0.5 +0.01 or more (class with >= 5 inside ids); val (320, halves pooled) not down more than 0.01;
Near-test veto: no class the change touches falls by more than 0.005 on a segment (before / inside) with >= 5 ids of that class
(logs/neartest_id_counts.json); for a Bicycle change Person (inside and before) is read with the recomputed copies. Before, AP@0.7: information.
Self-checks (STOP): inside Person 0.317989, Car 0.847262 (G3's v58 Car), Van = v54's, Bicycle 0.2105 (= v59's on both halves); val Car 0.896, Person 0.48.
    python scripts/igm_read.py lanes <m>          (either workspace: builds the Car / Van lanes only, cached by name)
    python scripts/igm_read.py near <m>           (in neartest_ws) -> logs/igm_<m>_near.json
    python scripts/igm_read.py val <m>            (main workspace, needs logs/igm_<m>_near.json) -> logs/igm_<m>_val.json + the gate table
    --classes Car,Van: only those classes' candidates and gate lines (bases and self-checks unchanged) -> logs/igm_<m>_carvan_{near,val}.json
    --check: dry run (bases, self-checks, and the member's candidates built where inputs exist; box counts only, no member AP, no json).
    python scripts/igm_read.py combo near|val --car X --van X --person X --bicycle X --motorcycle X [--fallback C]: the final-combination
    reading (scripts/igm_combo.py) -> logs/igm_combo_<key>_{near,val}.json; build_lumpi_v6x needs its PASS.
Motorcycle (dsvt_lumpi, vmamba_lumpi, pv_lumpi_s2 only, NOT fshnet_lumpi_s2): a
board-informed class choice. Base = v59's Motorcycle (the v29 file's, no second stage); STOP unless G3's rebuilt v29 lane (zG300, G3.B_SPEC) gives it
box for box (near and val). Candidate = the v29 lane + the member at weight 1 for Motorcycle (G3's lane machinery; PV s2: its 4-view kit
predictions); no second stage. Person = v59's Person + copies recomputed from v59's Bicycle and the candidate Motorcycle (W.copies, as v57).
Gate: Motorcycle inside +0.01 (5 ids), val pooled not down > 0.01; veto: Person inside / before (recomputed copies) not down > 0.005
(Motorcycle before has 3 ids: no veto there).
Truck / Van RS second stages, pseudo-member "rs3" (Truck and Van only; fixed before
any reading). Inputs output/lumpi/rs3/<r>_val_ref.json, r = vrcnn_lumpi_rs, vrcnn_lumpi_c_s2, pv_lumpi_s2 (x6_refine --min_score 0.05 on
lumpi_v29_<split>.json; in neartest_ws the near-test files, copied as _val_; scripts/rs3_fetch.sh). Per class C in {Truck, Van}: the refiners'
input boxes must equal the lref chain's (l11c/v29_val_ref.json) per frame, in order (STOP otherwise); R6 over the 3 refiners (ensemble_rules,
A = 0.71) for a box >= 0.05, else its v29 box and score (g3_search.bicycle's template); then rule37 on the refined height and T5 / V5 on the
refined box, the other class from lref (truck() mirrors IG.van). Self-checks (STOP): the rebuilt v59 Truck / Van equal SZ.build box for box;
inside Truck 0.05568 (logs/neartest_read.json v54) and Van (v54's), val Truck 0.4632 and Van 0.1708; the class not changed equals v59 box for box.
Gate: inside +0.01 (>= 5 ids), val pooled not down > 0.01, the veto (-0.005 on a segment with >= 5 ids, both classes), PLUS val a - base >= +0.005
unrounded.  python scripts/igm_read.py near|val rs3 [--check]  -> logs/igm_rs3_{near,val}.json, keys "<C>: add rs3".
v60-base mode (scripts/igm60.py, separate from the v59 path above, which it leaves unchanged): G3 member
safdnet_lumpi_rs (tag SR, one-stage, Motorcycle read), the g2 Bicycle refiner set and the moto2 Motorcycle second stage, each read against
lumpi_v60 rebuilt from logs/build_lumpi_v60.json's switches; gate inside +0.01 and val +0.005 vs v60 (moto2: no val bar; see scripts/igm60.py).
    python scripts/igm_read.py near60|val60 safdnet_lumpi_rs|g2|moto2 [--check [--assume-v59]]  -> logs/igm60_<m>_{near,val}.json
    python scripts/igm_read.py lanes60 safdnet_lumpi_rs [--assume-v59]   (G3's Car / Van / Truck / Motorcycle lanes on v60)
LUMPI #4: vrcnn_lumpi_b_fix2_rs (tag B2, REFINER: Person as pv_lumpi_s2) is read on v60
in G3's form: near60 / val60 / lanes60 vrcnn_lumpi_b_fix2_rs -> logs/igm60_vrcnn_lumpi_b_fix2_rs_{near,val}.json.
spec_lumpi_van_rs (tag VR): a Van-only Voxel R-CNN specialist (trained on an A100 GPU), one-stage, read on v60 in G3's form for Van ONLY (VAN_ONLY:
classes_of -> ("Van",); the Truck of the same T5 / V5 pass for the veto); no Car / Person / Bicycle / Motorcycle candidate or lane; v60 modes only:
near60 / val60 / lanes60 spec_lumpi_van_rs -> logs/igm60_spec_lumpi_van_rs_{near,val}.json (scripts/van_rs_chain.sh).
spec_lumpi_person_rs (tag PS): a Person-only Voxel R-CNN specialist (trained on an A100 GPU), read on v60 for
Person ONLY in REFINER form (PERSON_ONLY: classes_of -> ("Person",)): its own Person second stage (min 0.01, on the v29 lane) as a sixth member of
v60's own Person chain (G3.P_REF + output/lumpi/<m>/igm/v29_val_person_min01.json), plus v60's unchanged two-wheeler copies; no Car / Van / Truck /
Bicycle / Motorcycle candidate or lane; inputs_ok = that Person file; v60 modes only: near60 / val60 spec_lumpi_person_rs ->
logs/igm60_spec_lumpi_person_rs_{near,val}.json (scripts/person_rs_chain.sh)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ensemble_rules as E
import g3_search as G3
import ig_bike_read as IB
import ig_read as IG
import lumpi_copies_routes as W
import lumpi_static as S
import lumpi_v37_rules as R
import size_check_lumpi as SZ

TAGS = {"fshnet_lumpi_s2": "F2", "dsvt_lumpi": "DS", "vmamba_lumpi": "VM", "pv_lumpi_s2": "P2",
        "safdnet_lumpi_rs": "SR",   # SR: G3 (SAFDNet on real-size data), one-stage, read on v60 (scripts/igm60.py; igm_wait.sh T=SR)
        "vrcnn_lumpi_b_fix2_rs": "B2",   # B2: LUMPI #4, refiner-type member read on v60 (igm60.py; igm_wait.sh T=B2)
        "vmamba_lumpi_s2": "V2", "dsvt_lumpi_s2": "D2",   # Wave 2B pairs: seed 1 at 0.5 + seed 2 at 0.5 as one member (PAIR)
        "pv_lumpi_rs": "PR",   # PR: PV-RCNN++ seed 666 on ut_lumpi_fix2_rs, pairing with pv_lumpi_s2 (wiring only)
        "spec_lumpi_van_rs": "VR",   # VR: Van-only Voxel R-CNN specialist (A100-trained), one-stage, read on v60 for Van only (igm60.py; van_rs_chain.sh)
        "spec_lumpi_person_rs": "PS"}   # PS: Person-only Voxel R-CNN specialist (A100-trained), REFINER form, read on v60 for Person only (igm60.py; person_rs_chain.sh)
VAN_ONLY = {"spec_lumpi_van_rs"}   # members read for Van only (classes_of -> ("Van",)); v60 modes only (igm60.py)
PERSON_ONLY = {"spec_lumpi_person_rs"}   # members read for Person only, as a sixth Person second stage (classes_of -> ("Person",)); v60 modes only
PAIR = {"vmamba_lumpi_s2": "vmamba_lumpi", "dsvt_lumpi_s2": "dsvt_lumpi", "pv_lumpi_rs": "pv_lumpi_s2"}   # seed-2 model -> its seed 1 (zF2 form)
PAIR_W = (0.5, 0.5)   # (seed 1, seed 2); the 1.0 + 0.0 self-check must rebuild the seed-1 lane box for box
DRY = {"fshnet_lumpi": "FD"}   # --check only: a one-stage member whose near-test predictions exist (G3)
REFINER = {"pv_lumpi_s2", "vrcnn_lumpi_b_fix2_rs", "spec_lumpi_person_rs"}   # LUMPI #4: Person = its own second stage (min 0.01, v29 lane) as a sixth member, as pv_lumpi_s2
REF_ORDER = ("l11", "l12", "l11b_s2", "pv", "testa")   # lumpi_v51's order (P_REF / f2nt_read)
CLASSES = ("Person", "Car", "Van", "Bicycle")
ALL = CLASSES + ("Motorcycle",)
MOTO = {"dsvt_lumpi", "vmamba_lumpi", "pv_lumpi_s2", "safdnet_lumpi_rs", "vrcnn_lumpi_b_fix2_rs"}   # Motorcycle read for these only (G3
# safdnet_lumpi_rs and LUMPI #4 vrcnn_lumpi_b_fix2_rs: v60-base mode); fshnet_lumpi: --check code test only
MOTO_NOTE = "board-informed class choice (prompted by another team's public per-class results)"
# Motorcycle HOLD: no Motorcycle reading until its val rule is set (v59's val Motorcycle AP is 0.0051, so "not down more
# than 0.01" can never fail). The rule goes in logs/igm_moto_rule.json in the workspace that reads
# ({"val_rule": "not_down"} = a >= base, or {"val_rule": "minus_0.01"} = the usual class gate, a - base >= -0.01); until then real readings skip Motorcycle
# (--check dry runs, which compute no AP, still build it), and a later Motorcycle-only reading uses --classes Motorcycle (logs ..._motorcycle_...).
MOTO_RULE = Path("logs/igm_moto_rule.json")


def moto_val_ok(a, base):
    r = json.loads(MOTO_RULE.read_text()).get("val_rule") if MOTO_RULE.exists() else None
    if r == "not_down":
        return a >= base
    if r == "minus_0.01":
        return a - base >= -0.01
    sys.exit(f"STOP Motorcycle val rule not decided ({MOTO_RULE}: {r})")


classes_of = lambda m, check: ("Van",) if m in VAN_ONLY else ("Person",) if m in PERSON_ONLY else ALL if (m in MOTO and (check or MOTO_RULE.exists())) or (check and m in DRY) else CLASSES
suffix = lambda m, cls: "" if set(cls) == set(classes_of(m, False)) else "_" + "".join(c.lower() for c in ALL if c in cls)   # Car,Van -> _carvan
rd = lambda p: [l.strip() for l in open(p) if l.strip()]
LOAD = lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3]
RS3 = ("vrcnn_lumpi_rs", "vrcnn_lumpi_c_s2", "pv_lumpi_s2")   # L11-RS, L12-s2, PV-s2 (epoch 15), the spec's refiner order
RS3_DIR = Path("output/lumpi/rs3")
RS3_CLASSES = ("Truck", "Van")
RS3_BAR = 0.005   # rs3 rule: val a - base >= +0.005, unrounded, on top of the usual gate
RS3_VAL = {"Truck": 0.4632, "Van": 0.1708}   # v59's val Truck / Van (route_read_lumpi EXPECT, R0); inside: logs/neartest_read.json v54 (= v59)


def member(m, c, w=PAIR_W):
    if m == "fshnet_lumpi_s2":
        return [["file:g3in/fshnet_lumpi", 0.5, [c]], ["file:g3in/fshnet_lumpi_s2", 0.5, [c]]]
    if m in PAIR:   # Wave 2B pair: seed 1 at w[0] + seed 2 at w[1] as one member; w = (1.0, 0.0) only for the self-check
        return [[f"file:g3in/{PAIR[m]}", w[0], [c]], [f"file:g3in/{m}", w[1], [c]]]
    return [[f"file:g3in/{m}", 1, [c]]]


def tag(m, check):
    if m in TAGS:
        return TAGS[m]
    if check and m in DRY:
        return DRY[m]
    sys.exit(f"unknown member {m}")


def inputs_ok(m):
    if m in PERSON_ONLY:   # a Person-only refiner has no kit predictions in g3in: its input is its own Person second-stage file
        return Path(f"output/lumpi/{m}/igm/v29_val_person_min01.json").exists()
    need =["output/lumpi/g3in/fshnet_lumpi_val.json"] if m == "fshnet_lumpi_s2" else [f"output/lumpi/g3in/{PAIR[m]}_val.json"] if m in PAIR else []
    return all(Path(p).exists() for p in need + [f"output/lumpi/g3in/{m}_val.json"])


def lanes(m, t, ids):
    lref = json.loads((G3.RL.L / "v29_val_ref.json").read_text())
    car = G3.lane_class(G3.run_lane(f"IMc{t}", G3.B_SPEC + [IG.RS] + member(m, "Car"), {}, "val"), "Car", ids)
    van = IG.van(G3.run_lane(f"IMv{t}", G3.B_SPEC + member(m, "Van"), {}, "val"), lref, ids)
    return car, van


def same_boxes(a, b, ids):
    k = lambda v: sorted(json.dumps(SZ.keep(d), sort_keys=True) for d in v)
    return all(k(a.get(f, [])) == k(b.get(f, [])) for f in ids)


def build(m, t, ids, cls=CLASSES):
    """name -> (class, {frame: boxes}); bases first (v59, all four, for the self-checks), then the member's candidates of the classes cls
    where their inputs exist."""
    keys = SZ.PP.static_keys("lumpi")
    P = SZ.build("val", ids)
    cp = W.copies({f: [d for c in W.TW for d in P[c].get(f, [])] for f in ids})
    lref = json.loads((G3.RL.L / "v29_val_ref.json").read_text())
    person_of = lambda b: {f: P["Person"][f] + W.copies({f: b[f] + P["Motorcycle"][f]})[f] for f in ids}   # ig_bike_read's recomputed copies
    base_p = G3.person(G3.P_REF("val"), ids, LOAD, keys)
    out = {"Car: v59": ("Car", G3.lane_class(G3.run_lane("IGc00", G3.B_SPEC + [IG.RS], {}, "val"), "Car", ids)),
           "Van: v59": ("Van", IG.van(lref, lref, ids)),
           "Person: v59": ("Person", {f: base_p[f] + cp[f] for f in ids}),
           "Bicycle: v59": ("Bicycle", P["Bicycle"]),
           "Bicycle: v59 (lane check)": ("Bicycle", IB.bike(G3.RL.B / "v29_val_ref.json", ids, keys)),
           "Person: v59 (copies from v59 Bicycle)": ("Person", person_of(P["Bicycle"]))}
    person_mo = lambda mo: {f: P["Person"][f] + W.copies({f: P["Bicycle"][f] + mo[f]})[f] for f in ids}   # copies from v59's Bicycle + candidate Motorcycle
    if "Motorcycle" in cls:
        out["Motorcycle: v59"] = ("Motorcycle", P["Motorcycle"])
        out["Motorcycle: v59 (lane check)"] = ("Motorcycle", G3.lane_class(G3.run_lane("00", G3.B_SPEC, {}, "val"), "Motorcycle", ids))
    if not inputs_ok(m):
        print(f"no kit predictions for {m} in output/lumpi/g3in: member skipped", flush=True)
        return out
    if "Car" in cls or "Van" in cls:
        car, van = lanes(m, t, ids)
        if "Car" in cls:
            out[f"Car: add {m}"] = ("Car", car)
        if "Van" in cls:
            out[f"Van: add {m}"] = ("Van", van)
    d = Path(f"output/lumpi/{m}/igm")
    if m in REFINER:
        pp = d / "v29_val_person_min01.json"
        pfiles = G3.P_REF("val") + [pp] if pp.exists() else None
    else:
        pf = [d / f"person_{r}_val_ref.json" for r in REF_ORDER]
        pfiles = pf if all(p.exists() for p in pf) else None
    if "Person" not in cls:
        pass
    elif pfiles:
        p = G3.person(pfiles, ids, LOAD, keys)
        out[f"Person: add {m}"] = ("Person", {f: p[f] + cp[f] for f in ids})
    else:
        print(f"{m}: no Person second-stage files in {d}: Person skipped", flush=True)
    bf = d / "bicycle_val_ref.json"
    if "Bicycle" not in cls:
        pass
    elif bf.exists():
        b = IB.bike(bf, ids, keys)
        out[f"Bicycle: add {m}"] = ("Bicycle", b)
        out[f"Person: copies from Bicycle: add {m}"] = ("Person", person_of(b))
    else:
        print(f"{m}: no Bicycle second-stage file {bf}: Bicycle skipped", flush=True)
    if "Motorcycle" in cls:
        mo = G3.lane_class(G3.run_lane(f"IMm{t}", G3.B_SPEC + member(m, "Motorcycle"), {}, "val"), "Motorcycle", ids)
        out[f"Motorcycle: add {m}"] = ("Motorcycle", mo)
        out[f"Person: copies from Motorcycle: add {m}"] = ("Person", person_mo(mo))
    print("built", m, flush=True)
    return out


def counts(cand):
    for name, (c, p) in cand.items():
        if " add " in name:
            print(f"check: built {name:48s} {len(p)} frames, {sum(len(v) for v in p.values())} boxes", flush=True)


def ap2(p, gt, fr, c):
    return SZ.ap_from(*SZ.flags({f: p.get(f, []) for f in fr}, {f: gt[f] for f in fr}, c))   # as ig_read / ig_bike_read


def moto_check(cand, ids, where):
    if "Motorcycle: v59" not in cand:
        return True
    ok = same_boxes(cand["Motorcycle: v59 (lane check)"][1], cand["Motorcycle: v59"][1], ids)
    print(f"self-check {where} Motorcycle: rebuilt v29 lane equals v59's Motorcycle box for box: {ok}", flush=True)
    return ok


def near(m, check, cls=CLASSES):
    t = tag(m, check)
    ids = S.ids_of("val")
    assert len(ids) == 266
    H = {h: rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in ("before", "inside")}
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    cand = build(m, t, ids, cls)
    v54 = SZ.build("val", ids)
    want = {"Car": json.loads(Path("logs/g3_near.json").read_text())["v58_car"]["inside"]["v58"][0],
            "Person": json.loads(Path("logs/m3_neartest_copies.json").read_text())["copies_on"]["inside"][0],
            "Van": SZ.ap_from(*SZ.flags({f: v54["Van"][f] for f in H["inside"]}, {f: gt[f] for f in H["inside"]}, "Van"))[0]}
    ok = True
    for c, w in want.items():
        got = ap2(cand[f"{c}: v59"][1], gt, H["inside"], c)[0]
        print(f"self-check inside {c}: v59 base {got:.6f} vs {w:.6f}: {abs(got - w) < 1e-9}", flush=True)
        ok = ok and abs(got - w) < 1e-9
    bb = {h: ap2(cand["Bicycle: v59"][1], gt, H[h], "Bicycle")[0] for h in H}
    same = all(ap2(cand["Bicycle: v59 (lane check)"][1], gt, H[h], "Bicycle")[0] == bb[h] for h in H)
    pc = ap2(cand["Person: v59 (copies from v59 Bicycle)"][1], gt, H["inside"], "Person")[0]
    print(f"self-check inside Bicycle: v59 base {bb['inside']:.6f} (0.2105: {round(bb['inside'], 4) == 0.2105}); lane rebuild equals v59 on before / "
          f"inside: {same}; Person with copies recomputed from v59's Bicycle {pc:.6f} vs {want['Person']:.6f}: {abs(pc - want['Person']) < 1e-9}", flush=True)
    ok = ok and same and round(bb["inside"], 4) == 0.2105 and abs(pc - want["Person"]) < 1e-9
    ok = moto_check(cand, ids, "near") and ok
    if not ok:
        sys.exit("STOP a base does not reproduce v59 on inside")
    if check:
        counts(cand)
        print("check mode: no member AP computed, nothing written", flush=True)
        return
    res = {}
    for name, (c, p) in cand.items():
        res[name] = {"class": c, **{h: ap2(p, gt, fr, c) for h, fr in H.items()}}
    base_of = lambda name, c: "Person: v59 (copies from v59 Bicycle)" if name.startswith("Person: copies") else f"{c}: v59"   # like for like, as ig_bike_read
    for name, r in res.items():
        if "Motorcycle" in name:
            r["note"] = MOTO_NOTE
        b = res[base_of(name, r["class"])]
        r["d_inside"], r["d_before"] = r["inside"][0] - b["inside"][0], r["before"][0] - b["before"][0]
        print(f"{name:48s} inside {r['inside'][0]:.4f} ({r['d_inside']:+.4f}) | before {r['before'][0]:.4f} ({r['d_before']:+.4f}) | "
              f"AP@0.7 inside {r['inside'][1]:.4f}{' | ' + MOTO_NOTE if 'note' in r else ''}", flush=True)
    Path(f"logs/igm_{m}{suffix(m, cls)}_near.json").write_text(json.dumps(res, indent=1))


def val(m, check, cls=CLASSES):
    t = tag(m, check)
    ids = S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    near_res = None if check else json.loads(Path(f"logs/igm_{m}{suffix(m, cls)}_near.json").read_text())
    cand = build(m, t, ids, cls)
    for c, want in {"Car": 0.896, "Person": 0.48}.items():
        got = round(G3.ap(cand[f"{c}: v59"][1], gt, ids, c), 4)
        print(f"self-check val {c}: v59 base {got} vs {want}: {got == want}", flush=True)
        if got != want:
            sys.exit(f"STOP the val {c} base does not reproduce v59")
    for c, want in {"Van": 0.1708, "Bicycle": 0.5662}.items():   # v59 val bases, 4 decimals
        if f"{c}: v59" in cand:
            got = round(G3.ap(cand[f"{c}: v59"][1], gt, ids, c), 4)
            print(f"self-check val {c}: v59 base {got} vs {want}: {got == want}", flush=True)
            if got != want:
                sys.exit(f"STOP the val {c} base does not reproduce v59")
    if not moto_check(cand, ids, "val"):
        sys.exit("STOP the val Motorcycle base does not reproduce v59")
    if check:
        counts(cand)
        print("check mode: no member AP computed, nothing written", flush=True)
        return
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    nid = lambda h, c: cnt.get(h, {}).get(c, {}).get("distinct_ids", 0)
    res = {}
    print(f"gate table for {m} (inside +0.01, val not down > 0.01, veto: touched class not down > 0.005 on a segment with >= 5 ids)", flush=True)
    for c in [c for c in ALL if c in cls]:
        name = f"{c}: add {m}"
        if name not in cand or name not in near_res:
            print(f"{name:32s} not read (inputs missing)", flush=True)
            continue
        n = near_res[name]
        base = G3.ap(cand[f"{c}: v59"][1], gt, ids, c)
        a = G3.ap(cand[name][1], gt, ids, c)
        pb = near_res.get(f"Person: copies from {c}: add {m}") if c in ("Bicycle", "Motorcycle") else None
        touched = [(c, n)] + ([("Person", pb)] if pb else [])
        veto = [f"{k} {h} {r['d_' + h]:+.4f}" for k, r in touched for h in ("before", "inside") if nid(h, k) >= 5 and r["d_" + h] < -0.005]
        val_ok = moto_val_ok(a, base) if c == "Motorcycle" else a - base >= -0.01   # Motorcycle: its own val rule (moto_val_ok)
        pv = None
        if c in ("Bicycle", "Motorcycle") and f"Person: copies from {c}: add {m}" in cand:   # conflict 2c (igm_combo): Person val with the
            pv = (G3.ap(cand[f"Person: copies from {c}: add {m}"][1], gt, ids, "Person"),    # recomputed copies not down > 0.01 vs v59
                  G3.ap(cand["Person: v59 (copies from v59 Bicycle)"][1], gt, ids, "Person"))
            if pv[0] - pv[1] < -0.01:
                veto.append(f"Person val (copies) {pv[0] - pv[1]:+.4f}")
        ok = bool(nid("inside", c) >= 5 and n["d_inside"] >= 0.01 and val_ok and not veto)
        res[name] = dict(val=a, val_base=base, d_val=a - base, d_inside=n["d_inside"], d_before=n["d_before"], veto=veto, passed=ok)
        if pb:
            res[name].update(person_d_inside=pb["d_inside"], person_d_before=pb["d_before"])
        if pv:
            res[name].update(person_val=pv[0], person_val_base=pv[1], person_d_val=pv[0] - pv[1])
        if c == "Motorcycle":
            res[name]["note"] = MOTO_NOTE
        extra = f" | Person (copies) inside {pb['d_inside']:+.4f} before {pb['d_before']:+.4f}" if pb else ""
        extra += f" val {pv[0] - pv[1]:+.4f}" if pv else ""
        extra += f" | {MOTO_NOTE}" if c == "Motorcycle" else ""
        print(f"{name:32s} inside {n['inside'][0]:.4f} ({n['d_inside']:+.4f}, {nid('inside', c)} ids) | before {n['d_before']:+.4f} | "
              f"val {a:.4f} ({a - base:+.4f}){extra} | veto {'; '.join(veto) or 'none'} | {'PASS' if ok else 'fail'}", flush=True)
    Path(f"logs/igm_{m}{suffix(m, cls)}_val.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ rs3: Truck / Van RS second stages
def tv_pass(truck_src, van_src, ids, split="val"):
    """rule37 on truck_src's Truck boxes and van_src's Van boxes, then T5 / V5 in one pass (route_read_lumpi.Chain.cls's Truck / Van steps;
    each box's T5 / V5 features use only its own points). -> {"Truck": {frame: boxes}, "Van": {frame: boxes}}"""
    tv = {f: [R.rule37(G3.RL.keep(d)) for d in truck_src.get(f, []) if d["class"] == "Truck"]
          + [R.rule37(G3.RL.keep(d)) for d in van_src.get(f, []) if d["class"] == "Van"] for f in ids}
    tv = R.apply(tv, R.features(split, tv, ids), ["T5", "V5"])
    return {c: {f: [d for d in tv[f] if d["class"] == c] for f in ids} for c in RS3_CLASSES}


def truck(chain, lref, ids, split="val"):
    """IG.van's mirror for Truck (= route_read_lumpi.Chain.cls("Truck") with the chain's Truck): rule37 on the chain's Truck and lref's Van,
    rules T5 / V5; the Truck boxes. split: the point split of the T5 / V5 features (IG.van fixes it to "val")."""
    return tv_pass(chain, lref, ids, split)["Truck"]


def rs3_files(split="val"):
    return [RS3_DIR / f"{r}_{split}_ref.json" for r in RS3]


def rs3_fuse(lref, ids, split="val"):
    """R6 over the 3 RS refiners for every Truck / Van box, keyed per frame and box in lref's order: the refiners' input boxes (and scores) must
    equal lref's box for box, in order (STOP otherwise). A box >= 0.05: ensemble_rules.combine(.., "R6"); below: its v29 box and score
    (g3_search.bicycle's template). -> {"Truck": {frame: boxes}, "Van": {frame: boxes}}"""
    files = rs3_files(split)
    src = [{f: [d for d in v if d["class"] in RS3_CLASSES] for f, v in json.loads(p.read_text()).items()} for p in files]
    out = {c: {} for c in RS3_CLASSES}
    for c in RS3_CLASSES:
        for f in ids:
            ref = [d for d in lref[f] if d["class"] == c]
            rows = [[d for d in x.get(f, []) if d["class"] == c] for x in src]
            for p, r in zip(files, rows):
                if len(r) != len(ref) or any(a["box"] != b["box"] or a["score"] != b["score"] for a, b in zip(r, ref)):
                    sys.exit(f"STOP rs3 {c} frame {f}: the input boxes of {p} differ from the lref chain's ({G3.RL.L / f'v29_{split}_ref.json'})")
            out[c][f] = [E.combine(list(t), "R6") if t[0]["score"] >= 0.05 else SZ.keep(t[0]) for t in zip(*rows)]
    return out


def rs3_build(ids, where):
    """name -> (class, {frame: boxes}): the v59 bases (STOP unless equal to SZ.build box for box), then per class C the rs3 candidate "<C>: add rs3"
    and the other class from the same T5 / V5 pass, "<O>: other class with rs3 <C>" (STOP unless equal to v59 box for box)."""
    lref = json.loads((G3.RL.L / "v29_val_ref.json").read_text())
    P = SZ.build("val", ids)
    out = {"Truck: v59": ("Truck", truck(lref, lref, ids)), "Van: v59": ("Van", IG.van(lref, lref, ids))}
    ok = True
    for c in RS3_CLASSES:
        same = same_boxes(out[f"{c}: v59"][1], P[c], ids)
        print(f"self-check {where} {c}: the rebuilt v59 chain equals SZ.build box for box: {same}", flush=True)
        ok = ok and same
    del P
    if not ok:
        sys.exit(f"STOP the rebuilt v59 Truck / Van do not equal SZ.build on {where}")
    miss = [str(p) for p in rs3_files() if not p.exists()]
    if miss:
        print("rs3: second-stage files missing (scripts/rs3_fetch.sh): " + ", ".join(miss) + ": candidates skipped", flush=True)
        return out
    fu = rs3_fuse(lref, ids)
    for c in RS3_CLASSES:
        o = "Van" if c == "Truck" else "Truck"
        pair = tv_pass(fu["Truck"] if c == "Truck" else lref, fu["Van"] if c == "Van" else lref, ids)
        same = same_boxes(pair[o], out[f"{o}: v59"][1], ids)
        print(f"self-check {where} rs3 {c}: the class not changed ({o}) equals v59 box for box: {same}", flush=True)
        if not same:
            sys.exit(f"STOP rs3 {c}: the class not changed ({o}) differs from v59")
        out[f"{c}: add rs3"] = (c, pair[c])
        out[f"{o}: other class with rs3 {c}"] = (o, pair[o])
    print("built rs3", flush=True)
    return out


def near_rs3(check):
    ids = S.ids_of("val")
    assert len(ids) == 266
    H = {h: rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in ("before", "inside")}
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    cand = rs3_build(ids, "near")
    v54 = json.loads(Path("logs/neartest_read.json").read_text())["v54"]
    ok = True
    for c in RS3_CLASSES:
        w = v54[c]["inside"]["ap"][0]
        got = ap2(cand[f"{c}: v59"][1], gt, H["inside"], c)[0]
        print(f"self-check inside {c}: v59 base {got:.6f} vs {w:.6f} (logs/neartest_read.json v54): {abs(got - w) < 1e-9}", flush=True)
        ok = ok and abs(got - w) < 1e-9
    if not ok:
        sys.exit("STOP a Truck / Van base does not reproduce v59 on inside")
    if check:
        counts(cand)
        print("check mode: no member AP computed, nothing written", flush=True)
        return
    if any(f"{c}: add rs3" not in cand for c in RS3_CLASSES):
        sys.exit("STOP rs3: no candidates (second-stage files missing): nothing read")
    res = {}
    for name, (c, p) in cand.items():
        res[name] = {"class": c, **{h: ap2(p, gt, fr, c) for h, fr in H.items()}}
    for name, r in res.items():
        b = res[f"{r['class']}: v59"]
        r["d_inside"], r["d_before"] = r["inside"][0] - b["inside"][0], r["before"][0] - b["before"][0]
        print(f"{name:48s} inside {r['inside'][0]:.4f} ({r['d_inside']:+.4f}) | before {r['before'][0]:.4f} ({r['d_before']:+.4f}) | "
              f"AP@0.7 inside {r['inside'][1]:.4f}", flush=True)
    Path("logs/igm_rs3_near.json").write_text(json.dumps(res, indent=1))


def val_rs3(check):
    ids = S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    near_res = None if check else json.loads(Path("logs/igm_rs3_near.json").read_text())
    cand = rs3_build(ids, "val")
    for c, want in RS3_VAL.items():
        got = round(G3.ap(cand[f"{c}: v59"][1], gt, ids, c), 4)
        print(f"self-check val {c}: v59 base {got} vs {want}: {got == want}", flush=True)
        if got != want:
            sys.exit(f"STOP the val {c} base does not reproduce v59")
    if check:
        counts(cand)
        print("check mode: no member AP computed, nothing written", flush=True)
        return
    if any(f"{c}: add rs3" not in cand for c in RS3_CLASSES):
        sys.exit("STOP rs3: no candidates (second-stage files missing): nothing read")
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    nid = lambda h, c: cnt.get(h, {}).get(c, {}).get("distinct_ids", 0)
    res = {}
    print(f"gate table for rs3 (inside +0.01, val not down > 0.01, veto: Truck / Van not down > 0.005 on a segment with >= 5 ids, "
          f"PLUS val a - base >= +{RS3_BAR} unrounded)", flush=True)
    for c in RS3_CLASSES:
        o = "Van" if c == "Truck" else "Truck"
        name = f"{c}: add rs3"
        n, no = near_res[name], near_res[f"{o}: other class with rs3 {c}"]
        base = G3.ap(cand[f"{c}: v59"][1], gt, ids, c)
        a = G3.ap(cand[name][1], gt, ids, c)
        touched = [(c, n), (o, no)]
        veto = [f"{k} {h} {r['d_' + h]:+.4f}" for k, r in touched for h in ("before", "inside") if nid(h, k) >= 5 and r["d_" + h] < -0.005]
        up = a - base >= RS3_BAR
        ok = bool(nid("inside", c) >= 5 and n["d_inside"] >= 0.01 and a - base >= -0.01 and up and not veto)
        res[name] = dict(val=a, val_base=base, d_val=a - base, d_inside=n["d_inside"], d_before=n["d_before"], veto=veto, val_up_bar=bool(up),
                         passed=ok, other_class=o, other_d_inside=no["d_inside"], other_d_before=no["d_before"])
        print(f"{name:32s} inside {n['inside'][0]:.4f} ({n['d_inside']:+.4f}, {nid('inside', c)} ids) | before {n['d_before']:+.4f} | "
              f"val {a:.4f} ({a - base:+.4f}; +{RS3_BAR} bar {'met' if up else 'missed'}) | {o} inside {no['d_inside']:+.4f} before {no['d_before']:+.4f} | "
              f"veto {'; '.join(veto) or 'none'} | {'PASS' if ok else 'fail'}", flush=True)
    Path("logs/igm_rs3_val.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    argv = sys.argv[1:]
    a = [x for x in argv if x != "--check"]
    if len(a) >= 2 and a[1] == "rs3":   # Truck / Van RS second stages: igm_read.py near|val rs3 [--check]
        if len(a) != 2 or a[0] not in ("near", "val"):
            sys.exit("usage: python scripts/igm_read.py near|val rs3 [--check]")
        {"near": near_rs3, "val": val_rs3}[a[0]]("--check" in argv)
        sys.exit(0)
    if argv and argv[0] == "combo":   # final-combination reading; see scripts/igm_combo.py
        import igm_combo
        igm_combo.main(argv[1:])
        sys.exit(0)
    if argv and argv[0] == "joint":   # joint reading (the joint rule): igm_read.py joint near|val <Car|Van|Motorcycle> <m1> <m2> ...
        import igm_combo
        igm_combo.joint(argv[1:])
        sys.exit(0)
    if argv and argv[0] in ("near60", "val60", "lanes60"):   # v60-base mode; see scripts/igm60.py
        import igm60
        igm60.main(argv)
        sys.exit(0)
    vo = [x for x in a[1:] if x in VAN_ONLY or x in PERSON_ONLY]   # Van-only and Person-only members are read on v60 only (no v59 lanes / near / val)
    if vo:
        sys.exit(f"STOP {vo[0]} is read on v60 only: python scripts/igm_read.py near60|val60|lanes60 {vo[0]}")
    cls = None
    if "--classes" in argv:   # class selection only (e.g. Car / Van read before the GPU step; same gate, self-checks and numbers)
        i = argv.index("--classes")
        cls = tuple(argv[i + 1].split(","))
        argv = argv[:i] + argv[i + 2:]
    a = [x for x in argv if x != "--check"]
    chk = "--check" in argv
    cls = cls or classes_of(a[1], chk)
    assert set(cls) <= set(classes_of(a[1], chk)), cls
    if a[0] == "lanes":
        ids = S.ids_of("val")
        lanes(a[1], tag(a[1], chk), ids) if inputs_ok(a[1]) else print(f"no kit predictions for {a[1]}", flush=True)
        print("lanes built", a[1], flush=True)
    else:
        {"near": near, "val": val}[a[0]](a[1], chk, cls)
