"""v60-base readings for LUMPI (fixed before any reading; the moto2 candidate was added afterwards). A mode of
scripts/igm_read.py, separate from its v59 path: the v59 modes, their code and their outputs are untouched. Base = lumpi_v60 class by class, rebuilt
from v60's switch set (logs/build_lumpi_v60.json "switches"; never hardcoded) with igm_combo.classes_built, the code of the final-combination reading.
Three candidate types, each read add-one per class against v60:
  G3  member safdnet_lumpi_rs (tag SR, one-stage): v60's own chain for class C plus the member as one more lane member at weight 1.
        Car / Van / Motorcycle: v60's lane for C (v59 = the v29 lane, Car + vrcnn_lumpi_rs; a member = its IM<k><tag> lane; a joint = its zJ lane)
        + the member; Car: lane_class (rule37); Van / Truck: rule37 + T5 / V5 in one pass with the other class from v60 (igm_read.tv_pass, as rs3).
        Truck: the v29 lane + the member for Truck (lane IMt<tag>), then that pass.  Classes at v59 get exactly igm_read's member form (lanes
        IMc / IMv / IMm<tag>; Person: lumpi_v51's five second stages on the member's Person lane, R0, grow + length rule, P3; Bicycle: L11b's second
        stage on the member's Bicycle lane, R4 >= 0.05, P3; files output/lumpi/<m>/igm/, as igm_read). Person / Bicycle candidates are defined
        only while v60 keeps that class at v59 (else skipped, logged). Lanes: zG3<name>_chain_val.json (G3.run_lane); v60 classes not at v59 use
        "I6<k>_<v60 lane>_<tag>".
  g2  Bicycle refiner set on v60's Bicycle lane (defined only while v60's Bicycle is v59's, i.e. the v29 lane): R4 over four second stages at equal
        weight (g3_search.bicycle's template: E.combine of all refiners for a box >= 0.05, else the v29 box and score), then P3. Files, refiner
        order: output/lumpi/vrcnn_lumpi_b/l11bc/v29_val_ref.json (L11b, v59's Bicycle file), output/lumpi/rs3/vrcnn_lumpi_c_s2_val_ref.json,
        output/lumpi/rs3/pv_lumpi_s2_val_ref.json, output/lumpi/g2/vrcnn_lumpi_b_fix2_rs_val_ref.json (neartest_ws: the near-test files under
        _val_, as scripts/rs3_fetch.sh). Every refiner's Bicycle input boxes must equal L11b's per frame, in order (STOP otherwise).
  moto2  Motorcycle second stage on v60's Motorcycle lane: R4 over four refiners at equal weight (Bicycle's rule: E.combine "R4" for a box >= 0.05,
        else the lane box and score; no P3), files output/lumpi/moto2/<r>_val_ref.json, r = vrcnn_lumpi_b, vrcnn_lumpi_b_fix2_rs, vrcnn_lumpi_c_s2,
        pv_lumpi_s2 (x6_refine --only_class Motorcycle --min_score 0.05 on v60's Motorcycle lane file, printed by the reading); every refiner's
        Motorcycle input boxes must equal v60's Motorcycle per frame, in order (STOP otherwise). Low-evidence, board-informed class choice; one try.
A Bicycle or Motorcycle change recomputes Person: v60's own Person chain + copies (W.copies) from the candidate's Bicycle / Motorcycle and v60's other one.
Missing inputs: that candidate is skipped and logged (rs3_build's rule); a real reading with no candidate STOPs.
Gate per class against v60 (fixed; nothing loosened):
  G3, g2: inside AP@0.5 >= +0.01 on a class with >= 5 inside ids (logs/neartest_id_counts.json) AND val (320, pooled) a - base >= +0.005 unrounded
        (Motorcycle: also the Motorcycle val rule, logs/igm_moto_rule.json); veto -0.005 on every touched class on a segment (before / inside) with >= 5 ids:
        Van and Truck both for a Van or Truck change (one T5 / V5 pass), Person (recomputed copies) for a Bicycle / Motorcycle change; Person val
        with the copies not down > 0.01 vs v60. Before, AP@0.7: information.
  moto2: inside >= +0.01 (5 ids); val not down (a >= base, unrounded), NO +0.005 bar; Motorcycle before not down > 0.005 although it has 3 ids;
        the Person copy veto (-0.005, inside / before, >= 5 ids) and Person val with the copies not down > 0.01 vs v60.
Self-checks (STOP on a miss, before any candidate is read): v60's Truck / Van from the candidates' T5 / V5 sources equal v60's chains box for box;
v60's Person = its own chain + the copies; classes v60 keeps at v59 equal v59's box for box (Van, Truck, Bicycle, Motorcycle: size_check_lumpi.build;
Person's own chain: SZ.build's; Bicycle: also the L11b lane rebuild; Motorcycle: also G3's v29 lane zG300) and the v59 constants (inside: Car 0.847262
logs/g3_near.json, Person 0.317989 logs/m3_neartest_copies.json, Van = v54's, Truck = logs/neartest_read.json v54, Bicycle 0.2105; val: Car 0.896,
Person 0.48, Van 0.1708, Bicycle 0.5662, Truck 0.4632); every class v60 changed (Person when its copies changed) equals the combo reading of v60's
switches on inside / before (logs/igm_combo_<key>_near.json, 1e-9), and on val all six classes equal logs/build_lumpi_v60.json "val" "new" (1e-9)
and the touched ones logs/igm_combo_<key>_val.json (1e-9). g2: G2's fusion on L11b's file alone equals v59's Bicycle box for box.
--assume-v59 (with --check only): every switch at v59, no v60 files read; the v60 machinery must then give the v59 constants exactly.
    python scripts/igm_read.py lanes60 safdnet_lumpi_rs [--assume-v59]    (builds G3's Car / Van / Truck / Motorcycle lanes on v60, cached)
    python scripts/igm_read.py near60 safdnet_lumpi_rs|g2|moto2 [--check [--assume-v59]]   (neartest_ws; logs/build_lumpi_v60.json copied
        from the main workspace) -> logs/igm60_<m>_near.json
    python scripts/igm_read.py val60 safdnet_lumpi_rs|g2|moto2 [--check [--assume-v59]]    (main; needs logs/igm60_<m>_near.json from
        neartest_ws) -> logs/igm60_<m>_val.json + the gate table
    Real readings also write each Motorcycle candidate's boxes (G3, moto2) to output/lumpi/igm60/<m>/motorcycle_val.json (neartest_ws: near-test
    frames; main: val), path and md5 in the json, for a later stacked reading.
    --check: dry run (bases, self-checks, candidates built where inputs exist; box counts only, no candidate AP, nothing written). Under --check
    any igm_read member (TAGS / DRY) may stand in for safdnet_lumpi_rs (code tests); real readings take safdnet_lumpi_rs, g2, moto2 only.
LUMPI #4: vrcnn_lumpi_b_fix2_rs (tag B2) is read exactly as the G3 member above (same
lanes, chains, self-checks and gate; Person in pv_lumpi_s2's form, igm_read.REFINER): near60 / val60 / lanes60 vrcnn_lumpi_b_fix2_rs.
spec_lumpi_van_rs (tag VR; igm_read.VAN_ONLY): a Van-only specialist read as the G3 member above for Van ONLY: v60's Van lane + the member
at weight 1 for Van (IMvVR while v60's Van is v59), rule37 + T5 / V5 in one pass with v60's Truck (the Truck of that pass: self-check box for box and
the Truck veto); the same v60 bases, self-checks and gate (inside +0.01 with >= 5 ids, val a - base >= +0.005 unrounded, veto -0.005 on Van and
Truck on a segment with >= 5 ids). No Car / Truck / Person / Bicycle / Motorcycle candidate or lane is built for it; lanes60 builds its Van lane
only. Inputs: output/lumpi/g3in/spec_lumpi_van_rs_val.json (neartest_ws: the near-test predictions under _val_; scripts/van_rs_chain.sh).
spec_lumpi_person_rs (tag PS; igm_read.PERSON_ONLY and REFINER): a Person-only specialist read as
the G3 member above for Person ONLY, in REFINER form (as pv_lumpi_s2 / vrcnn_lumpi_b_fix2_rs): v60's own Person chain with this model's Person second
stage as a sixth member (G3.P_REF + output/lumpi/<m>/igm/v29_val_person_min01.json, G3.person), plus v60's unchanged two-wheeler copies (cp60); defined
only while v60's Person is v59's chain. The same v60 bases, self-checks and gate (inside +0.01 with >= 5 ids, val a - base >= +0.005 unrounded, veto
-0.005 on Person on a segment with >= 5 ids). No Car / Van / Truck / Bicycle / Motorcycle candidate or lane is built for it; lanes60 builds nothing.
Inputs: output/lumpi/spec_lumpi_person_rs/igm/v29_val_person_min01.json (neartest_ws: the near-test second stage under _val_; scripts/person_rs_chain.sh).
--base-v59 (spec_lumpi_person_rs near60 / val60 only): the reading on lumpi_v59 instead of v60. The switch
set is v59's for all six classes (the --assume-v59 set, here for a real reading): v59's Person chain and v59's two-wheeler copies from v59's Bicycle +
Motorcycle (682 on test) for base and candidate. Self-checks as --assume-v59: bases60's v59 box-for-box checks (Person with copies = v59's), check_near's
v59 constants, check_val's VAL_V59 (Person 0.48). The gate is the same; the json's "_v60" block says base_v59; val60 refuses a near json of the other base."""
import hashlib
import json
import sys
from pathlib import Path

import igm_combo as IC
import igm_read as M

G3M = "safdnet_lumpi_rs"
B2M = "vrcnn_lumpi_b_fix2_rs"   # LUMPI #4: read in G3's form
VRM = "spec_lumpi_van_rs"   # Van-only specialist, read in G3's form for Van only (igm_read.VAN_ONLY)
PSM = "spec_lumpi_person_rs"   # Person-only specialist, read in G3's form for Person only, REFINER form (igm_read.PERSON_ONLY)
G3_FORM = (G3M, B2M, VRM, PSM)
G2 = "g2"
MOTO2 = "moto2"
SIX = ("Person", "Car", "Van", "Truck", "Bicycle", "Motorcycle")
H = ("before", "inside")
BAR = 0.005   # G3 / g2: val a - base >= +0.005 unrounded vs v60 (not for moto2)
V60 = Path("logs/build_lumpi_v60.json")   # build_lumpi_v6x.py --name lumpi_v60 writes logs/build_<name>.json
N_NEAR = 266
BASE59 = False   # --base-v59 (main): the Person-only reading on lumpi_v59's switch set
NEAR_BICYCLE = 0.2105   # igm_read.near's Bicycle constant (4 decimals)
VAL_V59 = {"Car": 0.896, "Person": 0.48, "Van": 0.1708, "Bicycle": 0.5662, "Truck": 0.4632}   # igm_read.val / rs3 constants (4 decimals)
LANE_KEY = {"Car": "c", "Van": "v", "Truck": "t", "Motorcycle": "m"}
G2_FILES = lambda s="val": [M.G3.RL.B / f"v29_{s}_ref.json", M.RS3_DIR / f"vrcnn_lumpi_c_s2_{s}_ref.json", M.RS3_DIR / f"pv_lumpi_s2_{s}_ref.json",
                            Path(f"output/lumpi/g2/vrcnn_lumpi_b_fix2_rs_{s}_ref.json")]
MOTO2_REF = ("vrcnn_lumpi_b", "vrcnn_lumpi_b_fix2_rs", "vrcnn_lumpi_c_s2", "pv_lumpi_s2")
MOTO2_FILES = lambda s="val": [Path(f"output/lumpi/moto2/{r}_{s}_ref.json") for r in MOTO2_REF]
MOTO2_NOTE = ("low-evidence, board-informed class choice (Motorcycle: 5 inside / 3 before ids; second stage on v60's Motorcycle lane, "
              "an added candidate; one try)")
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ v60's switch set and readings
def ic_sw(sw):
    """igm_combo's form of a switch set: the five SW classes, Truck only when not v59 (key_of and the stored switches)."""
    return dict({c: sw[c] for c in IC.SW}, **({"Truck": sw["Truck"]} if sw["Truck"] != "v59" else {}))


def person_same(sw):
    return all(sw[c] == "v59" for c in ("Person", "Bicycle", "Motorcycle"))


def v60_set(assume, where):
    """-> (sw: {class: switch} over SIX, info). assume: every switch v59, no file read. Else logs/build_lumpi_v60.json's switches, validated by
    igm_combo.parse; with where = near / val also the PASS combo reading of exactly those switches (logs/igm_combo_<key>_<where>.json)."""
    if assume:
        if BASE59:
            print("switch set: --base-v59 (the Person reading on lumpi_v59: every class at v59, no v60 file read)", flush=True)
            return {c: "v59" for c in SIX}, {"assume_v59": True, "base_v59": True, "switches": {c: "v59" for c in SIX}}
        print("v60 switch set: --assume-v59 (dry run: every class at v59, no v60 file read)", flush=True)
        return {c: "v59" for c in SIX}, {"assume_v59": True, "switches": {c: "v59" for c in SIX}}
    if not V60.exists():
        sys.exit(f"STOP {V60} missing: build lumpi_v60 first (main); for near60 copy it into neartest_ws/logs")
    b = json.loads(V60.read_text())
    sw = b.get("switches", {})
    if set(sw) != set(SIX) or any(not isinstance(v, str) for v in sw.values()):
        sys.exit(f"STOP {V60}: switches {sw} are not one value per class {SIX}")
    if sw["Car"] == "v54":
        sys.exit(f"STOP {V60}: Car v54 is not a v60-base switch (no combo reading reads it)")
    IC.parse([x for c in IC.SW6 for x in (f"--{c.lower()}", sw[c])])   # igm_combo's validation (STOPs on an unknown or unread switch)
    key = b.get("final_combination")
    info = {"assume_v59": False, "switches": sw, "build_json": str(V60), "build_md5": md5(V60), "key": key,
            "fallback": b.get("fallback"), "only": b.get("only")}
    changed = [c for c in SIX if sw[c] != "v59"]
    if changed and not key:
        sys.exit(f"STOP {V60}: v60 changes {changed} but names no final-combination reading of exactly its switches (--only build?): needs a manual decision")
    if key and key != IC.key_of(ic_sw(sw)):
        sys.exit(f"STOP {V60}: final_combination {key} is not the key of its switches {IC.key_of(ic_sw(sw))}")
    if where and key:
        p = Path(f"logs/igm_combo_{key}_{where}.json")
        if not p.exists():
            sys.exit(f"STOP {p} missing (the combo reading of v60's switches; copy it into this workspace)")
        r = json.loads(p.read_text())
        if r.get("switches") != ic_sw(sw):
            sys.exit(f"STOP {p} is for switches {r.get('switches')}, not v60's {ic_sw(sw)}")
        if where == "val" and not r.get("passed"):
            sys.exit(f"STOP {p} did not pass")
        info["combo"], info["combo_classes"] = str(p), r["classes"]
    print(f"v60 switch set ({V60}, md5 {info['build_md5']}): {sw}" + (f" | combo reading {key}" if key else ""), flush=True)
    return sw, info


def kind_of(m, check):
    if m in (G2, MOTO2):
        return m
    if m in G3_FORM or (check and (m in M.TAGS or m in M.DRY)):   # --check only: another igm_read member may stand in for G3 (code tests)
        return "g3"
    sys.exit(f"STOP {m}: the v60-base mode reads {G3M}, {B2M}, {VRM}, {PSM}, {G2} or {MOTO2}" + ("" if check else " (stand-in members only with --check)"))


# ------------------------------------------------------------------ lanes
def lane_of(c, v):
    """v60's lane for class c in Car / Van / Truck / Motorcycle under switch v, as igm_combo / build_lumpi_v6x build it: (lane name, member spec);
    None for rs3 (R6 second stages on the v29 boxes: no lane to add a member to). v59: Car IGc00 (the v29 lane + vrcnn_lumpi_rs), others the v29
    lane (G3.B_SPEC, lane "00")."""
    if v == IC.RS3:
        return None
    if v.startswith("joint:"):
        ms = IC.joint_members(c, v)
        return IC.joint_tag(c, ms), IC.joint_spec(c, ms)
    if v == "v59":
        return ("IGc00", M.G3.B_SPEC + [M.IG.RS]) if c == "Car" else ("00", M.G3.B_SPEC)
    return f"IM{LANE_KEY[c]}{M.TAGS[v]}", M.G3.B_SPEC + ([M.IG.RS] if c == "Car" else []) + M.member(v, c)


def g3_lane(c, v, m, t):
    """v60's lane for class c plus member m at weight 1 for c -> (lane name, spec) or None. A class at v59 gets igm_read's member-form lane
    (IMc / IMv / IMm<t>; Truck IMt<t>), any other "I6<k>_<v60 lane>_<t>"."""
    b = lane_of(c, v)
    if b is None:
        return None
    return (f"IM{LANE_KEY[c]}{t}" if v == "v59" else f"I6{LANE_KEY[c]}_{b[0]}_{t}"), b[1] + M.member(m, c)


def tv_src(sw, lref, ids):
    """v60's Truck and Van inputs to the rule37 + T5 / V5 pass: v59 -> the L11-refined v29 file (lref); a member / joint -> its lane; rs3 -> the R6
    fusion of the 3 RS refiners (igm_read.rs3_fuse). Kept to the class's own boxes (tv_pass reads only those)."""
    src, fu = {}, None
    for c in ("Truck", "Van"):
        v = sw[c]
        if v == "v59":
            x = lref
        elif v == IC.RS3:
            fu = fu or M.rs3_fuse(lref, ids)
            x = fu[c]
        else:
            nm, spec = lane_of(c, v)
            x = M.G3.run_lane(nm, spec, {}, "val")
        src[c] = {f: [d for d in x.get(f, []) if d["class"] == c] for f in ids}
        del x
    return src


def load_cls(p, c, ids):
    x = M.W.only(p, c)
    return {f: x.get(f, []) for f in ids}


def refine_fuse(srcs, c, ref, ids, what):
    """R4 over second stages at equal weight (g3_search.bicycle's template). srcs: [(file, {frame: that file's class-c boxes})]; every file's
    input boxes (box and score) must equal ref's per frame, in order (STOP otherwise, as igm_read.rs3_fuse); a box >= 0.05 ->
    ensemble_rules.combine(all refiners, "R4"), else its input box and score. -> {frame: boxes}"""
    for p, x in srcs:
        for f in ids:
            r, b = x[f], ref[f]
            if len(r) != len(b) or any(a["box"] != q["box"] or a["score"] != q["score"] for a, q in zip(r, b)):
                sys.exit(f"STOP {what} {c} frame {f}: the input boxes of {p} differ from the reference ({len(r)} vs {len(b)} boxes)")
    return {f: [M.E.combine(list(t), "R4") if t[0]["score"] >= 0.05 else M.SZ.keep(t[0]) for t in zip(*[x[f] for _, x in srcs])] for f in ids}


def p3_bike(b, ids, keys):
    """P3 on a fused Bicycle (ig_bike_read.bike's last step)."""
    return {f: [M.SZ.keep(d) for d in v] for f, v in M.SZ.PP.filt({f: b[f] for f in ids}, "Bicycle", M.LOAD, keys).items()}


def close(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9)


# ------------------------------------------------------------------ v60 bases and their box-for-box self-checks
def bases60(sw, ids, where):
    """v60's six classes with igm_combo.classes_built on v60's switches ("<C>: v60"), and the context the candidates build on. Box-for-box
    self-checks (STOP on a miss): v60's Truck / Van from the candidates' T5 / V5 sources; v60's Person = own chain + copies; classes v60 keeps at
    v59 equal v59's (SZ.build; Bicycle also the L11b lane rebuild, Motorcycle also G3's v29 lane zG300)."""
    keys = M.SZ.PP.static_keys("lumpi")
    base59, new = IC.classes_built(ic_sw(sw), ids)
    del base59
    lref = json.loads((M.G3.RL.L / "v29_val_ref.json").read_text())
    src = tv_src(sw, lref, ids)
    del lref
    tv60 = M.tv_pass(src["Truck"], src["Van"], ids)
    new.setdefault("Truck", tv60["Truck"])   # Truck at v59: igm_read.truck(lref, lref) = this pass's Truck
    ok = True
    for c in ("Truck", "Van"):
        same = M.same_boxes(tv60[c], new[c], ids)
        print(f"self-check {where} {c}: v60's {c} from the candidates' T5 / V5 sources equals v60's chain box for box: {same}", flush=True)
        ok = ok and same
    del tv60
    cp60 = M.W.copies({f: new["Bicycle"][f] + new["Motorcycle"][f] for f in ids})
    own = {f: new["Person"][f][:len(new["Person"][f]) - len(cp60[f])] for f in ids}   # v60's own Person chain (classes_built: own + copies)
    same = all(new["Person"][f][len(own[f]):] == cp60[f] for f in ids)
    print(f"self-check {where} Person: v60's Person = its own chain + the copies from v60's Bicycle and Motorcycle: {same}", flush=True)
    ok = ok and same
    P = M.SZ.build("val", ids)   # v59 by size_check_lumpi's chains (route_read_lumpi.Chain / W.only / M1.person_chain)
    for c in ("Van", "Truck", "Bicycle", "Motorcycle"):
        if sw[c] == "v59":
            same = M.same_boxes(new[c], P[c], ids)
            print(f"self-check {where} {c}: v60 keeps v59; equals v59's (SZ.build) box for box: {same}", flush=True)
            ok = ok and same
    if sw["Person"] == "v59":
        same = M.same_boxes(own, P["Person"], ids)
        print(f"self-check {where} Person: v60 keeps v59's own chain; equals SZ.build's Person box for box: {same}", flush=True)
        ok = ok and same
    if person_same(sw):
        v59p = {f: P["Person"][f] + M.W.copies({f: P["Bicycle"][f] + P["Motorcycle"][f]})[f] for f in ids}
        same = M.same_boxes(new["Person"], v59p, ids)
        print(f"self-check {where} Person: v60's Person with copies equals v59's box for box: {same}", flush=True)
        ok = ok and same
    if sw["Bicycle"] == "v59":
        same = M.same_boxes(M.IB.bike(M.G3.RL.B / "v29_val_ref.json", ids, keys), new["Bicycle"], ids)
        print(f"self-check {where} Bicycle: L11b's lane rebuild (ig_bike_read.bike) equals v60's Bicycle box for box: {same}", flush=True)
        ok = ok and same
    if sw["Motorcycle"] == "v59":
        same = M.same_boxes(M.G3.lane_class(M.G3.run_lane("00", M.G3.B_SPEC, {}, "val"), "Motorcycle", ids), new["Motorcycle"], ids)
        print(f"self-check {where} Motorcycle: G3's rebuilt v29 lane equals v60's (= v59's) Motorcycle box for box: {same}", flush=True)
        ok = ok and same
    ctx = dict(keys=keys, src=src, new=new, cp60=cp60, v54_van=P["Van"],
               person_with=lambda b, mo: {f: own[f] + M.W.copies({f: b[f] + mo[f]})[f] for f in ids})
    del P
    if not ok:
        sys.exit(f"STOP a v60 base does not rebuild ({where})")
    return {f"{c}: v60": (c, new[c]) for c in SIX}, ctx


# ------------------------------------------------------------------ candidates
def cands60(out, sw, ids, m, kind, ctx, where):
    {"g3": g3_cands, G2: g2_cands, MOTO2: moto2_cands}[kind](out, sw, ids, m, ctx, where)


def g3_cands(out, sw, ids, m, ctx, where):
    t = M.TAGS.get(m) or M.DRY[m]
    new, src, keys, person_with = ctx["new"], ctx["src"], ctx["keys"], ctx["person_with"]
    person_only = m in M.PERSON_ONLY   # the Person candidate alone, REFINER form (v60's own chain + this sixth second stage + v60's copies)
    if not M.inputs_ok(m):
        print(f"no Person second-stage file output/lumpi/{m}/igm/v29_val_person_min01.json: G3 candidates skipped" if person_only else
              f"no kit predictions for {m} in output/lumpi/g3in: G3 candidates skipped", flush=True)
        return
    van_only = m in M.VAN_ONLY   # the Van candidate alone (+ the Truck of its T5 / V5 pass for the veto); nothing else built
    if not van_only and not person_only:
        x = g3_lane("Car", sw["Car"], m, t)
        out[f"Car: add {m}"] = ("Car", M.G3.lane_class(M.G3.run_lane(x[0], x[1], {}, "val"), "Car", ids))
    for c in () if person_only else ("Van",) if van_only else ("Van", "Truck"):
        o = "Truck" if c == "Van" else "Van"
        x = g3_lane(c, sw[c], m, t)
        if x is None:
            print(f"{c}: v60's {c} is rs3 (second stages on the v29 boxes): no lane to add {m} to; skipped", flush=True)
            continue
        ch = M.G3.run_lane(x[0], x[1], {}, "val")
        pair = M.tv_pass(ch if c == "Truck" else src["Truck"], ch if c == "Van" else src["Van"], ids)
        del ch
        same = M.same_boxes(pair[o], new[o], ids)
        print(f"self-check {where} {m} {c}: the class not changed ({o}) equals v60's box for box: {same}", flush=True)
        if not same:
            sys.exit(f"STOP {m} {c}: the class not changed ({o}) differs from v60")
        out[f"{c}: add {m}"] = (c, pair[c])
        out[f"{o}: other class with {m} {c}"] = (o, pair[o])
    if van_only:
        print(f"built G3 {m} (Van only: no Car / Truck / Person / Bicycle / Motorcycle candidate)", flush=True)
        return
    d = Path(f"output/lumpi/{m}/igm")
    if sw["Person"] != "v59":
        print(f"Person: v60's Person is {sw['Person']}, not v59's chain: G3 Person not defined; skipped", flush=True)
    else:
        pf = M.G3.P_REF("val") + [d / "v29_val_person_min01.json"] if m in M.REFINER else [d / f"person_{r}_val_ref.json" for r in M.REF_ORDER]
        if all(p.exists() for p in pf):
            p = M.G3.person(pf, ids, M.LOAD, keys)
            out[f"Person: add {m}"] = ("Person", {f: p[f] + ctx["cp60"][f] for f in ids})
        else:
            print(f"{m}: no Person second-stage files in {d}: Person skipped", flush=True)
    if person_only:
        print(f"built G3 {m} (Person only, REFINER form: no Car / Van / Truck / Bicycle / Motorcycle candidate)", flush=True)
        return
    bf = d / "bicycle_val_ref.json"
    if sw["Bicycle"] != "v59":
        print(f"Bicycle: v60's Bicycle is {sw['Bicycle']}, not v59's chain: G3 Bicycle not defined; skipped", flush=True)
    elif bf.exists():
        b = M.IB.bike(bf, ids, keys)
        out[f"Bicycle: add {m}"] = ("Bicycle", b)
        out[f"Person: copies from Bicycle: add {m}"] = ("Person", person_with(b, new["Motorcycle"]))
    else:
        print(f"{m}: no Bicycle second-stage file {bf}: Bicycle skipped", flush=True)
    x = g3_lane("Motorcycle", sw["Motorcycle"], m, t)
    mo = M.G3.lane_class(M.G3.run_lane(x[0], x[1], {}, "val"), "Motorcycle", ids)
    out[f"Motorcycle: add {m}"] = ("Motorcycle", mo)
    out[f"Person: copies from Motorcycle: add {m}"] = ("Person", person_with(new["Bicycle"], mo))
    print("built G3", m, flush=True)


def g2_cands(out, sw, ids, m, ctx, where):
    files = G2_FILES()
    new, keys = ctx["new"], ctx["keys"]
    if sw["Bicycle"] != "v59":
        print(f"g2: v60's Bicycle is {sw['Bicycle']}, not the v29 lane: g2 not defined; skipped", flush=True)
        return
    srcs = [(files[0], load_cls(files[0], "Bicycle", ids))]
    one = p3_bike(refine_fuse(srcs, "Bicycle", srcs[0][1], ids, "g2 template"), ids, keys)
    same = M.same_boxes(one, new["Bicycle"], ids)
    print(f"self-check {where} g2: the R4 fusion on L11b's file alone equals v60's (= v59's) Bicycle box for box: {same}", flush=True)
    if not same:
        sys.exit("STOP g2: the R4 template does not rebuild v59's Bicycle")
    miss = [str(p) for p in files if not p.exists()]
    if miss:
        print("g2: second-stage files missing: " + ", ".join(miss) + ": candidate skipped", flush=True)
        return
    srcs += [(p, load_cls(p, "Bicycle", ids)) for p in files[1:]]
    b = p3_bike(refine_fuse(srcs, "Bicycle", srcs[0][1], ids, "g2"), ids, keys)
    out["Bicycle: add g2"] = ("Bicycle", b)
    out["Person: copies from Bicycle: add g2"] = ("Person", ctx["person_with"](b, new["Motorcycle"]))
    print("built g2 from " + ", ".join(str(p) for p in files), flush=True)


def moto2_lane_file(sw, split="val"):
    """The file x6_refine reads (--pred) for moto2: v60's Motorcycle lane chain (G3.run_lane's cache); v59: the v29 file."""
    if sw["Motorcycle"] == "v59":
        return M.G3.O / f"lumpi_v29_{split}.json"
    return M.G3.O / f"zG3{lane_of('Motorcycle', sw['Motorcycle'])[0]}_chain_{split}.json"


def moto2_cands(out, sw, ids, m, ctx, where):
    files = MOTO2_FILES()
    new = ctx["new"]
    print(f"moto2: refiner input (--pred) = v60's Motorcycle lane {moto2_lane_file(sw)} (x6_refine --only_class Motorcycle --min_score 0.05)",
          flush=True)
    miss = [str(p) for p in files if not p.exists()]
    if miss:
        print("moto2: second-stage files missing: " + ", ".join(miss) + ": candidate skipped", flush=True)
        return
    srcs = [(p, load_cls(p, "Motorcycle", ids)) for p in files]
    mo = refine_fuse(srcs, "Motorcycle", new["Motorcycle"], ids, "moto2")
    out["Motorcycle: add moto2"] = ("Motorcycle", mo)
    out["Person: copies from Motorcycle: add moto2"] = ("Person", ctx["person_with"](new["Bicycle"], mo))
    print("built moto2", flush=True)


# ------------------------------------------------------------------ readings
def counts(cand):
    for name, (c, p) in cand.items():
        if not name.endswith(": v60"):
            print(f"check: built {name:52s} {len(p)} frames, {sum(len(v) for v in p.values())} boxes", flush=True)


def note_of(name, m):
    """Motorcycle readings carry their note (igm_read's MOTO_NOTE; moto2: low-evidence, board-informed)."""
    if "Motorcycle" not in name or name.endswith(": v60"):
        return None
    return MOTO2_NOTE if m == MOTO2 else M.MOTO_NOTE


def save_moto(cand, m, res):
    """Real readings: each Motorcycle candidate's boxes per split for a later stacked reading: output/lumpi/igm60/<m>/
    motorcycle_val.json (in neartest_ws "val" = the near-test frames, as the second-stage files); path and md5 go into the reading's json."""
    name = f"Motorcycle: add {m}"
    if name not in cand:
        return
    p = Path(f"output/lumpi/igm60/{m}/motorcycle_val.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cand[name][1]))
    res["_v60"]["motorcycle_boxes"] = {"path": str(p), "md5": md5(p)}
    print("wrote", p, flush=True)


def moved(sw, c):
    """v60 differs from v59 for class c (Person also when its copies changed)."""
    return sw[c] != "v59" or (c == "Person" and not person_same(sw))


def check_near(sw, info, cand, ctx, gt, HH):
    """AP self-checks on near-test, before any candidate: classes at v59 give igm_read's v59 constants (1e-9; Bicycle 4 decimals), classes v60
    changed equal the combo reading of v60's switches (AP@0.5 and AP@0.7 on inside and before, 1e-9)."""
    ok = True
    want = {}
    if sw["Car"] == "v59":
        want["Car"] = json.loads(Path("logs/g3_near.json").read_text())["v58_car"]["inside"]["v58"][0]
    if person_same(sw):
        want["Person"] = json.loads(Path("logs/m3_neartest_copies.json").read_text())["copies_on"]["inside"][0]
    if sw["Van"] == "v59":
        want["Van"] = M.ap2(ctx["v54_van"], gt, HH["inside"], "Van")[0]
    if sw["Truck"] == "v59":
        want["Truck"] = json.loads(Path("logs/neartest_read.json").read_text())["v54"]["Truck"]["inside"]["ap"][0]
    for c, w in want.items():
        got = M.ap2(cand[f"{c}: v60"][1], gt, HH["inside"], c)[0]
        print(f"self-check inside {c}: v60 (= v59) base {got:.6f} vs {w:.6f}: {close(got, w)}", flush=True)
        ok = ok and close(got, w)
    if sw["Bicycle"] == "v59":
        bb = M.ap2(cand["Bicycle: v60"][1], gt, HH["inside"], "Bicycle")[0]
        print(f"self-check inside Bicycle: v60 (= v59) base {bb:.6f} ({NEAR_BICYCLE}: {round(bb, 4) == NEAR_BICYCLE})", flush=True)
        ok = ok and round(bb, 4) == NEAR_BICYCLE
    cc = info.get("combo_classes", {})
    for c in SIX:
        if moved(sw, c) and c not in cc:
            print(f"self-check near {c}: v60 changed it but the combo reading has no {c}: False", flush=True)
            ok = False
        if c in cc:
            r = {h: M.ap2(cand[f"{c}: v60"][1], gt, HH[h], c) for h in H}
            same = all(close(r[h][t], cc[c][h]["new"][t]) for h in H for t in (0, 1))
            print(f"self-check near {c}: v60 inside {r['inside'][0]:.6f} / before {r['before'][0]:.6f} equal the combo reading {info['key']}: {same}",
                  flush=True)
            ok = ok and same
    if not ok:
        sys.exit("STOP a v60 base does not reproduce on near-test")


def check_val(sw, info, cand, gt, ids, assume):
    """AP self-checks on val, before any candidate: classes at v59 give igm_read's v59 constants (4 decimals); every class equals
    logs/build_lumpi_v60.json "val" "new" and the touched ones the combo val reading (1e-9)."""
    ok = True
    for c, want in VAL_V59.items():
        if not moved(sw, c):
            got = round(M.G3.ap(cand[f"{c}: v60"][1], gt, ids, c), 4)
            print(f"self-check val {c}: v60 (= v59) base {got} vs {want}: {got == want}", flush=True)
            ok = ok and got == want
    if not assume:
        bv = json.loads(V60.read_text())["val"]
        cc = info.get("combo_classes", {})
        for c in SIX:
            a = M.G3.ap(cand[f"{c}: v60"][1], gt, ids, c)
            same = close(a, bv[c]["new"]) and (c not in cc or close(a, cc[c]["val"]))
            print(f"self-check val {c}: v60 {a:.6f} equals {V60} {bv[c]['new']:.6f}" + (f" and the combo reading {cc[c]['val']:.6f}" if c in cc else "")
                  + f": {same}", flush=True)
            ok = ok and same
            if moved(sw, c) and c not in cc:
                print(f"self-check val {c}: v60 changed it but the combo reading has no {c}: False", flush=True)
                ok = False
    if not ok:
        sys.exit("STOP a v60 base does not reproduce on val")


def near60(m, check, assume):
    kind = kind_of(m, check)
    sw, info = v60_set(assume, "near")
    ids = M.S.ids_of("val")
    assert len(ids) == N_NEAR
    HH = {h: M.rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in H}
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in M.S.labels(ids).items()}
    cand, ctx = bases60(sw, ids, "near")
    check_near(sw, info, cand, ctx, gt, HH)
    cands60(cand, sw, ids, m, kind, ctx, "near")
    del ctx
    if check:
        counts(cand)
        print("check mode: no candidate AP computed, nothing written", flush=True)
        return
    if all(n.endswith(": v60") for n in cand):
        sys.exit(f"STOP {m}: no candidates (inputs missing): nothing read")
    res = {"_v60": {k: v for k, v in info.items() if k != "combo_classes"}}
    for name, (c, p) in cand.items():
        res[name] = {"class": c, **{h: M.ap2(p, gt, fr, c) for h, fr in HH.items()}}
    for name in cand:
        r = res[name]
        b = res[f"{r['class']}: v60"]   # like for like: "Person: copies from ..." -> v60's Person; the other T5 / V5 class -> its v60 chain
        r["d_inside"], r["d_before"] = r["inside"][0] - b["inside"][0], r["before"][0] - b["before"][0]
        if note_of(name, m):
            r["note"] = note_of(name, m)
        print(f"{name:52s} inside {r['inside'][0]:.4f} ({r['d_inside']:+.4f}) | before {r['before'][0]:.4f} ({r['d_before']:+.4f}) | "
              f"AP@0.7 inside {r['inside'][1]:.4f}{' | ' + r['note'] if 'note' in r else ''}", flush=True)
    save_moto(cand, m, res)
    Path(f"logs/igm60_{m}_near.json").write_text(json.dumps(res, indent=1))
    print("wrote", f"logs/igm60_{m}_near.json", flush=True)


def val60(m, check, assume):
    kind = kind_of(m, check)
    sw, info = v60_set(assume, "val")
    out_p = Path(f"logs/igm60_{m}_val.json")
    if m == MOTO2 and not check and out_p.exists():
        sys.exit(f"STOP moto2 is one try: {out_p} exists")
    near_res = None
    if not check:
        p = Path(f"logs/igm60_{m}_near.json")
        if not p.exists():
            sys.exit(f"STOP {p} missing (near60 in neartest_ws, then copy it here)")
        near_res = json.loads(p.read_text())
        nv = near_res.get("_v60", {})
        if (nv.get("switches") != sw or nv.get("key") != info.get("key") or bool(nv.get("assume_v59")) != BASE59
                or bool(nv.get("base_v59")) != BASE59):
            sys.exit(f"STOP {p} was read on another v60 switch set ({nv.get('switches')}, key {nv.get('key')})")
    ids = M.S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in M.S.labels(ids).items()}
    cand, ctx = bases60(sw, ids, "val")
    check_val(sw, info, cand, gt, ids, assume)
    cands60(cand, sw, ids, m, kind, ctx, "val")
    del ctx
    if check:
        counts(cand)
        print("check mode: no candidate AP computed, nothing written", flush=True)
        return
    mains = [n for n in cand if n.endswith(f": add {m}") and not n.startswith("Person: copies")]
    if not mains:
        sys.exit(f"STOP {m}: no candidates (inputs missing): nothing read")
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    nid = lambda h, c: cnt.get(h, {}).get(c, {}).get("distinct_ids", 0)
    res = {"_v60": {k: v for k, v in info.items() if k != "combo_classes"}}
    rule = ("inside +0.01 (>= 5 ids), val not down (a >= base), NO +0.005 bar; veto: Motorcycle before -0.005 (3 ids, fixed), Person copies "
            "-0.005 (>= 5 ids), Person val (copies) not down > 0.01") if m == MOTO2 else \
           ("inside +0.01 (>= 5 ids), val a - base >= +0.005 unrounded (Motorcycle: also val not down); veto: touched classes -0.005 on a segment with "
            ">= 5 ids (Van and Truck both; Person copies), Person val (copies) not down > 0.01")
    print(f"gate table for {m} on v60 ({rule})", flush=True)
    for name in mains:
        c = cand[name][0]
        if name not in near_res:
            sys.exit(f"STOP {name}: built here but not in the near reading (inputs differ between workspaces)")
        n = near_res[name]
        base = M.G3.ap(cand[f"{c}: v60"][1], gt, ids, c)
        a = M.G3.ap(cand[name][1], gt, ids, c)
        o = {"Van": "Truck", "Truck": "Van"}.get(c)
        other = near_res[f"{o}: other class with {m} {c}"] if o else None
        pb = near_res[f"Person: copies from {c}: add {m}"] if c in ("Bicycle", "Motorcycle") else None
        touched = [(c, n)] + ([(o, other)] if o else []) + ([("Person", pb)] if pb else [])
        seg = (lambda h, k: nid(h, k) >= 5 or (k == "Motorcycle" and h == "before")) if m == MOTO2 else (lambda h, k: nid(h, k) >= 5)
        veto = [f"{k} {h} {r['d_' + h]:+.4f}" for k, r in touched for h in H if seg(h, k) and r["d_" + h] < -0.005]
        pv = None
        if c in ("Bicycle", "Motorcycle"):   # Person val with the recomputed copies not down more than 0.01 vs v60
            pv = (M.G3.ap(cand[f"Person: copies from {c}: add {m}"][1], gt, ids, "Person"), M.G3.ap(cand["Person: v60"][1], gt, ids, "Person"))
            if pv[0] - pv[1] < -0.01:
                veto.append(f"Person val (copies) {pv[0] - pv[1]:+.4f}")
        up = a - base >= BAR
        val_ok = (M.moto_val_ok(a, base) if c == "Motorcycle" else True) and (m == MOTO2 or up)
        ok = bool(nid("inside", c) >= 5 and n["d_inside"] >= 0.01 and val_ok and not veto)
        r = dict(val=a, val_base=base, d_val=a - base, d_inside=n["d_inside"], d_before=n["d_before"], veto=veto, passed=ok)
        if m != MOTO2:
            r["val_up_bar"] = bool(up)
        if o:
            r.update(other_class=o, other_d_inside=other["d_inside"], other_d_before=other["d_before"])
        if pb:
            r.update(person_d_inside=pb["d_inside"], person_d_before=pb["d_before"])
        if pv:
            r.update(person_val=pv[0], person_val_base=pv[1], person_d_val=pv[0] - pv[1])
        nt = note_of(name, m)
        if nt:
            r["note"] = nt
        res[name] = r
        extra = f" | {o} inside {other['d_inside']:+.4f} before {other['d_before']:+.4f}" if o else ""
        extra += f" | Person (copies) inside {pb['d_inside']:+.4f} before {pb['d_before']:+.4f}" if pb else ""
        extra += f" val {pv[0] - pv[1]:+.4f}" if pv else ""
        bar = "" if m == MOTO2 else f"; +{BAR} bar {'met' if up else 'missed'}"
        print(f"{name:36s} inside {n['inside'][0]:.4f} ({n['d_inside']:+.4f}, {nid('inside', c)} ids) | before {n['d_before']:+.4f} | "
              f"val {a:.4f} ({a - base:+.4f}{bar}){extra}{' | ' + nt if nt else ''} | veto {'; '.join(veto) or 'none'} | {'PASS' if ok else 'fail'}",
              flush=True)
    save_moto(cand, m, res)
    out_p.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out_p, flush=True)


def lanes60(m, assume):
    if m not in G3_FORM:
        sys.exit(f"STOP lanes60 builds the lanes of {G3M}, {B2M} or {VRM} only")
    sw, _ = v60_set(assume, None)
    if m in M.PERSON_ONLY:   # a Person-only refiner has no lane member (Person = second stages on the v29 lane): nothing to build
        print(f"{m}: Person only (REFINER form): no lane to build", flush=True)
        return
    if not M.inputs_ok(m):
        sys.exit(f"STOP no kit predictions for {m} in output/lumpi/g3in")
    for c in ("Van",) if m in M.VAN_ONLY else ("Car", "Van", "Truck", "Motorcycle"):   # a Van-only member's Van lane alone
        x = g3_lane(c, sw[c], m, M.TAGS[m])
        if x is None:
            print(f"{c}: v60's {c} is rs3: no lane", flush=True)
            continue
        M.G3.run_lane(x[0], x[1], {}, "val")
        print(f"lane {c}: {M.G3.O}/zG3{x[0]}_chain_val.json", flush=True)
    print("lanes60 built", m, flush=True)


def main(argv):
    """argv: near60|val60|lanes60 <m> [--check] [--assume-v59] | near60|val60 spec_lumpi_person_rs --base-v59 [--check]"""
    global BASE59
    chk, assume, b59 = "--check" in argv, "--assume-v59" in argv, "--base-v59" in argv
    a = [x for x in argv if x not in ("--check", "--assume-v59", "--base-v59")]
    if len(a) != 2 or a[0] not in ("near60", "val60", "lanes60"):
        sys.exit("usage: python scripts/igm_read.py near60|val60 safdnet_lumpi_rs|vrcnn_lumpi_b_fix2_rs|spec_lumpi_van_rs|spec_lumpi_person_rs|g2|moto2 [--check [--assume-v59]] | "
                 "near60|val60 spec_lumpi_person_rs --base-v59 [--check] | lanes60 safdnet_lumpi_rs|vrcnn_lumpi_b_fix2_rs|spec_lumpi_van_rs [--assume-v59]")
    if b59 and a[0] == "lanes60":
        sys.exit("STOP --base-v59 is for near60 / val60 only")
    if a[0] == "lanes60":
        return lanes60(a[1], assume)
    if b59:   # the Person-only reading on lumpi_v59 (a real reading, unlike --assume-v59)
        if assume or a[1] != PSM:
            sys.exit(f"STOP --base-v59 is near60 / val60 {PSM} only, without --assume-v59")
        BASE59, assume = True, True
    elif assume and not chk:
        sys.exit("STOP --assume-v59 is a dry-run switch set: use it with --check")
    {"near60": near60, "val60": val60}[a[0]](a[1], chk, assume)
