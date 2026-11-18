"""Final-combination reading for LUMPI (the stacking rule's "one final combination reading
with one fallback", as a mode of scripts/igm_read.py with its machinery; no new scorer, nothing loosened). Called as
    python scripts/igm_read.py combo near --car X --van X --person X --bicycle X --motorcycle X [--fallback C]   (in neartest_ws)
    python scripts/igm_read.py combo val  --car X --van X --person X --bicycle X --motorcycle X [--fallback C]   (main workspace)
Each switch is v59 (default) or a member whose single reading passed for that class (logs/igm_<m>_val.json, "<C>: add <m>"); joint lanes are
refused (as build_lumpi_v6x). The switches given are always the FULL first-stage set; --fallback C reads that set with C back at v59.
Writes logs/igm_combo_<key>_{near,val}.json (key = sha1 of the switches read). scripts/build_lumpi_v6x.py refuses to write without its PASS.
  0. STOP self-check: every changed class other than Person equals its chosen single reading: built by the same igm_read code, and its AP@0.5
     on inside, before (near) and val equal to logs/igm_<m>_near.json / _val.json to 1e-9. Printed, not gated again. v59 bases as igm_read.
  1. Numbers: AP@0.5 inside, before, val (pooled) for every changed class, and for Person whenever Person or a copy source (Bicycle,
     Motorcycle) changed: Person = its own chain (v59's or the member's) + copies recomputed from the chosen Bicycle and Motorcycle (W.copies),
     as build_lumpi_v6x builds it. Shown against v59 and against the class's own chosen reading.
  2. Conflict: (a) a changed class (Person if it changed itself) misses inside +0.01 vs v59 (>= 5 inside ids) or val falls > 0.01 vs v59;
     (b) Person more than 0.005 below its reference on inside or before (segments with >= 5 ids), reference = Person's own chosen reading if
     Person changed itself, else v59's Person; (c) Person val with the recomputed copies falls > 0.01 vs v59.
  3. Fallback (computed): interacting changes = Person's own change and any Bicycle / Motorcycle change. Drop the smallest inside gain of its own
     chosen reading (a tie with Person keeps Person; any other tie STOPs for a manual decision). Rebuild only with --fallback <that class>, one
     more reading. If that also conflicts: the largest single inside gain among the first-stage interacting changes, built with --only <class>
     (build_lumpi_v6x; other interacting classes to v59, non-interacting changes stay); no more readings. No interacting change: the reading is
     step 0 plus the printed numbers, and any conflict STOPs.
Joint Person / Bicycle (runs only when 2+ members pass Person, or 2+ pass Bicycle, alone): lane zJ_<class>_<tags> =
the v29 lane + every passing member at weight 1 (fixed order; Person: pv_lumpi_s2 is not a lane member but stays the sixth refiner), G3.run_lane as
the CPU joints; second stages on that lane by scripts/igm_joint_refine.sh (on an A100 GPU or the RTX 3090), exactly the single readings'
refiners; files output/lumpi/joint/<tag>/igm/<p>_val_ref.json (near in neartest_ws, val in main; test: s2/<p>_test_ref.json). Read with the single
readings' chains and gates (Person: v59's copies; Bicycle: the Person copies veto and 2c):
    python scripts/igm_read.py joint lanes Person|Bicycle <m1> <m2> ... --split val|test       (builds the joint lane; neartest_ws: split val)
    python scripts/igm_read.py joint near|val Person|Bicycle <m1> <m2> ...                      -> logs/igm_joint_<class>_near.json / .json
    python scripts/igm_read.py joint selfcheck near|val Person|Bicycle <m> <single lane chain file> [test]   (ONE member, dry run: the joint lane equals the
        single lane box for box, and the joint reading on the member's own second-stage files equals its single reading to 1e-9; no json)
rs3 (Truck / Van RS second stages): --truck v59|rs3 and --van rs3. A switch dict holds "Truck" only when it is not
v59, so key_of (and every key of the readings without rs3) is unchanged. rs3 is exempt from the fixed member-choice check; its single reading is
logs/igm_rs3_{near,val}.json. Conflict (a) for an rs3 class also needs val a - base >= +0.005 vs v59 (the rs3 rule). Van, member vs rs3 (fixed before
reading): if the fixed Van member choice and rs3 Van both passed, the larger d_inside, then the larger d_val (4 decimals), then the member before rs3.
v60 base (lumpi_v61): --base v60. The set carries ALL of v60's switches (logs/build_lumpi_v60.json, via igm60.v60_set;
never hardcoded); a class not given keeps v60's switch, a class given another value takes a v60-base candidate read by scripts/igm60.py:
  safdnet_lumpi_rs (alias SR)  Person / Car / Van / Truck / Bicycle / Motorcycle: igm60's G3 candidate (lanes igm60.g3_lane: IMc / IMv / IMt<SR> on
                               a class at v59, else I6<k>_<v60 lane>_SR, e.g. Motorcycle I6m_zJ_Motorcycle_DS_VM_P2_SR; the same cache files)
  g2                           Bicycle: R4 over the 4 G2 refiners on v60's (= v59's) Bicycle lane, then P3 (igm60.g2_cands)
  moto2                        Motorcycle: R4 over the 4 moto2 refiners on v60's Motorcycle lane, no P3 (igm60.moto2_cands)
  mp1                          Motorcycle: the MP1 filter on v60's Motorcycle (B1 then P3; mp1() below, a pure copy of the reference MP1 code)
  moto2+mp1, safdnet_lumpi_rs+mp1 (SR+mp1)   Motorcycle: the winner's boxes, then MP1 (the stacked reading)
Single readings: logs/igm60_<m>_{near,val}.json "<C>: add <m>" (its "_v60" block must name this v60), the stacked ones the same at
logs/igm60_<w>+mp1_{near,val}.json; mp1: MP1's json logs/mp1_<v60 Motorcycle slug>_results.json (written by run_mp_experiments.py, copied into both
workspaces). Reference = v60 class by class (igm60.bases60 + its check_near / check_val self-checks, then the classes switched); conflict rules (a)-(c)
as above with v60 in place of v59 (an SR / g2 class also needs val a - base >= +0.005 vs v60, as its own reading; moto2 / mp1: val not down, a >= base);
fallback and --only return a class to v60's switch; a tie between two classes left after the val gains STOPs (no member order for v60-base
candidates; a tie inside a class cannot STOP). Fixed choice (final_choice60): Bicycle g2 vs SR by d_inside then d_val (4
decimals), a full tie g2; Motorcycle SR vs moto2 the same, a full tie moto2; that winner and mp1 both passed -> the stacked <w>+mp1 (its reading
logs/igm60_<w>+mp1_{near,val}.json must exist) is gated against the better single of {winner, mp1} (d_inside, then d_val, 4 decimals, a full tie the
winner) by moto2's Motorcycle gate on (stack - v60) - (single - v60): it ships if it passes, else that better single ships (no STOP). Other
classes SR if its reading passed. key_of adds "base": "v60" (v59 keys unchanged).
    python scripts/igm_read.py combo near --base v60 --bicycle g2 --motorcycle moto2   (neartest_ws; then copy logs/igm_combo_<key>_near.json)
    python scripts/igm_read.py combo val  --base v60 --bicycle g2 --motorcycle moto2   (main) -> logs/igm_combo_<key>_val.json
Wave 2B pairs on the v60 base (scope C: Car / Van / Truck / Motorcycle): a pair = seed 1 at 0.5 + seed 2 at 0.5 as
one member (igm_read.PAIR / member: vmamba_lumpi_s2, dsvt_lumpi_s2, pv_lumpi_rs). Switch values: "<m>" (one pair) or "pairs:<m1>+<m2>[+...]" (the
joint of 2+ passers of one class, fixed order PAIRS = the seed-1 order). Motorcycle: v60's joint with each pair's seed-1 entry replaced in place
(lane W2m<tags>); Car / Van / Truck: v60's lane + the pair(s) (G3 form; one pair = igm60.g3_lane's lane IM<k><tag>); pair_lane, which STOPs unless a
single pair's lane equals scripts/pair60.py's lanes_of (the cache its reading built). Readings: logs/pair60_<m1>[+<m2>...]_<moto|cvt>_<near|val>.json
"<C>: add <m1>[+<m2>...]" (pair60.py writes the single ones; the joint reading is a later pair60 step). Gate as SR (val +0.005 bar, bar60). Fixed
choice (final_choice60, the fixed member choice among the pairs): no pair passed -> the v60-base rule above; a pair passed while that rule also picked a
candidate -> STOP (needs a manual decision); one pair -> it; 2+ -> the joint if its reading passed (it must exist: STOP until it does), else the pair with the
larger d_inside, then d_val (4 decimals), then the fixed order. Person / Bicycle: no pair (scope C)."""
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import igm_read as M

SW = ("Car", "Van", "Person", "Bicycle", "Motorcycle")
SW6 = SW + ("Truck",)   # Truck: rs3 only; in a switch dict only when not v59
TWO = ("Bicycle", "Motorcycle")
H = ("before", "inside")
RS3 = "rs3"


def key_of(sw):
    k = {c: sw[c] for c in SW}
    if sw.get("Truck", "v59") != "v59":
        k["Truck"] = sw["Truck"]
    if sw.get("base"):   # v60-base sets only; v59 sets have no "base" and keep their keys
        k["base"] = sw["base"]
    return hashlib.sha1(json.dumps(k, sort_keys=True).encode()).hexdigest()[:10]


def parse(argv):
    if "--base" in argv:   # v60 base (lumpi_v61); without --base: the v59 path below, unchanged
        return parse60(argv)
    sw, fb = {c: "v59" for c in SW}, None
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--fallback":
            fb = argv[i + 1]
        elif k.startswith("--") and k[2:].capitalize() in SW6:
            sw[k[2:].capitalize()] = argv[i + 1]
        else:
            sys.exit(f"STOP unknown argument {k}")
        i += 2
    if sw.get("Truck") == "v59":
        del sw["Truck"]   # key_of and the stored switches stay as without rs3
    for c, m in sw.items():
        if m == "v59":
            continue
        if m == RS3 or c == "Truck":
            if m != RS3 or c not in M.RS3_CLASSES:
                sys.exit(f"STOP {c} = {m}: rs3 is read for {M.RS3_CLASSES} only, and Truck takes v59 or rs3")
            continue
        if m.startswith("joint:"):
            joint_members(c, m)   # STOPs unless a CPU-class joint in the fixed order
            continue
        if m not in M.TAGS or c not in M.classes_of(m, False):
            sys.exit(f"STOP {c} = {m}: not a member read for that class")
    if fb is not None and (fb not in SW or sw[fb] == "v59"):
        sys.exit(f"STOP --fallback {fb}: not a changed class of this switch set")
    return sw, fb


def find_reading(m, c, where, base=None):
    """The one single-reading file of member m that holds "<c>: add <m>" (a full reading logs/igm_<m>_<where>.json, or a class-subset one such
    as _carvan / _motorcycle). None if no file has it; STOP if more than one does. base "v60": the v60-base reading (find_reading60)."""
    if base == B60:
        return find_reading60(m, c, where)
    fs = [p for p in [Path(f"logs/igm_{m}_{where}.json")] + sorted(Path("logs").glob(f"igm_{m}_*_{where}.json"))
          if p.exists() and f"{c}: add {m}" in json.loads(p.read_text())]
    if len(fs) > 1:   # e.g. the _carvan reading and a later full reading: same code and inputs, so the numbers must agree; the full one is used
        rs = [json.loads(p.read_text())[f"{c}: add {m}"] for p in fs]
        num = lambda r: [r[k][0] if isinstance(r[k], list) else r[k] for k in ("inside", "before", "val", "d_inside", "passed") if k in r]
        if any(len(num(r)) != len(num(rs[0])) or any(abs(float(x) - float(y)) > 1e-9 for x, y in zip(num(r), num(rs[0]))) for r in rs[1:]):
            sys.exit(f"STOP {c} = {m}: readings disagree across {[str(p) for p in fs]}")
    return fs[0] if fs else None


def single(m, c, where, base=None):
    if base == B60:   # v60-base single reading (igm60 / MP1); STOP if none, or (val) if it did not pass
        r = reading60(m, c, where)
        if r is None:
            sys.exit(f"STOP {c} = {m}: no v60-base single reading '{c}: add {m}' ({file60(m, where)}"
                     + (f" / {mp1_json()}" if m == MP1 else "") + ")")
        if where == "val" and not (stack_gate60(m)[0] if m in STACKED else r.get("passed")):   # stacked: its gate vs the better single
            sys.exit(f"STOP {c} = {m}: its v60-base single reading did not pass ({find_reading60(m, c, where)}"
                     + ("; stack gate against the better single" if m in STACKED else "") + ")")
        return r
    if m.startswith("joint:"):
        ms = joint_members(c, m)
        p = Path(f"logs/igm_joint_{c}_near.json" if where == "near" else f"logs/igm_joint_{c}.json")
        r = json.loads(p.read_text()) if p.exists() else {}
        if r.get("members") != ms:
            sys.exit(f"STOP {c} = {m}: no joint reading of these members in {p}")
        r = r[f"{c}: {m}"]
        if where == "val" and not r.get("passed"):
            sys.exit(f"STOP {c} = {m}: its joint reading did not pass ({p})")
        return r
    p = find_reading(m, c, where)
    if p is None:
        sys.exit(f"STOP {c} = {m}: no single reading of '{c}: add {m}' in logs/igm_{m}*_{where}.json")
    r = json.loads(p.read_text()).get(f"{c}: add {m}")
    if where == "val" and not r.get("passed"):
        sys.exit(f"STOP {c} = {m}: its single reading did not pass ({p})")
    return r


def classes_built(sw, ids):
    """v59 bases and the switched combination, with igm_read's code. -> base {class: boxes}, new {class: boxes}."""
    keys = M.SZ.PP.static_keys("lumpi")
    P = M.SZ.build("val", ids)
    lref = json.loads((M.G3.RL.L / "v29_val_ref.json").read_text())
    own_p0 = M.G3.person(M.G3.P_REF("val"), ids, M.LOAD, keys)
    base = {"Car": M.G3.lane_class(M.G3.run_lane("IGc00", M.G3.B_SPEC + [M.IG.RS], {}, "val"), "Car", ids),
            "Van": M.IG.van(lref, lref, ids), "Bicycle": P["Bicycle"], "Motorcycle": P["Motorcycle"]}
    base["Person"] = {f: own_p0[f] + M.W.copies({f: base["Bicycle"][f] + base["Motorcycle"][f]})[f] for f in ids}
    new = dict(base)
    for c in ("Car", "Van"):
        if sw[c].startswith("joint:"):
            new[c] = joint_lane(c, joint_members(c, sw[c]), ids, lref)
        elif sw[c] != "v59" and sw[c] != RS3:
            car, van = M.lanes(sw[c], M.TAGS[sw[c]], ids)
            new[c] = car if c == "Car" else van
    rs = [c for c in M.RS3_CLASSES if sw.get(c, "v59") == RS3]
    if rs:   # rs3: R6 of the 3 RS refiners, then rule37 + T5 / V5 on the chosen Truck + Van pair in one pass (the other class from lref)
        miss = [str(p) for p in M.rs3_files() if not p.exists()]
        if miss:
            sys.exit("STOP rs3 second-stage files missing: " + ", ".join(miss))
        fu = M.rs3_fuse(lref, ids)
        pair = M.tv_pass(fu["Truck"] if "Truck" in rs else lref, fu["Van"] if "Van" in rs else lref, ids)
        for c in rs:
            new[c] = pair[c]
        if "Truck" in rs:
            base["Truck"] = M.truck(lref, lref, ids)
    if sw["Bicycle"].startswith("joint:"):
        new["Bicycle"] = M.IB.bike(joint_files("Bicycle", joint_members("Bicycle", sw["Bicycle"]))[0], ids, keys)
    elif sw["Bicycle"] != "v59":
        new["Bicycle"] = M.IB.bike(Path(f"output/lumpi/{sw['Bicycle']}/igm/bicycle_val_ref.json"), ids, keys)
    if sw["Motorcycle"].startswith("joint:"):
        new["Motorcycle"] = joint_lane("Motorcycle", joint_members("Motorcycle", sw["Motorcycle"]), ids, lref)
    elif sw["Motorcycle"] != "v59":
        m = sw["Motorcycle"]
        new["Motorcycle"] = M.G3.lane_class(M.G3.run_lane(f"IMm{M.TAGS[m]}", M.G3.B_SPEC + M.member(m, "Motorcycle"), {}, "val"), "Motorcycle", ids)
    own_p = own_p0
    if sw["Person"].startswith("joint:"):
        pf = joint_files("Person", joint_members("Person", sw["Person"]))
        miss = [str(p) for p in pf if not p.exists()]
        if miss:
            sys.exit("STOP joint Person second-stage files missing: " + ", ".join(miss))
        own_p = M.G3.person(pf, ids, M.LOAD, keys)
    elif sw["Person"] != "v59":
        m = sw["Person"]
        d = Path(f"output/lumpi/{m}/igm")
        pf = M.G3.P_REF("val") + [d / "v29_val_person_min01.json"] if m in M.REFINER else [d / f"person_{r}_val_ref.json" for r in M.REF_ORDER]
        miss = [str(p) for p in pf if not p.exists()]
        if miss:
            sys.exit("STOP Person second-stage files missing: " + ", ".join(miss))
        own_p = M.G3.person(pf, ids, M.LOAD, keys)
    new["Person"] = {f: own_p[f] + M.W.copies({f: new["Bicycle"][f] + new["Motorcycle"][f]})[f] for f in ids}
    return base, new


def changed_of(sw):
    ref = ref_of(sw)   # v59 sets: every class "v59" (unchanged); v60-base sets: v60's switches
    ch = [c for c in SW6 if sw.get(c, "v59") != ref[c]]   # Truck (rs3) last; without Truck as before
    touched = ch + (["Person"] if "Person" not in ch and any(sw[c] != ref[c] for c in TWO) else [])
    inter = [c for c in ("Person",) + TWO if sw[c] != ref[c]]
    return ch, touched, inter


def reduced(sw, fb):
    return dict(sw, **{fb: ref_of(sw)[fb]}) if fb else dict(sw)   # the fallback class back to v59 (v60-base sets: to v60's switch)


def near(argv):
    full, fb = parse(argv)
    sw = reduced(full, fb)
    ch, touched, _ = changed_of(sw)
    if not ch:
        sys.exit("STOP no class changed: nothing to read")
    bk = sw.get("base") or "v59"   # the reference: v59, or v60 for a v60-base set (its name in the json and the prints)
    ids = M.S.ids_of("val")
    assert len(ids) == 266
    HH = {h: M.rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in H}
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in M.S.labels(ids).items()}
    if bk == B60:   # v60's six classes with igm60's self-checks (bases60 box for box, check_near's AP checks), then the set's classes
        base, new = classes_built60(sw, ids, "near", gt, HH)
    else:
        base, new = classes_built(sw, ids)
        want_p = json.loads(Path("logs/m3_neartest_copies.json").read_text())["copies_on"]["inside"][0]
        got = M.ap2(base["Person"], gt, HH["inside"], "Person")[0]
        print(f"self-check inside Person: v59 base {got:.6f} vs {want_p:.6f}: {abs(got - want_p) < 1e-9}", flush=True)
        if abs(got - want_p) >= 1e-9:
            sys.exit("STOP the v59 Person base does not reproduce")
    res = {"switches": sw, "first_stage": full, "fallback": fb, "key": key_of(sw), "classes": {}}
    if bk == B60:
        res.update(info60(sw))
    ok = True
    for c in touched:
        r = {h: {bk: M.ap2(base[c], gt, HH[h], c), "new": M.ap2(new[c], gt, HH[h], c)} for h in H}
        res["classes"][c] = r
        if c != "Person":   # step 0: equals its single reading
            s = single(sw[c], c, "near", sw.get("base"))
            same = all(abs(r[h]["new"][0] - s[h][0]) < 1e-9 for h in H)
            print(f"step 0 near {c} = {sw[c]}: inside {r['inside']['new'][0]:.6f} / before {r['before']['new'][0]:.6f} equal its single reading: {same}", flush=True)
            ok = ok and same
        print(f"near {c:10s} inside {r['inside'][bk][0]:.4f} -> {r['inside']['new'][0]:.4f} | before {r['before'][bk][0]:.4f} -> "
              f"{r['before']['new'][0]:.4f}", flush=True)
    for (c, h, k), want in mp1_fields(sw, "near").items():   # MP1 alone: the combo recomputes MP1's own near-test numbers (1e-9)
        got = res["classes"][c][h][k][0]
        same = abs(got - want) < 1e-9
        print(f"step 0 near MP1 self-check {c} {h} ({'MP1' if k == 'new' else k}) {got:.6f} equals MP1's own json {want:.6f}: {same}", flush=True)
        ok = ok and same
    if not ok:
        sys.exit("STOP step 0: a changed class differs from its single reading on near-test")
    Path(f"logs/igm_combo_{key_of(sw)}_near.json").write_text(json.dumps(res, indent=1))
    print("wrote", f"logs/igm_combo_{key_of(sw)}_near.json", flush=True)


def val(argv):
    full, fb = parse(argv)
    sw = reduced(full, fb)
    ch, touched, inter = changed_of(sw)
    _, _, inter_full = changed_of(full)
    if not ch:
        sys.exit("STOP no class changed: nothing to read")
    bk, base_of = sw.get("base") or "v59", sw.get("base")   # the reference (v59, or v60 for a v60-base set) and the readings' base
    ref = ref_of(sw)
    fixed = final_choice60 if bk == B60 else final_choice
    for c in SW6:   # fixed member choice: a changed class must take the member (or joint) the fixed rule chooses; rs3 is exempt (not
        v = full.get(c, "v59")   # a member): Truck rs3 needs its own pass, Van member vs rs3 by the fixed Van conflict rule (final_choice)
        if v != ref[c] and v != fixed(c):   # v60-base sets: the fixed stacking rules (final_choice60)
            sys.exit(f"STOP {c} = {v}: the fixed choice for {c} is {fixed(c)} (" + ("v60-base stacking rule)" if bk == B60 else "fixed member-choice rule"
                     + ("; Van: member vs rs3 rule)" if c == "Van" else ")")))
    if fb:   # the fallback must be the one the first-stage reading computed
        p1 = Path(f"logs/igm_combo_{key_of(full)}_val.json")
        r1 = json.loads(p1.read_text()) if p1.exists() else None
        if not r1 or r1.get("passed") or r1.get("fallback_to_drop") != fb:
            sys.exit(f"STOP --fallback {fb}: the first-stage reading {p1} does not name it (or passed, or is missing)")
    nr = json.loads(Path(f"logs/igm_combo_{key_of(sw)}_near.json").read_text())
    assert nr["switches"] == sw, "near reading is for other switches"
    ids = M.S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in M.S.labels(ids).items()}
    if bk == B60:   # v60's six classes (bases60 box for box; check_val: v59 constants for the classes v60 keeps, build_lumpi_v60.json val 1e-9)
        i60 = info60(sw)
        if nr.get("v60") != i60["v60"] or (nr.get("mp1_json") or {}).get("md5") != (i60.get("mp1_json") or {}).get("md5"):
            sys.exit(f"STOP logs/igm_combo_{key_of(sw)}_near.json was read on another v60 / MP1 json ({nr.get('v60')}, {nr.get('mp1_json')})")
        base, new = classes_built60(sw, ids, "val", gt)
    else:
        base, new = classes_built(sw, ids)
        for c, want in {"Car": 0.896, "Person": 0.48}.items():
            got = round(M.G3.ap(base[c], gt, ids, c), 4)
            print(f"self-check val {c}: v59 base {got} vs {want}: {got == want}", flush=True)
            if got != want:
                sys.exit(f"STOP the val {c} base does not reproduce v59")
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    nid = lambda h, c: cnt.get(h, {}).get(c, {}).get("distinct_ids", 0)
    res = dict(nr, classes={})
    conflicts = []
    vals = {}
    print(f"final-combination reading {key_of(sw)}: switches {sw}" + (f" (fallback: {fb} back to {bk} from {full})" if fb else ""), flush=True)
    for c in touched:
        n = nr["classes"][c]
        a, b0 = M.G3.ap(new[c], gt, ids, c), M.G3.ap(base[c], gt, ids, c)
        vals[c] = {bk: b0, "new": a}
        r = {"inside": n["inside"]["new"][0], "before": n["before"]["new"][0], "val": a,
             f"d_inside_{bk}": n["inside"]["new"][0] - n["inside"][bk][0], f"d_before_{bk}": n["before"]["new"][0] - n["before"][bk][0],
             f"d_val_{bk}": a - b0}
        if c != "Person" or sw["Person"] != ref["Person"]:
            s = single(sw[c], c, "val", base_of)
            sn = single(sw[c], c, "near", base_of)
            r.update(own_inside=sn["inside"][0], own_before=sn["before"][0], own_val=s["val"], own_d_inside=s["d_inside"])
            if c != "Person":
                same = abs(a - s["val"]) < 1e-9
                print(f"step 0 val {c} = {sw[c]}: {a:.6f} equals its single reading {s['val']:.6f}: {same}", flush=True)
                if not same:
                    sys.exit("STOP step 0: a changed class differs from its single reading on val")
        if c in ch:   # (a)
            val_ok = M.moto_val_ok(a, b0) if c == "Motorcycle" else r[f"d_val_{bk}"] >= -0.01   # Motorcycle: its own val rule (moto_val_ok)
            if sw[c] == RS3:   # rs3 rule: an rs3 class also needs val a - base >= +0.005 vs v59, unrounded (as its single reading)
                val_ok = val_ok and r[f"d_val_{bk}"] >= M.RS3_BAR
            bar = bk == B60 and bar60(sw[c])   # an SR / g2 class: val a - base >= +0.005 vs v60, unrounded (as its own igm60 reading)
            if bar:
                val_ok = val_ok and r[f"d_val_{bk}"] >= BAR60
            if not (nid("inside", c) >= 5 and r[f"d_inside_{bk}"] >= 0.01 and val_ok):
                conflicts.append(f"(a) {c}: inside {r[f'd_inside_{bk}']:+.4f} vs {bk} ({nid('inside', c)} ids), val {r[f'd_val_{bk}']:+.4f}"
                                 + (f" (rs3: val bar +{M.RS3_BAR})" if sw[c] == RS3 else "") + (f" (val bar +{BAR60})" if bar else ""))
        if c == "Person":   # (b), (c)
            pref = {h: (r[f"own_{h}"] if sw["Person"] != ref["Person"] else n[h][bk][0]) for h in H}
            for h in H:
                if nid(h, "Person") >= 5 and r[h] - pref[h] < -0.005:
                    conflicts.append(f"(b) Person {h} {r[h]:.4f} vs reference {pref[h]:.4f} ({r[h] - pref[h]:+.4f})")
            if r[f"d_val_{bk}"] < -0.01:   # always a gate (2c)
                conflicts.append(f"(c) Person val with recomputed copies {r[f'd_val_{bk}']:+.4f} vs {bk}")
            r["reference"] = pref
        res["classes"][c] = r
        own = f" | own reading inside {r['own_inside']:.4f} val {r['own_val']:.4f}" if "own_inside" in r else ""
        print(f"{c:10s} inside {r['inside']:.4f} ({r[f'd_inside_{bk}']:+.4f} vs {bk}, {nid('inside', c)} ids) | before {r['before']:.4f} "
              f"({r[f'd_before_{bk}']:+.4f}) | val {a:.4f} ({r[f'd_val_{bk}']:+.4f}){own}", flush=True)
    for (c, h, k), want in mp1_fields(sw, "val").items():   # MP1 alone: the combo recomputes MP1's own val numbers (1e-9)
        same = abs(vals[c][k] - want) < 1e-9
        print(f"step 0 val MP1 self-check {c} ({'MP1' if k == 'new' else k}) {vals[c][k]:.6f} equals MP1's own json {want:.6f}: {same}", flush=True)
        if not same:
            sys.exit("STOP step 0: the combo does not reproduce MP1's own val numbers")
    res["conflicts"] = conflicts
    res["passed"] = not conflicts
    gain = lambda c: single(full[c], c, "val", base_of)["d_inside"]
    if conflicts:
        print("CONFLICT: " + "; ".join(conflicts), flush=True)
        if not inter:
            res["next"] = "STOP: no interacting change to drop; needs a manual decision"
        elif not fb:
            # drop the smallest inside gain (4 decimals); a tie with Person keeps Person; Bicycle vs Motorcycle: drop the smaller val gain,
            # then the one whose member is later in the fixed order; still tied: STOP
            # (v60-base sets: no member order is fixed for the v60-base candidates, so a tie left after the val gains STOPs)
            g = {c: round(gain(c), 4) for c in inter}
            low = [c for c in inter if g[c] == min(g.values())]
            if len(low) > 1 and "Person" in low:
                low.remove("Person")
            if len(low) > 1:
                dv = {c: round(single(full[c], c, "val", base_of)["d_val"], 4) for c in low}
                low = [c for c in low if dv[c] == min(dv.values())]
            if len(low) > 1 and bk != B60:
                om = {c: max(ORDER.index(x) for x in (joint_members(c, full[c]) if full[c].startswith("joint:") else [full[c]])) for c in low}
                low = [c for c in low if om[c] == max(om.values())]
            if len(low) > 1:
                res["next"] = f"STOP: tie for the smallest inside gain {low} {g}; needs a manual decision"
            elif not any(reduced(full, low[0]).get(c, "v59") != ref[c] for c in SW6):
                res["next"] = f"STOP: dropping {low[0]} leaves no change: the classes stay at {bk}, nothing to build"
            else:
                res["fallback_to_drop"] = low[0]
                res["next"] = f"rebuild with --fallback {low[0]} (inside gains of own readings {g}), then one more reading"
        else:
            g = {c: gain(c) for c in inter_full}
            hi = max(g.values())
            top = [c for c in inter_full if g[c] == hi]
            only_sw = dict(full, **{c: ref[c] for c in inter_full if c not in top[:1]})
            if len(top) > 1:
                res["next"] = f"STOP: tie for the largest inside gain {top} {g}; needs a manual decision"
            elif only_sw == sw:   # --only would build the very set that just failed
                res["next"] = f"STOP: --only {top[0]} is the set that just failed this reading; the interacting classes stay at {bk}"
            else:
                res["only"] = top[0]
                res["next"] = f"build with --only {top[0]} (largest single inside gain {g}); no more readings"
        print("NEXT:", res["next"], flush=True)
    print(f"final combination {key_of(sw)}: {'PASS' if res['passed'] else 'CONFLICT'}", flush=True)
    Path(f"logs/igm_combo_{key_of(sw)}_val.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", f"logs/igm_combo_{key_of(sw)}_val.json", flush=True)


# ------------------------------------------------------------------ joint readings (the joint rule; the fixed member choice)
ORDER = ("fshnet_lumpi_s2", "dsvt_lumpi", "vmamba_lumpi", "pv_lumpi_s2")
JOINT_CPU = ("Car", "Van", "Motorcycle")
JOINT_S2 = ("Person", "Bicycle")   # second stages on the joint lane: scripts/igm_joint_refine.sh


def joint_members(c, v, selfcheck=False):
    """selfcheck: ONE member, which may be igm_read's dry-run member (fshnet_lumpi), for the joint self-check only."""
    ms = v.split(":", 1)[1].split("+")
    order = ORDER + (tuple(M.DRY) if selfcheck else ())
    if c not in JOINT_CPU + JOINT_S2:
        sys.exit(f"STOP {c} = {v}: no joint for this class")
    if (len(ms) != 1 if selfcheck else len(ms) < 2) or len(set(ms)) != len(ms) or any(m not in order for m in ms) or ms != sorted(ms, key=order.index):
        sys.exit(f"STOP {c} = {v}: a joint needs 2+ distinct members in the fixed order {ORDER}" + (" (self-check: exactly one)" if selfcheck else ""))
    return ms


def joint_tag(c, ms):
    return f"zJ_{c}_" + "_".join(M.TAGS.get(m) or M.DRY[m] for m in ms)


def joint_spec(c, ms):
    lane_ms = [m for m in ms if not (c == "Person" and m in M.REFINER)]   # Person: pv_lumpi_s2 is a refiner (sixth), not a lane member
    return M.G3.B_SPEC + ([M.IG.RS] if c == "Car" else []) + [x for m in lane_ms for x in M.member(m, c)]


def joint_dir(c, ms):
    return Path("output/lumpi/joint") / joint_tag(c, ms)


def joint_refs(c, ms):
    return M.REF_ORDER + (("pv2",) if c == "Person" and "pv_lumpi_s2" in ms else ())   # lumpi_v51's five, then PV s2 as the sixth


def joint_files(c, ms, split="val"):
    """Second-stage files on the joint lane: split val = igm/ (near-test in neartest_ws, val in main); test = s2/."""
    d = joint_dir(c, ms) / ("igm" if split == "val" else "s2")
    if c == "Bicycle":
        return [d / f"bicycle_{split}_ref.json"]
    return [d / f"person_{r}_{split}_ref.json" for r in joint_refs(c, ms)]


def joint_lane(c, ms, ids, lref):
    ch = M.G3.run_lane(joint_tag(c, ms), joint_spec(c, ms), {}, "val")
    return M.IG.van(ch, lref, ids) if c == "Van" else M.G3.lane_class(ch, c, ids)


def passes(c):
    """Members (fixed order) whose single reading of class c passed."""
    out = []
    for m in ORDER:
        p = find_reading(m, c, "val")
        if p and json.loads(p.read_text())[f"{c}: add {m}"].get("passed"):
            out.append(m)
    return out


def chosen(c):
    """The fixed member choice, set before any reading: 0 passes -> v59; 1 -> that member; 2+ -> the joint if its reading passed, else the single member
    with the larger inside gain, a tie at 4 decimals -> the larger val gain, still tied -> the earlier member in the fixed order."""
    ps = passes(c)
    if len(ps) < 2:
        return ps[0] if ps else "v59"
    jv = "joint:" + "+".join(ps)
    p = Path(f"logs/igm_joint_{c}.json")
    r = json.loads(p.read_text()) if p.exists() else {}
    if r.get("members") != ps:
        pre = f"scripts/igm_joint_wait.sh {c} {' '.join(ps)} (second stages on the joint lane), then " if c in JOINT_S2 else ""
        sys.exit(f"STOP {c}: {len(ps)} members passed ({ps}); run the joint reading first: {pre}igm_read.py joint near|val {c} {' '.join(ps)}")
    if r[f"{c}: {jv}"]["passed"]:
        return jv
    sv = {m: json.loads(find_reading(m, c, "val").read_text())[f"{c}: add {m}"] for m in ps}
    return sorted(ps, key=lambda m: (-round(sv[m]["d_inside"], 4), -round(sv[m]["d_val"], 4), ORDER.index(m)))[0]


def rs3_reading(c):
    """rs3's single val reading of class c (logs/igm_rs3_val.json) if it passed, else None."""
    p = find_reading(RS3, c, "val")
    r = json.loads(p.read_text())[f"{c}: add {RS3}"] if p else None
    return r if r and r.get("passed") else None


def final_choice(c):
    """The class's fixed choice with rs3: Truck -> rs3 if its reading passed, else v59. Van -> the fixed member choice (chosen) unless rs3 Van passed too:
    rs3 alone -> rs3; both -> the larger d_inside, then the larger d_val (4 decimals), then the member before rs3.
    Other classes: chosen(c), unchanged."""
    if c == "Truck":
        return RS3 if rs3_reading("Truck") else "v59"
    m = chosen(c)
    if c != "Van":
        return m
    rs = rs3_reading("Van")
    if not rs:
        return m
    if m == "v59":
        return RS3
    if m.startswith("joint:"):
        p = Path("logs/igm_joint_Van.json")
        mr = json.loads(p.read_text())[f"Van: {m}"]
    else:
        mr = json.loads(find_reading(m, "Van", "val").read_text())[f"Van: add {m}"]
    k = lambda r: (round(r["d_inside"], 4), round(r["d_val"], 4))
    return RS3 if k(rs) > k(mr) else m   # a full tie keeps the member


def joint(argv):
    if argv[0] == "lanes":
        return joint_lanes(argv[1:])
    if argv[0] == "selfcheck":
        return joint_selfcheck(argv[1:])
    if argv[1] in JOINT_S2:
        return joint_ps(argv[0], argv[1], argv[2:])
    where, c, ms = argv[0], argv[1], argv[2:]
    v = "joint:" + "+".join(ms)
    ms = joint_members(c, v)
    if where == "val" and ms != passes(c):   # the single val readings live in the main workspace
        sys.exit(f"STOP joint {c}: members {ms} are not exactly the members that passed alone {passes(c)}")
    ids = M.S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": k} for b, k in vv] for f, vv in M.S.labels(ids).items()}
    sw = dict({k: "v59" for k in SW}, **{c: v})
    base, new = classes_built(sw, ids)
    touched = [c] + (["Person"] if c == "Motorcycle" else [])
    name = f"{c}: {v}"
    if where == "near":
        assert len(ids) == 266
        HH = {h: M.rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in H}
        want_p = json.loads(Path("logs/m3_neartest_copies.json").read_text())["copies_on"]["inside"][0]
        got = M.ap2(base["Person"], gt, HH["inside"], "Person")[0]
        print(f"self-check inside Person: v59 base {got:.6f} vs {want_p:.6f}: {abs(got - want_p) < 1e-9}", flush=True)
        if abs(got - want_p) >= 1e-9:
            sys.exit("STOP the v59 Person base does not reproduce")
        res = {"members": ms, "lane": joint_tag(c, ms)}
        for k in touched:
            r = {h: M.ap2(new[k], gt, HH[h], k) for h in H}
            b = {h: M.ap2(base[k], gt, HH[h], k) for h in H}
            r.update(d_inside=r["inside"][0] - b["inside"][0], d_before=r["before"][0] - b["before"][0])
            res[name if k == c else f"Person: copies from {name}"] = r
            print(f"near {k:10s} inside {b['inside'][0]:.4f} -> {r['inside'][0]:.4f} ({r['d_inside']:+.4f}) | before {b['before'][0]:.4f} -> "
                  f"{r['before'][0]:.4f} ({r['d_before']:+.4f})", flush=True)
        Path(f"logs/igm_joint_{c}_near.json").write_text(json.dumps(res, indent=1))
        print("wrote", f"logs/igm_joint_{c}_near.json", flush=True)
        return
    nr = json.loads(Path(f"logs/igm_joint_{c}_near.json").read_text())
    if nr.get("members") != ms:
        sys.exit("STOP the joint near reading is for other members")
    for k, want in {"Car": 0.896, "Person": 0.48}.items():
        got = round(M.G3.ap(base[k], gt, ids, k), 4)
        print(f"self-check val {k}: v59 base {got} vs {want}: {got == want}", flush=True)
        if got != want:
            sys.exit(f"STOP the val {k} base does not reproduce v59")
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    nid = lambda h, k: cnt.get(h, {}).get(k, {}).get("distinct_ids", 0)
    n = nr[name]
    a, b0 = M.G3.ap(new[c], gt, ids, c), M.G3.ap(base[c], gt, ids, c)
    tch = [(c, n)] + ([("Person", nr[f"Person: copies from {name}"])] if c == "Motorcycle" else [])
    veto = [f"{k} {h} {r['d_' + h]:+.4f}" for k, r in tch for h in H if nid(h, k) >= 5 and r["d_" + h] < -0.005]
    pv = None
    if c == "Motorcycle":   # 2c: Person val with the recomputed copies not down more than 0.01 vs v59
        pv = (M.G3.ap(new["Person"], gt, ids, "Person"), M.G3.ap(base["Person"], gt, ids, "Person"))
        if pv[0] - pv[1] < -0.01:
            veto.append(f"Person val (copies) {pv[0] - pv[1]:+.4f}")
    val_ok = M.moto_val_ok(a, b0) if c == "Motorcycle" else a - b0 >= -0.01
    ok = bool(nid("inside", c) >= 5 and n["d_inside"] >= 0.01 and val_ok and not veto)
    res = dict(nr)
    res[name] = dict(n, val=a, val_base=b0, d_val=a - b0, veto=veto, passed=ok)
    if pv:
        res[name].update(person_val=pv[0], person_val_base=pv[1], person_d_val=pv[0] - pv[1])
    print(f"joint {name} (lane {joint_tag(c, ms)}): inside {n['inside'][0]:.4f} ({n['d_inside']:+.4f}, {nid('inside', c)} ids) | before "
          f"{n['d_before']:+.4f} | val {a:.4f} ({a - b0:+.4f}) | veto {'; '.join(veto) or 'none'} | {'PASS' if ok else 'fail'}", flush=True)
    Path(f"logs/igm_joint_{c}.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", f"logs/igm_joint_{c}.json", "| class takes:", chosen(c), flush=True)


def joint_lanes(argv):
    """joint lanes <class> <m...> --split val|test [--selfcheck]: the joint lane, cached by name (G3.run_lane: zG3<tag>_chain_<split>.json)."""
    sel = "--selfcheck" in argv
    argv = [x for x in argv if x != "--selfcheck"]
    i = argv.index("--split")
    split, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    c, ms = argv[0], joint_members(argv[0], "joint:" + "+".join(argv[1:]), sel)
    M.G3.run_lane(joint_tag(c, ms), joint_spec(c, ms), {}, split)
    print(f"joint lane {joint_tag(c, ms)} ({split}): {M.G3.O}/zG3{joint_tag(c, ms)}_chain_{split}.json", flush=True)


def _bases(ids, keys):
    """igm_read.build's bases for Person / Bicycle, with its code (same functions, same order)."""
    P = M.SZ.build("val", ids)
    cp = M.W.copies({f: [d for c in M.W.TW for d in P[c].get(f, [])] for f in ids})
    person_of = lambda b: {f: P["Person"][f] + M.W.copies({f: b[f] + P["Motorcycle"][f]})[f] for f in ids}
    base_p = M.G3.person(M.G3.P_REF("val"), ids, M.LOAD, keys)
    out = {"Person: v59": ("Person", {f: base_p[f] + cp[f] for f in ids}), "Bicycle: v59": ("Bicycle", P["Bicycle"]),
           "Person: v59 (copies from v59 Bicycle)": ("Person", person_of(P["Bicycle"]))}
    return out, cp, person_of


def _cands(c, name, files, ids, keys, cp, person_of):
    miss = [str(p) for p in files if not p.exists()]
    if miss:
        sys.exit(f"STOP {c}: second-stage files missing: " + ", ".join(miss))
    if c == "Person":   # igm_read.build: the chain on the files, plus v59's copies
        p = M.G3.person(files, ids, M.LOAD, keys)
        return {name: ("Person", {f: p[f] + cp[f] for f in ids})}
    b = M.IB.bike(files[0], ids, keys)   # igm_read.build: L11b's file, R4 at >= 0.05, P3; Person copies recomputed
    return {name: ("Bicycle", b), f"Person: copies from {name}": ("Person", person_of(b))}


def _near_res(cand, gt, HH):
    res = {}
    for name, (k, p) in cand.items():
        res[name] = {"class": k, **{h: M.ap2(p, gt, fr, k) for h, fr in HH.items()}}
    base_of = lambda name, k: "Person: v59 (copies from v59 Bicycle)" if name.startswith("Person: copies") else f"{k}: v59"   # igm_read.near
    for name, r in res.items():
        b = res[base_of(name, r["class"])]
        r["d_inside"], r["d_before"] = r["inside"][0] - b["inside"][0], r["before"][0] - b["before"][0]
    return res


def _near_checks(cand, gt, HH):
    want = json.loads(Path("logs/m3_neartest_copies.json").read_text())["copies_on"]["inside"][0]
    got = M.ap2(cand["Person: v59"][1], gt, HH["inside"], "Person")[0]
    bb = M.ap2(cand["Bicycle: v59"][1], gt, HH["inside"], "Bicycle")[0]
    pc = M.ap2(cand["Person: v59 (copies from v59 Bicycle)"][1], gt, HH["inside"], "Person")[0]
    ok = abs(got - want) < 1e-9 and round(bb, 4) == 0.2105 and abs(pc - want) < 1e-9
    print(f"self-check inside Person: v59 base {got:.6f} vs {want:.6f}; Bicycle {bb:.6f} (0.2105); Person with copies from v59's Bicycle {pc:.6f}: {ok}", flush=True)
    if not ok:
        sys.exit("STOP a base does not reproduce v59 on inside")


def _val_gate(c, name, cand, near_res, gt, ids):
    """igm_read.val's gate for one Person / Bicycle candidate (unchanged): inside +0.01 (>= 5 ids), val not down > 0.01, the near-test veto on
    every touched class (Person copies for Bicycle) and, for Bicycle, Person val with the recomputed copies not down > 0.01 (2c)."""
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    nid = lambda h, k: cnt.get(h, {}).get(k, {}).get("distinct_ids", 0)
    n = near_res[name]
    base = M.G3.ap(cand[f"{c}: v59"][1], gt, ids, c)
    a = M.G3.ap(cand[name][1], gt, ids, c)
    pb = near_res.get(f"Person: copies from {name}") if c == "Bicycle" else None
    touched = [(c, n)] + ([("Person", pb)] if pb else [])
    veto = [f"{k} {h} {r['d_' + h]:+.4f}" for k, r in touched for h in H if nid(h, k) >= 5 and r["d_" + h] < -0.005]
    pv = None
    if c == "Bicycle":
        pv = (M.G3.ap(cand[f"Person: copies from {name}"][1], gt, ids, "Person"), M.G3.ap(cand["Person: v59 (copies from v59 Bicycle)"][1], gt, ids, "Person"))
        if pv[0] - pv[1] < -0.01:
            veto.append(f"Person val (copies) {pv[0] - pv[1]:+.4f}")
    ok = bool(nid("inside", c) >= 5 and n["d_inside"] >= 0.01 and a - base >= -0.01 and not veto)
    r = dict(n, val=a, val_base=base, d_val=a - base, veto=veto, passed=ok)
    if pb:
        r.update(person_d_inside=pb["d_inside"], person_d_before=pb["d_before"])
    if pv:
        r.update(person_val=pv[0], person_val_base=pv[1], person_d_val=pv[0] - pv[1])
    extra = (f" | Person (copies) inside {pb['d_inside']:+.4f} before {pb['d_before']:+.4f}" if pb else "") + (f" val {pv[0] - pv[1]:+.4f}" if pv else "")
    print(f"{name} inside {n['inside'][0]:.4f} ({n['d_inside']:+.4f}, {nid('inside', c)} ids) | before {n['d_before']:+.4f} | val {a:.4f} ({a - base:+.4f})"
          f"{extra} | veto {'; '.join(veto) or 'none'} | {'PASS' if ok else 'fail'}", flush=True)
    return r


def joint_ps(where, c, ms):
    """Joint Person / Bicycle reading: the single reading's chain and gate on the joint lane's second stages."""
    v = "joint:" + "+".join(ms)
    ms = joint_members(c, v)
    if where == "val" and ms != passes(c):
        sys.exit(f"STOP joint {c}: members {ms} are not exactly the members that passed alone {passes(c)}")
    ids = M.S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": k} for b, k in vv] for f, vv in M.S.labels(ids).items()}
    keys = M.SZ.PP.static_keys("lumpi")
    cand, cp, person_of = _bases(ids, keys)
    name = f"{c}: {v}"
    cand.update(_cands(c, name, joint_files(c, ms), ids, keys, cp, person_of))
    if where == "near":
        assert len(ids) == 266
        HH = {h: M.rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in H}
        _near_checks(cand, gt, HH)
        res = {"members": ms, "lane": joint_tag(c, ms), "files": [str(p) for p in joint_files(c, ms)]}
        for k, r in _near_res(cand, gt, HH).items():
            if k == name or k == f"Person: copies from {name}":
                res[k] = r
                print(f"near {k} inside {r['inside'][0]:.4f} ({r['d_inside']:+.4f}) | before {r['before'][0]:.4f} ({r['d_before']:+.4f}) | "
                      f"AP@0.7 inside {r['inside'][1]:.4f}", flush=True)
        Path(f"logs/igm_joint_{c}_near.json").write_text(json.dumps(res, indent=1))
        print("wrote", f"logs/igm_joint_{c}_near.json", flush=True)
        return
    nr = json.loads(Path(f"logs/igm_joint_{c}_near.json").read_text())
    if nr.get("members") != ms:
        sys.exit("STOP the joint near reading is for other members")
    got = round(M.G3.ap(cand["Person: v59"][1], gt, ids, "Person"), 4)
    print(f"self-check val Person: v59 base {got} vs 0.48: {got == 0.48}", flush=True)
    if got != 0.48:
        sys.exit("STOP the val Person base does not reproduce v59")
    res = dict(nr)
    res[name] = _val_gate(c, name, cand, nr, gt, ids)
    Path(f"logs/igm_joint_{c}.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", f"logs/igm_joint_{c}.json", "| class takes:", chosen(c), flush=True)


def joint_selfcheck(argv):
    """joint selfcheck near|val <class> <m> <single lane chain file>: ONE member. (1) the joint lane (built here, split val) equals the single lane
    box for box for that class; (2) with the joint second-stage files = the member's own single files (linked by the dry-run script), the joint
    reading's numbers equal the single reading's (igm_read.build on the member) to 1e-9: AP@0.5 / 0.7 on inside and before (near), val AP and the
    gate (val). Writes nothing to logs/."""
    where, c, m, single_lane = argv[0], argv[1], argv[2], Path(argv[3])
    split = argv[4] if len(argv) > 4 else "val"   # test: lane check only (the build's test lane)
    ms = joint_members(c, f"joint:{m}", True)
    ids = M.S.ids_of("val")
    jl = M.G3.run_lane(joint_tag(c, ms), joint_spec(c, ms), {}, split)
    sl = json.loads(single_lane.read_text())
    same = set(jl) == set(sl) and M.same_boxes({f: [d for d in jl[f] if d["class"] == c] for f in jl},
                                              {f: [d for d in sl[f] if d["class"] == c] for f in sl}, sorted(sl))
    nb = sum(1 for v in jl.values() for d in v if d["class"] == c)
    print(f"selfcheck {where} {c} lane {joint_tag(c, ms)} ({split}) equals the single lane {single_lane} box for box ({len(jl)} frames, {nb} {c} boxes): {same}", flush=True)
    files = joint_files(c, ms)
    if split != "val" or not all(p.exists() for p in files):
        print(f"selfcheck {where} {c}: no second-stage files at {files[0].parent} (the member has none here): reading part not run", flush=True)
        if not same:
            sys.exit("STOP joint self-check: lanes differ")
        return
    gt = {f: [{"box": [float(x) for x in b], "class": k} for b, k in vv] for f, vv in M.S.labels(ids).items()}
    keys = M.SZ.PP.static_keys("lumpi")
    cand, cp, person_of = _bases(ids, keys)
    name = f"{c}: joint:{m}"
    cand.update(_cands(c, name, files, ids, keys, cp, person_of))
    single = M.build(m, M.tag(m, True), ids, (c,))   # the single reading's own code path
    sname = f"{c}: add {m}"
    pairs = [(name, sname)] + ([(f"Person: copies from {name}", f"Person: copies from {sname}")] if c == "Bicycle" else [])
    ok = same
    if where == "near":
        HH = {h: M.rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in H}
        for a, b in pairs:
            x = {h: M.ap2(cand[a][1], gt, HH[h], cand[a][0]) for h in H}
            y = {h: M.ap2(single[b][1], gt, HH[h], single[b][0]) for h in H}
            eq = all(abs(x[h][t] - y[h][t]) < 1e-9 for h in H for t in (0, 1))
            print(f"selfcheck near {a}: inside {x['inside'][0]:.6f} / before {x['before'][0]:.6f} equal the single {b}: {eq}", flush=True)
            ok = ok and eq
    else:
        for a, b in pairs:
            x, y = M.G3.ap(cand[a][1], gt, ids, cand[a][0]), M.G3.ap(single[b][1], gt, ids, single[b][0])
            eq = abs(x - y) < 1e-9
            print(f"selfcheck val {a}: {x:.6f} equals the single {b} {y:.6f}: {eq}", flush=True)
            ok = ok and eq
        p = Path(f"logs/igm_{m}_val.json")
        if p.exists() and sname in json.loads(p.read_text()):
            r = json.loads(p.read_text())[sname]
            eq = abs(M.G3.ap(cand[name][1], gt, ids, c) - r["val"]) < 1e-9
            print(f"selfcheck val {name}: equals the logged single reading {p} ({r['val']:.6f}): {eq}", flush=True)
            ok = ok and eq
    print(f"joint self-check {where} {c} ({m}): {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        sys.exit("STOP joint self-check failed")


# ------------------------------------------------------------------ v60 base: lumpi_v61 (fixed)
B60 = "v60"
SR, G2, MOTO2, MP1 = "safdnet_lumpi_rs", "g2", "moto2", "mp1"
STACKED = (f"{MOTO2}+{MP1}", f"{SR}+{MP1}")
ALIAS60 = {"SR": SR, f"SR+{MP1}": f"{SR}+{MP1}"}   # the tag as a switch value: stored under the member's name
BAR60 = 0.005   # conflict (a) for an SR / g2 class (and SR+mp1): val a - base >= +0.005 vs v60, unrounded (its own igm60 gate; as rs3)
_V60 = {}
PAIRS = tuple(sorted(M.PAIR, key=lambda m: ORDER.index(M.PAIR[m])))   # Wave 2B pairs, fixed order = their seed-1 order (dsvt, vmamba, pv)
PAIR_CLASSES = ("Car", "Van", "Truck", "Motorcycle")   # scope C: no Person / Bicycle
PAIR_STANDIN = {}   # self-check only (build_lumpi_v6x --pair-selfcheck): {pair: {class: (w seed 1, w seed 2)}}; seed 1's file stands in for seed 2


def pair_members(v):
    """A pair switch value -> its pairs (fixed order), else None: "<m>" (one pair) or "pairs:<m1>+<m2>[+...]" (the joint of 2+)."""
    if v in M.PAIR:
        return (v,)
    if isinstance(v, str) and v.startswith("pairs:"):
        ms = tuple(v[len("pairs:"):].split("+"))
        if len(ms) < 2 or len(set(ms)) != len(ms) or any(m not in M.PAIR for m in ms) or list(ms) != sorted(ms, key=PAIRS.index):
            sys.exit(f"STOP {v}: a pair joint needs 2+ distinct pairs in the fixed order {PAIRS}")
        return ms
    return None


def add_key(m, c):
    """The reading key of candidate m for class c: "<c>: add <m>"; a pair joint "pairs:<a>+<b>" -> "<c>: add <a>+<b>"."""
    ms = pair_members(m)
    return f"{c}: add {'+'.join(ms) if ms else m}"


def pair_file(ms, c, where):
    return Path(f"logs/pair60_{'+'.join(ms)}_{'moto' if c == 'Motorcycle' else 'cvt'}_{where}.json")


def pair_ok(c, m, sw60):
    """Scope C: pair m may switch class c on v60. Motorcycle: v60's Motorcycle is a joint holding m's seed 1 (the pair replaces it there); Car /
    Van / Truck: v60's class is at v59 or a single-member lane (the pair as a new member, G3 form)."""
    v = sw60[c]
    if c == "Motorcycle":
        return v.startswith("joint:") and M.PAIR[m] in v.split(":", 1)[1].split("+")
    return c in PAIR_CLASSES and v != RS3 and not v.startswith("joint:")


def pair_lane(c, sw60, ms):
    """(lane name, spec) of class c on v60 with the pair(s) ms. Motorcycle: v60's joint lane with each pair's seed-1 entry replaced in place by
    the pair's two entries (lane W2m<tags>); Car / Van / Truck: v60's lane for c (igm60.lane_of) + each pair's entries (lane IM<k><tags> on a v59
    class, else I6<k>_<v60 lane>_<tags>; one pair: igm60.g3_lane's). One pair: STOP unless equal to pair60.lanes_of's (its reading's lane and
    cache). PAIR_STANDIN (self-check only): its weights, seed 1's own file in place of seed 2's, lane name prefixed W2sc."""
    import igm60 as I6
    if any(not pair_ok(c, m, sw60) for m in ms):
        sys.exit(f"STOP {c}: {ms} is not a pair switch on v60's {c} ({sw60[c]})")
    sub = lambda m: m in PAIR_STANDIN
    ent = lambda m: [[f"file:g3in/{M.PAIR[m]}" if i == 1 and sub(m) else e[0], e[1], e[2]]
                     for i, e in enumerate(M.member(m, c, PAIR_STANDIN[m][c] if sub(m) else M.PAIR_W))]
    tags = "".join(M.TAGS[m] for m in ms)
    if c == "Motorcycle":
        spec = joint_spec(c, joint_members(c, sw60[c]))
        for m in ms:
            one = M.member(M.PAIR[m], c)
            if len(one) != 1 or spec.count(one[0]) != 1:
                sys.exit(f"STOP Motorcycle: {M.PAIR[m]}'s entry {one} is not exactly once in v60's joint spec")
            i = spec.index(one[0])
            spec = spec[:i] + ent(m) + spec[i + 1:]
        name = f"W2m{tags}"
    else:
        nm, base = I6.lane_of(c, sw60[c])
        spec = base + [e for m in ms for e in ent(m)]
        name = f"IM{I6.LANE_KEY[c]}{tags}" if sw60[c] == "v59" else f"I6{I6.LANE_KEY[c]}_{nm}_{tags}"
    if any(sub(m) for m in ms):
        return "W2sc" + name, spec
    if len(ms) == 1:
        import pair60
        L = pair60.lanes_of(ms[0], sw60, "moto" if c == "Motorcycle" else "cvt")[c]
        if (L["name"], L["spec"]) != (name, spec):
            sys.exit(f"STOP {c}: pair_lane ({name}) differs from pair60.lanes_of ({L['name']}), the lane its reading used")
    return name, spec


def v60_base():
    """-> (v60's switches over igm60.SIX, info): logs/build_lumpi_v60.json validated by igm60.v60_set (never hardcoded); read once."""
    if "sw" not in _V60:
        import igm60
        _V60["sw"], _V60["info"] = igm60.v60_set(False, None)
    return _V60["sw"], _V60["info"]


def ref_of(sw):
    """The reference of a switch set: v59's (every class v59) or, for a v60-base set (sw["base"] == "v60"), v60's own switches."""
    return dict(v60_base()[0]) if sw.get("base") == B60 else {c: "v59" for c in SW6}


def cands60(c, sw60):
    """The v60-base candidates class c may take instead of v60's switch (igm60's definitions: SR's Person / Bicycle only while v60 keeps them at
    v59, SR's Van / Truck not on an rs3 class; g2 only on v60's (= v59's) Bicycle lane)."""
    out = []
    if (c in ("Car", "Motorcycle") or c in M.RS3_CLASSES and sw60[c] != RS3
            or c in ("Person", "Bicycle") and sw60[c] == "v59"):
        out.append(SR)
    if c == "Bicycle" and sw60["Bicycle"] == "v59":
        out.append(G2)
    if c == "Motorcycle":
        out += [MOTO2, MP1, *STACKED]
    ps = [m for m in PAIRS if pair_ok(c, m, sw60)]   # Wave 2B pairs (scope C): one pair, or the joint of 2+ (fixed order)
    out += ps + ["pairs:" + "+".join(x) for r in range(2, len(ps) + 1) for x in combinations(ps, r)]
    return tuple(out)


def bar60(v):
    return pair_members(v) is not None or v.split("+")[0] in (SR, G2)   # SR / g2 / pair classes: val a - base >= +0.005 vs v60


def parse60(argv):
    """--base v60 [--<class> <v>]... [--fallback C]: v60's switches with the given classes replaced by v60-base candidates (SR = safdnet_lumpi_rs).
    -> (sw with "base": "v60" and Truck only when not v59, fallback)."""
    sw60, _ = v60_base()
    sw, fb, base = {c: sw60[c] for c in SW6}, None, None
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--base":
            base = argv[i + 1]
        elif k == "--fallback":
            fb = argv[i + 1]
        elif k.startswith("--") and k[2:].capitalize() in SW6:
            sw[k[2:].capitalize()] = ALIAS60.get(argv[i + 1], argv[i + 1])
        else:
            sys.exit(f"STOP unknown argument {k}")
        i += 2
    if base != B60:
        sys.exit(f"STOP --base {base}: only v60 (lumpi_v61); the v59 path takes no --base")
    for c in SW6:
        if sw[c] != sw60[c] and sw[c] not in cands60(c, sw60):
            sys.exit(f"STOP {c} = {sw[c]}: on the v60 base a class keeps v60's switch ({sw60[c]}) or takes a v60-base candidate {cands60(c, sw60)}")
    if fb is not None and (fb not in SW or sw[fb] == sw60[fb]):
        sys.exit(f"STOP --fallback {fb}: not a class this set changes against v60")
    if sw["Truck"] == "v59":
        del sw["Truck"]
    sw["base"] = B60
    return sw, fb


def file60(m, where):
    return Path(f"logs/igm60_{m}_{where}.json")


def mp1_json():
    """MP1's output (run_mp_experiments.py --mode mp1 --build-log logs/build_lumpi_v60.json writes mp1_<slug>_results.json, slug = v60's
    Motorcycle switch with ':' and '+' as '_'), copied into logs/ of the workspace that reads."""
    slug = v60_base()[0]["Motorcycle"].replace(":", "_").replace("+", "_")
    return Path(f"logs/mp1_{slug}_results.json")


def find_reading60(m, c, where):
    """The v60-base single-reading file holding class c of candidate m, or None: logs/igm60_<m>_<where>.json (SR, g2, moto2, stacked <w>+mp1),
    MP1's json for mp1 (Motorcycle)."""
    if m == MP1:
        p = mp1_json()
        return p if c == "Motorcycle" and p.exists() else None
    ms = pair_members(m)
    p = pair_file(ms, c, where) if ms else file60(m, where)
    return p if p.exists() and add_key(m, c) in json.loads(p.read_text()) else None


def reading60(m, c, where):
    """v60-base single reading of class c by candidate m (None if none). igm60's own entry "<c>: add <m>" (its "_v60" block must name this v60: the
    same switches and combo key); mp1: MP1's json mapped to the same fields (near: inside / before [AP@0.5, AP@0.7], d_inside, d_before; val: val,
    val_base, d_val, d_inside, d_before, the Person copy deltas person_d_inside / person_d_before / person_d_val with person_val / person_val_base,
    passed = its dual gate), its switches checked against v60's."""
    p = find_reading60(m, c, where)
    if p is None:
        return None
    sw60, info = v60_base()
    j = json.loads(p.read_text())
    if m == MP1:
        j = j.get("MP1_preregistered", {})
        want = {k: sw60[k] for k in ("Person", "Bicycle", "Motorcycle")}
        if j.get("switches") != want:
            sys.exit(f"STOP {p}: MP1 was read on switches {j.get('switches')}, not v60's {want}")
        if where == "near":
            return {"inside": j["moto_inside_mp1"], "before": j["moto_before_mp1"], "d_inside": j["d_moto_inside"], "d_before": j["d_moto_before"]}
        r = {"val": j["moto_val_mp1"][0], "val_base": j["moto_val_base"][0], "d_val": j["d_moto_val"], "d_inside": j["d_moto_inside"],
             "d_before": j["d_moto_before"], "passed": bool(j["dual_gate_pass"])}
        pf = {"person_d_inside": "d_person_inside", "person_d_before": "d_person_before", "person_d_val": "d_person_val", "person_val": "person_val_mp1",
              "person_val_base": "person_val_base"}   # MP1 writes them all; one missing STOPs in stack_gate60, the only consumer
        r.update({k: j[x][0] if isinstance(j[x], list) else j[x] for k, x in pf.items() if x in j})
        return r
    v = j.get("_v60", {})
    if v.get("assume_v59") or v.get("switches") != sw60 or v.get("key") != info.get("key"):
        sys.exit(f"STOP {p} was read on another v60 switch set ({v.get('switches')}, key {v.get('key')}), not {sw60} / {info.get('key')}")
    return j[add_key(m, c)]


def mp1_reading():
    p = mp1_json()
    if not p.exists():
        sys.exit(f"STOP {p} missing (copy MP1's json into logs/)")
    return json.loads(p.read_text())["MP1_preregistered"]


STACK_KEYS = ("d_inside", "d_before", "d_val", "person_d_inside", "person_d_before", "person_d_val")


def base_numbers60(m, where):
    """The v60 base numbers a Motorcycle reading of candidate m was taken against: AP@0.5 of v60's Motorcycle and Person (with v60's copies) on
    inside / before (near) or val. Two readings with equal base numbers (1e-9) were read on the same split and frames. igm60's format:
    near "Motorcycle: v60" / "Person: v60", val val_base / person_val_base; MP1: its moto_* / person_*_base."""
    p = find_reading60(m, "Motorcycle", where)
    try:
        j = json.loads(p.read_text())
        if m == MP1:
            j = j["MP1_preregistered"]
            return {(k, h): j[f"{x}_{h}_base"][0] for h in (H if where == "near" else ("val",)) for k, x in (("Motorcycle", "moto"), ("Person", "person"))}
        if where == "near":
            return {(k, h): j[f"{k}: v60"][h][0] for k in ("Motorcycle", "Person") for h in H}
        r = j[f"Motorcycle: add {m}"]
        return {("Motorcycle", "val"): r["val_base"], ("Person", "val"): r["person_val_base"]}
    except (KeyError, TypeError, AttributeError) as e:
        sys.exit(f"STOP {p}: no v60 base number {e} (igm60's format: near 'Motorcycle: v60' / 'Person: v60', val val_base / person_val_base)")


def better_single60(w):
    """The better single of {winner w, mp1}: the larger d_inside vs v60, then the larger d_val (4 decimals); a full tie keeps the winner."""
    k = lambda r: (round(r["d_inside"], 4), round(r["d_val"], 4))
    rw, rp = reading60(w, "Motorcycle", "val"), reading60(MP1, "Motorcycle", "val")
    if rw is None or rp is None:
        sys.exit(f"STOP Motorcycle: the stack gate needs the val readings of {w} and mp1")
    return MP1 if k(rp) > k(rw) else w


def stack_gate60(st):
    """The stacked <w>+mp1 against the better single of {w, mp1} with moto2's Motorcycle gate, on the difference of the two v60-base
    readings, (stack - v60) - (single - v60), per split, after checking both were read on the same splits and frames (equal v60 base numbers on
    near-test and val, 1e-9; STOP otherwise): Motorcycle inside >= +0.01 (>= 5 inside ids); val not down (igm_read.moto_val_ok);
    Motorcycle before not down > 0.005 (although it has 3 ids); the Person copy veto (-0.005 on inside / before, segments with >= 5 ids); Person val
    with the copies not down > 0.01. -> (passed, the better single, {delta: value}, [failed parts])"""
    w = st.split("+")[0]
    single = better_single60(w)
    for where in ("near", "val"):
        if reading60(st, "Motorcycle", where) is None:
            sys.exit(f"STOP Motorcycle: no stacked reading of {st} ({file60(st, where)})")
        a, b = base_numbers60(st, where), base_numbers60(single, where)
        if set(a) != set(b) or any(abs(a[x] - b[x]) >= 1e-9 for x in a):
            sys.exit(f"STOP Motorcycle: the stacked reading {st} and the single {single} were not read on the same {where} split / frames "
                     f"(v60 base numbers {a} vs {b})")
    rs, r1 = reading60(st, "Motorcycle", "val"), reading60(single, "Motorcycle", "val")
    miss = [x for x in STACK_KEYS if x not in rs or x not in r1]
    if miss:
        sys.exit(f"STOP Motorcycle: the readings of {st} / {single} lack {miss} (igm60's Motorcycle val fields)")
    d = {x: rs[x] - r1[x] for x in STACK_KEYS}
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    nid = lambda h, c: cnt.get(h, {}).get(c, {}).get("distinct_ids", 0)
    fails = []
    if not (nid("inside", "Motorcycle") >= 5 and d["d_inside"] >= 0.01):
        fails.append(f"Motorcycle inside {d['d_inside']:+.4f} ({nid('inside', 'Motorcycle')} ids)")
    if not M.moto_val_ok(d["d_val"], 0.0):
        fails.append(f"Motorcycle val {d['d_val']:+.4f} (val not down)")
    if d["d_before"] < -0.005:
        fails.append(f"Motorcycle before {d['d_before']:+.4f}")
    for h in H:
        if nid(h, "Person") >= 5 and d[f"person_d_{h}"] < -0.005:
            fails.append(f"Person (copies) {h} {d[f'person_d_{h}']:+.4f}")
    if d["person_d_val"] < -0.01:
        fails.append(f"Person val (copies) {d['person_d_val']:+.4f}")
    return not fails, single, d, fails


def final_choice60(c):
    """The fixed v60-base choice with the Wave 2B pairs on top (scope C; the fixed member choice among the pairs, fixed before any pair reading): the rule
    of _final_choice60_v60 unchanged; then the pairs that passed class c: none -> that rule's choice; any while that rule picked a candidate -> STOP
    (no fixed order between them: needs a manual decision); one -> that pair; 2+ -> "pairs:<passers>" if its joint reading passed (it must exist, near and val:
    STOP until it does), else the passer with the larger d_inside, then d_val (4 decimals), then the fixed order PAIRS."""
    x = _final_choice60_v60(c)
    sw60, _ = v60_base()
    ok = lambda m: (lambda r: r if r and r.get("passed") else None)(reading60(m, c, "val"))
    ps = [m for m in PAIRS if pair_ok(c, m, sw60) and ok(m)]
    if not ps:
        return x
    if x != sw60[c]:
        sys.exit(f"STOP {c}: the v60-base rule picks {x} and the pair(s) {ps} passed too: no fixed rule between them (needs a manual decision)")
    if len(ps) == 1:
        return ps[0]
    jv = "pairs:" + "+".join(ps)
    if reading60(jv, c, "near") is None or reading60(jv, c, "val") is None:
        sys.exit(f"STOP {c}: {len(ps)} pairs passed ({ps}); run their joint reading first: {pair_file(tuple(ps), c, 'near')} / "
                 f"{pair_file(tuple(ps), c, 'val')}")
    if reading60(jv, c, "val").get("passed"):
        return jv
    sv = {m: ok(m) for m in ps}
    return sorted(ps, key=lambda m: (-round(sv[m]["d_inside"], 4), -round(sv[m]["d_val"], 4), PAIRS.index(m)))[0]


def _final_choice60_v60(c):
    """The fixed v60-base choice. Bicycle: g2 and SR both passed -> the larger d_inside, then the larger d_val (4
    decimals), a full tie g2; one -> that one. Motorcycle: SR vs moto2 the same way, a full tie moto2; that winner and mp1 both passed -> the stacked
    <w>+mp1 (its reading must exist: STOP until it does) if it passes stack_gate60 against the better single of {winner, mp1}, else that better
    single (no STOP); one of them -> that one. Person / Car / Van / Truck: SR if its reading passed. Nothing passed: v60's switch."""
    sw60, _ = v60_base()
    ok = lambda m: (lambda r: r if r and r.get("passed") else None)(reading60(m, c, "val"))
    k = lambda r: (round(r["d_inside"], 4), round(r["d_val"], 4))
    pick = lambda a, b, ra, rb: b if k(rb) > k(ra) else a   # a keeps a full tie
    if c == "Bicycle":
        g, s = ok(G2), ok(SR)
        return pick(G2, SR, g, s) if g and s else G2 if g else SR if s else sw60[c]
    if c != "Motorcycle":
        return SR if SR in cands60(c, sw60) and ok(SR) else sw60[c]
    s, o, p = ok(SR), ok(MOTO2), ok(MP1)
    w = pick(MOTO2, SR, o, s) if o and s else MOTO2 if o else SR if s else None
    if w and p:
        st = f"{w}+{MP1}"
        if reading60(st, c, "near") is None or reading60(st, c, "val") is None:
            sys.exit(f"STOP Motorcycle: {w} and mp1 both passed: run the stacked reading of {st} first ({file60(st, 'near')} / {file60(st, 'val')})")
        passed, single, d, fails = stack_gate60(st)
        print(f"Motorcycle stack gate: {st} vs the better single {single}: "
              + ", ".join(f"{x} {v:+.4f}" for x, v in d.items()) + f" | {'PASS: ' + st if passed else 'fail (' + '; '.join(fails) + '): ' + single}",
              flush=True)
        return st if passed else single
    return w or (MP1 if p else sw60[c])


def info60(sw):
    """What a v60-base reading records of its base: v60's build json (path, md5) and combo key; MP1's json (path, md5) when mp1 is read."""
    _, info = v60_base()
    out = {"v60": {k: info.get(k) for k in ("build_json", "build_md5", "key")}}
    if sw.get("Motorcycle") == MP1:
        p = mp1_json()
        out["mp1_json"] = {"path": str(p), "md5": hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None}
    return out


def mp1(moto, load, keys):
    """MP1, copied as a pure function from the reference run_mp_experiments.py (mode mp1: apply_b1_cls, then apply_p3_cls; "B1 + P3 before
    W.copies"), on {frame: Motorcycle boxes}: B1 (lumpi_rider) a box >= 0.05 whose point height span (2-98 %) is < 1.35 m -> score x 0.3; then P3
    (postproc_p123.inbox_frac) every box with >= 0.8 of its points in the static map -> score x 0.3. -> (boxes, B1 down-weighted, P3 down-weighted)"""
    import lumpi_rider as LR
    b1, n1 = {}, 0
    for f, v in moto.items():
        P = None
        dd = []
        for d in v:
            if d["class"] == "Motorcycle" and d["score"] >= 0.05:
                P = load(f) if P is None else P
                s = LR.span(d, P)
                if s is not None and s < LR.H:
                    d = {**d, "score": d["score"] * LR.F}
                    n1 += 1
            dd.append(d)
        b1[f] = dd
    out, n3 = {}, 0
    for f, v in b1.items():
        bx = [d for d in v if d["class"] == "Motorcycle"]
        fr = M.SZ.PP.inbox_frac(load(f), [d["box"] for d in bx], keys) if bx else []
        it = iter(fr)
        o = []
        for d in v:
            if d["class"] == "Motorcycle":
                x = next(it)
                if x is not None and x >= 0.8:
                    d = {**d, "score": d["score"] * 0.3}
                    n3 += 1
            o.append(d)
        out[f] = o
    return out, n1, n3


def mp1_fields(sw, where):
    """MP1 alone (v60's Motorcycle + MP1): MP1's own numbers the combo recomputes, (class, segment, "v60" | "new") -> AP@0.5: Motorcycle (base and
    MP1) on inside / before (near) or val, and Person (base and with MP1's copies) while Person and Bicycle stay v60's. {} otherwise."""
    if sw.get("base") != B60 or sw.get("Motorcycle") != MP1:
        return {}
    sw60, _ = v60_base()
    j = mp1_reading()
    out = {}
    for h in (H if where == "near" else ("val",)):
        out[("Motorcycle", h, B60)], out[("Motorcycle", h, "new")] = j[f"moto_{h}_base"][0], j[f"moto_{h}_mp1"][0]
        if sw["Person"] == sw60["Person"] and sw["Bicycle"] == sw60["Bicycle"]:
            out[("Person", h, B60)], out[("Person", h, "new")] = j[f"person_{h}_base"][0], j[f"person_{h}_mp1"][0]
    return out


def classes_built60(sw, ids, where, gt, HH=None):
    """v60-base sets. base = lumpi_v60's six classes by igm60.bases60 (classes_built on v60's switches; box-for-box self-checks), then igm60's AP
    self-checks (check_near: v59 constants and the combo reading of v60's switches; check_val: also logs/build_lumpi_v60.json val, 1e-9). new =
    base with each switched class built by igm60's candidate code on the same lanes / cache files / second-stage files as its reading: SR lanes by
    igm60.g3_lane (Van / Truck: one rule37 + T5 / V5 pass on the set's two sources), g2 / moto2 by igm60.g2_cands / moto2_cands, mp1 = mp1() on
    v60's Motorcycle or on the stacked winner's; Person = its own chain (v60's, or SR's five second stages) + copies from the new Bicycle and
    Motorcycle (W.copies), as classes_built. -> (base, new)"""
    import igm60 as I6
    sw60, _ = v60_base()
    _, info = I6.v60_set(False, where)   # with the combo reading of v60's switches for the AP self-checks
    cand, ctx = I6.bases60(sw60, ids, where)
    if where == "near":
        I6.check_near(sw60, info, cand, ctx, gt, HH)
    else:
        I6.check_val(sw60, info, cand, gt, ids, False)
    base = {c: cand[f"{c}: v60"][1] for c in I6.SIX}
    del cand
    new = dict(base)
    keys, t, v = ctx["keys"], M.TAGS[SR], lambda c: sw.get(c, "v59")
    def lane(c):   # the switched class's lane: SR's (igm60.g3_lane) or a pair's (pair_lane); the cache files its reading used
        ms = pair_members(v(c))
        return M.G3.run_lane(*(pair_lane(c, sw60, ms) if ms else I6.g3_lane(c, sw60[c], SR, t)), {}, "val")
    if v("Car") != sw60["Car"]:
        new["Car"] = M.G3.lane_class(lane("Car"), "Car", ids)
    tv = [c for c in ("Truck", "Van") if v(c) != sw60[c]]
    if tv:   # igm60's G3 Van / Truck: the class's lane + SR, the other class v60's source; both switched: one pass with both SR lanes
        src = dict(ctx["src"])
        for c in tv:
            ch = lane(c)
            src[c] = {f: [d for d in ch.get(f, []) if d["class"] == c] for f in ids}
            del ch
        pair = M.tv_pass(src["Truck"], src["Van"], ids)
        for c in tv:
            new[c] = pair[c]
        del src, pair
    if v("Bicycle") == G2:
        out = {}
        I6.g2_cands(out, sw60, ids, G2, ctx, where)
        if "Bicycle: add g2" not in out:
            sys.exit("STOP Bicycle = g2: second-stage files missing (igm60.G2_FILES)")
        new["Bicycle"] = out["Bicycle: add g2"][1]
    elif v("Bicycle") != sw60["Bicycle"]:   # SR: L11b's second stage on SR's Bicycle lane, R4 >= 0.05, P3 (igm60.g3_cands)
        bf = Path(f"output/lumpi/{SR}/igm/bicycle_val_ref.json")
        if not bf.exists():
            sys.exit(f"STOP Bicycle = {SR}: {bf} missing")
        new["Bicycle"] = M.IB.bike(bf, ids, keys)
    mv = v("Motorcycle")
    if mv != sw60["Motorcycle"]:
        w = mv.split("+")[0]
        if pair_members(mv):   # a pair (or the joint of pairs): v60's joint with the seed-1 entries replaced (pair_lane, lane W2m<tags>)
            mo = M.G3.lane_class(lane("Motorcycle"), "Motorcycle", ids)
        elif w == MOTO2:
            out = {}
            I6.moto2_cands(out, sw60, ids, MOTO2, ctx, where)
            if "Motorcycle: add moto2" not in out:
                sys.exit("STOP Motorcycle = moto2: second-stage files missing (igm60.MOTO2_FILES)")
            mo = out["Motorcycle: add moto2"][1]
        elif w == SR:   # v60's Motorcycle lane + SR: lane I6m_<v60 lane>_SR (igm60.g3_lane; the cache name the reading used)
            mo = M.G3.lane_class(lane("Motorcycle"), "Motorcycle", ids)
        else:   # mp1 alone: on v60's Motorcycle
            mo = base["Motorcycle"]
        if mv == MP1 or mv.endswith(f"+{MP1}"):
            mo, n1, n3 = mp1(mo, M.LOAD, keys)
            print(f"MP1 on Motorcycle ({mv}): B1 down-weighted {n1}, P3 down-weighted {n3}", flush=True)
        new["Motorcycle"] = mo
    if v("Person") != sw60["Person"]:   # SR: lumpi_v51's five second stages on SR's Person lane, R0, grow + length rule, P3
        d = Path(f"output/lumpi/{SR}/igm")
        pf = [d / f"person_{r}_val_ref.json" for r in M.REF_ORDER]
        miss = [str(p) for p in pf if not p.exists()]
        if miss:
            sys.exit("STOP Person = SR: second-stage files missing: " + ", ".join(miss))
        own = M.G3.person(pf, ids, M.LOAD, keys)
        new["Person"] = {f: own[f] + M.W.copies({f: new["Bicycle"][f] + new["Motorcycle"][f]})[f] for f in ids}
    elif v("Bicycle") != sw60["Bicycle"] or mv != sw60["Motorcycle"]:   # v60's own chain + copies recomputed (igm60's person_with)
        new["Person"] = ctx["person_with"](new["Bicycle"], new["Motorcycle"])
    return base, new


def main(argv):
    {"near": near, "val": val}[argv[0]](argv[1:])
