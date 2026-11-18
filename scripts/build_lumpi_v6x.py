"""Next LUMPI test file from lumpi_v59 with per-class switches for the four classes read on the inside gate
(readings by scripts/igm_read.py, logs/igm_<m>_val.json). CPU only, no new settings, nothing searched.
Every class is lumpi_v59's box for box unless its switch names something else; the realism frames stay v59's (output/lumpi/realism_v2, checked
byte-identical to the v59 zip's). Per class, each switch value maps to exactly the chain igm_read.py read for that class and member:
  Car      v59     v59's Car lane: the v29 lane + vrcnn_lumpi_rs at weight 1 (G3 lane "IGc00"), save_lumpi_chain, rule37
           v54     lumpi_v54's Car (lumpi_v41 chain on the v29 boxes, rule37; output/lumpi/ens_x/lumpi_p3_both_test.json): a manual choice
           <m>     v59's Car lane + the member for Car (G3 lane "IMc<tag>"), save_lumpi_chain, rule37
  Van      v59     the v29 lane's Van + the v29 Truck, rule37, rules T5 / V5 (ig_read.van on the L11-refined v29 file)
           <m>     the v29 lane + the member for Van (G3 lane "IMv<tag>"), then the same Van steps with the v29 Truck
  Person   v59     R0 of lumpi_v51's five second stages on the v29 boxes, grow + length rule, P3, plus the two-wheeler copies
           <m>     one-stage members: the five second stages run on the member's Person lane (output/lumpi/<m>/s2/person_<r>_<split>_ref.json;
                   val reads igm/, the copies of the route's s2 files), same chain, plus the copies; pv_lumpi_s2: its own Person second stage
                   (s2/v29_<split>_min01.json) as a sixth member of v59's chain
  Bicycle  v59     L11b's second stage on the v29 lane, R4 at >= 0.05, P3
           <m>     L11b's second stage on the member's Bicycle lane (s2/bicycle_<split>_ref.json), R4 at >= 0.05, P3; Person then gets the copies
                   recomputed from the new Bicycle and v59's Motorcycle (lumpi_copies_routes.copies: every Bicycle / Motorcycle box >= 0.3 also
                   becomes a Person box, same box, score x 0.3, after P3)
  Motorcycle v59  the v29 file's Motorcycle (no second stage)
  Truck    v59     the v29 Truck + the v29 Van, rule37, rules T5 / V5 (igm_read.truck on the L11-refined v29 file; route_read_lumpi.Chain.cls)
           rs3     (also --van rs3) R6 of the 3 RS refiners (vrcnn_lumpi_rs, vrcnn_lumpi_c_s2, pv_lumpi_s2;
                   output/lumpi/rs3/<r>_<split>_ref.json, x6_refine on lumpi_v29_<split>.json) per v29 box >= 0.05, else the v29 box and score;
                   then rule37 + T5 / V5 with the other class from the v29 file (igm_read.rs3_fuse / tv_pass). Needs logs/igm_rs3_val.json's PASS;
                   val and test files must match the rs3_tv run's logs/rs3_refine.md5; on val the class equals igm_read's rs3 candidate box for box.
           <m>     the v29 lane + the member at weight 1 for Motorcycle (G3 lane "IMm<tag>"), no second stage (dsvt_lumpi, vmamba_lumpi, pv_lumpi_s2;
                   a board-informed class choice); Person then gets the copies recomputed from the chosen Bicycle and the new Motorcycle
  Final combination: when any class changes, the build needs the PASS of igm_read.py's combo reading for the
  switches it writes (logs/igm_combo_<key>_val.json): no PASS, nothing written. On a conflict that reading names the one fallback; the rebuild
  gives the same switches plus --fallback <class> (that class back to v59; its own combo reading must pass). If that also conflicts, its reading
  names --only <class>: the other interacting classes (Person's own change, Bicycle, Motorcycle) go back to v59, non-interacting changes stay,
  and no more readings. Car v54 is a manual choice with its own reading: the combo key reads Car as v59.
  joint:<a>+<b>    Car / Van / Motorcycle: the v29 / v59 lane + all members at weight 1 (lane zJ_<class>_<tags>, member order fshnet_lumpi_s2,
                   dsvt_lumpi, vmamba_lumpi, pv_lumpi_s2; reading logs/igm_joint_<class>.json; the joint rule). Any changed class must take the fixed member choice (igm_combo.chosen).
  joint:<a>+<b>    Person / Bicycle: the single chain on the joint lane's second stages (scripts/igm_joint_refine.sh;
                   test files output/lumpi/joint/<tag>/s2/<p>_test_ref.json, val read from igm/ = that run's s2 val files); Person: PV s2 as the sixth.
Members: fshnet_lumpi_s2 (zF2 form: fshnet_lumpi 0.5 + fshnet_lumpi_s2 0.5), dsvt_lumpi, vmamba_lumpi, pv_lumpi_s2 (igm_read.member / TAGS).
A member is accepted only if logs/igm_<m>_val.json marks "<Class>: add <m>" passed; its test-side inputs must exist, else the script refuses and
names the files (the 3090 sources in brackets): output/lumpi/g3in/<m>_test.json (<m>/kit/pred_test.json; fshnet_lumpi_s2 also
g3in/fshnet_lumpi_test.json), output/lumpi/<m>/s2/person_<r>_test_ref.json, s2/v29_test_min01.json (pv_lumpi_s2), s2/bicycle_test_ref.json.
Checks (STOP on any miss, nothing written):
  - v59 bases rebuilt from their chains: val Car 0.896 / Person 0.48 (as igm_read), realism_check 160 Car 0.9002 / Person 0.4923
    (logs/build_lumpi_v59.json); on test each class left at v59 among the four, rebuilt from its chain, equals v59 box for box;
  - each changed class's val from the combined predictions equals its igm reading (1e-9); fresh: igm_read logs no fresh reading, so a changed
    class STOPs here unless --val-only is given (a manual decision); Person with copies recomputed for a Bicycle change has no val reading (igm reads
    it on near-test only): printed for information under --val-only;
  - Car v54: test equals lumpi_v54's Car box for box; val 0.8858 / realism_check 0.8891 (logs/build_lumpi_l11rs.json);
  - every class not switched (Truck, Bus, Motorcycle, Unknown and the four at v59) equals v59 box for box on test.
Writes <outdir>/<name>_test.json and <outdir>/build_<name>.json (defaults output/lumpi/ens_x and logs/).
    python scripts/build_lumpi_v6x.py --name lumpi_v60 --car v59 --van v59 --person v59 --bicycle v59 [--truck v59|rs3] [--van rs3]
    python scripts/build_lumpi_v6x.py --name lumpi_v6x_dryrun --outdir submissions/lumpi_v6x_dryrun   (every switch at v59: must equal v59)
Inputs read for a member are md5-asserted (no silent skip): a read val file (igm/) must exist and equal the route's s2 val file,
and every file a fetch / stage record names must match it (logs/rs3_refine.md5, g2_bike_refine.md5, moto2_refine.md5, safdnet_lumpi_rs_s2.md5,
the "-> <file> <md5> : match" lines of logs/g3in_test_stage.log).
v60 base (lumpi_v61): --base v60. Every class not given keeps lumpi_v60's switch (logs/build_lumpi_v60.json, never
hardcoded); a class given another value takes a v60-base candidate (igm_combo.cands60: safdnet_lumpi_rs / SR on any of the six classes, g2 on
Bicycle, moto2 / mp1 / moto2+mp1 / safdnet_lumpi_rs+mp1 on Motorcycle), built on each split by igm60's candidate code (the same lanes, cache names
and second-stage files as the readings; MP1 = igm_combo.mp1, a pure copy of the reference MP1 code in run_mp_experiments.py). Needs the PASS of the combo reading of the set
(igm_read.py combo near|val --base v60 ...) and each candidate's fixed choice (igm_combo.final_choice60). Checks: lumpi_v60_test.json has md5
V60_MD5; v60's six classes rebuilt from its chains equal it box for box on test, and their val equals logs/build_lumpi_v60.json "val" (1e-9);
every class unchanged vs v60 is spliced from lumpi_v60_test.json and must equal it box for box (STOP otherwise); MP1 alone: its test counts
equal MP1's own json. New inputs need a record: output/lumpi/g3in/safdnet_lumpi_rs_test.json (3090 output/lumpi/safdnet_lumpi_rs/kit/pred_test.json)
a "-> ... : match" line in logs/g3in_test_stage.log; SR's Person / Bicycle s2 files a line in logs/safdnet_lumpi_rs_s2.md5.
    python scripts/build_lumpi_v6x.py --name lumpi_v61 --base v60 --motorcycle moto2 --bicycle g2 --val-only
Wave 2B pairs (scope C: Car / Van / Truck / Motorcycle; igm_combo's pair section): --base v60 --motorcycle <pair> (v60's joint with the pair's seed-1
entry replaced in place) and --car / --van / --truck <pair> (the pair as a new member on v60's lane, G3 form), or "pairs:<m1>+<m2>" (the pairs' joint),
each on the lanes igm_combo.pair_lane names (the readings' caches; one pair = pair60.lanes_of). Needs the pair's reading and fixed choice
(final_choice60) as any v60-base candidate; inputs: every g3in/<x>_test.json of the lane, a pair's own with its pair_chain.sh stage record.
    python scripts/build_lumpi_v6x.py --name pair_selfcheck --base v60 --pair-selfcheck vmamba_lumpi_s2|dsvt_lumpi_s2|pv_lumpi_rs
        wiring self-check (seed 1's own file stands in for seed 2): Motorcycle at 1.0 + 0.0 and Car / Van / Truck at 0.0 + 0.0, built on test
        through build60, must equal lumpi_v60_test.json (06c70a89) box for box, class by class; no reading, nothing written."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ensemble_rules as E
import g3_search as G3
import ig_read as IG
import igm60 as I6
import igm_combo as IC
import igm_read as IGM
import lumpi_copies_routes as W
import lumpi_static as S
import lumpi_v37_rules as R
import route_read_lumpi as RL
import size_check_lumpi as SZ
from utl import guard

O = Path("output/lumpi/ens_x")
V59, V54 = O / "lumpi_v59_test.json", O / "lumpi_p3_both_test.json"
V60, V60_MD5 = O / "lumpi_v60_test.json", "06c70a896b1b24a8a839bb4215e1d9db"   # the v60 submission file (logs/build_lumpi_v60.log)
REC_FILES = ("logs/rs3_refine.md5", "logs/g2_bike_refine.md5", "logs/moto2_refine.md5", "logs/safdnet_lumpi_rs_s2.md5")   # md5sum lines
G3IN_REC = Path("logs/g3in_test_stage.log")   # "... -> output/lumpi/g3in/<m>_test.json <md5> : match"
_REC, _MD5 = {}, {}
CLASSES4 = ("Car", "Van", "Person", "Bicycle")
SW5 = CLASSES4 + ("Motorcycle",)
SW6 = SW5 + ("Truck",)   # Truck: v59 or rs3 (Truck / Van RS second stages)
RS3 = "rs3"
_RS3 = {}   # (split, frames) -> igm_read.rs3_fuse output (fused once); "md5": rs3 inputs checked
MEMBERS = tuple(IGM.TAGS)
LOAD = {"val": lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3],
        "fresh": lambda f: np.load(f"data/pcdet/lumpi_real_fresh50k/points/{f}.npy")[:, :3],
        "test": lambda f: np.fromfile(f"data/kit/lumpi/detection_test_frames/{f}.bin", np.float32).reshape(-1, 3)}   # postproc_p123's P3 loaders
CHK = sorted(l.strip() for l in open("data/pcdet/lumpi_real/ImageSets/realism_check.txt") if l.strip())
keep = lambda d: {"box": [float(x) for x in d["box"]], "score": float(d["score"]), "class": d["class"]}
canon = lambda v: sorted(json.dumps([round(x, 6) for x in d["box"]] + [round(d["score"], 8), d["class"]]) for d in v)
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
ok = True


def chk(name, c):
    global ok
    ok &= bool(c)
    print(f"{name}: {bool(c)}", flush=True)


def refuse(msg):
    sys.exit(f"REFUSED {msg}")


def cls_of(p, c, ids):
    return {f: [keep(d) for d in p.get(f, []) if d["class"] == c] for f in ids}


# ------------------------------------------------------------------ chains (split in val / fresh / test; igm_read's code with the split open)
def lref(split):
    return json.loads((RL.L / f"v29_{split}_ref.json").read_text())


def van(chain, lr, ids, split):
    """ig_read.van with the point split open (ig_read fixes it to "val")."""
    tv = {f: [R.rule37(RL.keep(d)) for d in chain.get(f, []) if d["class"] == "Van"] + [R.rule37(RL.keep(d)) for d in lr[f] if d["class"] == "Truck"]
          for f in ids}
    tv = R.apply(tv, R.features(split, tv, ids), ["T5", "V5"])
    return {f: [d for d in tv[f] if d["class"] == "Van"] for f in ids}


def rs3_fused(split, ids, lr):
    """igm_read.rs3_fuse on this split (input boxes asserted equal to lref's, STOP otherwise), cached per split and frame list."""
    if (split, tuple(ids)) not in _RS3:
        _RS3[(split, tuple(ids))] = IGM.rs3_fuse(lr, ids, split)
    return _RS3[(split, tuple(ids))]


def truck_lane(choice, split, ids):
    """Truck: v59 = igm_read.truck(lref, lref) (route_read_lumpi.Chain.cls("Truck")); rs3 = igm_read.truck on the R6-fused Truck, other class lref."""
    if split == "fresh" and choice != "v59":
        refuse(f"Truck = {choice}: no fresh reading")
    lr = lref(split)
    ch = rs3_fused(split, ids, lr)["Truck"] if choice == RS3 else lr
    return {f: [keep(d) for d in v] for f, v in IGM.truck(ch, lr, ids, split).items()}


def bike(path, ids, keys, split):
    """ig_bike_read.bike with the point split open."""
    bi = W.only(path, "Bicycle")
    out = {f: [E.combine([d], "R4") if d["score"] >= 0.05 else SZ.keep(d) for d in bi.get(f, [])] for f in ids}
    return {f: [SZ.keep(d) for d in v] for f, v in SZ.PP.filt(out, "Bicycle", LOAD[split], keys).items()}


def person_files(choice, split):
    if choice == "v59":
        return G3.P_REF(split)
    if choice.startswith("joint:"):
        return IC.joint_files("Person", IC.joint_members("Person", choice), split)
    d = Path(f"output/lumpi/{choice}") / ("igm" if split == "val" else "s2")
    if choice in IGM.REFINER:
        return G3.P_REF(split) + [d / ("v29_val_person_min01.json" if split == "val" else f"v29_{split}_min01.json")]
    return [d / f"person_{r}_{split}_ref.json" for r in IGM.REF_ORDER]


def bike_file(choice, split):
    if choice == "v59":
        return RL.B / f"v29_{split}_ref.json"
    if choice.startswith("joint:"):
        return IC.joint_files("Bicycle", IC.joint_members("Bicycle", choice), split)[0]
    return Path(f"output/lumpi/{choice}") / ("igm" if split == "val" else "s2") / f"bicycle_{split}_ref.json"


def lane(c, choice, split, ids):
    if c == "Car":
        if choice == "v54":   # lumpi_v41's chain on the v29 boxes (route_read_lumpi.Chain without a model)
            return {f: [keep(d) for d in v] for f, v in RL.Chain(split, ids).cls("Car").items()}
        if split == "fresh":   # no G3 fresh lanes: v59's Car on realism_check is the L11-RS route lane it was read on (zQR), lumpi_v58's reading
            if choice != "v59":
                refuse(f"Car = {choice}: no fresh lane for a member")
            RL.TAG["vrcnn_lumpi_rs"] = "zQR"   # l11rs_read's tag (route_lanes_lumpi.sh)
            return {f: [keep(d) for d in v] for f, v in RL.Chain("fresh", ids, "vrcnn_lumpi_rs").cls("Car").items()}
        if choice.startswith("joint:"):
            ms = IC.joint_members("Car", choice)
            spec, name = IC.joint_spec("Car", ms), IC.joint_tag("Car", ms)
        else:
            spec = G3.B_SPEC + [IG.RS] + ([] if choice == "v59" else IGM.member(choice, "Car"))
            name = "IGc00" if choice == "v59" else f"IMc{IGM.TAGS[choice]}"
        return {f: [keep(d) for d in v] for f, v in G3.lane_class(G3.run_lane(name, spec, {}, split), "Car", ids).items()}
    if split == "fresh" and choice != "v59":
        refuse(f"Van = {choice}: no fresh lane for a member")
    lr = lref(split)
    if choice.startswith("joint:"):
        ms = IC.joint_members("Van", choice)
        ch = G3.run_lane(IC.joint_tag("Van", ms), IC.joint_spec("Van", ms), {}, split)
    elif choice == RS3:
        ch = rs3_fused(split, ids, lr)["Van"]
    else:
        ch = lr if choice == "v59" else G3.run_lane(f"IMv{IGM.TAGS[choice]}", G3.B_SPEC + IGM.member(choice, "Van"), {}, split)
    return {f: [keep(d) for d in v] for f, v in van(ch, lr, ids, split).items()}


def build(split, ids, sw, keys):
    """class -> {frame: boxes} for the four classes under the switches, on one split; Person includes the copies."""
    out = {c: lane(c, sw[c], split, ids) for c in ("Car", "Van")}
    if split != "fresh":   # Truck has no realism_check reading (the fresh build serves Car / Person only)
        out["Truck"] = truck_lane(sw.get("Truck", "v59"), split, ids)
    out["Bicycle"] = {f: [keep(d) for d in v] for f, v in bike(bike_file(sw["Bicycle"], split), ids, keys, split).items()}
    p = G3.person(person_files(sw["Person"], split), ids, LOAD[split], keys)
    if sw.get("Motorcycle", "v59") == "v59":
        moto = W.only(RL.L / f"v29_{split}_ref.json", "Motorcycle")
    else:
        if split == "fresh":
            refuse(f"Motorcycle = {sw['Motorcycle']}: no fresh lane for a member")
        m = sw["Motorcycle"]
        if m.startswith("joint:"):
            ms = IC.joint_members("Motorcycle", m)
            moto = G3.lane_class(G3.run_lane(IC.joint_tag("Motorcycle", ms), IC.joint_spec("Motorcycle", ms), {}, split), "Motorcycle", ids)
        else:
            moto = G3.lane_class(G3.run_lane(f"IMm{IGM.TAGS[m]}", G3.B_SPEC + IGM.member(m, "Motorcycle"), {}, split), "Motorcycle", ids)
    out["Motorcycle"] = {f: [keep(d) for d in moto.get(f, [])] for f in ids}
    cp = W.copies({f: out["Bicycle"][f] + out["Motorcycle"][f] for f in ids})
    out["Person"] = {f: [keep(d) for d in p.get(f, [])] + cp[f] for f in ids}
    out["_copies"] = cp
    return out


# ------------------------------------------------------------------ v60 base (lumpi_v61)
def sr_lane(c, sw60, split):
    """v60's lane for class c + safdnet_lumpi_rs (igm60.g3_lane: the lane name, spec and cache file zG3<name>_chain_<split>.json its reading used;
    e.g. Motorcycle on v60's joint: I6m_zJ_Motorcycle_DS_VM_P2_SR)."""
    x = I6.g3_lane(c, sw60[c], IC.SR, IGM.TAGS[IC.SR])
    if x is None:
        refuse(f"{c} = {IC.SR}: v60's {c} is rs3, no lane to add it to")
    return G3.run_lane(x[0], x[1], {}, split)


def mem_lane(c, v, sw60, split):
    """The lane of a v60-base switched Car / Van / Truck / Motorcycle: a Wave 2B pair's (igm_combo.pair_lane, the cache its reading used) or SR's."""
    ms = IC.pair_members(v)
    if ms:
        return G3.run_lane(*IC.pair_lane(c, sw60, ms), {}, split)
    return sr_lane(c, sw60, split)


def tv_source(c, v, split, ids, lr):
    """v60's Truck / Van input to the rule37 + T5 / V5 pass (igm60.tv_src with the split open): v59 -> lref, rs3 -> the R6 fusion, else its lane."""
    if v == "v59":
        x = lr
    elif v == RS3:
        x = rs3_fused(split, ids, lr)[c]
    else:
        nm, spec = I6.lane_of(c, v)
        x = G3.run_lane(nm, spec, {}, split)
    return {f: [d for d in x.get(f, []) if d["class"] == c] for f in ids}


def g2_bike(split, ids, keys, b0):
    """g2 (igm60.g2_cands with the split open): R4 over the 4 refiners at equal weight on L11b's Bicycle boxes (their input boxes asserted equal
    per frame, in order), then P3; first its template self-check: the fusion on L11b's file alone equals v60's (= v59's) Bicycle box for box."""
    files = I6.G2_FILES(split)
    p3 = lambda b: {f: [SZ.keep(d) for d in v] for f, v in SZ.PP.filt({f: b[f] for f in ids}, "Bicycle", LOAD[split], keys).items()}
    srcs = [(files[0], I6.load_cls(files[0], "Bicycle", ids))]
    one = p3(I6.refine_fuse(srcs, "Bicycle", srcs[0][1], ids, "g2 template"))
    chk(f"{split} g2: the R4 fusion on L11b's file alone equals v60's Bicycle box for box", all(canon(one[f]) == canon(b0[f]) for f in ids))
    del one
    srcs += [(p, I6.load_cls(p, "Bicycle", ids)) for p in files[1:]]
    return p3(I6.refine_fuse(srcs, "Bicycle", srcs[0][1], ids, "g2"))


def build60(split, ids, sw, sw60, keys, b0):
    """v60 base on one split: b0 = build(split, ids, sw60) (v60's classes from its chains); each class switched to a v60-base candidate is rebuilt
    with igm60's candidate code (Car: v60's lane + SR; Van / Truck: one rule37 + T5 / V5 pass on the set's two sources; Bicycle: g2 or SR's L11b
    second stage; Motorcycle: v60's lane + SR, or moto2's R4 on v60's Motorcycle, then MP1 for mp1 / <w>+mp1); Person = its own chain (v60's, or
    SR's five second stages) + copies from the new Bicycle and Motorcycle. "_mp1": MP1's down-weight counts when it ran."""
    out = {c: b0[c] for c in b0 if c != "_copies"}
    ch = {c for c in SW6 if sw[c] != sw60[c]}
    if "Car" in ch:
        out["Car"] = {f: [keep(d) for d in v] for f, v in G3.lane_class(mem_lane("Car", sw["Car"], sw60, split), "Car", ids).items()}
    tv = [c for c in ("Truck", "Van") if c in ch]
    if tv:
        lr = lref(split)
        src = {}
        for c in ("Truck", "Van"):
            x = mem_lane(c, sw[c], sw60, split) if c in tv else None
            src[c] = {f: [d for d in x.get(f, []) if d["class"] == c] for f in ids} if c in tv else tv_source(c, sw60[c], split, ids, lr)
            del x
        pair = IGM.tv_pass(src["Truck"], src["Van"], ids, split)
        for c in tv:
            out[c] = {f: [keep(d) for d in pair[c][f]] for f in ids}
        del lr, src, pair
    if "Bicycle" in ch:
        b = g2_bike(split, ids, keys, b0["Bicycle"]) if sw["Bicycle"] == IC.G2 else bike(bike_file(IC.SR, split), ids, keys, split)
        out["Bicycle"] = {f: [keep(d) for d in b[f]] for f in ids}
    if "Motorcycle" in ch:
        mv = sw["Motorcycle"]
        w = mv.split("+")[0]
        if IC.pair_members(mv):   # a Wave 2B pair (or the joint of pairs): v60's joint with the seed-1 entries replaced (pair_lane)
            mo = G3.lane_class(mem_lane("Motorcycle", mv, sw60, split), "Motorcycle", ids)
        elif w == IC.MOTO2:   # igm60.moto2_cands: R4 over the 4 refiners, input boxes asserted equal to v60's Motorcycle per frame, in order; no P3
            mo = I6.refine_fuse([(p, I6.load_cls(p, "Motorcycle", ids)) for p in I6.MOTO2_FILES(split)], "Motorcycle", b0["Motorcycle"], ids, "moto2")
        elif w == IC.SR:
            mo = G3.lane_class(sr_lane("Motorcycle", sw60, split), "Motorcycle", ids)
        else:
            mo = b0["Motorcycle"]
        if mv == IC.MP1 or mv.endswith(f"+{IC.MP1}"):
            mo, n1, n3 = IC.mp1(mo, LOAD[split], keys)
            out["_mp1"] = {"b1_down": n1, "p3_down": n3}
            print(f"{split} MP1 on Motorcycle ({mv}): B1 down-weighted {n1}, P3 down-weighted {n3}", flush=True)
        out["Motorcycle"] = {f: [keep(d) for d in mo.get(f, [])] for f in ids}
    if "Person" in ch:
        own = {f: [keep(d) for d in v] for f, v in G3.person(person_files(IC.SR, split), ids, LOAD[split], keys).items()}
    else:   # v60's own chain: build() appends the copies after it
        own = {f: b0["Person"][f][:len(b0["Person"][f]) - len(b0["_copies"][f])] for f in ids}
    cp = W.copies({f: out["Bicycle"][f] + out["Motorcycle"][f] for f in ids})
    out["Person"] = {f: own[f] + cp[f] for f in ids}
    out["_copies"] = cp
    return out


def md5f(p):
    """md5 of a file, read in chunks (val files are ~0.5 GB), cached per path."""
    p = str(p)
    if p not in _MD5:
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        _MD5[p] = h.hexdigest()
    return _MD5[p]


def records():
    """path -> {md5} from the fetch / stage records (REC_FILES: md5sum lines; G3IN_REC: "-> <path> <md5> : match")."""
    if not _REC:
        for rf in map(Path, REC_FILES):
            for l in (rf.read_text().splitlines() if rf.exists() else []):
                x = l.split()
                if len(x) == 2 and re.fullmatch(r"[0-9a-f]{32}", x[0]):
                    _REC.setdefault(x[1].lstrip("*"), set()).add(x[0])
        for l in (G3IN_REC.read_text().splitlines() if G3IN_REC.exists() else []):
            m = re.search(r"-> (\S+) ([0-9a-f]{32}) : match", l)
            if m:
                _REC.setdefault(m.group(1), set()).add(m.group(2))
        _REC.setdefault("_", set())
    return _REC


def rec_check(p, what, required):
    """md5 assert of an input against its record. No record: required -> REFUSED; else a printed note (an input of v59's / v60's
    own chains with no fetch record)."""
    p = Path(p)
    want = records().get(str(p))
    if not want:
        if required:
            refuse(f"{what}: {p} has no md5 record ({', '.join(REC_FILES)}, {G3IN_REC} '-> {p} <md5> : match'): record its fetch / staging first")
        print(f"note: {what}: {p} has no md5 record (not a staged input)", flush=True)
        return
    if len(want) > 1:
        refuse(f"{what}: the md5 records of {p} disagree ({sorted(want)})")
    if not p.exists():
        refuse(f"{what}: {p} missing (its record: {next(iter(want))})")
    chk(f"{what}: {p} md5 equals its record", md5f(p) in want)


def need_inputs60(c, v, sw60):
    """A v60-base candidate's val- and test-side inputs, md5-asserted against their records."""
    w = v.split("+")[0]
    ms = IC.pair_members(v)
    if ms:   # Wave 2B pair: every kit prediction of its lane on test; each pair's own (seed 2) needs its stage record (pair_chain.sh)
        for m in sorted({x[0][len("file:g3in/"):] for x in IC.pair_lane(c, sw60, ms)[1] if x[0].startswith("file:g3in/")}):
            p = Path(f"output/lumpi/g3in/{m}_test.json")
            if not p.exists():
                refuse(f"{c} = {v}: {p} missing")
            rec_check(p, f"{c} = {v}", m in ms)
        return
    if v == IC.G2:   # L11b's file is v59's own Bicycle file (no record; the template self-check and v60's rebuild cover it)
        for s in ("val", "test"):
            fs = I6.G2_FILES(s)
            if not fs[0].exists():
                refuse(f"Bicycle = g2: {fs[0]} missing")
            for p in fs[1:]:
                rec_check(p, "Bicycle = g2", True)
    elif w == IC.MOTO2:
        for s in ("val", "test"):
            for p in I6.MOTO2_FILES(s):
                rec_check(p, f"Motorcycle = {v}", True)
    if w == IC.SR and c in ("Car", "Van", "Truck", "Motorcycle"):   # every kit prediction of the SR lane on test; SR's own needs its stage record
        for m in sorted({x[0][len("file:g3in/"):] for x in I6.g3_lane(c, sw60[c], IC.SR, IGM.TAGS[IC.SR])[1] if x[0].startswith("file:g3in/")}):
            p = Path(f"output/lumpi/g3in/{m}_test.json")
            if not p.exists():
                refuse(f"{c} = {v}: {p} missing (3090 output/lumpi/{m}/kit/pred_test.json)")
            rec_check(p, f"{c} = {v}", m == IC.SR)
    elif w == IC.SR:   # Person / Bicycle: the read val files (igm/) are the route's s2 val files, and the s2 val / test files match their record
        pairs = (list(zip(person_files(IC.SR, "val"), [Path(f"output/lumpi/{IC.SR}/s2/person_{r}_val_ref.json") for r in IGM.REF_ORDER]))
                 if c == "Person" else [(bike_file(IC.SR, "val"), Path(f"output/lumpi/{IC.SR}/s2/bicycle_val_ref.json"))])
        tests = person_files(IC.SR, "test") if c == "Person" else [bike_file(IC.SR, "test")]
        for a, b in pairs:
            pair_md5(c, v, a, b)
            rec_check(b, f"{c} = {v}", True)
        for p in tests:
            rec_check(p, f"{c} = {v}", True)
    if v == IC.MP1 or v.endswith(f"+{IC.MP1}"):
        if not IC.mp1_json().exists():
            refuse(f"Motorcycle = {v}: MP1's json {IC.mp1_json()} missing (copy it into logs/)")


def pair_md5(c, m, a, b):
    """The read val file a (igm/) must be the route's own s2 val file b (same s2 run as the test file): both present, md5 equal (no silent skip)."""
    if not (Path(a).exists() and Path(b).exists()):
        refuse(f"{c} = {m}: cannot compare {a} with {b} (" + " and ".join(str(p) for p in (a, b) if not Path(p).exists()) + " missing): "
               "the md5 assert needs both")
    chk(f"{c} = {m}: read val file {a} equals the route's {Path(b).name} (same s2 run as the test file)", md5f(a) == md5f(b))


# ------------------------------------------------------------------ inputs and readings
def need_test_inputs(c, m):
    files = []
    if c in ("Car", "Van", "Motorcycle"):
        files = [f"output/lumpi/g3in/{m}_test.json"] + (["output/lumpi/g3in/fshnet_lumpi_test.json"] if m == "fshnet_lumpi_s2" else [])
    elif c == "Person":
        files = [str(p) for p in person_files(m, "test")]
    elif c == "Bicycle":
        files = [str(bike_file(m, "test"))]
    miss = [f for f in files if not Path(f).exists()]
    if miss:
        refuse(f"{c} = {m}: test-side inputs missing (copy from the 3090 output/lumpi/{m}/kit/pred_test.json -> g3in/{m}_test.json, "
               f"output/lumpi/{m}/s2/*_test_ref.json, s2/v29_test_min01.json): " + ", ".join(miss))
    for f in files:   # a test input a fetch / stage record names must match it (g3in: logs/g3in_test_stage.log)
        rec_check(f, f"{c} = {m}", False)
    pairs = []   # the read val file (igm/) must be the route's own s2 val file, so the test file comes from the same run
    if c == "Person":
        pairs = list(zip(person_files(m, "val")[-1:] if m in IGM.REFINER else person_files(m, "val"),
                         [Path(f"output/lumpi/{m}/s2/v29_val_min01.json")] if m in IGM.REFINER
                         else [Path(f"output/lumpi/{m}/s2/person_{r}_val_ref.json") for r in IGM.REF_ORDER]))
    elif c == "Bicycle":
        pairs = [(bike_file(m, "val"), Path(f"output/lumpi/{m}/s2/bicycle_val_ref.json"))]
    for a, b in pairs:   # md5 assert, both files required (was a note when one was missing)
        pair_md5(c, m, a, b)


def need_rs3_inputs(c):
    """rs3: the 3 refiners' test files must exist, and the val files read (igm_read rs3) and the test files must both be the rs3_tv run's
    (md5 equal to logs/rs3_refine.md5, fetched from the 3090 by scripts/rs3_fetch.sh)."""
    if "md5" in _RS3:
        return
    miss = [str(p) for p in IGM.rs3_files("test") if not p.exists()]
    if miss:
        refuse(f"{c} = rs3: test-side inputs missing (scripts/rs3_fetch.sh: 3090 output/lumpi/rs3/): " + ", ".join(miss))
    mf = Path("logs/rs3_refine.md5")
    if not mf.exists():
        refuse(f"{c} = rs3: {mf} missing (scripts/rs3_fetch.sh fetches it from the 3090)")
    want = {Path(l.split()[1]).name: l.split()[0] for l in mf.read_text().splitlines() if l.strip()}
    for p in IGM.rs3_files("val") + IGM.rs3_files("test"):
        chk(f"{c} = rs3: {p} md5 equals the rs3_tv run's (logs/rs3_refine.md5)", p.exists() and md5(p) == want.get(p.name))
    _RS3["md5"] = True


def need_joint_inputs(c, choice):
    """Person / Bicycle joint: its test files, and its read val files (igm/) equal to the same run's s2 val files."""
    ms = IC.joint_members(c, choice)
    files = IC.joint_files(c, ms, "test")
    miss = [str(f) for f in files if not f.exists()]
    if miss:
        refuse(f"{c} = {choice}: joint test-side inputs missing (copy from the 3090 output/lumpi/joint/{IC.joint_tag(c, ms)}/s2/): " + ", ".join(miss))
    for a in IC.joint_files(c, ms, "val"):   # md5 assert, both files required (was a note when one was missing)
        b = IC.joint_dir(c, ms) / "s2" / a.name
        if not (a.exists() and b.exists()):
            refuse(f"{c} = {choice}: cannot compare {a} with {b} (" + " and ".join(str(p) for p in (a, b) if not p.exists()) + " missing): "
                   "the md5 assert needs both")
        chk(f"{c} = {choice}: read val file {a} equals the joint run's {b} (same s2 run as the test file)", md5f(a) == md5f(b))
    for f in files:
        rec_check(f, f"{c} = {choice}", False)


def reading(c, m):
    p = IC.find_reading(m, c, "val")   # the full reading or a class-subset one (_carvan, _motorcycle)
    if p is None:
        refuse(f"{c} = {m}: no reading of '{c}: add {m}' in logs/igm_{m}*_val.json")
    r = json.loads(p.read_text()).get(f"{c}: add {m}")
    if not r or not r.get("passed"):
        refuse(f"{c} = {m}: '{c}: add {m}' did not pass in {p} ({'not read' if not r else 'fail'})")
    return r


def parse(v, c):
    if v.startswith("joint:"):
        IC.joint_members(c, v)   # 2+ members in the fixed order (the joint rule); Person / Bicycle: second stages on the joint lane
        return v
    if c == "Truck":
        allowed = ("v59", RS3)
    else:
        allowed = ("v59",) + (("v54",) if c == "Car" else ()) + (tuple(sorted(IGM.MOTO)) if c == "Motorcycle" else MEMBERS) + ((RS3,) if c == "Van" else ())
    if v not in allowed:
        refuse(f"{c} = {v}: not one of {allowed}")
    return v


def parse60(v, c, sw60):
    """v60 base: v60's own switch, or a v60-base candidate for the class (igm_combo.cands60; SR = safdnet_lumpi_rs)."""
    v = IC.ALIAS60.get(v, v)
    if v != sw60[c] and v not in IC.cands60(c, sw60):
        refuse(f"{c} = {v}: on the v60 base a class keeps v60's switch ({sw60[c]}) or takes a v60-base candidate {IC.cands60(c, sw60)}")
    return v


# ------------------------------------------------------------------ final combination
def combo_json(sw, base=None):
    k = IC.key_of(dict(sw, Car="v59" if sw["Car"] == "v54" else sw["Car"], **({"base": base} if base else {})))
    p = Path(f"logs/igm_combo_{k}_val.json")
    return json.loads(p.read_text()) if p.exists() else None, p


def final_combination(full, fallback, only, ref=None, base=None):
    """-> (the switches to build, the PASS reading of exactly those switches or None). Refuses to go on without the reading the rule asks for.
    ref: the reference switches (v59's, or v60's for --base v60: fallback / --only return a class to it)."""
    ref = ref or {c: "v59" for c in SW6}
    inter = lambda sw: [c for c in ("Person", "Bicycle", "Motorcycle") if sw[c] != ref[c]]
    if fallback and only:
        refuse("--fallback and --only together")
    if all(full[c] == ref[c] or (not base and full[c] == "v54") for c in SW6):
        if fallback or only:
            refuse("--fallback / --only without a member change")
        return full, None   # no member change: the dry run, or Car v54 alone (its own reading); --base v60: v60's own switches
    r1, p1 = combo_json(full, base)
    if r1 is None:
        refuse(f"no final-combination reading {p1} for {full}: run igm_read.py combo near / val with these switches first"
               + (" (--base v60)" if base else ""))
    if not fallback and not only:
        if not r1["passed"]:
            refuse(f"the final-combination reading {p1} is a CONFLICT ({'; '.join(r1['conflicts'])}); next: {r1.get('next')}")
        print(f"final-combination reading {r1['key']}: PASS", flush=True)
        return full, r1
    if r1["passed"] or not r1.get("fallback_to_drop"):
        refuse(f"{p1} passed or names no fallback: --fallback / --only do not apply")
    if fallback:
        if fallback != r1["fallback_to_drop"]:
            refuse(f"--fallback {fallback}: the reading {p1} names {r1['fallback_to_drop']}")
        sw = dict(full, **{fallback: ref[fallback]})
        r2, p2 = combo_json(sw, base)
        fs = {c: full[c] if not (c == "Car" and full[c] == "v54") else "v59" for c in SW5}
        if full["Truck"] != "v59":   # igm_combo stores Truck only when it is not v59
            fs["Truck"] = full["Truck"]
        if base:
            fs["base"] = base
        if r2 is None or r2.get("fallback") != fallback or r2.get("first_stage") != fs:
            refuse(f"no fallback reading {p2} for --fallback {fallback} (run igm_read.py combo near / val with the full switches and --fallback {fallback})")
        if not r2["passed"]:
            refuse(f"the fallback reading {p2} is a CONFLICT ({'; '.join(r2['conflicts'])}); next: {r2.get('next')}")
        print(f"fallback reading {r2['key']} (--fallback {fallback}): PASS", flush=True)
        return sw, r2
    sw_fb = dict(full, **{r1["fallback_to_drop"]: ref[r1["fallback_to_drop"]]})
    r2, p2 = combo_json(sw_fb, base)
    if r2 is None or r2["passed"] or r2.get("only") != only:
        refuse(f"--only {only}: the fallback reading {p2} does not name it (missing, passed, or names {r2 and r2.get('only')})")
    sw = dict(full, **{c: ref[c] for c in inter(full) if c != only})
    print(f"--only {only}: interacting classes {inter(full)} -> keep {only}; no more readings (final-combination rule)", flush=True)
    return sw, None


def test_build(sw, changed, sw60, refv, tid, keys):
    """2. test: the reference's classes (v59's, or v60's for --base v60) rebuilt from their chains must equal its file box for box; the changed
    classes from their chains; every class not changed is spliced from the reference file (lumpi_v59_test.json / lumpi_v60_test.json) and must
    equal it box for box (STOP otherwise). -> (out, nt)"""
    bk = "v60" if sw60 else "v59"
    base = dict(sw60) if sw60 else {c: "v59" for c in SW5}
    bt = build("test", tid, base, keys)
    refc = lambda c: {f: [keep(d) for d in refv[f] if d["class"] == c] for f in tid}
    for c in SW6:
        chk(f"test {c}: {bk} chain rebuilt equals {bk} box for box", all(canon(bt[c][f]) == canon(refc(c)[f]) for f in tid))
    nt = bt if not changed else build60("test", tid, sw, sw60, keys, bt) if sw60 else build("test", tid, sw, keys)
    del bt
    if sw["Car"] == "v54":
        v54 = json.loads(V54.read_text())
        chk("test Car v54: lumpi_v41 chain rebuilt equals lumpi_v54's Car box for box",
            all(canon(nt["Car"][f]) == canon([keep(d) for d in v54[f] if d["class"] == "Car"]) for f in tid))
    out = {f: [d for d in refv[f] if d["class"] not in changed] + [d for c in sorted(changed) for d in nt[c][f]] for f in tid}
    allc = sorted({d["class"] for v in refv.values() for d in v} | set(SW6))
    for c in allc:
        if c in changed:
            n0, n1 = sum(len(refc(c)[f]) for f in tid), sum(len(nt[c][f]) for f in tid)
            print(f"test {c}: {sw.get(c, 'copies')} ({n0:,} -> {n1:,} boxes)", flush=True)
        else:   # the splice guard (also for v60): a class not changed equals the reference file box for box
            chk(f"test {c}: equals {bk} box for box", all(canon([d for d in out[f] if d["class"] == c]) == canon(refc(c)[f]) for f in tid))
    if sw60 and sw["Motorcycle"] == IC.MP1:   # MP1 alone: its own test counts (on lumpi_v60_test.json's Motorcycle) must be reproduced
        j = IC.mp1_reading().get("counts_test_unlabeled") or {}
        if j.get("source") == V60.name:
            got = dict(nt["_mp1"], ge03_before_after=[sum(1 for f in tid for d in refc("Motorcycle")[f] if d["score"] >= 0.3),
                                                      sum(1 for f in tid for d in nt["Motorcycle"][f] if d["score"] >= 0.3)])
            for k in ("b1_down", "p3_down", "ge03_before_after"):
                chk(f"test MP1 {k} {got[k]} = MP1's own json {j[k]}", got[k] == j[k])
        else:
            print(f"note: MP1's json has no test counts on {V60.name} ({j.get('source')}): not compared", flush=True)
    return out, nt


# ------------------------------------------------------------------ Wave 2B pair wiring self-check
def pair_selfcheck(m):
    """Seed 1's own file stands in for seed 2 (igm_combo.PAIR_STANDIN; lanes W2sc...): Motorcycle at 1.0 + 0.0 (the pair replacing seed 1 in v60's
    joint = v60's joint lane), Car / Van / Truck at 0.0 + 0.0 (the new member absent = v60's lane). The test build with every scope-C class
    switched to the pair, through build60 / test_build, must equal lumpi_v60_test.json (06c70a89) box for box, class by class. First the real
    pair lanes (no stand-in) must equal pair60.lanes_of (pair_lane's own STOP). No reading, combo, val or fresh step; nothing written."""
    if m not in IGM.PAIR:
        refuse(f"--pair-selfcheck {m}: not a pair {tuple(IGM.PAIR)}")
    sw60 = IC.v60_base()[0]
    cls = [c for c in IC.PAIR_CLASSES if IC.pair_ok(c, m, sw60)]
    for c in cls:
        print(f"{c}: real pair lane zG3{IC.pair_lane(c, sw60, (m,))[0]} (= pair60.lanes_of); v60-base candidates {IC.cands60(c, sw60)}; "
              f"fixed choice now {IC.final_choice60(c)}", flush=True)
    IC.PAIR_STANDIN[m] = {c: (1.0, 0.0) if c == "Motorcycle" else (0.0, 0.0) for c in cls}
    for c in cls:
        nm, spec = IC.pair_lane(c, sw60, (m,))
        print(f"{c}: stand-in lane zG3{nm} = {json.dumps(spec)}", flush=True)
    sw = dict(sw60, **{c: m for c in cls})
    chk(f"{V60} md5 {md5f(V60)} = v60's {V60_MD5}", md5f(V60) == V60_MD5)
    refv = json.loads(V60.read_text())
    tid = sorted(refv)
    keys = SZ.PP.static_keys("lumpi")
    out, nt = test_build(sw, set(cls) | {"Person"}, sw60, refv, tid, keys)
    for c in sorted({d["class"] for v in refv.values() for d in v} | set(SW6)):
        chk(f"pair self-check {m} test {c}: equals lumpi_v60 box for box",
            all(canon([d for d in out[f] if d["class"] == c]) == canon([d for d in refv[f] if d["class"] == c]) for f in tid))
    chk(f"pair self-check {m} test: the whole file equals lumpi_v60 ({V60_MD5}) box for box", all(canon(out[f]) == canon(refv[f]) for f in tid))
    if not ok:
        sys.exit(f"STOP the pair self-check of {m} failed")
    print(f"pair self-check {m}: PASS", flush=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--base", default="v59", choices=("v59", "v60"),
                    help="v60: lumpi_v61 = v60's switches + v60-base candidates; default v59 (unchanged)")
    for c in CLASSES4:
        ap.add_argument(f"--{c.lower()}", default=None, help="default: the base's switch (v59, or v60's own for --base v60)")
    ap.add_argument("--motorcycle", default=None)
    ap.add_argument("--truck", default=None, help="v59 or rs3 (Truck / Van RS second stages)")
    ap.add_argument("--val-only", action="store_true", help="accept changed classes without a fresh reading (igm_read logs none): a manual decision")
    ap.add_argument("--fallback", default=None, help="the class the final-combination reading named to drop")
    ap.add_argument("--only", default=None, help="the class the fallback reading named to keep alone among the interacting changes")
    ap.add_argument("--pair-selfcheck", default=None, help="Wave 2B pair wiring self-check with --base v60 (seed 1 stands in for seed 2); nothing written")
    a = ap.parse_args()
    if a.pair_selfcheck:
        if a.base != "v60":
            refuse("--pair-selfcheck needs --base v60")
        return pair_selfcheck(a.pair_selfcheck)
    b60 = a.base == "v60"
    bk = a.base
    sw60 = IC.v60_base()[0] if b60 else None   # logs/build_lumpi_v60.json's switches, validated (igm60.v60_set)
    ref = dict(sw60) if b60 else {c: "v59" for c in SW6}
    sw = {}
    for c in SW6:
        v = getattr(a, c.lower())
        v = ref[c] if v is None else v
        sw[c] = parse60(v, c, sw60) if b60 else parse(v, c)
    sw, combo = final_combination(sw, a.fallback, a.only, ref if b60 else None, IC.B60 if b60 else None)
    changed = {c for c in SW6 if sw[c] != ref[c]}
    if sw["Bicycle"] != ref["Bicycle"] or sw["Motorcycle"] != ref["Motorcycle"]:
        changed.add("Person")   # recomputed copies
    print("switches:", sw, "| classes changed:", sorted(changed) or "none", *([f"(vs v60: {sw60})"] if b60 else []), flush=True)
    for c in SW6:
        if b60:   # v60 base: the fixed stacking rules (final_choice60), and every input md5-asserted against its record
            if sw[c] != ref[c]:
                if sw[c] != IC.final_choice60(c):
                    refuse(f"{c} = {sw[c]}: the fixed choice for {c} is {IC.final_choice60(c)} (v60-base stacking rule)")
                need_inputs60(c, sw[c], sw60)
            continue
        if sw[c] not in ("v59", "v54") and sw[c] != IC.final_choice(c):   # rs3: exempt from the fixed member choice; Van member vs rs3 by the fixed Van rule
            refuse(f"{c} = {sw[c]}: the fixed choice for {c} is {IC.final_choice(c)} (fixed member-choice rule" + ("; Van: member vs rs3 rule)" if c == "Van" else ")"))
        if sw[c] == RS3:
            need_rs3_inputs(c)
            continue
        if sw[c].startswith("joint:") and c in IC.JOINT_S2:
            need_joint_inputs(c, sw[c])
            continue
        for m in (IC.joint_members(c, sw[c]) if sw[c].startswith("joint:") else [sw[c]]):
            if m in MEMBERS:
                need_test_inputs(c, m)
    if not ok:
        sys.exit("STOP an input check failed")
    if b60:
        rd = {c: IC.single(sw[c], c, "val", IC.B60) for c in SW6 if sw[c] != ref[c]}
    else:
        rd = {c: (IC.single(sw[c], c, "val") if sw[c].startswith("joint:") else reading(c, sw[c])) for c in SW6
              if sw[c] in MEMBERS or sw[c].startswith("joint:") or sw[c] == RS3}
    if changed and not a.val_only and rd:
        sys.exit("STOP igm_read logs no fresh (realism_check) reading for a member; rerun with --val-only only after a manual decision")

    keys = SZ.PP.static_keys("lumpi")
    if b60:
        chk(f"{V60} md5 {md5f(V60)} = v60's {V60_MD5}", md5f(V60) == V60_MD5)
    refv = json.loads((V60 if b60 else V59).read_text())   # the splice source: lumpi_v59_test.json, or lumpi_v60_test.json for --base v60
    tid = sorted(refv)
    chk(f"{bk} test file has the 50 test frames", len(tid) == 50)
    B = json.loads(Path("logs/build_lumpi_v59.json").read_text())["readings"]
    L = json.loads(Path("logs/build_lumpi_l11rs.json").read_text())["swapped"]["Car"]
    res = {"switches": sw, "changed": sorted(changed), "val": {}, "fresh": {}, "final_combination": combo and combo["key"],
           "fallback": a.fallback, "only": a.only}
    if b60:
        i60 = IC.info60(dict(sw, base=IC.B60))   # v60's build json (path, md5) and combo key; MP1's json (path, md5) when mp1 is built
        res.update(base=IC.B60, v60=dict(i60["v60"], test_file=str(V60), test_md5=V60_MD5), **({"mp1_json": i60["mp1_json"]} if "mp1_json" in i60 else {}))

    # 1. readings: val (320) and realism_check (160)
    vid = S.ids_of("val")
    gtv = {f: [{"box": [float(x) for x in b], "class": k} for b, k in v] for f, v in S.labels(vid).items()}
    base = {c: "v59" for c in SW5}
    if b60:   # v60's classes rebuilt from its chains must give its build's val numbers (1e-9); the set's classes on top (build60)
        bv = build("val", vid, ref, keys)
        nv = bv if not changed else build60("val", vid, sw, sw60, keys, bv)
        bj = json.loads(Path("logs/build_lumpi_v60.json").read_text())["val"]
        for c in SW6:
            res["val"][c] = {"v60": G3.ap(bv[c], gtv, vid, c), "new": G3.ap(nv[c], gtv, vid, c)}
            chk(f"val {c} v60 base {res['val'][c]['v60']:.6f} = logs/build_lumpi_v60.json {bj[c]['new']:.6f} (v60 rebuilt from its chains)",
                abs(res["val"][c]["v60"] - bj[c]["new"]) < 1e-9)
        del bv, nv
    else:
        bv = build("val", vid, base, keys)
        nv = bv if not changed else build("val", vid, sw, keys)
        for c in SW6:
            res["val"][c] = {"v59": G3.ap(bv[c], gtv, vid, c), "new": G3.ap(nv[c], gtv, vid, c)}
        for c, want in (("Car", B["Car"]["val"][0]), ("Person", B["Person"]["val"][0])):
            got = round(res["val"][c]["v59"], 4)
            chk(f"val {c} v59 base {got} = v59's reading {want} (igm_read self-check)", got == want)
        rs = [c for c in IGM.RS3_CLASSES if sw[c] == RS3]
        for c, want in IGM.RS3_VAL.items():   # igm_read rs3's self-check values; gated only when rs3 is used
            got = round(res["val"][c]["v59"], 4)
            if rs:
                chk(f"val {c} v59 base {got} = v59's {want} (igm_read rs3 self-check)", got == want)
            else:
                print(f"information: val {c} v59 base {got} (v59's {want})", flush=True)
        if rs:   # each rs3 class equals igm_read's rs3 candidate box for box (its single reading's chain; other class from lref)
            lv = lref("val")
            fu = rs3_fused("val", vid, lv)
            for c in rs:
                o = "Van" if c == "Truck" else "Truck"
                single = IGM.tv_pass(fu[c] if c == "Truck" else lv, fu[c] if c == "Van" else lv, vid)[c]
                chk(f"val {c} (rs3) equals igm_read's rs3 candidate box for box ({o} from lref)",
                    all(canon(nv[c][f]) == canon([keep(d) for d in single[f]]) for f in vid))
            del lv
    for c in SW6:
        if c not in changed:
            continue
        x = res["val"][c]["new"]
        if combo and c in combo["classes"]:   # the final-combination reading of exactly these switches
            chk(f"val {c} ({sw[c]}) {x:.6f} = the final-combination reading {combo['key']} {combo['classes'][c]['val']:.6f}",
                abs(x - combo["classes"][c]["val"]) < 1e-9)
            if c in rd and c != "Person":
                chk(f"val {c} ({sw[c]}) {x:.6f} = its igm reading {rd[c]['val']:.6f}", abs(x - rd[c]["val"]) < 1e-9)
            continue
        if sw[c] == "v54":
            chk(f"val Car v54 {round(x, 4)} = lumpi_v54's reading {L['val']['ref_v53_v54']['all'][0]}", round(x, 4) == L["val"]["ref_v53_v54"]["all"][0])
        elif c in rd:
            if c == "Person" and (sw["Bicycle"] != "v59" or sw["Motorcycle"] != "v59"):
                chk(f"val Person ({sw['Person']}, copies from Bicycle {sw['Bicycle']}): {x:.6f} vs its reading {rd[c]['val']:.6f} (read with v59's "
                    "copies; the stacked pair was not read)", abs(x - rd[c]["val"]) < 1e-9)
            else:
                chk(f"val {c} ({sw[c]}) {x:.6f} = its igm reading {rd[c]['val']:.6f}", abs(x - rd[c]["val"]) < 1e-9)
        else:
            print(f"information: val Person with copies from Bicycle {sw['Bicycle']} / Motorcycle {sw['Motorcycle']}: {res['val']['Person']['v59']:.4f} -> {x:.4f} "
                  "(igm reads this on near-test only; no val reading)", flush=True)
    for i in CHK:
        guard.check("lumpi", i)
    gtf = S.labels(CHK)
    sf = dict(base, Car="v54" if sw["Car"] == "v54" else "v59")
    bf = build("fresh", CHK, base, keys)
    nf = build("fresh", CHK, sf, keys) if sw["Car"] == "v54" else bf
    for c, want in (("Car", B["Car"]["fresh"][0]), ("Person", B["Person"]["fresh"][0])):
        got = S.ap(cls_of(bf[c], c, CHK), gtf, CHK, [c])[0.5][c]
        res["fresh"][c] = {"v59": got}
        chk(f"realism_check 160 {c} v59 base {got} = v59's reading {want}", got == want)
    if sw["Car"] == "v54":
        got = S.ap(cls_of(nf["Car"], "Car", CHK), gtf, CHK, ["Car"])[0.5]["Car"]
        res["fresh"]["Car"]["new"] = got
        chk(f"realism_check 160 Car v54 {got} = lumpi_v54's reading {L['fresh']['ref_v53_v54']['all'][0]}", got == L["fresh"]["ref_v53_v54"]["all"][0])
    for c in sorted(changed - {"Car"} if sw["Car"] == "v54" else changed):
        print(f"information: {c} has no realism_check reading in igm_read (val + near-test only); accepted under --val-only", flush=True)

    # 2. test: bases rebuilt from their chains must equal v59 (v60); changed classes from their chains; the others spliced and guarded
    out, nt = test_build(sw, changed, sw60, refv, tid, keys)
    ncp = sum(len(v) for v in nt["_copies"].values())
    ref_cp = json.loads(Path("logs/build_lumpi_v60.json").read_text())["copies"] if b60 else 682
    print(f"copies on test: {ncp} ({bk}: {ref_cp}) | boxes {bk} {sum(map(len, refv.values())):,} -> new {sum(map(len, out.values())):,}", flush=True)
    res.update(copies=ncp, boxes=[sum(map(len, refv.values())), sum(map(len, out.values()))])
    if "_mp1" in nt:
        res["mp1_test"] = nt["_mp1"]
    if not ok:
        sys.exit("STOP a check failed: nothing written")
    od = Path(a.outdir) if a.outdir else O
    od.mkdir(parents=True, exist_ok=True)
    pj = od / f"{a.name}_test.json"
    pj.write_text(json.dumps(out))
    lg = (Path(a.outdir) if a.outdir else Path("logs")) / f"build_{a.name}.json"
    lg.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", pj, "and", lg, f"| identical to {bk} box for box:", all(canon(out[f]) == canon(refv[f]) for f in tid), flush=True)


if __name__ == "__main__":
    main()
