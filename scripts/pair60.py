"""Wave 2B pair readings against lumpi_v60, scope C (Car / Van / Truck / Motorcycle). One reading per seed
model, in two class batches so that Motorcycle can run first on the Mac:
  moto  Motorcycle: v60's joint lane (dsvt_lumpi + vmamba_lumpi + pv_lumpi_s2) with seed 1's entry replaced, in place, by the pair (seed 1 at 0.5 +
        seed 2 at 0.5, igm_read.member's zF2 form); lane zG3W2m<tag>. Person = v60's own chain + copies from v60's Bicycle and the candidate
        Motorcycle (igm60's person_with), as G3's Motorcycle.
  cvt   Car / Van / Truck: the pair as ONE new member on v60's lane for the class, G3's form unchanged (igm60.g3_lane: lanes IMc / IMv / IMt<tag>;
        Car lane_class with rule37; Van / Truck one rule37 + T5 / V5 pass with the other class from v60, which must equal v60 box for box).
  Person / Bicycle: not read (scope C). Tags: vmamba_lumpi_s2 V2, dsvt_lumpi_s2 D2 (igm_read.TAGS / PAIR).
igm60.py is reused unchanged (v60_set, bases60, check_near, check_val, g3_lane, tv_pass via igm_read); this file adds the pair candidates and
copies igm60.val60's G3 gate code (Motorcycle: val not down; Person copies veto; Person val (copies) not down > 0.01).
Self-checks, all before any candidate number (STOP on a miss):
  igm60.bases60 + check_near / check_val (v60 rebuilt box for box and to its readings); the seed-2 file covers exactly the seed-1 file's frames
  (utl.ensemble asserts it per lane); for every class of the batch in CHK (default Motorcycle), the pair lane at 1.0 + 0.0 with the REAL seed-2
  file (zG3W2chk<k><tag>, built by the lanes step) equals the seed-1 lane box for box (dry run on seed 1, all 8 lanes byte-identical: logs/
  pair_selfcheck_*_dry_*.json).
Gate (G3's, fixed): inside AP@0.5 >= +0.01 on a class with >= 5 inside ids AND val (320, pooled) a - base >= +0.005 unrounded (Motorcycle: also
val not down (a >= base), logs/igm_moto_rule.json); veto -0.005 on every touched class on a segment with >= 5 ids (Van and Truck both; Person copies for
Motorcycle); Person val with the copies not down > 0.01. The fixed member choice (both pairs pass the same class -> one joint reading of exactly those passers)
is a later, separate step.
    python scripts/pair60.py lanes <m> moto|cvt            (either workspace: builds the batch's pair lanes and 1.0 + 0.0 check lanes; no AP)
    python scripts/pair60.py near <m> moto|cvt [--check]   (neartest_ws) -> logs/pair60_<m>_<batch>_near.json
    python scripts/pair60.py val <m> moto|cvt [--check]    (main; needs the near json from neartest_ws) -> logs/pair60_<m>_<batch>_val.json
A real near / val reading refuses to run when its output json exists (one reading). --check: bases, self-checks and candidates built, box counts
only, no candidate AP, nothing written."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import igm_combo as IC
import igm60 as I6

M = I6.M   # scripts/igm_read.py
BATCH = {"moto": ("Motorcycle",), "cvt": ("Car", "Van", "Truck")}
KEY = {"Car": "c", "Van": "v", "Truck": "t", "Motorcycle": "m"}
CHK = ("Motorcycle",)   # classes whose 1.0 + 0.0 lane is rebuilt with the real seed-2 file (all 8 were byte-identical in the dry run)
OUT = lambda m, b, where: Path(f"logs/pair60_{m}_{b}_{where}.json")
NOTE = ("Wave 2B pair (scope C): seed 1 at 0.5 + seed 2 at 0.5 as one member; Motorcycle replaces seed 1 in "
        "v60's joint, Car / Van / Truck add it as a new member; Person / Bicycle not read (for time)")


def stop(msg):
    sys.exit(f"STOP {msg}")


def files(m):
    return Path(f"output/lumpi/g3in/{M.PAIR[m]}_val.json"), Path(f"output/lumpi/g3in/{m}_val.json")


def lanes_of(m, sw, b):
    """class -> dict(form, name, spec, ref, ref_spec, chk, chk_spec): the batch's pair lane (0.5 + 0.5), the seed-1 lane it must reproduce at
    1.0 + 0.0, and that check lane (CHK classes only; else chk = None)."""
    s1, t = M.PAIR[m], M.TAGS[m]
    out = {}
    for c in BATCH[b]:
        v = sw[c]
        if c == "Motorcycle":
            if not (v.startswith("joint:") and s1 in IC.joint_members(c, v)):
                stop(f"v60's Motorcycle is {v}: scope C reads the pair as a replacement of {s1} in v60's joint only")
            ms = IC.joint_members(c, v)
            ref, ref_spec = IC.joint_tag(c, ms), IC.joint_spec(c, ms)
            one = M.member(s1, c)
            if len(one) != 1 or ref_spec.count(one[0]) != 1:
                stop(f"{s1}'s entry {one} is not exactly once in v60's joint spec")
            i = ref_spec.index(one[0])
            mk = lambda w: ref_spec[:i] + M.member(m, c, w) + ref_spec[i + 1:]
            name, form = f"W2m{t}", f"replacement of {s1} in v60's Motorcycle joint {v}"
        else:
            if v.startswith("joint:") or v == IC.RS3:
                stop(f"v60's {c} is {v}: the G3-form new-member lane is defined on a v59 / member lane only")
            ref, ref_spec = I6.g3_lane(c, v, s1, M.TAGS[s1])
            if ref_spec[-1] != M.member(s1, c)[0]:
                stop(f"{c}: the seed-1 lane does not end with {s1}'s member entry")
            mk = lambda w: ref_spec[:-1] + M.member(m, c, w)
            name, spec_g3 = I6.g3_lane(c, v, m, t)
            if spec_g3 != mk(M.PAIR_W):
                stop(f"{c}: igm60.g3_lane's pair lane differs from the seed-1 lane with the member swapped")
            form = f"new member on v60's {c} lane ({v}), G3 form"
        out[c] = dict(form=form, name=name, spec=mk(M.PAIR_W), ref=ref, ref_spec=ref_spec,
                      chk=f"W2chk{KEY[c]}{t}" if c in CHK else None, chk_spec=mk((1.0, 0.0)) if c in CHK else None)
    return out


def coverage(m, where):
    f1, f2 = files(m)
    for f in (f1, f2):
        if not f.exists():
            stop(f"{f} missing")
    a, b = set(json.loads(f1.read_text())), set(json.loads(f2.read_text()))
    ok = a == b
    print(f"self-check {where} pair inputs: {f2} (md5 {I6.md5(f2)}) covers exactly the {len(a)} frames of {f1} (md5 {I6.md5(f1)}): {ok}", flush=True)
    if not ok:
        stop(f"the seed-2 file covers other frames than seed 1's ({len(b)} vs {len(a)})")


def lanes(m, b):
    sw, _ = I6.v60_set(False, None)
    coverage(m, "lanes")
    for c, L in lanes_of(m, sw, b).items():
        print(f"{c}: {L['form']}; pair lane zG3{L['name']} = {json.dumps(L['spec'])}", flush=True)
        M.G3.run_lane(L["name"], L["spec"], {}, "val")
        if L["chk"]:
            M.G3.run_lane(L["chk"], L["chk_spec"], {}, "val")
            M.G3.run_lane(L["ref"], L["ref_spec"], {}, "val")
    print("lanes built", m, b, flush=True)


def frames_equal(a, b):
    k = lambda v: sorted(json.dumps(d, sort_keys=True) for d in v)
    return set(a) == set(b) and all(k(a[f]) == k(b[f]) for f in b)


def check_pair(m, Ls, where):
    """1.0 + 0.0 = seed 1, box for box, for the CHK classes of the batch (the lanes step built both lanes)."""
    ok = True
    for c, L in Ls.items():
        if not L["chk"]:
            continue
        got, ref = M.G3.run_lane(L["chk"], L["chk_spec"], {}, "val"), M.G3.run_lane(L["ref"], L["ref_spec"], {}, "val")
        same = frames_equal(got, ref)
        print(f"self-check {where} {c}: the pair lane at 1.0 + 0.0 (zG3{L['chk']}, real seed-2 file) equals the seed-1 lane zG3{L['ref']} box for box: "
              f"{same} (md5 {I6.md5(M.G3.O / ('zG3' + L['chk'] + '_chain_val.json'))} vs {I6.md5(M.G3.O / ('zG3' + L['ref'] + '_chain_val.json'))})",
              flush=True)
        ok = ok and same
        del got, ref
    if not ok:
        stop(f"a pair lane at 1.0 + 0.0 does not reproduce the seed-1 lane ({where})")


def pair_cands(out, sw, ids, m, ctx, where, Ls):
    """The batch's candidates, as igm60.g3_cands builds G3's (Car / Van / Truck unchanged in form; Motorcycle on the replacement lane)."""
    new, src, person_with = ctx["new"], ctx["src"], ctx["person_with"]
    for c, L in Ls.items():
        if c == "Car":
            out[f"Car: add {m}"] = ("Car", M.G3.lane_class(M.G3.run_lane(L["name"], L["spec"], {}, "val"), "Car", ids))
        elif c in ("Van", "Truck"):
            o = "Truck" if c == "Van" else "Van"
            ch = M.G3.run_lane(L["name"], L["spec"], {}, "val")
            pair = M.tv_pass(ch if c == "Truck" else src["Truck"], ch if c == "Van" else src["Van"], ids)
            del ch
            same = M.same_boxes(pair[o], new[o], ids)
            print(f"self-check {where} {m} {c}: the class not changed ({o}) equals v60's box for box: {same}", flush=True)
            if not same:
                stop(f"{m} {c}: the class not changed ({o}) differs from v60")
            out[f"{c}: add {m}"] = (c, pair[c])
            out[f"{o}: other class with {m} {c}"] = (o, pair[o])
        else:   # Motorcycle: the joint with seed 1 replaced by the pair
            mo = M.G3.lane_class(M.G3.run_lane(L["name"], L["spec"], {}, "val"), "Motorcycle", ids)
            out[f"Motorcycle: add {m}"] = ("Motorcycle", mo)
            out[f"Person: copies from Motorcycle: add {m}"] = ("Person", person_with(new["Bicycle"], mo))
    print("built pair", m, ", ".join(Ls), flush=True)


def counts(cand):
    for name, (c, p) in cand.items():
        if not name.endswith(": v60"):
            print(f"check: built {name:52s} {len(p)} frames, {sum(len(v) for v in p.values())} boxes", flush=True)


def meta(m, b, info, Ls):
    f1, f2 = files(m)
    return {"_v60": {k: v for k, v in info.items() if k != "combo_classes"},
            "_pair": {"member": m, "seed1": M.PAIR[m], "weights": list(M.PAIR_W), "batch": b, "seed1_file": str(f1), "seed1_md5": I6.md5(f1),
                      "seed2_file": str(f2), "seed2_md5": I6.md5(f2), "script_md5": I6.md5(__file__), "note": NOTE,
                      "lanes": {c: dict(form=L["form"], lane=f"zG3{L['name']}", spec=L["spec"]) for c, L in Ls.items()}}}


def save_moto(cand, m, res):
    name = f"Motorcycle: add {m}"
    if name in cand:
        p = Path(f"output/lumpi/pair60/{m}/motorcycle_val.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cand[name][1]))
        res["_pair"]["motorcycle_boxes"] = {"path": str(p), "md5": I6.md5(p)}
        print("wrote", p, flush=True)


def near(m, b, check):
    out_p = OUT(m, b, "near")
    if not check and out_p.exists():
        stop(f"one reading: {out_p} exists")
    sw, info = I6.v60_set(False, "near")
    Ls = lanes_of(m, sw, b)
    ids = M.S.ids_of("val")
    assert len(ids) == I6.N_NEAR
    HH = {h: M.rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in I6.H}
    gt = {f: [{"box": [float(x) for x in bb], "class": c} for bb, c in v] for f, v in M.S.labels(ids).items()}
    cand, ctx = I6.bases60(sw, ids, "near")
    I6.check_near(sw, info, cand, ctx, gt, HH)
    coverage(m, "near")
    check_pair(m, Ls, "near")
    pair_cands(cand, sw, ids, m, ctx, "near", Ls)
    del ctx
    if check:
        counts(cand)
        print("check mode: no candidate AP computed, nothing written", flush=True)
        return
    keep = {f"{c}: v60" for c in BATCH[b]} | ({"Person: v60"} if b == "moto" else {"Van: v60", "Truck: v60"})
    res = meta(m, b, info, Ls)
    names = [n for n in cand if n in keep or not n.endswith(": v60")]
    for name in names:
        c, p = cand[name]
        res[name] = {"class": c, **{h: M.ap2(p, gt, fr, c) for h, fr in HH.items()}}
    for name in names:
        r = res[name]
        bb = res[f"{r['class']}: v60"]
        r["d_inside"], r["d_before"] = r["inside"][0] - bb["inside"][0], r["before"][0] - bb["before"][0]
        if "Motorcycle" in name and not name.endswith(": v60"):
            r["note"] = M.MOTO_NOTE
        print(f"{name:52s} inside {r['inside'][0]:.4f} ({r['d_inside']:+.4f}) | before {r['before'][0]:.4f} ({r['d_before']:+.4f}) | "
              f"AP@0.7 inside {r['inside'][1]:.4f}", flush=True)
    save_moto(cand, m, res)
    out_p.write_text(json.dumps(res, indent=1))
    print("wrote", out_p, flush=True)


def val(m, b, check):
    out_p = OUT(m, b, "val")
    if not check and out_p.exists():
        stop(f"one reading: {out_p} exists")
    sw, info = I6.v60_set(False, "val")
    Ls = lanes_of(m, sw, b)
    near_res = None
    if not check:
        p = OUT(m, b, "near")
        if not p.exists():
            stop(f"{p} missing (near in neartest_ws, then copy it here)")
        near_res = json.loads(p.read_text())
        nv = near_res.get("_v60", {})
        if nv.get("switches") != sw or nv.get("key") != info.get("key") or nv.get("assume_v59"):
            stop(f"{p} was read on another v60 switch set ({nv.get('switches')}, key {nv.get('key')})")
        if {c: L["spec"] for c, L in near_res["_pair"]["lanes"].items()} != {c: L["spec"] for c, L in Ls.items()}:
            stop(f"{p} was read on other pair lanes")
    ids = M.S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in bb], "class": c} for bb, c in v] for f, v in M.S.labels(ids).items()}
    cand, ctx = I6.bases60(sw, ids, "val")
    I6.check_val(sw, info, cand, gt, ids, False)
    coverage(m, "val")
    check_pair(m, Ls, "val")
    pair_cands(cand, sw, ids, m, ctx, "val", Ls)
    del ctx
    if check:
        counts(cand)
        print("check mode: no candidate AP computed, nothing written", flush=True)
        return
    mains = [n for n in cand if n.endswith(f": add {m}") and not n.startswith("Person: copies")]
    cnt = json.loads(Path("logs/neartest_id_counts.json").read_text())
    nid = lambda h, c: cnt.get(h, {}).get(c, {}).get("distinct_ids", 0)
    res = meta(m, b, info, Ls)
    print(f"gate table for pair {m} ({b}) on v60 (G3's rule: inside +0.01 (>= 5 ids), val a - base >= +{I6.BAR} unrounded (Motorcycle: also "
          f"val not down); veto: touched classes -0.005 on a segment with >= 5 ids (Van and Truck both; Person copies), Person val (copies) not down "
          f"> 0.01)", flush=True)
    H = I6.H
    for name in mains:   # igm60.val60's G3 branch (m != moto2), unchanged
        c = cand[name][0]
        if name not in near_res:
            stop(f"{name}: built here but not in the near reading")
        n = near_res[name]
        base = M.G3.ap(cand[f"{c}: v60"][1], gt, ids, c)
        a = M.G3.ap(cand[name][1], gt, ids, c)
        o = {"Van": "Truck", "Truck": "Van"}.get(c)
        other = near_res[f"{o}: other class with {m} {c}"] if o else None
        pb = near_res[f"Person: copies from {c}: add {m}"] if c in ("Bicycle", "Motorcycle") else None
        touched = [(c, n)] + ([(o, other)] if o else []) + ([("Person", pb)] if pb else [])
        veto = [f"{k} {h} {r['d_' + h]:+.4f}" for k, r in touched for h in H if nid(h, k) >= 5 and r["d_" + h] < -0.005]
        pv = None
        if c in ("Bicycle", "Motorcycle"):
            pv = (M.G3.ap(cand[f"Person: copies from {c}: add {m}"][1], gt, ids, "Person"), M.G3.ap(cand["Person: v60"][1], gt, ids, "Person"))
            if pv[0] - pv[1] < -0.01:
                veto.append(f"Person val (copies) {pv[0] - pv[1]:+.4f}")
        up = a - base >= I6.BAR
        val_ok = (M.moto_val_ok(a, base) if c == "Motorcycle" else True) and up
        ok = bool(nid("inside", c) >= 5 and n["d_inside"] >= 0.01 and val_ok and not veto)
        r = dict(val=a, val_base=base, d_val=a - base, d_inside=n["d_inside"], d_before=n["d_before"], veto=veto, passed=ok, val_up_bar=bool(up),
                 form=Ls[c]["form"])
        if o:
            r.update(other_class=o, other_d_inside=other["d_inside"], other_d_before=other["d_before"])
        if pb:
            r.update(person_d_inside=pb["d_inside"], person_d_before=pb["d_before"])
        if pv:
            r.update(person_val=pv[0], person_val_base=pv[1], person_d_val=pv[0] - pv[1])
        if c == "Motorcycle":
            r["note"] = M.MOTO_NOTE
        res[name] = r
        extra = f" | {o} inside {other['d_inside']:+.4f} before {other['d_before']:+.4f}" if o else ""
        extra += f" | Person (copies) inside {pb['d_inside']:+.4f} before {pb['d_before']:+.4f}" if pb else ""
        extra += f" val {pv[0] - pv[1]:+.4f}" if pv else ""
        print(f"{name:36s} inside {n['inside'][0]:.4f} ({n['d_inside']:+.4f}, {nid('inside', c)} ids) | before {n['d_before']:+.4f} | "
              f"val {a:.4f} ({a - base:+.4f}; +{I6.BAR} bar {'met' if up else 'missed'}){extra} | veto {'; '.join(veto) or 'none'} | "
              f"{'PASS' if ok else 'fail'}", flush=True)
    save_moto(cand, m, res)
    out_p.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out_p, flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    chk = "--check" in argv
    a = [x for x in argv if x != "--check"]
    if len(a) != 3 or a[0] not in ("lanes", "near", "val") or a[1] not in M.PAIR or a[2] not in BATCH or (chk and a[0] == "lanes"):
        sys.exit("usage: python scripts/pair60.py lanes|near|val vmamba_lumpi_s2|dsvt_lumpi_s2 moto|cvt [--check]")
    if a[0] == "lanes":
        lanes(a[1], a[2])
    else:
        {"near": near, "val": val}[a[0]](a[1], a[2], chk)
