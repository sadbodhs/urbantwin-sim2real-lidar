"""dsvt_v2x reading.
Mac CPU. ONE reading, settings fixed before it runs; refuses to run if its output json exists. Builds nothing. v2x_v31's three class chains, each
with only its stage-1 lane rebuilt with dsvt_v2x (trained on an A100 GPU, epoch 20) as one more fusion member at weight 1 (scripts/dsvt_v2x_lanes.py ->
output/v2x/dsvt_v2x_j/s2_in) and v31's own second stages rerun on it on an A100 (x6_refine defaults, epoch 20,
4 views -> output/v2x/dsvt_v2x_j/s2):
  vehicle     X6 R0 on the dsvt_v2x joint lane, X8 decided on the steps-off build, V-G's frozen steps (v2x_opt2_read.build_pair with
              the one refiner file = v2x_vgeom_read.build, the v31 chain); against v31 0.8446 / 0.7532.
              Diagnostic only (not gated): raw X6 R0 on the joint lane against X6 R0 on v31's own lane (v2x_v30's vehicle, 0.8296 / 0.7226).
  pedestrian  R6 over X6b + negatives on the dsvt_v2x joint lane, then P3 (static_keys("v2x")) (= build_v2x_v29.ped, the v31 chain); against
              v31 0.3416 / 0.3321.
  truck       [v31 truck base + dsvt_v2x]: X6b on the joint truck lane (v18 trucks + dsvt_v2x), v2x_plan.build (c12, cap 1 m,
              width >= 2.55 m), then P.size (= build_v2x_v3x.py's truck chain); against v31 0.7804 / 0.8472.
Inputs: output/v2x/dsvt_v2x_j/s2/{vehicle_x6,pedestrian_x6b,pedestrian_neg,truck_x6b}_<s>_ref.json (s = val_i1, test, fresh), logs/dsvt_v2x_j_refine.md5
(the A100 job's record), logs/dsvt_v2x_j_lanes.json, logs/dsvt_v2x_j_lanes.md5.
STOP self-checks, all before any candidate number is computed:
  (3) logs/dsvt_v2x_j_lanes.json is a full lane run whose lane-off check passed box for box on val_i1 / test / fresh x vehicle / pedestrian / truck
      and whose new-lane md5s equal logs/dsvt_v2x_j_lanes.md5; the 12 s2 files equal logs/dsvt_v2x_j_refine.md5 (exactly those 12 files);
  (2) lane-off = v31: v31's chains on v31's own second-stage files reproduce v31 exactly: vehicle (the one-file pair build equals
      v2x_vgeom_read.build box for box) 0.8446 / 0.7532, pedestrian 0.3416 / 0.3321, truck 0.7804 / 0.8472; the diagnostic's X6 R0 base 0.8296 /
      0.7226; P3's static map has v31's 16542 voxels; UT_REALISM_FIT unset; output/v2x/ens/v2x_v31_test.json has md5 6a4360ac.
Gate per class against v31 (one try): the standard V2X gate (r6r7_read_v2x.line: val_i1 +0.003 on all, h1 and h2 with
AP@0.7 not down; fresh +0.003 on all and h2 with AP@0.7 not down). Q7 joint rule: only if two or more V2X tries pass a class.
Writes logs/dsvt_v2x_j_read.json ({"baseline": "v2x_v31", "vehicle": {new, ref, val_gate, fresh_gate, passed, ...}, "pedestrian": {...},
"truck": {...}, "vehicle_x6_r0_diagnostic": {...}, "md5": {12 s2 files}}). --selfcheck-only runs check (2) alone and writes nothing.
    python scripts/dsvt_v2x_read.py [--selfcheck-only]        (Docker CPU, under logs/v2x.lock) -> logs/dsvt_v2x_j_read.json"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import postproc_p123 as PP
import r6r7_read_v2x as RR
import route_read_v2x as RV
import v2x_opt2_read as O2
import v2x_plan as P
import v2x_vgeom_read as VG

LANES = "dsvt_v2x_j"
J, O, XB = Path("output/v2x/fsh_saf_v2x/s2"), Path("output/v2x/fshnet_v2x/s2"), Path("output/v2x/vrcnn_v2x_b/x6bc")
N = Path(f"output/v2x/{LANES}/s2")
V31 = {"vehicle": (0.8446, 0.7532), "pedestrian": (0.3416, 0.3321), "truck": (0.7804, 0.8472)}
V30_X6R0 = (0.8296, 0.7226)
V31_KEYS = 16542
V31_FILE, V31_MD5 = Path("output/v2x/ens/v2x_v31_test.json"), "6a4360ac5b715559f364bbae3edcdb75"   # the baseline being read against
SPLITS = ("val_i1", "test", "fresh")
CL = ("vehicle", "pedestrian", "truck")
FILES = {"vehicle": ["vehicle_x6_{s}_ref.json"], "pedestrian": ["pedestrian_x6b_{s}_ref.json", "pedestrian_neg_{s}_ref.json"],
         "truck": ["truck_x6b_{s}_ref.json"]}
LANE_FILES = [f"output/v2x/{LANES}/s2_in/{c}_{s}.json" for s in SPLITS for c in CL]
OUTJ = Path(f"logs/{LANES}_read.json")


def stop_unless(ok, msg):
    if not ok:
        sys.exit("STOP " + msg)


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def md5_lines(p):
    return dict(reversed(l.split()) for l in Path(p).read_text().splitlines() if l.strip())


def row(c, s, p):
    return (P.score if s == "val_i1" else RV.fresh_score)(p, c)


def ped(s, files, keys):     # = build_v2x_v29.ped / build_v2x_v3x.py --pedestrian v30 / v2x2_read.ped
    return PP.filt(RV.build(files, "pedestrian", RR.KIND[s], "R6"), "pedestrian", RR.LOAD[s], keys)


def trk(s, files):           # = build_v2x_v3x.py's truck: P.size(P.build(v25_paths("truck"), "truck", kind), "truck")
    return P.size(P.build(files, "truck", RR.KIND[s]), "truck")


def inputs():
    """Check (3); returns {path: md5} of the 12 dsvt_v2x second-stage files."""
    lj = json.loads(Path(f"logs/{LANES}_lanes.json").read_text())
    stop_unless(lj.get("mode") == "full" and lj.get("lane_off_passed") is True and "stopped" not in lj and lj.get("lanes") == LANES,
                f"logs/{LANES}_lanes.json is not a full lane run with a passed lane-off check")
    for c in CL:
        for s in SPLITS:
            stop_unless(lj["lane_off"][c].get(s, {}).get("box_for_box") is True, f"lane-off {c} {s} not passed box for box in logs/{LANES}_lanes.json")
    lm = md5_lines(f"logs/{LANES}_lanes.md5")
    stop_unless(set(lm) == set(LANE_FILES), f"logs/{LANES}_lanes.md5 lists {sorted(lm)}, not the 9 dsvt_v2x lanes")
    stop_unless(all(lm[f"output/v2x/{LANES}/s2_in/{c}_{s}.json"] == lj["new"][c][s]["md5"] for c in CL for s in SPLITS),
                f"logs/{LANES}_lanes.md5 differs from logs/{LANES}_lanes.json")
    print(f"check (3): lane run full, lane-off box for box on {list(SPLITS)} x {list(CL)}, lane md5s consistent: True", flush=True)
    rm = md5_lines(f"logs/{LANES}_refine.md5")
    want = {str(N / n.format(s=s)) for c in FILES for n in FILES[c] for s in SPLITS}
    stop_unless(set(rm) == want, f"logs/{LANES}_refine.md5 lists {sorted(rm)}, not the 12 dsvt_v2x refines")
    out = {}
    for p in sorted(want):
        got = md5f(p)
        print(f"{p}: md5 {got} vs logs/{LANES}_refine.md5 {rm[p]}: {got == rm[p]}", flush=True)
        stop_unless(got == rm[p], f"{p} md5 differs from logs/{LANES}_refine.md5")
        out[p] = got
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck-only", action="store_true", help="check (2) only (v31 rebuilt from its own files); writes nothing")
    a = ap.parse_args()
    t0 = time.time()
    if not a.selfcheck_only:
        stop_unless(not OUTJ.exists(), f"{OUTJ} exists: this is a one-try reading (no rerun)")
    stop_unless(not os.environ.get("UT_REALISM_FIT"), "UT_REALISM_FIT is set: it changes P3's static map; unset it")
    stop_unless(VG.D == J, f"v2x_vgeom_read.D is {VG.D}, not v31's vehicle lane {J}")
    got = md5f(V31_FILE)
    print(f"v2x_v31 test file {V31_FILE}: md5 {got} vs {V31_MD5}: {got == V31_MD5}", flush=True)
    stop_unless(got == V31_MD5, f"{V31_FILE} is not v2x_v31 (md5 {V31_MD5})")
    res = {"baseline": "v2x_v31", "baseline_md5": V31_MD5,
           "experiment": "dsvt_v2x (epoch 20) as one more fusion member (weight 1) in v31's vehicle, pedestrian and truck lanes; "
                         "v31's second stages rerun on an A100",
           "gate": "standard V2X gate (val_i1 +0.003 all / h1 / h2, AP@0.7 not down; fresh +0.003 all / h2, AP@0.7 not down), per class, vs v2x_v31"}
    if not a.selfcheck_only:
        res["md5"] = inputs()
    keys = PP.static_keys("v2x")
    print(f"P3 static map voxels {len(keys)} vs {V31_KEYS}: {len(keys) == V31_KEYS}", flush=True)
    stop_unless(len(keys) == V31_KEYS, "P3's static map is not v31's")
    # (2) lane-off = v31, every class, before any candidate number
    base = {c: {} for c in CL}
    diag_base = {}
    for s in RR.LOAD:
        p = ped(s, RR.paths("pedestrian", s, O), keys)
        base["pedestrian"][s] = row("pedestrian", s, p)
        del p
        gc.collect()
        one = O2.build_pair(s, [J / f"vehicle_x6_{s}_ref.json"], True)
        vg = VG.build(s, True)
        same = one == vg
        del vg
        print(f"self-check {s}: one-refiner pair build equals v2x_vgeom_read.build box for box: {same}", flush=True)
        stop_unless(same, "the pair build differs from V-G's build")
        base["vehicle"][s] = row("vehicle", s, one)
        del one
        gc.collect()
        diag_base[s] = row("vehicle", s, RV.build([J / f"vehicle_x6_{s}_ref.json"], "vehicle", RR.KIND[s], "R0"))
        base["truck"][s] = row("truck", s, trk(s, [XB / f"v18_{s}_ref.json"]))
        gc.collect()
    for c in CL:
        got = (base[c]["val_i1"]["all"][0], base[c]["fresh"]["all"][0])
        ok = got == V31[c]
        print(f"self-check (lane-off) v2x_v31 {c}: val_i1 {got[0]} / fresh {got[1]} vs {V31[c]}: {ok}", flush=True)
        stop_unless(ok, f"the base does not rebuild v2x_v31's {c}")
    got = (diag_base["val_i1"]["all"][0], diag_base["fresh"]["all"][0])
    print(f"self-check X6 R0 on v31's own vehicle lane (diagnostic base, v2x_v30's vehicle): val_i1 {got[0]} / fresh {got[1]} vs {V30_X6R0}: "
          f"{got == V30_X6R0}", flush=True)
    stop_unless(got == V30_X6R0, "X6 R0 on v31's own vehicle lane does not rebuild v2x_v30's vehicle")
    if a.selfcheck_only:
        print(f"check (2) passed: v31's vehicle, pedestrian and truck rebuild exactly ({time.time() - t0:.0f} s); nothing written", flush=True)
        return
    # candidate
    new = {c: {} for c in CL}
    diag_new = {}
    for s in RR.LOAD:
        p = ped(s, [N / n.format(s=s) for n in FILES["pedestrian"]], keys)
        new["pedestrian"][s] = row("pedestrian", s, p)
        del p
        gc.collect()
        p = O2.build_pair(s, [N / f"vehicle_x6_{s}_ref.json"], True)
        new["vehicle"][s] = row("vehicle", s, p)
        del p
        gc.collect()
        diag_new[s] = row("vehicle", s, RV.build([N / f"vehicle_x6_{s}_ref.json"], "vehicle", RR.KIND[s], "R0"))
        new["truck"][s] = row("truck", s, trk(s, [N / f"truck_x6b_{s}_ref.json"]))
        gc.collect()
    name = {"vehicle": "dsvt_v2x vehicle (joint lane + dsvt_v2x; X6 R0, X8, V-G frozen steps)",
            "pedestrian": "dsvt_v2x pedestrian (joint lane + dsvt_v2x; X6b + negatives, R6 + P3)",
            "truck": "dsvt_v2x truck (v18 trucks + dsvt_v2x; X6b, c12, P.size)"}
    for c in CL:
        res[c] = RR.line(name[c], new[c], base[c], "v2x_v31")
        res[c]["files"] = [str(N / n) for n in FILES[c]]
    res["vehicle_x6_r0_diagnostic"] = RR.line("(diagnostic, not gated) dsvt_v2x vehicle raw X6 R0", diag_new, diag_base, "X6 R0 on v31's own lane (v2x_v30 vehicle)")
    res["vehicle_x6_r0_diagnostic"]["note"] = "information only; the vehicle decision is the V-G joint line above"
    passed = [c for c in CL if res[c]["passed"]]
    res["passed_classes"] = passed
    res["q7_note"] = "Q7 joint reading only if two or more V2X tries pass the same class"
    print(f"dsvt_v2x reading vs v2x_v31: passed classes {passed if passed else 'none'} ({time.time() - t0:.0f} s)", flush=True)
    OUTJ.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUTJ}", flush=True)


if __name__ == "__main__":
    main()
