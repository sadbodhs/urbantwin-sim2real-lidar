"""v2x_v32 build: v2x_v31 with each class taken from the source its switch names. CPU only, no new readings or
settings. Anchor: output/v2x/ens/v2x_v31_test.json, the shipped file (md5 V31_MD5 = predictions.json of v2x_v31_vgeom-vehicle-flagged.zip).
Nothing falls back to v30: there is no v30 switch, no v30 file is read, and every class left at v31 is written from v2x_v31's own file.
A switch value other than v31 is accepted only if its reading json (baseline v2x_v31) marks it passed. Stacking rule: two
candidates passing on one class get one joint reading, and the build takes the switch value that reading names (an entry with a "switch" key
must name the value asked for); otherwise the build only prints which other candidates of the class passed.
  --vehicle     v31   V-G: X6 on output/v2x/fsh_saf_v2x/s2, X8 decided on the steps-off build, V-G steps on (scripts/v2x_vgeom_read.py
                      build(split, True)); 0.8446 / 0.7532
                2c    [X6, neg_g] 0.5 / 0.5 per lane box, X8 on the steps-off build, V-G steps (scripts/v2x_opt2_read.py build_pair; neg_g =
                      output/v2x/neg_g/vehicle_neg_g_<s>_ref.json), as scripts/v2x_opt3_read.py reads it; logs/v2x_opt3_read.json "vehicle"
                dsvt_j  dsvt_v2x: X6 on the dsvt_v2x joint lane (v31's lane + dsvt_v2x at weight 1,
                      scripts/dsvt_v2x_lanes.py; A100 refine output/v2x/dsvt_v2x_j/s2/vehicle_x6_<s>_ref.json), X8 decided on its own
                      steps-off build, V-G frozen steps (build_pair with that one file), exactly as scripts/dsvt_v2x_read.py reads it;
                      logs/dsvt_v2x_j_read.json "vehicle". Locked inputs (DSVT_J_MD5): the reading json, the refine and lane md5 lists, and
                      the 3 vehicle files (val_i1 / fresh / test, also checked against the reading json's own md5 record)
                q7_dsvt_vmamba  the Q7 joint: X6 on output/v2x/q7_dsvt_vmamba_v2x_j/s2 (v31's vehicle lane + dsvt_v2x + vmamba_v2x
                      at weight 1, scripts/q7_v2x_lanes.py / q7_v2x_refine.sh on the 3090), the same chain as dsvt_j (build_pair with that one
                      file); its reading is AGAINST v2x_v32 (scripts/q7_v2x_read.py -> logs/q7_dsvt_vmamba_v2x_j_read.json "vehicle": baseline
                      v2x_v32 / adc281c6, ref = v32's vehicle 0.856 / 0.7663, passed), and the 3 vehicle files (val_i1 / fresh / test) must equal
                      that reading's md5 record. --q7-selfcheck (test only, --outdir not output/v2x/ens): the q7 path with v32's own dsvt_j files
                      in place of the joint files must rebuild v2x_v32_test.json byte for byte (adc281c6); the reading checks are not applied.
                q7_dsvt_vmamba_pvn  the second Q7 joint: as q7_dsvt_vmamba with pv_v2x_n added (dsvt_v2x + vmamba_v2x + pv_v2x_n
                      at weight 1, output/v2x/q7_dsvt_vmamba_pvn_v2x_j/s2); reading logs/q7_dsvt_vmamba_pvn_v2x_j_read.json, the same checks.
  --pedestrian  v31   R6 over X6b + negatives on output/v2x/fshnet_v2x/s2, then P3 (scripts/build_v2x_v29.py ped, static_keys("v2x")); 0.3416 / 0.3321
                opt3  R6 over X6b (fshnet_v2x/s2) + neg_g (output/v2x/neg_g/pedestrian_neg_g_<s>_ref.json), then P3, as v2x_opt3_read.py
                      reads it; logs/v2x_opt3_read.json "pedestrian"
                5b    TODO (see P5B below): FSHNet clean-walker retrain fshnet_v2x_g in place of fshnet_v2x in the pedestrian lane,
                      with the then-best pedestrian second-stage set; R6 over the P5B files (or --p5b-files), then P3; logs/v2x_5b_read.json
                      "pedestrian"; refuses unless that json marks it passed
                pz    P-Z: v31's chain (B.ped) with z <- z + 0.03 on every pedestrian
                      box; its reading is AGAINST v2x_v32 (scripts/v2x_pz_read.py -> logs/v2x_pz_read.json "pedestrian": baseline v2x_v32 /
                      adc281c6, dz 0.03, ref = v32's pedestrian 0.3416 / 0.3321, passed). A val-informed constant: the package stays flagged.
                      --pz-selfcheck (test only, --outdir not output/v2x/ens): --vehicle dsvt_j --pedestrian pz --truck v31 with the shift set to 0
                      must rebuild v2x_v32_test.json byte for byte (adc281c6); the reading checks are not applied.
  --truck       v31   P.size(P.build(P.v25_paths("truck", s), "truck", kind), "truck") (X6b on the v18 trucks, scripts/build_v2x_v30.py:49); 0.7804 / 0.8472
                2b    as v31 with output/v2x/neg_g/truck_neg_g_<s>_ref.json as a second refiner at equal weight, as v2x_opt3_read.py reads it;
                      logs/v2x_opt3_read.json "truck"
  synthetic/    not built here; package with output/v2x/realism_r2 exactly as v31 (scripts/package_check.sh).
Checks (STOP on any miss, all before anything is written): the anchor's md5; P3's static map has v31's 16542 voxels; per class, val_i1 / fresh
rebuilt from the class's chain equal v31's constants (v31 classes, AP@0.5 all) or the reading's "new" rows (candidates: all / h1 / h2, AP@0.5 and
AP@0.7, to the 4 decimals stored); on test, every v31 class rebuilt from its chain equals v2x_v31 box for box and is written from v2x_v31's file,
every changed class equals its own build box for box; with all three classes at v31 the output must be byte-equal to v2x_v31_test.json.
Test-file assembly as scripts/build_v2x_v3x.py: per frame, the anchor's boxes of unchanged classes (file order) + the changed classes' boxes
(vehicle, pedestrian, truck), fields box / score / class, no rounding, sorted by score descending (stable), json.dumps.
Writes <outdir>/<version>_test.json (default output/v2x/ens/v2x_v32_test.json); never v2x_v29 / v2x_v30 / v2x_v31; no overwrite without --force.
    python scripts/build_v2x_v32.py --vehicle v31 --pedestrian opt3 --truck v31 2>&1 | tee logs/build_v2x_v32.log      (Docker CPU)"""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import build_v2x_v29 as B
import postproc_p123 as PP
import route_read_v2x as RV
import v2x_fresh_read as VF
import v2x_opt2_read as O2
import v2x_plan as P
import v2x_vgeom_read as VG

CL = ("vehicle", "pedestrian", "truck")
V31 = {"vehicle": (0.8446, 0.7532), "pedestrian": (0.3416, 0.3321), "truck": (0.7804, 0.8472)}   # logs/build_v2x_v31.log
V31_FILE = P.E / "v2x_v31_test.json"
V31_MD5 = "6a4360ac5b715559f364bbae3edcdb75"   # v2x_v31_test.json = predictions.json in the submitted v2x_v31_vgeom-vehicle-flagged.zip
V31_KEYS = 16542                               # "P3 static map voxels" in logs/build_v2x_v31.log
PROTECTED = ("v2x_v29", "v2x_v30", "v2x_v31", "v2x_v32")
V32_MD5 = "adc281c6c553c2409f0ef2dedc449db0"   # v2x_v32_test.json = predictions.json of v2x_v32_dsvt-vehicle-flagged.zip (the incumbent)
V32_VEH = (0.856, 0.7663)                      # v32's vehicle (logs/dsvt_v2x_j_read.json "vehicle" new, AP@0.5 all)
V32_PED = (0.3416, 0.3321)                     # v32's pedestrian (= v31's chain)
PZ = {"dz": 0.03}                              # P-Z shift; --pz-selfcheck sets 0.0
J = Path("output/v2x/fsh_saf_v2x/s2")          # v31's vehicle lane (= VG.D)
O = Path("output/v2x/fshnet_v2x/s2")           # v31's pedestrian lane
NG = Path("output/v2x/neg_g")                  # VG-1f refiner vrcnn_v2x_neg_g epoch 20 on v31's lanes (3090, scripts/vg1f_refine.sh)
# TODO (5b): fill in from the 5b reading once its spec is fixed: the fshnet_v2x_g lane's second-stage files for the then-best
# pedestrian set (X6b + negatives, or X6b + neg_g if option 3 passed), one pattern per member with {s} for the split. --p5b-files overrides
# this list (comma-separated). If logs/v2x_5b_read.json's "pedestrian" entry carries "files", they must equal these patterns.
P5B = ["output/v2x/fshnet_v2x_g/s2/pedestrian_x6b_{s}_ref.json", "output/v2x/fshnet_v2x_g/s2/pedestrian_neg_{s}_ref.json"]
DJ = Path("output/v2x/dsvt_v2x_j/s2")        # dsvt_v2x joint lanes' second stages (run on an A100)
DSVT_J_MD5 = {"logs/dsvt_v2x_j_read.json": "3e6f5f40c000c11aebf54226d5d39db3",      # the one reading (vehicle passed)
              "logs/dsvt_v2x_j_refine.md5": "ae7081045552a47d3ef9bf2c7ecb1b12",     # the A100 job's record of the 12 refine outputs
              "logs/dsvt_v2x_j_lanes.md5": "48f6a720d17d9fd23ac0e3790a566556",      # the 9 lane inputs staged for the A100 run
              "output/v2x/dsvt_v2x_j/s2/vehicle_x6_val_i1_ref.json": "66e4a636744ba661cba779d4780c2c61",
              "output/v2x/dsvt_v2x_j/s2/vehicle_x6_fresh_ref.json": "aa052d3a65eff209ed79afa036f6692c",
              "output/v2x/dsvt_v2x_j/s2/vehicle_x6_test_ref.json": "8c73cc7d2b2fea76f0099150ef247e6f"}
Q7DIR = {"q7_dsvt_vmamba": Path("output/v2x/q7_dsvt_vmamba_v2x_j/s2"),
         "q7_dsvt_vmamba_pvn": Path("output/v2x/q7_dsvt_vmamba_pvn_v2x_j/s2")}   # Q7 joints (vehicle); --q7-selfcheck points the chosen one at DJ
READING = {("vehicle", "2c"): ("logs/v2x_opt3_read.json", "vehicle"),
           ("vehicle", "q7_dsvt_vmamba"): ("logs/q7_dsvt_vmamba_v2x_j_read.json", "vehicle"),
           ("vehicle", "q7_dsvt_vmamba_pvn"): ("logs/q7_dsvt_vmamba_pvn_v2x_j_read.json", "vehicle"),
           ("pedestrian", "pz"): ("logs/v2x_pz_read.json", "pedestrian"),
           ("pedestrian", "opt3"): ("logs/v2x_opt3_read.json", "pedestrian"),
           ("truck", "2b"): ("logs/v2x_opt3_read.json", "truck"),
           ("pedestrian", "5b"): ("logs/v2x_5b_read.json", "pedestrian"),
           ("vehicle", "dsvt_j"): ("logs/dsvt_v2x_j_read.json", "vehicle")}
ok_all = True


def stop(msg):
    sys.exit("STOP " + msg)


def expect(name, got, want):
    global ok_all
    same = got == want
    ok_all &= same
    print(f"{name}: {got} vs {want} | equal: {same}", flush=True)


def same_class(name, a, b, c):
    global ok_all
    same = set(a) == set(b) and all(B.canon([d for d in a[f] if d["class"] == c]) == B.canon([d for d in b[f] if d["class"] == c]) for f in a)
    ok_all &= same
    print(f"test {name} {c}: box for box equal: {same}", flush=True)


def ped_r6(files, s, keys):
    return PP.filt(RV.build(files, "pedestrian", B.KIND[s], "R6"), "pedestrian", B.LOAD[s], keys)


def pz_shift(p, dz):
    """P-Z: every pedestrian box z + dz (box[2]); every other field, class and the order unchanged (= scripts/v2x_pz_read.py shift)."""
    return {f: [dict(d, box=[d["box"][0], d["box"][1], d["box"][2] + dz] + list(d["box"][3:])) if d["class"] == "pedestrian" else d for d in v]
            for f, v in p.items()}


def truck(s, extra=None):
    return P.size(P.build(P.v25_paths("truck", s) + ([extra] if extra else []), "truck", B.KIND[s]), "truck")


BUILD = {"vehicle": {"v31": lambda s, keys: VG.build(s, True),
                     "2c": lambda s, keys: O2.build_pair(s, [J / f"vehicle_x6_{s}_ref.json", NG / f"vehicle_neg_g_{s}_ref.json"], True),
                     "dsvt_j": lambda s, keys: O2.build_pair(s, [DJ / f"vehicle_x6_{s}_ref.json"], True),
                     "q7_dsvt_vmamba": lambda s, keys: O2.build_pair(s, [Q7DIR["q7_dsvt_vmamba"] / f"vehicle_x6_{s}_ref.json"], True),
                     "q7_dsvt_vmamba_pvn": lambda s, keys: O2.build_pair(s, [Q7DIR["q7_dsvt_vmamba_pvn"] / f"vehicle_x6_{s}_ref.json"], True)},
         "pedestrian": {"v31": lambda s, keys: B.ped(s, keys),
                        "opt3": lambda s, keys: ped_r6([O / f"pedestrian_x6b_{s}_ref.json", NG / f"pedestrian_neg_g_{s}_ref.json"], s, keys),
                        "5b": lambda s, keys: ped_r6([Path(p.format(s=s)) for p in P5B], s, keys),
                        "pz": lambda s, keys: pz_shift(B.ped(s, keys), PZ["dz"])},
         "truck": {"v31": lambda s, keys: truck(s),
                   "2b": lambda s, keys: truck(s, NG / f"truck_neg_g_{s}_ref.json")}}


def score(c, s, p):
    if s == "val_i1":
        return P.score(p, c)
    return (VF.score if c == "truck" else RV.fresh_score)(p, c)


def entry(c, sw):
    """The reading's entry for --<c> <sw>, or None if the json or the entry is missing."""
    p, key = READING[(c, sw)]
    if not Path(p).exists():
        return None, p
    return json.loads(Path(p).read_text()), p


def wanted(c, sw):
    """{split: row} the class must score: v31's AP@0.5 all, or the passed reading's full "new" rows after its baseline / ref checks."""
    if sw == "v31":
        return {"val_i1": V31[c][0], "fresh": V31[c][1]}
    r, p = entry(c, sw)
    if r is None:
        stop(f"{p} not found: no reading for --{c} {sw} yet")
    if sw in Q7DIR:   # Q7 joint: read against the incumbent v2x_v32
        e = r.get(READING[(c, sw)][1])
        if r.get("baseline") != "v2x_v32" or r.get("baseline_md5") != V32_MD5:
            stop(f"{p} baseline is {r.get('baseline')} / {r.get('baseline_md5')}, not v2x_v32 / {V32_MD5}")
        if not isinstance(e, dict) or e.get("passed") is not True:
            stop(f"{p} does not mark --{c} {sw} passed")
        ref = (e["ref"]["val_i1"]["all"][0], e["ref"]["fresh"]["all"][0])
        if ref != V32_VEH:
            stop(f"{p} {c} ref {ref} is not v32's vehicle {V32_VEH}")
        rec = r.get("md5", {})
        for s in ("val_i1", "fresh", "test"):
            f = Q7DIR[sw] / f"vehicle_x6_{s}_ref.json"
            got = hashlib.md5(f.read_bytes()).hexdigest()
            print(f"{f}: md5 {got} vs {p}'s record {rec.get(str(f))}: {got == rec.get(str(f))}", flush=True)
            if got != rec.get(str(f)):
                stop(f"{f} is not the file {p} read")
        return {s: e["new"][s] for s in ("val_i1", "fresh")}
    if sw == "pz":   # P-Z: read against the incumbent v2x_v32
        e = r.get(READING[(c, sw)][1])
        if r.get("baseline") != "v2x_v32" or r.get("baseline_md5") != V32_MD5 or r.get("dz") != PZ["dz"]:
            stop(f"{p} is not the P-Z reading against v2x_v32 / {V32_MD5} with dz {PZ['dz']} ({r.get('baseline')}, {r.get('baseline_md5')}, {r.get('dz')})")
        if not isinstance(e, dict) or e.get("passed") is not True:
            stop(f"{p} does not mark --{c} {sw} passed")
        ref = (e["ref"]["val_i1"]["all"][0], e["ref"]["fresh"]["all"][0])
        if ref != V32_PED:
            stop(f"{p} {c} ref {ref} is not v32's pedestrian {V32_PED}")
        return {s: e["new"][s] for s in ("val_i1", "fresh")}
    if r.get("baseline") != "v2x_v31":
        stop(f"{p} baseline is {r.get('baseline')}, not v2x_v31: this build adds readings against v31 only")
    e = r.get(READING[(c, sw)][1])
    if not isinstance(e, dict) or e.get("passed") is not True:
        stop(f"{p} does not mark --{c} {sw} passed")
    if e.get("switch", sw) != sw:
        stop(f"{p} (joint reading) names --{c} {e['switch']}, not {sw}")
    ref = (e["ref"]["val_i1"]["all"][0], e["ref"]["fresh"]["all"][0])
    if ref != V31[c]:
        stop(f"{p} {c} ref {ref} is not v31's {V31[c]}")
    if sw == "5b" and "files" in e and e["files"] != P5B:
        stop(f"{p} read the files {e['files']}, this build would use {P5B}")
    if sw == "5b" and "files" not in e:
        print(f"NOTE {p} does not list its files; the 5b files {P5B} are checked only by the scores below", flush=True)
    if sw == "dsvt_j":   # locked inputs; the vehicle files must also be the ones the reading md5-checked (test included; test is not scored)
        for f, m in DSVT_J_MD5.items():
            got = hashlib.md5(Path(f).read_bytes()).hexdigest()
            print(f"{f}: md5 {got} vs locked {m}: {got == m}", flush=True)
            if got != m:
                stop(f"{f} md5 {got} is not the locked {m}")
        rec = r.get("md5", {})
        for s in ("val_i1", "fresh", "test"):
            f = str(DJ / f"vehicle_x6_{s}_ref.json")
            if rec.get(f) != DSVT_J_MD5[f]:
                stop(f"{f} is not the file {p} read ({rec.get(f)})")
        print(f"dsvt_j: the 3 vehicle files equal {p}'s md5 record: True", flush=True)
    return {s: e["new"][s] for s in ("val_i1", "fresh")}


def others(c, sw):
    out = []
    for (cc, v) in READING:
        if cc == c and v != sw:
            r, _ = entry(cc, v)
            e = r.get(READING[(cc, v)][1]) if r and r.get("baseline") == "v2x_v31" else None
            if isinstance(e, dict) and e.get("passed") is True:
                out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for c in CL:
        ap.add_argument(f"--{c}", choices=sorted(BUILD[c]), default="v31")
    ap.add_argument("--version", default="v2x_v32", help="output name: <outdir>/<version>_test.json")
    ap.add_argument("--outdir", default=str(P.E), help="output directory (tests: a scratch directory)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing <outdir>/<version>_test.json")
    ap.add_argument("--p5b-files", help="TODO: --pedestrian 5b member files, comma-separated patterns with {s}")
    ap.add_argument("--q7-selfcheck", action="store_true", help="test: the --vehicle q7_* path on v32's dsvt_j files must rebuild v2x_v32 (adc281c6)")
    ap.add_argument("--pz-selfcheck", action="store_true", help="test: --pedestrian pz with the shift 0 (and --vehicle dsvt_j) must rebuild v2x_v32 (adc281c6)")
    a = ap.parse_args()
    if a.pz_selfcheck:
        if a.pedestrian != "pz" or a.vehicle != "dsvt_j" or a.truck != "v31" or a.q7_selfcheck:
            stop("--pz-selfcheck needs --vehicle dsvt_j --pedestrian pz --truck v31 (and no --q7-selfcheck)")
        if Path(a.outdir).resolve() == Path(P.E).resolve():
            stop("--pz-selfcheck writes only to a scratch --outdir")
        PZ["dz"] = 0.0
        print("pz self-check: --pedestrian pz with dz 0.0; the output must be v2x_v32 (adc281c6)", flush=True)
    if a.q7_selfcheck:
        if a.vehicle not in Q7DIR or a.pedestrian != "v31" or a.truck != "v31":
            stop("--q7-selfcheck needs --vehicle q7_* with pedestrian and truck v31")
        if Path(a.outdir).resolve() == Path(P.E).resolve():
            stop("--q7-selfcheck writes only to a scratch --outdir")
        Q7DIR[a.vehicle] = DJ
        print(f"q7 self-check: --vehicle {a.vehicle} reads {DJ} (v32's dsvt_j files); the output must be v2x_v32 ({V32_MD5})", flush=True)
    src = {c: getattr(a, c) for c in CL}
    out_p = Path(a.outdir) / f"{a.version}_test.json"
    if a.version in PROTECTED:
        stop(f"{a.version} is a built version; this script never writes it")
    if not Path(a.outdir).is_dir():
        stop(f"{a.outdir} is not a directory")
    if out_p.exists() and not a.force:
        stop(f"{out_p} exists (pass --force to overwrite)")
    if os.environ.get("UT_REALISM_FIT"):
        stop("UT_REALISM_FIT is set: it changes P3's static map; unset it")
    if a.p5b_files:
        P5B[:] = a.p5b_files.split(",")
    raw = V31_FILE.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != V31_MD5:
        stop(f"{V31_FILE} md5 {md5} is not the shipped v31's {V31_MD5}")
    if VG.D != J:
        stop(f"v2x_vgeom_read.D is {VG.D}, not v31's vehicle lane {J}")
    want = {c: wanted(c, "dsvt_j" if (a.q7_selfcheck and c == "vehicle") else "v31" if (a.pz_selfcheck and c == "pedestrian") else src[c]) for c in CL}
    print(f"{a.version}: vehicle {src['vehicle']}, pedestrian {src['pedestrian']}, truck {src['truck']} | anchor {V31_FILE} (md5 {md5})", flush=True)
    if src["pedestrian"] == "5b":
        print("5b files:", P5B, flush=True)
    for c in CL:
        o = others(c, src[c])
        if o:
            print(f"NOTE other {c} candidates marked passed: {o}; stacking rule: one joint reading names the value to take", flush=True)
    keys = PP.static_keys("v2x")
    expect("P3 static map voxels", len(keys), V31_KEYS)
    for s in ("val_i1", "fresh"):
        for c in CL:
            b = BUILD[c][src[c]](s, keys)
            row = score(c, s, b)
            del b
            gc.collect()
            if src[c] == "v31" or (a.pz_selfcheck and c == "pedestrian"):
                expect(f"{a.version} {s} {c} ({src[c]})", round(float(row["all"][0]), 4), round(want[c][s], 4))
            else:
                print(f"{a.version} {s} {c} ({src[c]}) AP@0.5 all {row['all'][0]} vs reading {want[c][s]['all'][0]}", flush=True)
                expect(f"{a.version} {s} {c} ({src[c]}) all / h1 / h2 rows", {h: [round(float(x), 4) for x in row[h]] for h in ("all", "h1", "h2")},
                       {h: [round(float(x), 4) for x in want[c][s][h]] for h in ("all", "h1", "h2")})
    v31 = json.loads(raw)
    new = {}
    for c in CL:
        b = BUILD[c][src[c]]("test", keys)
        if set(b) != set(v31):
            stop(f"test frames differ ({c})")
        if src[c] == "v31":
            same_class(f"{c} rebuilt from v31's chain vs v2x_v31", b, v31, c)
        else:
            new[c] = b
        del b
        gc.collect()
    out = {f: sorted([B.keep(d) for d in v31[f] if d["class"] not in new] + [B.keep(d) for c in CL if c in new for d in new[c][f] if d["class"] == c],
                     key=lambda d: -d["score"]) for f in v31}
    for c in CL:
        if c in new:
            same_class(f"{a.version} vs {src[c]} build", out, new[c], c)
        else:
            same_class(f"{a.version} vs v2x_v31", out, v31, c)
    n = {c: sum(d["class"] == c for x in out.values() for d in x) for c in CL}
    n31 = {c: sum(d["class"] == c for x in v31.values() for d in x) for c in CL}
    print(f"boxes {a.version}", n, sum(n.values()), "| v31", n31, sum(n31.values()), flush=True)
    for c in new:
        cn = sum(d["class"] == c and d["score"] >= 0.3 for x in out.values() for d in x)
        c31 = sum(d["class"] == c and d["score"] >= 0.3 for x in v31.values() for d in x)
        print(f"confident {c} (score >= 0.3) v31", c31, f"-> {a.version}", cn, flush=True)
    text = json.dumps(out)
    out_md5 = hashlib.md5(text.encode()).hexdigest()
    if a.q7_selfcheck:
        expect("q7 self-check: output md5 vs v2x_v32_test.json", out_md5, V32_MD5)
    if a.pz_selfcheck:
        expect("pz self-check (dz 0): output md5 vs v2x_v32_test.json", out_md5, V32_MD5)
    if not new:
        expect("all classes v31: output equals v2x_v31 in file order", out == v31, True)
        expect("all classes v31: output md5 vs v2x_v31_test.json", out_md5, V31_MD5)
    if not ok_all:
        stop("a check failed; nothing written")
    if out_p.exists() and not a.force:
        stop(f"{out_p} appeared during the run (pass --force to overwrite)")
    out_p.write_text(text)
    print(f"wrote {out_p} (md5 {out_md5})", flush=True)


if __name__ == "__main__":
    main()
