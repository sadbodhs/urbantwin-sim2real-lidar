"""Next V2X build after v2x_v30: v2x_v30 with each class taken from the source its switch names (Nov 12; prepared before the FSHNet V2X seed 2
reading). CPU only, no new readings or settings. A switch value other than v30 is only allowed when that class's reading passed.
  --vehicle     v30   joint second stage (output/v2x/fsh_saf_v2x/s2), route R0, no P3 (scripts/build_v2x_v30.py veh)
                fs2   FSHNet seed 2 as a both-seeds member (output/v2x/fsh2_v2x/s2/vehicle_x6_*), route R0, no P3; exactly as
                      scripts/r6r7_read_v2x.py fs2 builds "vehicle"; expected scores from logs/fs2_read_v2x.json
                vg    V-G (rule exception; frozen spec): v30's vehicle route with V-G's steps, exactly as
                      scripts/v2x_vgeom_read.py builds it (VG.build(split, True), scores = v30's); expected 0.8446 / 0.7532 from
                      logs/vgeom_read_v2x.json; on test it must equal output/v2x/ens/vgeom_vehicle_test.json box for box
                saf2  TODO: SAFDNet V2X seed 2 (joint lane as v30 with SAFDNet s1 0.5 + s2 0.5, X6 refiner); refuses to run until its reading exists
                joint TODO: both seed-2 models in one lane; refuses to run until its reading exists
  --pedestrian  x6bz  X6b-Z: output/v2x/fshnet_v2x/s2 X6b + negatives + output/v2x/x6bz/s2 X6b-Z, route R6, then P3; exactly as
                      r6r7_read_v2x.py x6bz builds "pedestrian"; expected scores from logs/x6bz_read_v2x.json (passed, baseline v2x_v30)
  --pedestrian  v30   FSHNet second stage (output/v2x/fshnet_v2x/s2) X6b + negatives, route R6, then P3 (static_keys("v2x")) (= v29 = v30)
                fs2   output/v2x/fsh2_v2x/s2 X6b + negatives, route R6, then P3; exactly as r6r7_read_v2x.py fs2 builds "pedestrian" on a v30 baseline
  truck         always v2x_v30's (= v29 = v27), box for box from output/v2x/ens/v2x_v30_test.json; val_i1 / fresh from the v25 truck chain.
  synthetic/    not built here; package with output/v2x/realism_r2 exactly as v30 (scripts/package_check.sh).
Checks (STOP on any miss): per class, val_i1 / fresh from the combined predictions equal the class's reading (v30: vehicle 0.8296 / 0.7226,
pedestrian 0.3416 / 0.3321, truck 0.7804 / 0.8472; fs2: the "new" values in the reading json, which must say passed and baseline v2x_v30);
on test, every v30 class rebuilt from its sources equals v2x_v30 box for box and is written from v2x_v30's file; changed classes equal their own
build box for box. Writes output/v2x/ens/<version>_test.json (never v2x_v29 / v2x_v30; no overwrite without --overwrite).
    python scripts/build_v2x_v3x.py --version v2x_v31 --vehicle fs2 --pedestrian v30      (Docker CPU)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import build_v2x_v29 as B
import build_v2x_v30 as B30
import postproc_p123 as PP
import route_read_v2x as RV
import v2x_fresh_read as VF
import v2x_plan as P
import v2x_vgeom_read as VG

F2 = Path("output/v2x/fsh2_v2x/s2")
V30 = {"vehicle": (0.8296, 0.7226), "pedestrian": (0.3416, 0.3321), "truck": (0.7804, 0.8472)}
READING = {"fs2": "logs/fs2_read_v2x.json", "vg": "logs/vgeom_read_v2x.json", "x6bz": "logs/x6bz_read_v2x.json"}
O = Path("output/v2x/fshnet_v2x/s2")
XZ = Path("output/v2x/x6bz/s2")
ok_all = True


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


def todo(what):
    def stub(s, keys):
        sys.exit(f"STOP {what} is a TODO stub: write its reading script first, then fill in exactly that reading's files and route rule here")
    return stub


VEH = {"v30": lambda s, keys: B30.veh(s),
       "fs2": lambda s, keys: RV.build([F2 / f"vehicle_x6_{s}_ref.json"], "vehicle", B.KIND[s], "R0"),
       "vg": lambda s, keys: VG.build(s, True),   # VG.D = v30's joint second stages (output/v2x/fsh_saf_v2x/s2), unchanged
       # TODO saf2: fill in from the SAFDNet V2X seed 2 reading (joint lane as v30, SAFDNet s1 0.5 + s2 0.5, X6 refiner, R0) once it exists.
       "saf2": todo("--vehicle saf2"),
       # TODO joint: fill in from the joint reading (both seed-2 models) once it exists.
       "joint": todo("--vehicle joint")}
PED = {"v30": lambda s, keys: B.ped(s, keys),
       "fs2": lambda s, keys: PP.filt(RV.build([F2 / f"pedestrian_x6b_{s}_ref.json", F2 / f"pedestrian_neg_{s}_ref.json"], "pedestrian",
                                               B.KIND[s], "R6"), "pedestrian", B.LOAD[s], keys),
       "x6bz": lambda s, keys: PP.filt(RV.build([O / f"pedestrian_x6b_{s}_ref.json", O / f"pedestrian_neg_{s}_ref.json",
                                                 XZ / f"pedestrian_x6bz_{s}_ref.json"], "pedestrian", B.KIND[s], "R6"), "pedestrian", B.LOAD[s], keys)}


def wanted(cls, src):
    """(val_i1, fresh) the class must score: v30's values, or the reading's "new" values after its pass / baseline checks."""
    if src == "v30":
        return V30[cls]
    p = Path(READING[src])
    if not p.exists():
        sys.exit(f"STOP {p} not found: no {src} reading yet")
    r = json.loads(p.read_text())
    if src == "vg":   # v2x_vgeom_read.py's json: base / new / passed (vehicle only)
        if cls != "vehicle" or not r.get("passed"):
            sys.exit(f"STOP the V-G reading did not pass ({p})")
        if (r["base"]["val_i1"]["all"][0], r["base"]["fresh"]["all"][0]) != V30["vehicle"]:
            sys.exit(f"STOP the V-G reading's base is not v30's vehicle {V30['vehicle']}")
        return r["new"]["val_i1"]["all"][0], r["new"]["fresh"]["all"][0]
    if r.get("baseline") != "v2x_v30":
        sys.exit(f"STOP {p} baseline is {r.get('baseline')}, not v2x_v30: this build adds {src} to v30 only")
    if not r[cls]["passed"]:
        sys.exit(f"STOP the {src} {cls} reading did not pass")
    ref = (r[cls]["ref"]["val_i1"]["all"][0], r[cls]["ref"]["fresh"]["all"][0])
    if ref != V30[cls]:
        sys.exit(f"STOP the {src} {cls} reading's baseline {ref} is not v30's {V30[cls]}")
    return r[cls]["new"]["val_i1"]["all"][0], r[cls]["new"]["fresh"]["all"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="output name, e.g. v2x_v31 -> output/v2x/ens/v2x_v31_test.json")
    ap.add_argument("--vehicle", choices=sorted(VEH), default="v30")
    ap.add_argument("--pedestrian", choices=sorted(PED), default="v30")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out_p = P.E / f"{a.version}_test.json"
    if a.version in ("v2x_v29", "v2x_v30"):
        sys.exit(f"STOP {out_p} is a built version")
    if out_p.exists() and not a.overwrite:
        sys.exit(f"STOP {out_p} exists (pass --overwrite)")
    src = {"vehicle": a.vehicle, "pedestrian": a.pedestrian, "truck": "v30"}
    fn = {"vehicle": VEH[a.vehicle], "pedestrian": PED[a.pedestrian]}
    if a.vehicle in ("saf2", "joint"):
        fn["vehicle"]("val_i1", None)
    want = {c: wanted(c, s) for c, s in src.items()}
    print(f"{a.version}: vehicle {a.vehicle}, pedestrian {a.pedestrian}, truck v30 | expected (val_i1, fresh) {want}", flush=True)
    keys = PP.static_keys("v2x")
    print("P3 static map voxels", len(keys), flush=True)
    for i, s in enumerate(("val_i1", "fresh")):
        v, p = fn["vehicle"](s, keys), fn["pedestrian"](s, keys)
        t = P.size(P.build(P.v25_paths("truck", s), "truck", B.KIND[s]), "truck")
        fr = set(v) | set(p) | set(t)
        comb = {f: [d for d in v.get(f, []) if d["class"] == "vehicle"] + [d for d in p.get(f, []) if d["class"] == "pedestrian"]
                + [d for d in t.get(f, []) if d["class"] == "truck"] for f in fr}
        sc = P.score if s == "val_i1" else RV.fresh_score
        for c in ("vehicle", "pedestrian"):
            expect(f"{a.version} {s} {c} ({src[c]})", sc(comb, c)["all"][0], want[c][i])
        expect(f"{a.version} {s} truck (v30)", (P.score if s == "val_i1" else VF.score)(comb, "truck")["all"][0], want["truck"][i])
    v30 = json.loads((P.E / "v2x_v30_test.json").read_text())
    new = {}
    for c in ("vehicle", "pedestrian"):
        b = fn[c]("test", keys)
        assert set(b) == set(v30), f"test frames differ ({c})"
        if src[c] == "v30":
            same_class(f"{c} rebuilt from v30's sources vs v2x_v30", b, v30, c)
        else:
            new[c] = b
    out = {f: sorted([B.keep(d) for d in v30[f] if d["class"] not in new] + [B.keep(d) for c in ("vehicle", "pedestrian") if c in new
                                                                              for d in new[c][f] if d["class"] == c], key=lambda d: -d["score"]) for f in v30}
    if a.vehicle == "vg":   # check: the test rebuild equals the reading's test file box for box
        vgt = json.loads((P.E / "vgeom_vehicle_test.json").read_text())
        same_class(f"{a.version} vehicle vs output/v2x/ens/vgeom_vehicle_test.json (the V-G reading's test file)", out, vgt, "vehicle")
    for c in ("vehicle", "pedestrian", "truck"):
        if c in new:
            same_class(f"{a.version} vs {src[c]} build", out, new[c], c)
        else:
            same_class(f"{a.version} vs v2x_v30", out, v30, c)
    n = {c: sum(d["class"] == c for x in out.values() for d in x) for c in ("vehicle", "pedestrian", "truck")}
    n30 = {c: sum(d["class"] == c for x in v30.values() for d in x) for c in ("vehicle", "pedestrian", "truck")}
    print(f"boxes {a.version}", n, sum(n.values()), "| v30", n30, sum(n30.values()), flush=True)
    for c in new:
        cn = sum(d["class"] == c and d["score"] >= 0.3 for x in out.values() for d in x)
        c30 = sum(d["class"] == c and d["score"] >= 0.3 for x in v30.values() for d in x)
        print(f"confident {c} (score >= 0.3) v30", c30, f"-> {a.version}", cn, flush=True)
    if not new:
        print("all classes v30: output equals v2x_v30 file order too:", out == {f: [B.keep(d) for d in v30[f]] for f in v30}, flush=True)
    if not ok_all:
        sys.exit("STOP a check failed")
    if a.version == "v2x_v31":
        print("NOTE output/v2x/ens/v2x_v31_test.json now exists: scripts/r6r7_read_v2x.py fs2 rerun would take v31 as its baseline", flush=True)
    out_p.write_text(json.dumps(out))
    print(f"wrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
