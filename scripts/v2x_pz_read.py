"""P-Z reading. Mac CPU. ONE reading,
settings fixed before it runs; refuses to run if its output json exists. Builds nothing.
  base       v2x_v32's pedestrian = v31's chain: build_v2x_v29.ped (R6 over X6b + negatives on output/v2x/fshnet_v2x/s2, then P3); 0.3416 / 0.3321.
  candidate  the same boxes with z <- z + 0.03 (box[2], the box centre) on EVERY pedestrian box; x, y, l, w, h, yaw, score, class and order unchanged.
             A val-informed constant (BEV-matched pedestrian centres sit low, median -0.026 m on val_i1), the same kind of step as
             V-G's z_ref + 0.06 and the truck size step; the package stays flagged and the report lists it.
STOP self-checks, before any candidate number: v2x_v32's test file has md5 adc281c6 (the incumbent); P3's static map has v31's 16542 voxels; the base
rebuilds v32's pedestrian exactly (val_i1 0.3416 / fresh 0.3321); the candidate differs from the base in z only, by exactly +0.03, on every
pedestrian box, same frames, order, scores and other fields.
Order: val_i1 candidate first; its row must equal the exploratory row (all / h1 / h2 0.3540 / 0.3326 / 0.3809, AP@0.7 all 0.0055, 4 dp);
if it does not, STOP before fresh is read (the two code paths would not be the same step). Then fresh.
Gate: the standard V2X gate (r6r7_read_v2x.line: val_i1 +0.003 on all, h1 and h2 with AP@0.7 not down; fresh +0.003 on all and
h2 with AP@0.7 not down) against v2x_v32's pedestrian. Fails -> dropped, no second value.
Writes logs/v2x_pz_read.json ({"baseline": "v2x_v32", "baseline_md5", "dz": 0.03, "pedestrian": {new, ref, val_gate, fresh_gate, passed}, ...}).
    python scripts/v2x_pz_read.py        (Docker CPU, under logs/v2x.lock) -> logs/v2x_pz_read.json"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import build_v2x_v29 as B
import postproc_p123 as PP
import r6r7_read_v2x as RR
import route_read_v2x as RV
import v2x_plan as P

DZ = 0.03
V32_PED = (0.3416, 0.3321)
V31_KEYS = 16542
V32_FILE, V32_MD5 = Path("output/v2x/ens/v2x_v32_test.json"), "adc281c6c553c2409f0ef2dedc449db0"
EXPECT_VAL = {"all": 0.3540, "h1": 0.3326, "h2": 0.3809, "ap07_all": 0.0055}   # the exploratory run (val_i1 only)
OUTJ = Path("logs/v2x_pz_read.json")


def stop_unless(ok, msg):
    if not ok:
        sys.exit("STOP " + msg)


def row(s, p):
    return (P.score if s == "val_i1" else RV.fresh_score)(p, "pedestrian")


def shift(p, dz):
    """Every pedestrian box: z + dz (box[2]); every other field, every other class and the order unchanged."""
    return {f: [dict(d, box=[d["box"][0], d["box"][1], d["box"][2] + dz] + list(d["box"][3:])) if d["class"] == "pedestrian" else d for d in v]
            for f, v in p.items()}


def same_but_z(a, b, dz):
    if list(a) != list(b):
        return False
    for f in a:
        if len(a[f]) != len(b[f]):
            return False
        for x, y in zip(a[f], b[f]):
            if set(x) != set(y) or any(x[k] != y[k] for k in x if k != "box"):
                return False
            if x["class"] != "pedestrian":
                if x["box"] != y["box"]:
                    return False
            elif x["box"][:2] != y["box"][:2] or x["box"][3:] != y["box"][3:] or y["box"][2] != x["box"][2] + dz:
                return False
    return True


def main():
    t0 = time.time()
    stop_unless(not OUTJ.exists(), f"{OUTJ} exists: this is a one-try reading (no rerun)")
    got = hashlib.md5(V32_FILE.read_bytes()).hexdigest()
    print(f"v2x_v32 test file {V32_FILE}: md5 {got} vs {V32_MD5}: {got == V32_MD5}", flush=True)
    stop_unless(got == V32_MD5, f"{V32_FILE} is not v2x_v32 (md5 {V32_MD5})")
    keys = PP.static_keys("v2x")
    print(f"P3 static map voxels {len(keys)} vs {V31_KEYS}: {len(keys) == V31_KEYS}", flush=True)
    stop_unless(len(keys) == V31_KEYS, "P3's static map is not v31's")
    res = {"baseline": "v2x_v32", "baseline_md5": V32_MD5, "dz": DZ,
           "experiment": "P-Z: v32's pedestrian chain (build_v2x_v29.ped: R6 + P3) with z + 0.03 on every pedestrian box",
           "gate": "standard V2X gate (val_i1 +0.003 all / h1 / h2, AP@0.7 not down; fresh +0.003 all / h2, AP@0.7 not down), pedestrian, vs v2x_v32"}
    base, new = {}, {}
    for s in ("val_i1", "fresh"):   # the base, both splits, before any candidate number
        p = B.ped(s, keys)
        base[s] = row(s, p)
        q = shift(p, DZ)
        ok = same_but_z(p, q, DZ)
        n = sum(d["class"] == "pedestrian" for v in p.values() for d in v)
        print(f"self-check {s}: candidate = base with z + {DZ} on all {n} pedestrian boxes, nothing else changed: {ok}", flush=True)
        stop_unless(ok, f"the shift changes more than z on {s}")
        del p, q
        gc.collect()
    got = (base["val_i1"]["all"][0], base["fresh"]["all"][0])
    print(f"self-check v2x_v32 pedestrian (v31 chain): val_i1 {got[0]} / fresh {got[1]} vs {V32_PED}: {got == V32_PED}", flush=True)
    stop_unless(got == V32_PED, "the base does not rebuild v2x_v32's pedestrian")
    p = shift(B.ped("val_i1", keys), DZ)
    new["val_i1"] = row("val_i1", p)
    del p
    gc.collect()
    r = new["val_i1"]
    seen = {"all": round(float(r["all"][0]), 4), "h1": round(float(r["h1"][0]), 4), "h2": round(float(r["h2"][0]), 4), "ap07_all": round(float(r["all"][1]), 4)}
    print(f"val_i1 candidate row {seen} vs the exploratory row {EXPECT_VAL}: {seen == EXPECT_VAL}", flush=True)
    stop_unless(seen == EXPECT_VAL, "the val_i1 row differs from the exploratory row: not the same step; fresh not read")
    p = shift(B.ped("fresh", keys), DZ)
    new["fresh"] = row("fresh", p)
    del p
    gc.collect()
    res["pedestrian"] = RR.line(f"P-Z pedestrian (v32's chain, z + {DZ})", new, base, "v2x_v32")
    res["passed_classes"] = ["pedestrian"] if res["pedestrian"]["passed"] else []
    print(f"P-Z reading vs v2x_v32: passed classes {res['passed_classes'] or 'none'} ({time.time() - t0:.0f} s)", flush=True)
    OUTJ.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUTJ}", flush=True)


if __name__ == "__main__":
    main()
