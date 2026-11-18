"""Q7 V2X vehicle joint reading. Mac CPU. ONE reading, settings fixed before it runs; refuses to run if its output
json exists. Builds nothing. Vehicle only.
  candidate  v2x_v32's own vehicle chain (X6 R0 on the lane, X8 decided on the steps-off build, V-G frozen steps = v2x_opt2_read.build_pair
             with one refiner file, exactly as scripts/dsvt_v2x_read.py builds a vehicle candidate) on the joint lane of --members (each at weight 1):
             output/v2x/<lanes>/s2/vehicle_x6_<s>_ref.json (scripts/q7_v2x_lanes.py on the 3090 CPU, scripts/q7_v2x_refine.sh on the 3090 GPU).
  base       v2x_v32's vehicle = the same chain on output/v2x/dsvt_v2x_j/s2/vehicle_x6_<s>_ref.json (dsvt_v2x alone, A100 refine): 0.8560 / 0.7663.
STOP self-checks, all before any candidate number is computed:
  (3) logs/<lanes>_lanes.json is a full lane run for exactly --members with the lane-off (= v31's vehicle lane) AND one-member (each member
      alone = its own single-reading lane) checks passed box for box on val_i1 / test / fresh, and its new-lane md5s equal logs/<lanes>_lanes.md5;
      the 3 joint refines equal logs/<lanes>_refine.md5 (exactly those 3); the 3 v32 refines equal logs/dsvt_v2x_j_refine.md5;
  (2) output/v2x/ens/v2x_v32_test.json has md5 adc281c6 (the incumbent); the base rebuilds v32's vehicle exactly (val_i1 0.856 / fresh 0.7663).
Gate: the standard V2X gate (r6r7_read_v2x.line: val_i1 +0.003 on all, h1 and h2 with AP@0.7 not down; fresh +0.003 on all and
h2 with AP@0.7 not down) against v2x_v32's vehicle. Passed -> the joint may replace v32's vehicle (v33).
Writes logs/<lanes>_read.json. --selfcheck-only runs check (2) alone (no joint file needed) and writes nothing.
    python scripts/q7_v2x_read.py --members dsvt_v2x vmamba_v2x [--selfcheck-only]   (Docker CPU, under logs/v2x.lock) -> logs/<lanes>_read.json"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import r6r7_read_v2x as RR
import route_read_v2x as RV
import v2x_opt2_read as O2
import v2x_plan as P

ORDER = ("dsvt_v2x", "vmamba_v2x", "voxelnext_v2x", "pv_v2x_n", "parta2_v2x")   # Q7 fixed order (= scripts/q7_v2x_lanes.py)
SHORT = {"dsvt_v2x": "dsvt", "vmamba_v2x": "vmamba", "voxelnext_v2x": "voxelnext", "pv_v2x_n": "pvn", "parta2_v2x": "parta2"}   # lanes-name parts (= scripts/q7_v2x_lanes.py); m[:-len("_v2x")] gave "pv_v" for pv_v2x_n
assert set(SHORT) == set(ORDER)
LANES = N = LANE_FILES = REFINE_MD5 = OUTJ = None   # set by configure()
DJ = Path("output/v2x/dsvt_v2x_j/s2")   # v32's vehicle second stages (dsvt_v2x alone)
SPLITS = ("val_i1", "test", "fresh")
V32 = (0.856, 0.7663)
V32_FILE, V32_MD5 = Path("output/v2x/ens/v2x_v32_test.json"), "adc281c6c553c2409f0ef2dedc449db0"
V32_REFINE_MD5 = "logs/dsvt_v2x_j_refine.md5"
MEMBERS = []


def configure(members):
    global LANES, N, LANE_FILES, REFINE_MD5, OUTJ
    if len(members) < 2 or len(set(members)) != len(members) or any(m not in ORDER for m in members) or list(members) != [m for m in ORDER if m in members]:
        sys.exit(f"STOP --members {members}: 2 or more distinct members of {ORDER}, in that order")
    MEMBERS[:] = list(members)
    LANES = "q7_" + "_".join(SHORT[m] for m in MEMBERS) + "_v2x_j"
    N = Path(f"output/v2x/{LANES}/s2")
    LANE_FILES = [f"output/v2x/{LANES}/s2_in/vehicle_{s}.json" for s in SPLITS]
    REFINE_MD5 = f"logs/{LANES}_refine.md5"
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


def row(s, p):
    return (P.score if s == "val_i1" else RV.fresh_score)(p, "vehicle")


def inputs():
    """Check (3); returns {path: md5} of the 3 joint and the 3 v32 vehicle second-stage files."""
    lj = json.loads(Path(f"logs/{LANES}_lanes.json").read_text())
    stop_unless(lj.get("mode") == "full" and lj.get("lane_off_passed") is True and lj.get("one_member_passed") is True and "stopped" not in lj
                and lj.get("lanes") == LANES and lj.get("members") == MEMBERS,
                f"logs/{LANES}_lanes.json is not a full lane run of {MEMBERS} with passed lane-off and one-member checks")
    for s in SPLITS:
        stop_unless(lj["lane_off"]["vehicle"].get(s, {}).get("box_for_box") is True, f"lane-off vehicle {s} not passed in logs/{LANES}_lanes.json")
        for m in MEMBERS:
            stop_unless(lj["one_member"]["vehicle"].get(m, {}).get(s, {}).get("box_for_box") is True, f"one-member {m} vehicle {s} not passed in logs/{LANES}_lanes.json")
    lm = md5_lines(f"logs/{LANES}_lanes.md5")
    stop_unless(set(lm) == set(LANE_FILES), f"logs/{LANES}_lanes.md5 lists {sorted(lm)}, not the 3 {LANES} vehicle lanes")
    stop_unless(all(lm[f"output/v2x/{LANES}/s2_in/vehicle_{s}.json"] == lj["new"]["vehicle"][s]["md5"] for s in SPLITS),
                f"logs/{LANES}_lanes.md5 differs from logs/{LANES}_lanes.json")
    print(f"check (3): lane run full, lane-off and one-member box for box on {list(SPLITS)}, lane md5s consistent: True", flush=True)
    out = {}
    for rec, d in ((REFINE_MD5, N), (V32_REFINE_MD5, DJ)):
        rm = {p: m for p, m in md5_lines(rec).items() if p.startswith(f"{d}/vehicle_x6_")}
        want = {str(d / f"vehicle_x6_{s}_ref.json") for s in SPLITS}
        stop_unless(set(rm) == want, f"{rec} lists {sorted(rm)} for {d}, not the 3 vehicle X6 refines")
        if rec == REFINE_MD5:
            stop_unless(len(md5_lines(rec)) == 3, f"{rec} lists more than the 3 joint vehicle refines")
        for p in sorted(want):
            got = md5f(p)
            print(f"{p}: md5 {got} vs {rec} {rm[p]}: {got == rm[p]}", flush=True)
            stop_unless(got == rm[p], f"{p} md5 differs from {rec}")
            out[p] = got
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True, help=f"2 or more of {ORDER}, in that order")
    ap.add_argument("--selfcheck-only", action="store_true", help="check (2) only (v32's vehicle rebuilt from its own files); writes nothing")
    a = ap.parse_args()
    configure(a.members)
    t0 = time.time()
    if not a.selfcheck_only:
        stop_unless(not OUTJ.exists(), f"{OUTJ} exists: this is a one-try reading (no rerun)")
    got = md5f(V32_FILE)
    print(f"v2x_v32 test file {V32_FILE}: md5 {got} vs {V32_MD5}: {got == V32_MD5}", flush=True)
    stop_unless(got == V32_MD5, f"{V32_FILE} is not v2x_v32 (md5 {V32_MD5})")
    res = {"baseline": "v2x_v32", "baseline_md5": V32_MD5,
           "experiment": f"Q7 joint: v31's vehicle lane + {' + '.join(MEMBERS)} (weight 1 each), v32's vehicle chain (X6 R0, X8, V-G frozen steps)",
           "lanes": LANES, "members": list(MEMBERS),
           "gate": "standard V2X gate (val_i1 +0.003 all / h1 / h2, AP@0.7 not down; fresh +0.003 all / h2, AP@0.7 not down), vehicle, vs v2x_v32"}
    if not a.selfcheck_only:
        res["md5"] = inputs()
    base, new = {}, {}
    for s in RR.LOAD:   # (2) the base = v32's vehicle, before any candidate number
        p = O2.build_pair(s, [DJ / f"vehicle_x6_{s}_ref.json"], True)
        base[s] = row(s, p)
        del p
        gc.collect()
    got = (base["val_i1"]["all"][0], base["fresh"]["all"][0])
    print(f"self-check v2x_v32 vehicle (dsvt_v2x_j chain): val_i1 {got[0]} / fresh {got[1]} vs {V32}: {got == V32}", flush=True)
    stop_unless(got == V32, "the base does not rebuild v2x_v32's vehicle")
    if a.selfcheck_only:
        print(f"check (2) passed: v32's vehicle rebuilds exactly ({time.time() - t0:.0f} s); nothing written", flush=True)
        return
    for s in RR.LOAD:
        p = O2.build_pair(s, [N / f"vehicle_x6_{s}_ref.json"], True)
        new[s] = row(s, p)
        del p
        gc.collect()
    res["vehicle"] = RR.line(f"{LANES} vehicle (v31 lane + {' + '.join(MEMBERS)}; X6 R0, X8, V-G frozen steps)", new, base, "v2x_v32")
    res["vehicle"]["files"] = [str(N / f"vehicle_x6_{s}_ref.json") for s in SPLITS]
    res["passed_classes"] = ["vehicle"] if res["vehicle"]["passed"] else []
    print(f"{LANES} joint reading vs v2x_v32: passed classes {res['passed_classes'] or 'none'} ({time.time() - t0:.0f} s)", flush=True)
    OUTJ.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUTJ}", flush=True)


if __name__ == "__main__":
    main()
