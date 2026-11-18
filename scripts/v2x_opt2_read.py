"""V2X option 2 reading: pv_v2x epoch 20 as one more second-stage refiner on v2x_v31's own chains
(3090 refines scripts/v2x_opt2_pv.sh -> output/v2x/pv_refiner, md5s in logs/v2x_opt2_pv.md5). Mac CPU.
One reading, settings fixed before it runs. Builds nothing; a test file is written only by a later build.
  pedestrian: v31's pedestrian = R6 over X6b + negatives on fshnet_v2x/s2 (fshnet_v2x/s2_in lane), then P3 (0.3416 / 0.3321).
              Candidate: R6 over X6b + negatives + PV at equal weight (1/3 each), then P3 (r6r7_read_v2x.py's pattern, as X6b-Z).
  vehicle:    v31's vehicle = V-G on v30's vehicle route (scripts/v2x_vgeom_read.py: X6 on fsh_saf_v2x/s2_in, X8 decided on the steps-off build,
              V-G steps 1-4 on, scores copied) (0.8446 / 0.7532). Candidate: the same with the refiner pair [X6, PV] per lane box, averaged
              0.5 / 0.5 by v2x_plan.avg, in BOTH the steps-off build (X8's decision) and the steps-on build (V-G steps).
STOP self-checks before any candidate: the rebuilt pedestrian equals 0.3416 / 0.3321; the one-refiner pair build equals v2x_vgeom_read.build
box for box and scores 0.8446 / 0.7532. Gate per class: val_i1 +0.003 on all, h1 and h2 with AP@0.7 not down; fresh +0.003 on all
and h2 with AP@0.7 not down; against v31. Truck not read.
    python scripts/v2x_opt2_read.py        (Docker CPU, under logs/igm.lock) -> logs/v2x_opt2_read.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import postproc_p123 as PP
import r6r7_read_v2x as RR
import v2x_plan as P
import v2x_vgeom_read as VG

J, O, PV = Path("output/v2x/fsh_saf_v2x/s2"), Path("output/v2x/fshnet_v2x/s2"), Path("output/v2x/pv_refiner")
V31 = {"vehicle": (0.8446, 0.7532), "pedestrian": (0.3416, 0.3321)}


def load(p):
    return {f: [d for d in v if d["class"] == "vehicle"] for f, v in json.loads(Path(p).read_text()).items()}


def build_pair(split, files, on):
    """v2x_vgeom_read.build with a list of refiner files zipped per lane box (same input box asserted), X8 decided on the steps-off build."""
    src = [load(p) for p in files]
    rows = {f: [s[f] for s in src] for f in src[0]}
    for f, r in rows.items():
        assert all(len(x) == len(r[0]) for x in r), f
        assert all(t["box"] == tup[0]["box"] for tup in zip(*r) for t in tup), f
    off = {f: [VG.finish_vg(list(tup), None, False) for tup in zip(*rows[f])] for f in rows}
    off = P.R.apply(off, P.R.features(VG.KIND[split], off, sorted(off)), ["X8"])
    if not on:
        return off
    out = {}
    for f in rows:
        pts = VG.PTS[split](f)
        new = [VG.finish_vg(list(tup), pts, True) for tup in zip(*rows[f])]
        assert len(new) == len(off[f])
        out[f] = [{**d, "score": o["score"]} for d, o in zip(new, off[f])]
    return out


def main():
    keys = PP.static_keys("v2x")
    res = {"baseline": "v2x_v31", "refiner": "pv_v2x epoch 20 (output/v2x/pv_refiner)"}
    # pedestrian
    bp = {s: RR.read("pedestrian", s, RR.paths("pedestrian", s, O), keys) for s in RR.LOAD}
    ok = (bp["val_i1"]["all"][0], bp["fresh"]["all"][0]) == V31["pedestrian"]
    print(f"self-check v2x_v31 pedestrian: val_i1 {bp['val_i1']['all'][0]} / fresh {bp['fresh']['all'][0]} vs {V31['pedestrian']}: {ok}", flush=True)
    if not ok:
        sys.exit("STOP the base does not rebuild v2x_v31's pedestrian")
    np_ = {s: RR.read("pedestrian", s, RR.paths("pedestrian", s, O, PV / f"pedestrian_pv_{s}_ref.json"), keys) for s in RR.LOAD}
    res["pedestrian"] = RR.line("option 2 pedestrian (X6b + negatives + PV, R6 + P3)", np_, bp, "v2x_v31")
    # vehicle
    bv, nv = {}, {}
    for s in RR.LOAD:
        one = build_pair(s, [J / f"vehicle_x6_{s}_ref.json"], True)
        same = one == VG.build(s, True)
        bv[s] = VG.score(s, one)
        print(f"self-check {s}: one-refiner pair build equals v2x_vgeom_read.build box for box: {same}", flush=True)
        if not same:
            sys.exit("STOP the pair build differs from V-G's build")
        nv[s] = VG.score(s, build_pair(s, [J / f"vehicle_x6_{s}_ref.json", PV / f"vehicle_pv_{s}_ref.json"], True))
    ok = (bv["val_i1"]["all"][0], bv["fresh"]["all"][0]) == V31["vehicle"]
    print(f"self-check v2x_v31 vehicle (V-G): val_i1 {bv['val_i1']['all'][0]} / fresh {bv['fresh']['all'][0]} vs {V31['vehicle']}: {ok}", flush=True)
    if not ok:
        sys.exit("STOP the base does not rebuild v2x_v31's vehicle")
    res["vehicle"] = RR.line("option 2 vehicle (X6 + PV 0.5 / 0.5, X8, V-G steps)", nv, bv, "v2x_v31")
    Path("logs/v2x_opt2_read.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
