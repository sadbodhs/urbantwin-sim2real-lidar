"""R6 / R7 readings on top of v2x_v29 (3090 GPU; settings fixed
before running, one reading each). Mac CPU. Builds nothing.
  Base: v2x_v29's classes exactly as scripts/build_v2x_v29.py builds them (FSHNet V2X route second stages output/v2x/fshnet_v2x/s2):
    vehicle route R0 (no P3); pedestrian route R6 over X6b + negatives, then P3. Self-check (STOP): vehicle 0.8237 / 0.7138, pedestrian 0.3416 / 0.3321.
  r7: the same builds from the 12-view files (output/v2x/fshnet_v2x/r7/v12) against the paired 4-view files of the same job (r7/v4). Each class gated
      against its paired 4-view build; the stored v29 value is printed next to it (information).
  r6: pedestrian route R6 over X6b + negatives + the OS1-64-only X6b (output/v2x/fshnet_v2x/r6/pedestrian_o64_*), then P3, against v2x_v29's pedestrian.
      Vehicle for information: route R0 over X6 + the new model's vehicle refine (r6/vehicle_o64_*), against v2x_v29's vehicle.
  Gate per class: val_i1 +0.003 overall and on both halves, AP@0.7 not down; fresh +0.003 overall and on h2 with fresh AP@0.7 not down (h2 in every mode since Nov 13;
  before that only saf2 and V-G coded it; no past decision changes).
  saf: SAFDNet on v29: v29's routes (vehicle R0; pedestrian R6 over X6b + negatives, then P3)
      on the joint second stages output/v2x/fsh_saf_v2x/s2 (lanes v25 + FSHNet + SAFDNet, scripts/saf_joint.sh), against v2x_v29. Truck not read.
    python scripts/r6r7_read_v2x.py r7 | r6 | saf | fs2 | saf2 v2x_v30 | x6bz     (Docker CPU) -> logs/<mode>_read_v2x.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import postproc_p123 as PP
import route_read_v2x as RV
import v2x_plan as P

nz = lambda p: p[np.abs(p).sum(1) > 0]
LOAD = {"val_i1": lambda f: nz(np.load(f"data/pcdet/v2x_real_50k/points/{f}.npy")[:, :3]),
        "fresh": lambda f: nz(np.load(f"data/pcdet/v2x_fresh_50k/points/{f}.npy")[:, :3])}
KIND = {"val_i1": "val", "fresh": "fresh"}
O = Path("output/v2x/fshnet_v2x")
V29 = {"vehicle": (0.8237, 0.7138), "pedestrian": (0.3416, 0.3321)}
gate_v = lambda r, b: bool(all(round(1e4 * (r[h][0] - b[h][0])) >= 30 for h in ("all", "h1", "h2")) and r["all"][1] >= b["all"][1])
gate_f = lambda a, b: bool(all(round(1e4 * (a[h][0] - b[h][0])) >= 30 for h in ("all", "h2")) and a["all"][1] >= b["all"][1])   # fresh all + h2, AP@0.7 not down (as route_read_v2x.py:139)


def paths(cls, s, d, extra=None):
    p = {"vehicle": [d / f"vehicle_x6_{s}_ref.json"], "pedestrian": [d / f"pedestrian_x6b_{s}_ref.json", d / f"pedestrian_neg_{s}_ref.json"]}[cls]
    return p + ([extra] if extra else [])


def read(cls, s, ps, keys):
    rule = "R0" if cls == "vehicle" else "R6"
    p = RV.build(ps, cls, KIND[s], rule)
    if cls == "pedestrian":
        p = PP.filt(p, "pedestrian", LOAD[s], keys)
    return (P.score if s == "val_i1" else RV.fresh_score)(p, cls)


def line(name, n, b, ref_name):
    d = {h: n["val_i1"][h][0] - b["val_i1"][h][0] for h in ("all", "h1", "h2")}
    gv, gf = gate_v(n["val_i1"], b["val_i1"]), gate_f(n["fresh"], b["fresh"])
    print(f"{name} vs {ref_name}: val_i1 {b['val_i1']['all'][0]:.4f} -> {n['val_i1']['all'][0]:.4f} ({d['all']:+.4f}; h1 {d['h1']:+.4f}, h2 {d['h2']:+.4f}; "
          f"AP@0.7 {b['val_i1']['all'][1]:.4f} -> {n['val_i1']['all'][1]:.4f}) val gate {'PASS' if gv else 'fail'} | fresh {b['fresh']['all'][0]:.4f} -> "
          f"{n['fresh']['all'][0]:.4f} ({n['fresh']['all'][0] - b['fresh']['all'][0]:+.4f}; h2 {n['fresh']['h2'][0] - b['fresh']['h2'][0]:+.4f}) fresh gate (all + h2, AP@0.7 {b['fresh']['all'][1]:.4f} -> {n['fresh']['all'][1]:.4f}) "
          f"{'PASS' if gf else 'fail'} -> {'PASS' if gv and gf else 'fail'}", flush=True)
    return dict(new=n, ref=b, val_gate=gv, fresh_gate=gf, passed=bool(gv and gf))


def main(mode):
    keys = PP.static_keys("v2x")
    base = {c: {s: read(c, s, paths(c, s, O / "s2"), keys) for s in LOAD} for c in ("vehicle", "pedestrian")}
    for c, (wv, wf) in V29.items():
        ok = base[c]["val_i1"]["all"][0] == wv and base[c]["fresh"]["all"][0] == wf
        print(f"self-check v2x_v29 {c}: val_i1 {base[c]['val_i1']['all'][0]} vs {wv}, fresh {base[c]['fresh']['all'][0]} vs {wf}: {ok}", flush=True)
        if not ok:
            sys.exit("STOP the base does not rebuild v2x_v29")
    res = {"v29": base}
    if mode == "r7":
        for c in ("vehicle", "pedestrian"):
            v4 = {s: read(c, s, paths(c, s, O / "r7/v4"), keys) for s in LOAD}
            v12 = {s: read(c, s, paths(c, s, O / "r7/v12"), keys) for s in LOAD}
            print(f"information {c}: paired 4-view rerun vs stored v29 files: val_i1 {v4['val_i1']['all'][0]:.4f} vs {base[c]['val_i1']['all'][0]:.4f}, "
                  f"fresh {v4['fresh']['all'][0]:.4f} vs {base[c]['fresh']['all'][0]:.4f}", flush=True)
            res[c] = line(f"R7 {c} 12-view", v12, v4, "paired 4-view")
            res[c]["vs_v29"] = line(f"(information) R7 {c} 12-view", v12, base[c], "stored v2x_v29")
    elif mode == "saf":
        J = Path("output/v2x/fsh_saf_v2x/s2")
        for c in ("vehicle", "pedestrian"):
            n = {s: read(c, s, paths(c, s, J), keys) for s in LOAD}
            res[c] = line(f"SAFDNet on v29 {c} (joint lanes v25 + FSHNet + SAFDNet, v29's route)", n, base[c], "v2x_v29")
    elif mode == "saf2":
        # SAFDNet V2X seed 2: vehicle only. Lanes with both SAFDNet seeds
        # as one member (scripts/route_v2x_lanes_saf2.py), X6, R0, against an explicitly named baseline (argv[2]): v2x_v30 (vehicle = the joint s2
        # files fsh_saf_v2x/s2) or v2x_v31 (vehicle = fs2's files fsh2_v2x/s2 if v31 took fs2 vehicles, else v30's; given as argv[3] = fs2 | v30).
        # Gates: val_i1 +0.003 on all / h1 / h2 with AP@0.7 not down; fresh +0.003 on all and on h2.
        bname = sys.argv[2]
        if bname != "v2x_v30":   # Nov 13: v2x_v31 is now v30 + V-G; part (b) below reads against V-G
            sys.exit("STOP saf2 reads as: saf2 v2x_v30 (part (a) vs v30, part (b) joint with V-G vs V-G's vehicle)")
        bveh = sys.argv[3] if bname == "v2x_v31" else "v30"
        BD = Path("output/v2x/fsh2_v2x/s2") if bveh == "fs2" else Path("output/v2x/fsh_saf_v2x/s2")
        want = (0.8296, 0.7226)
        if bveh == "fs2":
            r = json.loads(Path("logs/fs2_read_v2x.json").read_text())["vehicle"]["new"]
            want = (r["val_i1"]["all"][0], r["fresh"]["all"][0])
        b = {s: read("vehicle", s, paths("vehicle", s, BD), keys) for s in LOAD}
        ok = b["val_i1"]["all"][0] == want[0] and b["fresh"]["all"][0] == want[1]
        print(f"self-check {bname} vehicle ({bveh}): val_i1 {b['val_i1']['all'][0]} vs {want[0]}, fresh {b['fresh']['all'][0]} vs {want[1]}: {ok}", flush=True)
        if not ok:
            sys.exit(f"STOP the base does not rebuild {bname}'s vehicle")
        n = {s: read("vehicle", s, paths("vehicle", s, Path("output/v2x/fsh_saf2_v2x/s2")), keys) for s in LOAD}
        res["vehicle"] = line(f"SAFDNet seed 2 on {bname} vehicle (both SAFDNet seeds as one member, v30's lane, X6, R0)", n, b, bname)
        dh2 = n["fresh"]["h2"][0] - b["fresh"]["h2"][0]
        h2ok = round(1e4 * dh2) >= 30
        print(f"fresh h2 {b['fresh']['h2'][0]:.4f} -> {n['fresh']['h2'][0]:.4f} ({dh2:+.4f}): {'PASS' if h2ok else 'fail'} (part of this reading's gate)", flush=True)
        res["vehicle"]["fresh_h2_gate"] = bool(h2ok)
        res["vehicle"]["passed"] = bool(res["vehicle"]["passed"] and h2ok)
        print(f"(a) SAFDNet seed 2 vehicle alone vs v2x_v30: {'PASS' if res['vehicle']['passed'] else 'fail'}", flush=True)
        # (b) fixed before this reading: the joint lane (saf2 lane -> X6 R0 -> V-G's frozen steps, scripts/v2x_vgeom_read.py
        # as run on Nov 12, unchanged) must pass the same gates against V-G's vehicle (val_i1 0.8446 / fresh 0.7532). saf2 enters only if (a) and (b).
        import v2x_vgeom_read as VG
        VG.D = Path("output/v2x/fsh_saf_v2x/s2")
        vb = {s: VG.score(s, VG.build(s, True)) for s in LOAD}
        ok = (vb["val_i1"]["all"][0], vb["fresh"]["all"][0]) == (0.8446, 0.7532)
        print(f"self-check V-G vehicle on v30's lane: val_i1 {vb['val_i1']['all'][0]} / fresh {vb['fresh']['all'][0]} vs (0.8446, 0.7532): {ok}", flush=True)
        if not ok:
            sys.exit("STOP V-G's vehicle does not rebuild")
        VG.D = Path("output/v2x/fsh_saf2_v2x/s2")
        off = {s: VG.score(s, VG.build(s, False)) for s in LOAD}
        ok = all(off[s]["all"] == n[s]["all"] for s in LOAD)
        print(f"self-check saf2 lane with V-G steps off = (a)'s saf2 build: val_i1 {off['val_i1']['all'][0]} / fresh {off['fresh']['all'][0]}: {ok}", flush=True)
        if not ok:
            sys.exit("STOP the V-G steps-off build of the saf2 lane differs from (a)")
        vj = {s: VG.score(s, VG.build(s, True)) for s in LOAD}
        res["vehicle_joint_vg"] = line("(b) SAFDNet seed 2 + V-G vehicle (saf2 lane, X6 R0, V-G steps)", vj, vb, "V-G vehicle (v30 + V-G)")
        res["vehicle"]["passed_alone_vs_v30"] = res["vehicle"]["passed"]
        res["vehicle"]["passed"] = bool(res["vehicle"]["passed"] and res["vehicle_joint_vg"]["passed"])
        print(f"SAFDNet seed 2 vehicle enters only if (a) and (b): (a) {'PASS' if res['vehicle']['passed_alone_vs_v30'] else 'fail'}, "
              f"(b) {'PASS' if res['vehicle_joint_vg']['passed'] else 'fail'} -> {'PASS' if res['vehicle']['passed'] else 'fail'}", flush=True)
        res["baseline"] = bname
    elif mode == "fs2":
        # FSHNet V2X seed 2: lanes with both FSHNet seeds as one member
        # (scripts/route_v2x_lanes_fs2.py) and the baseline's own second stages (output/v2x/fsh2_v2x/s2), against the newest built V2X file:
        # v2x_v31 if output/v2x/ens/v2x_v31_test.json exists (pedestrian refiners X6b + negatives + R6's OS1-64 X6b), else v2x_v30.
        J, F2 = Path("output/v2x/fsh_saf_v2x/s2"), Path("output/v2x/fsh2_v2x/s2")
        v31 = Path("output/v2x/ens/v2x_v31_test.json").exists()
        if v31:   # after a v31 build every V2X reading names its baseline explicitly; no auto-pick
            sys.exit("STOP v2x_v31_test.json exists: the fs2 auto-pick is retired; read with an explicit, named baseline")
        bname = "v2x_v31" if v31 else "v2x_v30"
        o64 = lambda d, s: d / f"pedestrian_o64_{s}_ref.json"
        bp = {"vehicle": lambda s: paths("vehicle", s, J),
              "pedestrian": (lambda s: paths("pedestrian", s, O / "s2", O / f"r6/pedestrian_o64_{s}_ref.json")) if v31 else (lambda s: paths("pedestrian", s, O / "s2"))}
        np_ = {"vehicle": lambda s: paths("vehicle", s, F2),
               "pedestrian": (lambda s: paths("pedestrian", s, F2, o64(F2, s))) if v31 else (lambda s: paths("pedestrian", s, F2))}
        want = {"vehicle": (0.8296, 0.7226), "pedestrian": V29["pedestrian"]}
        if v31:
            r6 = json.loads(Path("logs/r6_read_v2x.json").read_text())["pedestrian"]["new"]
            want["pedestrian"] = (r6["val_i1"]["all"][0], r6["fresh"]["all"][0])
        for c in ("vehicle", "pedestrian"):
            b = {s: read(c, s, bp[c](s), keys) for s in LOAD}
            ok = b["val_i1"]["all"][0] == want[c][0] and b["fresh"]["all"][0] == want[c][1]
            print(f"self-check {bname} {c}: val_i1 {b['val_i1']['all'][0]} vs {want[c][0]}, fresh {b['fresh']['all'][0]} vs {want[c][1]}: {ok}", flush=True)
            if not ok:
                sys.exit(f"STOP the base does not rebuild {bname}")
            n = {s: read(c, s, np_[c](s), keys) for s in LOAD}
            res[c] = line(f"FSHNet seed 2 on {bname} {c} (both seeds as one member, {bname}'s route)", n, b, bname)
        res["baseline"] = bname
    elif mode == "x6bz":
        # X6b-Z: X6b retrained with VOXEL_SIZE z 0.125
        # (configs/pcdet/vrcnn_v2x_b_z.yaml), its epoch-20 second stage (4 views) on v2x_v30's lanes, from an A100 run: output/v2x/x6bz/s2/.
        # pedestrian (GATED): R6 over X6b + negatives + X6b-Z on fshnet_v2x/s2_in, then P3, against v2x_v30's pedestrian (O / "s2", 0.3416 / 0.3321).
        # vehicle (INFORMATION ONLY): R0 over X6 + X6b-Z on the joint lane fsh_saf_v2x/s2_in, against v2x_v30's vehicle (J, 0.8296 / 0.7226).
        J, X = Path("output/v2x/fsh_saf_v2x/s2"), Path("output/v2x/x6bz/s2")
        bv = {s: read("vehicle", s, paths("vehicle", s, J), keys) for s in LOAD}
        ok = (bv["val_i1"]["all"][0], bv["fresh"]["all"][0]) == (0.8296, 0.7226)
        print(f"self-check v2x_v30 vehicle: val_i1 {bv['val_i1']['all'][0]} / fresh {bv['fresh']['all'][0]} vs (0.8296, 0.7226): {ok}", flush=True)
        if not ok:
            sys.exit("STOP the base does not rebuild v2x_v30's vehicle")
        ped = {s: read("pedestrian", s, paths("pedestrian", s, O / "s2", X / f"pedestrian_x6bz_{s}_ref.json"), keys) for s in LOAD}
        res["pedestrian"] = line("X6b-Z pedestrian (X6b + negatives + X6b-Z, R6 + P3)", ped, base["pedestrian"], "v2x_v30")
        veh = {s: read("vehicle", s, paths("vehicle", s, J, X / f"vehicle_x6bz_{s}_ref.json"), keys) for s in LOAD}
        res["vehicle_information"] = line("(information) X6b-Z vehicle (X6 + X6b-Z, R0)", veh, bv, "v2x_v30")
        res["baseline"] = "v2x_v30"
    else:
        ped = {s: read("pedestrian", s, paths("pedestrian", s, O / "s2", O / f"r6/pedestrian_o64_{s}_ref.json"), keys) for s in LOAD}
        res["pedestrian"] = line("R6 pedestrian (X6b + negatives + OS1-64 X6b, R6 + P3)", ped, base["pedestrian"], "v2x_v29")
        veh = {s: read("vehicle", s, paths("vehicle", s, O / "s2", O / f"r6/vehicle_o64_{s}_ref.json"), keys) for s in LOAD}
        res["vehicle_information"] = line("(information) R6 vehicle (X6 + OS1-64 X6b, R0)", veh, base["vehicle"], "v2x_v29")
    Path(f"logs/{mode}_read_v2x.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    assert sys.argv[1] in ("r7", "r6", "saf", "fs2", "saf2", "x6bz"), "mode: r7 | r6 | saf | fs2 | saf2 | x6bz"
    main(sys.argv[1])
