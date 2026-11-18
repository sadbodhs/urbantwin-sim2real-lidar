"""V-G, vehicle geometry on v2x_v30 (RULE EXCEPTION:
second try of the vehicle size step, val-informed constants, board-informed; spec frozen before the reading).
CPU only. X8 decided on v30's own boxes, scores copied (must equal v30's). v30's vehicle route (R0 on output/v2x/fsh_saf_v2x/s2/vehicle_x6_<split>_ref.json, v2x_plan.build: finish_one then rule X8) with one change
inside finish_one, for vehicle boxes that took the X6 box (X6 centre within 1 m of the lane box); scores and every other box unchanged:
  1. yaw = 50/50 axial mean of the lane and X6 yaws, pi-aligned to the lane yaw;
  2. l = 0.5 l_ref + 0.5 (l_lane - 0.125); w = 0.5 w_ref + 0.5 (w_lane + 0.067);
  3. h = h_ref + 0.18; z = z_ref + 0.06;
  4. grow-only roof lock: non-zero points inside the footprint (height_refine.inside_footprint, margin 0) with z in (bottom + 0.2, bottom + 2.35);
     if >= 25 and their p98 > top: top = min(p98, bottom + 2.3), h = top - bottom, z = bottom + h / 2.
STOP self-check: with every step off the rebuild equals v30's vehicle (val_i1 0.8296, fresh 0.7226). Gate: val_i1 +0.003 on all, h1 and h2 with AP@0.7
not down; fresh +0.003 on all and on h2. One reading of steps 1-4 together. Writes logs/vgeom_read_v2x.{log,json}; test file only if it passes
(output/v2x/ens/vgeom_vehicle_test.json, vehicle class only).
    python scripts/v2x_vgeom_read.py        (Docker CPU)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import route_read_v2x as RV
import v2x_plan as P
from utl.height_refine import inside_footprint

D = Path("output/v2x/fsh_saf_v2x/s2")
nz = lambda p: p[np.abs(p).sum(1) > 0]
PTS = {"val_i1": lambda f: nz(np.load(f"data/pcdet/v2x_real_50k/points/{f}.npy")[:, :3]),
       "fresh": lambda f: nz(np.load(f"data/pcdet/v2x_fresh_50k/points/{f}.npy")[:, :3]),
       "test": lambda f: nz(np.fromfile(f"data/kit/v2x/detection_test_frames/{f}.bin", np.float32).reshape(-1, 3))}
KIND = {"val_i1": "val", "fresh": "fresh", "test": "test"}
V30 = (0.8296, 0.7226)
stats = {"taken": 0, "roof": 0, "boxes": 0}


def finish_vg(ds, pts, on):
    d0 = ds[0]
    r = P.avg(ds, "vehicle")
    if r is None:
        return P.keep(d0)
    ref, _ = r
    b = list(d0["box"])
    if on:
        stats["boxes"] += 1
    if np.hypot(ref[0] - b[0], ref[1] - b[1]) <= P.CAP["vehicle"]:
        if not on:
            bot = ref[2] - ref[5] / 2
            b = [ref[0], ref[1], bot + ref[5] / 2, b[3], b[4], ref[5], b[6]]
        else:
            stats["taken"] += 1
            lane_yaw = b[6]                                                                        # 1: 50/50 axial mean
            t = 0.5 * np.arctan2(np.sin(2 * lane_yaw) + np.sin(2 * ref[6]), np.cos(2 * lane_yaw) + np.cos(2 * ref[6]))
            yaw = float(t + np.round((lane_yaw - t) / np.pi) * np.pi)
            l = 0.5 * ref[3] + 0.5 * (b[3] - 0.125)                                                # 2
            w = 0.5 * ref[4] + 0.5 * (b[4] + 0.067)
            h = ref[5] + 0.18                                                                      # 3
            z = ref[2] + 0.06
            bot = z - h / 2
            top = bot + h
            box = [ref[0], ref[1], z, l, w, h, yaw]
            q = pts[inside_footprint(pts, box, 0.0)]                                               # 4
            q = q[(q[:, 2] > bot + 0.2) & (q[:, 2] < bot + 2.35)]
            if len(q) >= 25:
                p98 = float(np.percentile(q[:, 2], 98))
                if p98 > top:
                    top = min(p98, bot + 2.3)
                    h = top - bot
                    z = bot + h / 2
                    stats["roof"] += 1
            b = [ref[0], ref[1], z, l, w, h, yaw]
    return {"box": b, "score": d0["score"], "class": "vehicle"}


def build(split, on):
    """X8 is decided on v30's own boxes (the off build); with steps on, those scores are copied onto the V-G boxes, and every
    V-G score must equal v30's, per frame and per box."""
    src = [{f: [d for d in v if d["class"] == "vehicle"] for f, v in json.loads(Path(p).read_text()).items()} for p in [D / f"vehicle_x6_{split}_ref.json"]]
    off = {f: [finish_vg([t], None, False) for t in src[0][f]] for f in src[0]}
    off = P.R.apply(off, P.R.features(KIND[split], off, sorted(off)), ["X8"])
    if not on:
        return off
    out = {}
    for f in src[0]:
        pts = PTS[split](f)
        new = [finish_vg([t], pts, True) for t in src[0][f]]
        assert len(new) == len(off[f])
        out[f] = [{**d, "score": o["score"]} for d, o in zip(new, off[f])]
    eq = set(out) == set(off) and all(len(out[f]) == len(off[f]) and all(a["score"] == b["score"] for a, b in zip(out[f], off[f])) for f in off)
    print(f"{split}: scores equal v30: {eq}", flush=True)
    if not eq:
        sys.exit("STOP V-G scores differ from v30's")
    return out


def score(split, p):
    return (P.score if split == "val_i1" else RV.fresh_score)(p, "vehicle")


def main():
    base = {s: score(s, build(s, False)) for s in ("val_i1", "fresh")}
    ref = {s: score(s, RV.build([D / f"vehicle_x6_{s}_ref.json"], "vehicle", KIND[s], "R0")) for s in ("val_i1", "fresh")}
    ok = all(base[s]["all"] == ref[s]["all"] for s in base) and (base["val_i1"]["all"][0], base["fresh"]["all"][0]) == V30
    print(f"self-check (steps off) = v30's vehicle route: val_i1 {base['val_i1']['all'][0]} / fresh {base['fresh']['all'][0]} vs {V30}: {ok}", flush=True)
    if not ok:
        sys.exit("STOP the base does not rebuild v2x_v30's vehicle")
    new = {}
    for s in ("val_i1", "fresh"):
        for k in stats:
            stats[k] = 0
        new[s] = score(s, build(s, True))
        print(f"{s}: boxes {stats['boxes']}, took the X6 box (changed) {stats['taken']}, roof lock raised {stats['roof']}", flush=True)
    d = lambda s, h, i=0: new[s][h][i] - base[s][h][i]
    gv = all(round(1e4 * d("val_i1", h)) >= 30 for h in ("all", "h1", "h2")) and new["val_i1"]["all"][1] >= base["val_i1"]["all"][1]
    gf = all(round(1e4 * d("fresh", h)) >= 30 for h in ("all", "h2"))
    print(f"V-G vehicle vs v2x_v30: val_i1 {base['val_i1']['all'][0]:.4f} -> {new['val_i1']['all'][0]:.4f} ({d('val_i1','all'):+.4f}; h1 {d('val_i1','h1'):+.4f}, "
          f"h2 {d('val_i1','h2'):+.4f}; AP@0.7 {base['val_i1']['all'][1]:.4f} -> {new['val_i1']['all'][1]:.4f}) val gate {'PASS' if gv else 'fail'} | "
          f"fresh {base['fresh']['all'][0]:.4f} -> {new['fresh']['all'][0]:.4f} ({d('fresh','all'):+.4f}; h2 {d('fresh','h2'):+.4f}) fresh gate {'PASS' if gf else 'fail'} "
          f"-> {'PASS' if gv and gf else 'fail'}", flush=True)
    res = dict(base=base, new=new, val_gate=bool(gv), fresh_gate=bool(gf), passed=bool(gv and gf),
               flag="rule exception: second try of the vehicle size step; val-informed constants; board-informed")
    if gv and gf:
        t = build("test", True)
        (P.E / "vgeom_vehicle_test.json").write_text(json.dumps(t))
        print("wrote output/v2x/ens/vgeom_vehicle_test.json (vehicle only)", flush=True)
    Path("logs/vgeom_read_v2x.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
