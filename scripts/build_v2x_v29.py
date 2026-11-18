"""v2x_v29 build. CPU only, no new
readings or settings; only parts with a passing R0 reading (scripts/r0_fshnet_v2x.py):
  vehicle     FSHNet V2X second stage (output/v2x/fshnet_v2x/s2), route R0, no P3;
  pedestrian  FSHNet V2X second stage, route R6, then P3 (postproc_p123 filt, static_keys("v2x"));
  truck       v2x_v27's (= v26), box for box, from v27's predictions.json (output/v2x/ens/v2x_v27_predictions.json, unzipped from the uploaded zip).
Checks (STOP on any miss): val_i1 / fresh per class from the combined predictions equal the R0 readings (vehicle 0.8237 / 0.7138, pedestrian
0.3416 / 0.3321) and v27's truck (0.7804 / 0.8472); on test, v25's pedestrian chain + P3 rebuilds v27's pedestrians box for box (test points and
static keys as v27), and v29's trucks equal v27's box for box. Writes output/v2x/ens/v2x_v29_test.json.
    python scripts/build_v2x_v29.py      (Docker CPU)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import postproc_p123 as PP
import route_read_v2x as RV
import v2x_fresh_read as VF
import v2x_plan as P

M = "fshnet_v2x"
nz = lambda p: p[np.abs(p).sum(1) > 0]
LOAD = {"val_i1": lambda f: nz(np.load(f"data/pcdet/v2x_real_50k/points/{f}.npy")[:, :3]),
        "fresh": lambda f: nz(np.load(f"data/pcdet/v2x_fresh_50k/points/{f}.npy")[:, :3]),
        "test": lambda f: nz(np.fromfile(f"data/kit/v2x/detection_test_frames/{f}.bin", np.float32).reshape(-1, 3))}
KIND = {"val_i1": "val", "fresh": "fresh", "test": "test"}
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}
canon = lambda v: sorted(json.dumps([round(x, 6) for x in d["box"]] + [round(d["score"], 8), d["class"]]) for d in v)
ok_all = True


def expect(name, got, want):
    global ok_all
    same = got == want
    ok_all &= same
    print(f"{name}: {got} vs {want} | equal: {same}", flush=True)


def same_class(name, a, b, c):
    global ok_all
    same = set(a) == set(b) and all(canon([d for d in a[f] if d["class"] == c]) == canon([d for d in b[f] if d["class"] == c]) for f in a)
    ok_all &= same
    print(f"test {name} {c}: box for box equal: {same}", flush=True)


def veh(s):
    return RV.build(RV.new_paths(M, "vehicle", s), "vehicle", KIND[s], "R0")


def ped(s, keys):
    return PP.filt(RV.build(RV.new_paths(M, "pedestrian", s), "pedestrian", KIND[s], "R6"), "pedestrian", LOAD[s], keys)


def main():
    keys = PP.static_keys("v2x")
    print("P3 static map voxels", len(keys), flush=True)
    for s, want in (("val_i1", (0.8237, 0.3416, 0.7804)), ("fresh", (0.7138, 0.3321, 0.8472))):
        v, p = veh(s), ped(s, keys)
        t = P.size(P.build(P.v25_paths("truck", s), "truck", KIND[s]), "truck")
        fr = set(v) | set(p) | set(t)
        comb = {f: [d for d in v.get(f, []) if d["class"] == "vehicle"] + [d for d in p.get(f, []) if d["class"] == "pedestrian"]
                + [d for d in t.get(f, []) if d["class"] == "truck"] for f in fr}
        for c, w in zip(("vehicle", "pedestrian"), want[:2]):
            expect(f"v29 {s} {c}", (P.score if s == "val_i1" else RV.fresh_score)(comb, c)["all"][0], w)
        expect(f"v29 {s} truck", (P.score if s == "val_i1" else VF.score)(comb, "truck")["all"][0], want[2])
    v27 = json.loads(Path("output/v2x/ens/v2x_v27_predictions.json").read_text())
    b = PP.filt(RV.build(P.v25_paths("pedestrian", "test"), "pedestrian", "test", "R0"), "pedestrian", LOAD["test"], keys)
    same_class("v25 chain + P3 rebuilt vs v2x_v27", {f: b.get(f, []) for f in v27}, v27, "pedestrian")
    v, p = veh("test"), ped("test", keys)
    assert set(v) == set(p) == set(v27), "test frames differ"
    out = {f: sorted([keep(d) for d in v27[f] if d["class"] == "truck"] + [keep(d) for d in v[f] if d["class"] == "vehicle"]
                     + [keep(d) for d in p[f] if d["class"] == "pedestrian"], key=lambda d: -d["score"]) for f in v27}
    same_class("v29 vs v2x_v27", out, v27, "truck")
    same_class("v29 vs FSHNet R0 build", out, v, "vehicle")
    same_class("v29 vs FSHNet R6 + P3 build", out, p, "pedestrian")
    n = {c: sum(d["class"] == c for x in out.values() for d in x) for c in ("vehicle", "pedestrian", "truck")}
    n27 = {c: sum(d["class"] == c for x in v27.values() for d in x) for c in ("vehicle", "pedestrian", "truck")}
    print("boxes v29", n, sum(n.values()), "| v27", n27, sum(n27.values()), flush=True)
    if not ok_all:
        sys.exit("STOP a check failed")
    (P.E / "v2x_v29_test.json").write_text(json.dumps(out))
    print("wrote v2x_v29_test.json", flush=True)


if __name__ == "__main__":
    main()
