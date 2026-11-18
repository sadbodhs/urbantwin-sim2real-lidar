"""v2x_v30 build.
CPU only, no new readings or settings. v2x_v29 with one change:
  vehicle     joint second stage (lanes v25 + FSHNet + SAFDNet, scripts/saf_joint.sh, output/v2x/fsh_saf_v2x/s2), route R0, no P3,
              exactly as scripts/r6r7_read_v2x.py saf builds "vehicle";
  pedestrian  v2x_v29's (FSHNet s2 X6b + negatives, R6, then P3), box for box;
  truck       v2x_v29's (= v27 = v26), box for box.
Checks (STOP on any miss): val_i1 / fresh per class from the combined predictions: vehicle 0.8296 / 0.7226, pedestrian 0.3416 / 0.3321,
truck 0.7804 / 0.8472; on test, pedestrians and trucks equal v2x_v29 box for box, vehicles equal the joint R0 build. Writes output/v2x/ens/v2x_v30_test.json.
    python scripts/build_v2x_v30.py      (Docker CPU)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import build_v2x_v29 as B
import postproc_p123 as PP
import route_read_v2x as RV
import v2x_plan as P
import v2x_fresh_read as VF

J = Path("output/v2x/fsh_saf_v2x/s2")
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


def veh(s):
    return RV.build([J / f"vehicle_x6_{s}_ref.json"], "vehicle", B.KIND[s], "R0")


def main():
    keys = PP.static_keys("v2x")
    print("P3 static map voxels", len(keys), flush=True)
    for s, want in (("val_i1", (0.8296, 0.3416, 0.7804)), ("fresh", (0.7226, 0.3321, 0.8472))):
        v, p = veh(s), B.ped(s, keys)
        t = P.size(P.build(P.v25_paths("truck", s), "truck", B.KIND[s]), "truck")
        fr = set(v) | set(p) | set(t)
        comb = {f: [d for d in v.get(f, []) if d["class"] == "vehicle"] + [d for d in p.get(f, []) if d["class"] == "pedestrian"]
                + [d for d in t.get(f, []) if d["class"] == "truck"] for f in fr}
        for c, w in zip(("vehicle", "pedestrian"), want[:2]):
            expect(f"v30 {s} {c}", (P.score if s == "val_i1" else RV.fresh_score)(comb, c)["all"][0], w)
        expect(f"v30 {s} truck", (P.score if s == "val_i1" else VF.score)(comb, "truck")["all"][0], want[2])
    v29 = json.loads((P.E / "v2x_v29_test.json").read_text())
    v = veh("test")
    assert set(v) == set(v29), "test frames differ"
    out = {f: sorted([B.keep(d) for d in v29[f] if d["class"] in ("truck", "pedestrian")] + [B.keep(d) for d in v[f] if d["class"] == "vehicle"],
                     key=lambda d: -d["score"]) for f in v29}
    same_class("v30 vs v2x_v29", out, v29, "truck")
    same_class("v30 vs v2x_v29", out, v29, "pedestrian")
    same_class("v30 vs joint R0 build", out, v, "vehicle")
    n = {c: sum(d["class"] == c for x in out.values() for d in x) for c in ("vehicle", "pedestrian", "truck")}
    n29 = {c: sum(d["class"] == c for x in v29.values() for d in x) for c in ("vehicle", "pedestrian", "truck")}
    print("boxes v30", n, sum(n.values()), "| v29", n29, sum(n29.values()), flush=True)
    c30 = sum(d["class"] == "vehicle" and d["score"] >= 0.3 for x in out.values() for d in x)
    c29 = sum(d["class"] == "vehicle" and d["score"] >= 0.3 for x in v29.values() for d in x)
    print("confident vehicles (score >= 0.3) v29", c29, "-> v30", c30, flush=True)
    if not ok_all:
        sys.exit("STOP a check failed")
    (P.E / "v2x_v30_test.json").write_text(json.dumps(out))
    print("wrote v2x_v30_test.json", flush=True)


if __name__ == "__main__":
    main()
