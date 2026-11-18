"""Combined zips on both tracks (the class gains of each track combined in one file).
CPU only, built only from parts
that already have their own readings; no new readings or settings. Each class is checked box for box against its source file on test, and its
val / fresh reading is rebuilt from the combined val / fresh predictions and must equal the source reading, else STOP.
  v2x <v26 predictions.json>   (3090 CPU: the V2X class builders rebuild exactly there)
      combined: vehicles = v2x_v25 (= v26), pedestrians = output/v2x/ens/v2x_p3_test.json (P3), trucks = v2x_v26
        -> output/v2x/ens/v2x_comb_test.json. Val_i1: vehicle 0.8177, pedestrian 0.3339, truck 0.7804; fresh: 0.7046, 0.3283, 0.8472.
      P3-free twin: predictions exactly v2x_v26 (the v26 predictions.json is packaged as it is).
  lumpi <fresh ids> <v47 predictions.json>   (Mac)
      combined: lumpi_v54 (= lumpi_p3_both_test.json) with Car from lumpi_v47 -> output/lumpi/ens_x/lumpi_comb_test.json
      P3-free twin: lumpi_v53 (= lumpi_bicycle_r4_test.json) with Car from lumpi_v47 -> output/lumpi/ens_x/lumpi_comb_free_test.json
      Car val 0.8924 / fresh 0.8941 (all 320 fresh frames); Person 0.4841 val / 0.4938 (160 check frames), Bicycle 0.5662 / 0.5713 (P3);
      twin Person 0.4366 val / 0.4481 fresh, Bicycle 0.5469 / 0.5476 (v53)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

keep = lambda d: {k: d[k] for k in ("box", "score", "class")}
canon = lambda v: sorted(json.dumps([round(x, 6) for x in d["box"]] + [round(d["score"], 8), d["class"]]) for d in v)
ok_all = True


def expect(name, got, want):
    global ok_all
    same = got == want
    ok_all &= same
    print(f"{name}: {got} vs source {want} | equal: {same}", flush=True)


def same_class(name, a, b, c):
    global ok_all
    same = set(a) == set(b) and all(canon([d for d in a[f] if d["class"] == c]) == canon([d for d in b[f] if d["class"] == c]) for f in a)
    ok_all &= same
    print(f"test {name} {c}: box for box equal: {same}", flush=True)


def v2x(v26_path):
    import postproc_p123 as PP
    import v2x_fresh_read as VF
    import v2x_plan as P
    import v2x_v20_rules as VR
    E = PP.E
    nz = lambda p: p[np.abs(p).sum(1) > 0]
    load = {"val_i1": lambda f: nz(np.load(f"data/pcdet/v2x_real_50k/points/{f}.npy")[:, :3]),
            "fresh": lambda f: nz(np.load(f"data/pcdet/v2x_fresh_50k/points/{f}.npy")[:, :3])}
    keys = PP.static_keys("v2x")
    print("P3 static map voxels", len(keys), flush=True)

    def ped(s, kind):   # postproc_p123.v2x's P3 pedestrian builder
        p = E.fuse(E.load(P.v25_paths("pedestrian", s), "pedestrian"), "R0")
        return PP.filt(VR.apply(p, VR.features(kind, p, sorted(p)), ["X1", "X3"]), "pedestrian", load[s], keys)

    for s, kind, sc, want in (("val_i1", "val", P.score, (0.8177, 0.3339, 0.7804)), ("fresh", "fresh", VF.score, (0.7046, 0.3283, 0.8472))):
        veh = P.build(P.v25_paths("vehicle", s), "vehicle", kind)
        trk = P.size(P.build(P.v25_paths("truck", s), "truck", kind), "truck")
        pd = ped(s, kind)
        fr = set(veh) | set(trk) | set(pd)
        comb = {f: [d for d in veh.get(f, []) if d["class"] == "vehicle"] + [d for d in pd.get(f, []) if d["class"] == "pedestrian"]
                + [d for d in trk.get(f, []) if d["class"] == "truck"] for f in fr}
        for c, w in zip(P.CL, want):
            expect(f"combined {s} {c}", sc(comb, c)["all"][0], w)
    v26 = json.loads(Path(v26_path).read_text())
    v25 = json.loads((P.E / "v25_test.json").read_text())
    p3 = json.loads((P.E / "v2x_p3_test.json").read_text())
    assert set(v26) == set(v25) == set(p3), "test frames differ"
    comb = {f: sorted([keep(d) for d in v26[f] if d["class"] != "pedestrian"] + [keep(d) for d in p3[f] if d["class"] == "pedestrian"],
                      key=lambda d: -d["score"]) for f in v26}
    same_class("combined vs v2x_v25", comb, v25, "vehicle")
    same_class("combined vs v2x_v26", comb, v26, "vehicle")
    same_class("combined vs P3 file", comb, p3, "pedestrian")
    same_class("combined vs v2x_v26", comb, v26, "truck")
    same_class("v2x_v26 vs v2x_v25", v26, v25, "pedestrian")
    if not ok_all:
        sys.exit("STOP a class does not equal its source")
    (P.E / "v2x_comb_test.json").write_text(json.dumps(comb))
    print("wrote v2x_comb_test.json", sum(len(v) for v in comb.values()), "boxes", flush=True)


def lumpi(fr, v47_path):
    import lumpi_member_read as M
    import lumpi_static as S
    import postproc_p123 as PP
    import route_read_lumpi as RL
    O = M.O
    vid = S.ids_of("val")
    chk = sorted(l.strip() for l in open("data/pcdet/lumpi_real/ImageSets/realism_check.txt") if l.strip())
    sets = {"val": vid, "fresh": fr}
    load = {"val": lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3],
            "fresh": lambda f: np.load(f"data/pcdet/lumpi_real_fresh50k/points/{f}.npy")[:, :3]}
    keys = PP.static_keys("lumpi")
    chain = {"Person": lambda s: (lambda p: {f: p.get(f, []) for f in sets[s]})(RL.person(M.paths(s, None), "R0")),
             "Bicycle": lambda s: RL.Chain(s, sets[s]).cls("Bicycle", "R4")}
    car_src = {"val": Path("output/lumpi/ens_x/zPVc_chain_val.json"), "fresh": Path("fresh_ws/output/lumpi/ens_x/zPVc_chain_fresh.json")}
    want = {"comb": {("Car", "val"): 0.8924, ("Car", "fresh"): 0.8941, ("Person", "val"): 0.4841, ("Person", "check"): 0.4938,
                     ("Bicycle", "val"): 0.5662, ("Bicycle", "check"): 0.5713},
            "free": {("Person", "val"): 0.4366, ("Person", "fresh"): 0.4481, ("Bicycle", "val"): 0.5469, ("Bicycle", "fresh"): 0.5476}}
    for s in ("val", "fresh"):
        car = {f: [keep(d) for d in v if d["class"] == "Car"] for f, v in json.loads(car_src[s].read_text()).items()}
        pb = {c: {f: [d for d in v if d["class"] == c] for f, v in chain[c](s).items()} for c in ("Person", "Bicycle")}
        p3 = {c: {f: [d for d in v if d["class"] == c] for f, v in PP.filt(pb[c], c, load[s], keys).items()} for c in ("Person", "Bicycle")}
        gt = S.labels(sets[s])
        for name, pc in (("comb", p3), ("free", pb)):
            whole = {f: car.get(f, []) + pc["Person"].get(f, []) + pc["Bicycle"].get(f, []) for f in sets[s]}
            for c in ("Car", "Person", "Bicycle"):
                for k, ids in ((s, sets[s]), ("check", chk)) if s == "fresh" else ((s, sets[s]),):
                    if (c, k) in want[name]:
                        g = gt if k != "check" else S.labels(ids)
                        expect(f"{name} {c} {k}", S.ap({f: [d for d in whole[f] if d["class"] == c] for f in ids}, g, ids, [c])[0.5][c], want[name][(c, k)])
    v47 = json.loads(Path(v47_path).read_text())
    v54 = json.loads((O / "lumpi_p3_both_test.json").read_text())
    v53 = json.loads((O / "lumpi_bicycle_r4_test.json").read_text())
    assert set(v47) == set(v54) == set(v53), "test frames differ"
    out = {}
    for name, base in (("comb", v54), ("free", v53)):
        p = {f: [keep(d) for d in base[f] if d["class"] != "Car"] + [keep(d) for d in v47[f] if d["class"] == "Car"] for f in base}
        same_class(f"{name} vs lumpi_v47", p, v47, "Car")
        for c in sorted({d["class"] for v in v53.values() for d in v} - {"Car"}):
            same_class(f"{name} vs {'lumpi_v54' if name == 'comb' else 'lumpi_v53'}", p, base, c)
            if name == "comb" and c not in ("Person", "Bicycle"):
                same_class("comb vs lumpi_v53", p, v53, c)
        out[name] = p
    if not ok_all:
        sys.exit("STOP a class does not equal its source")
    (O / "lumpi_comb_test.json").write_text(json.dumps(out["comb"]))
    (O / "lumpi_comb_free_test.json").write_text(json.dumps(out["free"]))
    print("wrote lumpi_comb_test.json and lumpi_comb_free_test.json", flush=True)


if __name__ == "__main__":
    v2x(sys.argv[2]) if sys.argv[1] == "v2x" else lumpi([l.strip() for l in open(sys.argv[2]) if l.strip()], sys.argv[3])
