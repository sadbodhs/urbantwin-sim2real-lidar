"""LUMPI P3 time-transfer check and combined file.
Mac CPU, one try, no zip here. Fixed before any reading:
  map_early: built exactly like R1's LUMPI map (0.2 m voxels occupied in at least half the frames) from the realism_fit frames with id < 003200 only.
  Read on late frames only: val frames with id >= 003200, and fresh realism_check frames with id >= 003200. Per class (Person, Bicycle), each on
  lumpi_v53's chain (Person R0, Bicycle R4) with the same P3 rule (>= 80 % of a box's points in background voxels -> score x 0.3):
    (a) without P3; (b) P3 with map_early; (c) P3 with the full map (information).
  Pass per class: gain(b vs a) >= +0.003 on val-late AND on fresh-late, AND gain(b) >= 0.5 x gain(c) on the same frames (ten-thousandths, F4).
  A pass keeps the class's P3 file as an upload candidate; a fail keeps it flagged "in-time only".
Combined file: lumpi_v53 with P3 Person and P3 Bicycle together (classes add exactly); read on val and on the 160 check frames, each class must equal its
single-class reading (Person 0.4841 val / 0.4938 check; Bicycle 0.5662 / 0.5713), else stop. Writes output/lumpi/ens_x/lumpi_p3_both_test.json.
    python scripts/postproc_p3_time.py logs/fresh_ids.txt"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import postproc_p123 as PP


def main(fr):
    import lumpi_member_read as M
    import lumpi_static as S
    import realism_r1 as R1
    import route_read_lumpi as RL
    from utl.realism import nonzero
    c1 = R1.CFG["lumpi"]
    fit_ids = [l.strip() for l in open(c1["fit"]) if l.strip()]
    early = [i for i in fit_ids if int(i) < 3200]
    c_early = dict(c1, fit=None)
    tmp = Path("logs/p3_time_fit_early.txt")
    tmp.write_text("\n".join(early) + "\n")
    c_early["fit"] = str(tmp)
    m_early = R1.static_map([nonzero(f) for f in R1.real_frames("lumpi", c_early, "fit")])
    keys = {"early": np.unique(PP.enc(np.floor(m_early / 0.2).astype(np.int64))), "full": PP.static_keys("lumpi")}
    print("map_early from", len(early), "of", len(fit_ids), "realism_fit frames (id < 003200) | map voxels: early", len(keys["early"]), "| full", len(keys["full"]), flush=True)
    vid = S.ids_of("val")
    chk = sorted(l.strip() for l in open("data/pcdet/lumpi_real/ImageSets/realism_check.txt") if l.strip())
    late = {"val": [i for i in vid if int(i) >= 3200], "fresh": [i for i in chk if int(i) >= 3200]}
    assert set(late["fresh"]) <= set(fr) and not set(late["fresh"]) & set(fit_ids)
    print("late frames: val", len(late["val"]), "| fresh realism_check", len(late["fresh"]), flush=True)
    load = {"val": lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3],
            "fresh": lambda f: np.load(f"data/pcdet/lumpi_real_fresh50k/points/{f}.npy")[:, :3]}
    sets = {"val": vid, "fresh": fr}
    base_rule = {"Person": "R0", "Bicycle": "R4"}
    chain = {"Person": lambda s: (lambda p: {f: p.get(f, []) for f in sets[s]})(RL.person(M.paths(s, None), "R0")),
             "Bicycle": lambda s: RL.Chain(s, sets[s]).cls("Bicycle", "R4")}
    gts = {s: S.labels(late[s]) for s in late}
    ap = lambda p, s, c: S.ap({f: [d for d in p[f] if d["class"] == c] for f in late[s]}, gts[s], late[s], [c])[0.5][c]
    gain = lambda x, y: round(1e4 * (x - y))   # ten-thousandths
    res = {"map_voxels": {k: int(len(v)) for k, v in keys.items()}, "late_frames": {k: len(v) for k, v in late.items()}}
    full = {}
    for c in ("Person", "Bicycle"):
        e = {}
        for s in ("val", "fresh"):
            b0 = chain[c](s)
            b0 = {f: b0.get(f, []) for f in sets[s]}
            a = ap(b0, s, c)
            pb = PP.filt(b0, c, load[s], keys["early"])
            pc = PP.filt(b0, c, load[s], keys["full"])
            b, cc = ap(pb, s, c), ap(pc, s, c)
            nb = sum(d["class"] == c for f in late[s] for d in b0[f])
            share = lambda p: sum(1 for f in late[s] for d, d0 in zip(p[f], b0[f]) if d["class"] == c and d["score"] != d0["score"]) / max(nb, 1)
            e[s] = dict(a=a, b_early=b, c_full=cc, gain_b=gain(b, a) / 1e4, gain_c=gain(cc, a) / 1e4, share_down_early=share(pb), share_down_full=share(pc))
            print(c, s + "-late", "(a) no P3 %.4f | (b) P3 map_early %.4f (%+.4f) | (c) P3 full map %.4f (%+.4f) | boxes down-weighted: early %.3f, full %.3f"
                  % (a, b, e[s]["gain_b"], cc, e[s]["gain_c"], e[s]["share_down_early"], e[s]["share_down_full"]), flush=True)
            if s == "val":
                full[c] = pc
        g = all(gain(e[s]["b_early"], e[s]["a"]) >= 30 and gain(e[s]["b_early"], e[s]["a"]) >= 0.5 * gain(e[s]["c_full"], e[s]["a"]) for s in ("val", "fresh"))
        e["pass"] = bool(g)
        print(c, "time-transfer check:", "PASS (upload candidate)" if g else "fail (file flagged in-time only)", flush=True)
        res[c] = e
    # combined file
    O = M.O
    v53 = json.loads((O / "lumpi_bicycle_r4_test.json").read_text())
    tp = json.loads((O / "lumpi_p3_person_test.json").read_text())
    tb = json.loads((O / "lumpi_p3_bicycle_test.json").read_text())
    assert set(v53) == set(tp) == set(tb)
    canon = lambda v: sorted(json.dumps([round(x, 6) for x in d["box"]] + [round(d["score"], 8), d["class"]]) for d in v)
    for f in v53:   # each single-class file is v53 with only its class changed
        assert canon([d for d in tp[f] if d["class"] != "Person"]) == canon([d for d in v53[f] if d["class"] != "Person"])
        assert canon([d for d in tb[f] if d["class"] != "Bicycle"]) == canon([d for d in v53[f] if d["class"] != "Bicycle"])
    both = {f: [d for d in v53[f] if d["class"] not in ("Person", "Bicycle")] + [d for d in tp[f] if d["class"] == "Person"]
            + [d for d in tb[f] if d["class"] == "Bicycle"] for f in v53}
    # whole-file reading: both classes' P3 predictions in one file, val and the 160 check frames
    pv = PP.filt({f: v for f, v in chain["Person"]("val").items()}, "Person", load["val"], keys["full"])
    bv = PP.filt({f: v for f, v in chain["Bicycle"]("val").items()}, "Bicycle", load["val"], keys["full"])
    pf = PP.filt({f: v for f, v in chain["Person"]("fresh").items()}, "Person", load["fresh"], keys["full"])
    bf = PP.filt({f: v for f, v in chain["Bicycle"]("fresh").items()}, "Bicycle", load["fresh"], keys["full"])
    whole_val = {f: [d for d in pv.get(f, []) if d["class"] == "Person"] + [d for d in bv.get(f, []) if d["class"] == "Bicycle"] for f in vid}
    whole_chk = {f: [d for d in pf.get(f, []) if d["class"] == "Person"] + [d for d in bf.get(f, []) if d["class"] == "Bicycle"] for f in chk}
    gv, gc = S.labels(vid), S.labels(chk)
    expect = {("Person", "val"): 0.4841, ("Person", "check"): 0.4938, ("Bicycle", "val"): 0.5662, ("Bicycle", "check"): 0.5713}
    ok = True
    for c in ("Person", "Bicycle"):
        rv = S.ap({f: [d for d in whole_val[f] if d["class"] == c] for f in vid}, gv, vid, [c])[0.5][c]
        rc = S.ap({f: [d for d in whole_chk[f] if d["class"] == c] for f in chk}, gc, chk, [c])[0.5][c]
        for k, r in (("val", rv), ("check", rc)):
            same = round(r, 4) == expect[(c, k)]
            ok &= same
            print("combined file", c, k, "%.4f" % r, "vs single-class reading", expect[(c, k)], "| equal:", same, flush=True)
    res["combined_equal"] = bool(ok)
    if not ok:
        Path("logs/postproc_p3_time.json").write_text(json.dumps(res, indent=1))
        sys.exit("STOP the combined file does not equal the single-class readings")
    (O / "lumpi_p3_both_test.json").write_text(json.dumps(both))
    print("wrote lumpi_p3_both_test.json: lumpi_v53 with P3 Person and P3 Bicycle", flush=True)
    Path("logs/postproc_p3_time.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main([l.strip() for l in open(sys.argv[1]) if l.strip()])
