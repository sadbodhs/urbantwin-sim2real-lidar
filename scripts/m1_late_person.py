"""M1, late-window Person study.
Read-only, Mac CPU, information for the final-file choice (not a gate). Real labels: val and fresh scoring only; test labels are never read.
Base = lumpi_v55's Person on val / fresh, rebuilt as scripts/combined_zips.py builds it: R0 fusion of lumpi_v51's five Person second stages
(route_read_lumpi.person: box grow, then the v37 length rule score x 0.3 for boxes >= 0.85 m long), then P3 (postproc_p123.filt). The base is
checked against the logged v55 readings (val 0.4841, check 0.4938) before anything else.
Variants, each changing one thing against the base (like for like):
  (a) length rule off: the same fusion and grow without the x 0.3 for long boxes, then P3;
  (b) two-wheeler copies on: base + v46's rule on v55's own two-wheelers (Bicycle = v53 chain R4 then P3, Motorcycle = the v29 lane), every
      box with score >= 0.3 also as Person, same box, score x 0.3, added after P3;
  (c) P3 off: the fusion with the length rule, no P3.
Frames: val (320) and fresh = realism_check only (160; P3's static map is built from the realism_fit frames, so v55's Person is read on the check
frames, as in its own reading). 400-frame windows from 0 to 6400 by frame number; per window: Person labels, share of Person labels with length
>= 1 m, Person AP (R40) at IoU 0.5 for base and each variant. Pooled: frames >= 4800, val and check together (frame keys prefixed per split).
Writes logs/m1_late_person.json.
    python scripts/m1_late_person.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_copies_routes as W
import lumpi_member_read as M
import lumpi_static as S
import postproc_p123 as PP
import route_read_lumpi as RL

WIN = 400
POOL_FROM = 4800


def person_chain(paths, length_rule=True):
    p = RL.E.fuse(RL.E.load(paths, "Person"), "R0")
    out = {}
    for f, v in p.items():
        out[f] = []
        for d in v:
            b = RL.W.grow(d["box"])
            out[f].append({"box": b, "score": d["score"] * 0.3 if (length_rule and b[3] >= 0.85) else d["score"], "class": "Person"})
    return out


def main():
    vid = S.ids_of("val")
    chk = sorted(l.strip() for l in open("data/pcdet/lumpi_real/ImageSets/realism_check.txt") if l.strip())
    sets = {"val": vid, "fresh": chk}
    load = {"val": lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3],
            "fresh": lambda f: np.load(f"data/pcdet/lumpi_real_fresh50k/points/{f}.npy")[:, :3]}
    keys = PP.static_keys("lumpi")
    V = {}   # variant -> split -> {frame: boxes}
    for s, ids in sets.items():
        paths = M.paths(s, None)
        on = {f: v for f, v in person_chain(paths, True).items() if f in ids}
        off = {f: v for f, v in person_chain(paths, False).items() if f in ids}
        on = {f: on.get(f, []) for f in ids}
        off = {f: off.get(f, []) for f in ids}
        base = PP.filt(on, "Person", load[s], keys)
        bi = RL.Chain(s, ids).cls("Bicycle", "R4")
        bi = PP.filt({f: bi.get(f, []) for f in ids}, "Bicycle", load[s], keys)
        mo = W.only(W.L / f"v29_{s}_ref.json", "Motorcycle")
        wheels = {f: [d for d in bi[f] if d["class"] == "Bicycle"] + [W.keep(d) for d in mo.get(f, [])] for f in ids}
        cp = W.copies(wheels)
        V.setdefault("base", {})[s] = base
        V.setdefault("len_off", {})[s] = PP.filt(off, "Person", load[s], keys)
        V.setdefault("copies_on", {})[s] = {f: base[f] + cp[f] for f in ids}
        V.setdefault("p3_off", {})[s] = on
        print(s, "frames", len(ids), "copies", sum(map(len, cp.values())), flush=True)
    gt = {s: S.labels(ids) for s, ids in sets.items()}
    for s, want in (("val", 0.4841), ("fresh", 0.4938)):
        got = S.ap(V["base"][s], gt[s], sets[s], ["Person"])[0.5]["Person"]
        print(f"base check {s}: Person {got} vs logged v55 {want}", flush=True)
        if got != want:
            sys.exit("STOP the base does not rebuild v55's Person reading")
    # pooled sets with prefixed keys
    P = {k: {f"{s[0]}{f}": v for s in sets for f, v in V[k][s].items()} for k in V}
    G = {f"{s[0]}{f}": v for s in sets for f, v in gt[s].items()}
    num = lambda key: int(key[1:])
    rows = []
    for lo in range(0, 6400, WIN):
        fr = sorted(k for k in G if lo <= num(k) < lo + WIN)
        lab = [b for k in fr for b, c in G[k] if c == "Person"]
        r = {"window": f"{lo}-{lo + WIN - 1}", "frames_val": sum(k[0] == "v" for k in fr), "frames_check": sum(k[0] == "f" for k in fr),
             "person_labels": len(lab), "share_len_ge_1m": round(float(np.mean([b[3] >= 1.0 for b in lab])), 3) if lab else None}
        for k in V:
            r[k] = S.ap(P[k], G, fr, ["Person"])[0.5]["Person"] if lab else None
        for k in ("len_off", "copies_on", "p3_off"):
            r[f"d_{k}"] = None if r[k] is None or r["base"] is None else round(r[k] - r["base"], 4)
        rows.append(r)
        print(json.dumps(r), flush=True)
    late = sorted(k for k in G if num(k) >= POOL_FROM)
    pooled = {"frames": len(late), "frames_val": sum(k[0] == "v" for k in late), "frames_check": sum(k[0] == "f" for k in late)}
    for k in V:
        a = S.ap(P[k], G, late, ["Person"])
        pooled[k] = [a[0.5]["Person"], a[0.7]["Person"]]
    for k in ("len_off", "copies_on", "p3_off"):
        pooled[f"d_{k}"] = round(pooled[k][0] - pooled["base"][0], 4)
    lab = [b for k in late for b, c in G[k] if c == "Person"]
    pooled["person_labels"] = len(lab)
    pooled["share_len_ge_1m"] = round(float(np.mean([b[3] >= 1.0 for b in lab])), 3)
    whole = {}
    for k in V:
        for s in sets:
            a = S.ap(V[k][s], gt[s], sets[s], ["Person"])
            whole.setdefault(k, {})[s] = [a[0.5]["Person"], a[0.7]["Person"]]
    print("pooled frames >= 4800 (val + check):", json.dumps(pooled), flush=True)
    print("whole split (val 320 / check 160):", json.dumps(whole), flush=True)
    Path("logs/m1_late_person.json").write_text(json.dumps({"windows": rows, "pooled_ge_4800": pooled, "whole": whole}, indent=1))
    print("wrote logs/m1_late_person.json", flush=True)


if __name__ == "__main__":
    main()
