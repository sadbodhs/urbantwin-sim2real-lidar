"""Inside-gate Bicycle reading for the new LUMPI members on lumpi_v59 (Bicycle included,
with the Person copies recomputed). v59's Bicycle = L11b's Bicycle second stage on the v29 lane, R4, P3 (size_check_lumpi.build). Candidate = L11b's
Bicycle second stage on the member's Bicycle lane (the v29 lane + the member at weight 1 for Bicycle; as every member route), R4, P3. Person = v59's
Person chain plus v57's copies recomputed from the candidate's Bicycle and v59's Motorcycle.
Gate: Bicycle AP@0.5 on the 120 inside frames +0.01 or more (10 inside ids); Person on inside (with the recomputed copies) not down more than 0.005;
Bicycle val (320) not down more than 0.01. Before, AP@0.7: information. Self-check: the base reproduces v59's Bicycle and Person on inside.
    python scripts/ig_bike_read.py near   (neartest_ws: member refines at output/lumpi/<m>/g3/bicycle_val_ref.json) -> logs/ig_bike_near.json
    python scripts/ig_bike_read.py val    (main: the members' route refines output/lumpi/<m>/s2/bicycle_val_ref.json) -> logs/ig_bike.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ensemble_rules as E
import g3_search as G3
import lumpi_copies_routes as W
import lumpi_static as S
import size_check_lumpi as SZ

MODELS = ["vrcnn_lumpi_fix_s3", "vrcnn_lumpi_c_s2", "vrcnn_lumpi_rs_wa"]
rd = lambda p: [l.strip() for l in open(p) if l.strip()]
LOAD = lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3]


def bike(path, ids, keys):
    bi = W.only(path, "Bicycle")
    out = {f: [E.combine([d], "R4") if d["score"] >= 0.05 else SZ.keep(d) for d in bi.get(f, [])] for f in ids}
    return {f: [SZ.keep(d) for d in v] for f, v in SZ.PP.filt(out, "Bicycle", LOAD, keys).items()}


def ap(p, gt, fr, c):
    return SZ.ap_from(*SZ.flags({f: p.get(f, []) for f in fr}, {f: gt[f] for f in fr}, c))


def main(mode):
    ids = S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in S.labels(ids).items()}
    keys = SZ.PP.static_keys("lumpi")
    P = SZ.build("val", ids)
    src = {m: Path(f"output/lumpi/{m}/g3/bicycle_val_ref.json" if mode == "near" else f"output/lumpi/{m}/s2/bicycle_val_ref.json") for m in MODELS}
    if mode == "near":
        H = {h: rd(f"data/pcdet/lumpi_real_50k/ImageSets/{h}.txt") for h in ("before", "inside")}
        base_b = bike(G3.RL.B / "v29_val_ref.json", ids, keys)
        same = all(ap(base_b, gt, H[h], "Bicycle")[0] == ap(P["Bicycle"], gt, H[h], "Bicycle")[0] for h in H)
        m3 = json.loads(Path("logs/m3_neartest_copies.json").read_text())["copies_on"]["inside"][0]
        person = lambda b: {f: P["Person"][f] + W.copies({f: b[f] + P["Motorcycle"][f]})[f] for f in ids}
        pb = ap(person(P["Bicycle"]), gt, H["inside"], "Person")[0]
        print(f"self-check: base Bicycle equals v59's on before / inside: {same}; v59 Person inside {pb:.6f} vs M3 {m3:.6f}: {abs(pb - m3) < 1e-9}", flush=True)
        if not same or abs(pb - m3) >= 1e-9:
            sys.exit("STOP the base does not reproduce v59")
        res = {}
        for m, p in src.items():
            if not p.exists():
                print(f"{m}: no near-test Bicycle second stage, skipped", flush=True)
                continue
            b = bike(p, ids, keys)
            r = {h: dict(bicycle=ap(b, gt, fr, "Bicycle"), bicycle_v59=ap(P["Bicycle"], gt, fr, "Bicycle")) for h, fr in H.items()}
            pn = ap(person(b), gt, H["inside"], "Person")[0]
            r.update(person_inside=pn, person_inside_v59=pb, d_inside=r["inside"]["bicycle"][0] - r["inside"]["bicycle_v59"][0],
                     d_before=r["before"]["bicycle"][0] - r["before"]["bicycle_v59"][0], d_person=pn - pb)
            res[m] = r
            print(f"Bicycle: add {m}: inside {r['inside']['bicycle_v59'][0]:.4f} -> {r['inside']['bicycle'][0]:.4f} ({r['d_inside']:+.4f}) | before {r['d_before']:+.4f} | "
                  f"Person inside with recomputed copies {pb:.4f} -> {pn:.4f} ({pn - pb:+.4f})", flush=True)
        Path("logs/ig_bike_near.json").write_text(json.dumps(res, indent=1))
        return
    near = json.loads(Path("logs/ig_bike_near.json").read_text())
    vb = ap(P["Bicycle"], gt, ids, "Bicycle")[0]
    print(f"val: v59 Bicycle {vb:.4f}", flush=True)
    out = {}
    for m, r in near.items():
        a = ap(bike(src[m], ids, keys), gt, ids, "Bicycle")[0] if src[m].exists() else None
        ok = bool(r["d_inside"] >= 0.01 and r["d_person"] >= -0.005 and a is not None and a - vb >= -0.01)
        out[m] = dict(r, val=a, val_v59=vb, passed=ok)
        print(f"Bicycle: add {m}: inside {r['d_inside']:+.4f} (10 ids) | Person inside {r['d_person']:+.4f} | val "
              f"{'n/a (route refine not ready)' if a is None else f'{a:.4f} ({a - vb:+.4f})'} | {'PASS' if ok else 'fail'}", flush=True)
    Path("logs/ig_bike.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
