"""Task B, size check on lumpi_v54. Read-only measurement:
builds nothing for upload, writes no zip. Rule (kit submission_format.md): total uncompressed size "must be under ~75 MB"; target <= 72,000,000 B.
lumpi_v54 = output/lumpi/ens_x/lumpi_p3_both_test.json, synthetic frames = output/lumpi/realism_v2 (byte-identical to the v54 zip's).
Val (320) and fresh (realism_check, 160) predictions are v54's own chains, rebuilt per class and checked against the logged readings:
  Person: R0 fusion of lumpi_v51's five second stages, grow + v37 length rule, P3 (scripts/m1_late_person.person_chain + postproc_p123.filt)
  Bicycle: L11b second stage R4 (route_read_lumpi.Chain), P3; Car / Truck / Van: lumpi_v41 chain (route_read_lumpi.Chain)
  Motorcycle / Bus / Unknown: the v29 boxes as they are (v54's test boxes equal v29's for these classes and Car).
AP: the kit's own scoring_program/detection_eval.evaluate_3d_detection (3D IoU, AP@0.5 and AP@0.7, R40), full precision.
  0. Compact JSON (no spaces) with box values rounded to 3 decimals, scores untouched: size; every class's AP on val and fresh before / after.
  1. Per class, drop boxes below a score floor (FLOORS): test bytes saved (compact, rounded), AP@0.5 / AP@0.7 cost on val and fresh; board cost
     = mean(val, fresh) AP@0.5 cost x 0.12 (+0.01 AP in one class = +0.0012 on the board).
  2. Synthetic frames thinned uniformly (exactly round(q n) rows per frame, chosen with default_rng(frame number), original order kept):
     kit realism (the kit's compute_realism, seed 1234, against data/kit_ref/lumpi/realism_reference, the kit's normalisation; v54 gives
     0.941798) and the 5-seed mean on realism_check (scripts/realism_r1.per_seed, seeds 1234, 1, 2, 3, 4); board cost = 0.4 x realism loss.
  3. Cheapest combination (option 0 + one floor per class + one thinning fraction) with total <= 72,000,000 B, its size measured exactly.
Writes logs/size_check_lumpi.json.
    python scripts/size_check_lumpi.py"""
import importlib.util
import itertools
import json
import multiprocessing as mp
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_copies_routes as W
import lumpi_member_read as M
import lumpi_static as S
import lumpi_v37_rules as R
import m1_late_person as M1
import postproc_p123 as PP
import realism_r1 as R1
import route_read_lumpi as RL
from utl import guard
from utl.realism import _load, nonzero
from utl.tracks import TRACKS

TARGET = 72_000_000
FLOORS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
QS = [0.75, 0.5, 0.35, 0.25, 0.15, 0.1]
BOARD_AP = 0.12   # board per 1.0 AP in one class (0.0012 per 0.01)
O = Path("output/lumpi/ens_x")
SYN = Path("output/lumpi/realism_v2")
V54ZIP = Path("tmp_size/lumpi_v54_p3-person-bicycle-flagged.zip")   # temporary copy of the uploaded zip (the container sees only the repo)
KIT = Path("data/kit/lumpi/scoring_program")
REF = Path("data/kit_ref/lumpi")


def mod(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


sys.path.insert(0, str(KIT))
DE = mod("kit_detection_eval", KIT / "detection_eval.py")
KM = mod("kit_metrics", KIT / "metrics.py")
SC = mod("kit_score", KIT / "score.py")   # its load_bin
CFG = json.loads((REF / "config.json").read_text())
CLASSES = CFG["class_names"]
keep = lambda d: {"box": [float(x) for x in d["box"]], "score": float(d["score"]), "class": d["class"]}
r3 = lambda d: {"box": [round(x, 3) for x in d["box"]], "score": d["score"], "class": d["class"]}
compact = lambda p: json.dumps(p, separators=(",", ":"))


def flags(pred, gt, c):
    """The kit's _ap_for_class matching, step for step (same frame order, per-frame score sort, greedy over unused labels in order, strict >),
    with each pair's IoU computed once by the kit's own _iou_3d and shared by both thresholds; pairs whose BEV circles cannot touch have IoU 0.0
    in the kit too (no polygon intersection), so they are skipped. Returns per threshold the kit's (scores, tp) lists in its append order."""
    n_gt = sum(sum(1 for g in v if g["class"].lower() == c.lower()) for v in gt.values())
    out = {t: ([], []) for t in (0.5, 0.7)}
    for f, gts in gt.items():
        cg = [np.asarray(g["box"], dtype=float) for g in gts if g["class"].lower() == c.lower()]
        ps = [p for p in pred.get(f, []) if p["class"].lower() == c.lower()]
        if not cg:
            for t in out:
                out[t][0].extend(p["score"] for p in ps)
                out[t][1].extend(0 for _ in ps)
            continue
        ps = sorted(ps, key=lambda d: -d["score"])
        if not ps:
            continue
        pb = np.asarray([p["box"] for p in ps], dtype=float)
        gb = np.asarray(cg)
        rp, rg = np.hypot(pb[:, 3], pb[:, 4]) / 2, np.hypot(gb[:, 3], gb[:, 4]) / 2
        dd = np.hypot(pb[:, None, 0] - gb[None, :, 0], pb[:, None, 1] - gb[None, :, 1])
        iou = np.zeros((len(ps), len(cg)))
        for i, j in np.argwhere(dd <= rp[:, None] + rg[None] + 1e-6):
            iou[i, j] = DE._iou_3d(pb[i], gb[j])
        for t in out:
            used = [False] * len(cg)
            for i, p in enumerate(ps):
                best, bj = 0.0, -1
                for j in range(len(cg)):
                    if used[j]:
                        continue
                    if iou[i, j] > best:
                        best, bj = iou[i, j], j
                out[t][0].append(p["score"])
                if best >= t and bj >= 0:
                    out[t][1].append(1)
                    used[bj] = True
                else:
                    out[t][1].append(0)
    return n_gt, out


def ap_from(n_gt, fl, floor=None):
    """The kit's AP@0.5 / AP@0.7 R40 from the matching lists; with a floor, the boxes below it are left out (the kit would build exactly these
    lists: the matches of higher-scored boxes in a frame do not depend on lower ones)."""
    if n_gt == 0:
        return [None, None]
    r = []
    for t in (0.5, 0.7):
        sc, tp = np.asarray(fl[t][0], dtype=float), np.asarray(fl[t][1])
        if floor is not None:
            k = sc >= floor
            sc, tp = sc[k], tp[k]
        if not len(sc):
            r.append(0.0)
            continue
        o = np.argsort(-sc)
        a = np.cumsum(tp[o])
        b = np.cumsum(1 - tp[o])
        r.append(float(DE._compute_ap_40(a / np.maximum(a + b, 1), a / n_gt)))
    return r


_P = _G = None
NPROC = 6


def job(a):
    s, c, v = a
    pred = _P[s][c] if v == "base" else {f: [r3(d) for d in x] for f, x in _P[s][c].items()}
    return flags(pred, _G[s], c)


def kit_ap(pred, gt, c):
    r = DE.evaluate_3d_detection(pred, gt, [c], iou_thresholds=(0.5, 0.7), iou_type="3d")["per_class"][c]
    return [r.get("AP_0.5_R40"), r.get("AP_0.7_R40")]


def build(s, ids):
    load = {"val": lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3],
            "fresh": lambda f: np.load(f"data/pcdet/lumpi_real_fresh50k/points/{f}.npy")[:, :3]}[s]
    keys = PP.static_keys("lumpi")
    ch = RL.Chain(s, ids)
    per = M1.person_chain(M.paths(s, None), True)
    out = {"Person": PP.filt({f: per.get(f, []) for f in ids}, "Person", load, keys)}
    bi = ch.cls("Bicycle", "R4")
    out["Bicycle"] = PP.filt({f: bi.get(f, []) for f in ids}, "Bicycle", load, keys)
    for c in ("Car", "Truck", "Van"):
        out[c] = ch.cls(c)
    for c in ("Motorcycle", "Bus", "Unknown"):
        x = W.only(W.L / f"v29_{s}_ref.json", c)
        out[c] = {f: x.get(f, []) for f in ids}
    return {c: {f: [keep(d) for d in v.get(f, []) if d["class"] == c] for f in ids} for c, v in out.items()}


def thin(frames, q):
    out = []
    for i, f in enumerate(frames, 1):
        n = len(f)
        idx = np.sort(np.random.default_rng(i).choice(n, int(round(q * n)), replace=False))
        out.append(f[idx])
    return out


def kit_realism(frames, real):
    raw = KM.compute_realism([f[:, :3] for f in frames], real, tuple(CFG["point_cloud_range"]), seed=1234)
    n = CFG["normalization"]
    norm = lambda v, lo, hi: 1 - float(np.clip((v - lo) / (hi - lo), 0, 1))
    return float(np.mean([norm(raw[k], *n[k]) for k in ("CD", "MMD", "EMD", "FPD")]))


def main():
    res = {"target_bytes": TARGET}
    with zipfile.ZipFile(V54ZIP) as z:
        info = {i.filename: i.file_size for i in z.infolist()}
        syn_zip = {n: z.read(n) for n in info if n.startswith("synthetic/")}
    syn_files = [SYN / f"{i:04d}.bin" for i in range(1, 51)]
    same = all(syn_zip[f"synthetic/{i:04d}.bin"] == p.read_bytes() for i, p in enumerate(syn_files, 1))
    total, pj, decl = sum(info.values()), info["predictions.json"], info["declaration.txt"]
    syn_b = sum(v for k, v in info.items() if k.startswith("synthetic/"))
    print(f"lumpi_v54 zip: {len(info)} files, {total:,} B unpacked = synthetic {syn_b:,} + predictions.json {pj:,} + declaration {decl:,} | "
          f"realism_v2 frames byte-identical to the zip's: {same}", flush=True)
    if not same:
        sys.exit("STOP realism_v2 is not v54's synthetic set")
    res["v54"] = dict(files=len(info), total=total, synthetic=syn_b, predictions=pj, declaration=decl)
    test = json.loads((O / "lumpi_p3_both_test.json").read_text())
    test = {f: sorted([keep(d) for d in v], key=lambda d: -d["score"]) for f, v in test.items()}
    full = len(json.dumps(test).encode())
    print(f"predictions.json as packaged (json.dumps, score order): {full:,} B (zip member {pj:,} B)", flush=True)
    rt = {f: [r3(d) for d in v] for f, v in test.items()}
    c_only, c_r3 = len(compact(test).encode()), len(compact(rt).encode())
    print(f"option 0: compact only {c_only:,} B; compact + boxes to 3 decimals {c_r3:,} B ({c_r3 - pj:+,} B)", flush=True)
    res["option0"] = dict(compact_only=c_only, compact_r3=c_r3)

    # val / fresh rebuilds
    vid = S.ids_of("val")
    chk = sorted(l.strip() for l in open("data/pcdet/lumpi_real/ImageSets/realism_check.txt") if l.strip())
    for i in chk:
        guard.check("lumpi", i)
    sets = {"val": vid, "fresh": chk}
    want = {("val", "Person"): 0.4841, ("fresh", "Person"): 0.4938, ("val", "Bicycle"): 0.5662, ("fresh", "Bicycle"): 0.5713,
            ("val", "Car"): 0.8858, ("val", "Truck"): 0.4632, ("val", "Van"): 0.1708}
    P, G = {}, {}
    for s, ids in sets.items():
        P[s] = build(s, ids)
        G[s] = {f: [{"box": [float(x) for x in b], "class": c} for b, c in g] for f, g in S.labels(ids).items()}
        for c in P[s]:
            ours = S.ap(P[s][c], S.labels(ids), ids, [c])[0.5][c]
            w = want.get((s, c))
            print(f"rebuild {s} {c}: {sum(map(len, P[s][c].values())):,} boxes | AP@0.5 (ours, 4 dp) {ours}"
                  f"{' vs logged ' + str(w) if w is not None else ''}", flush=True)
            if w is not None and ours != w:
                sys.exit("STOP rebuild does not equal the logged reading")
    # self-check: the shared-IoU matching equals the kit's evaluate_3d_detection exactly (40 val frames, Person and Car)
    sub = sets["val"][:40]
    for c in ("Person", "Car"):
        pp, gg = {f: P["val"][c][f] for f in sub}, {f: G["val"][f] for f in sub}
        k, m = kit_ap(pp, gg, c), ap_from(*flags(pp, gg, c))
        print(f"self-check {c} (40 val frames): kit {k} | shared-IoU matching {m} | equal: {k == m}", flush=True)
        if k != m:
            sys.exit("STOP the shared-IoU matching does not equal the kit")
    global _P, _G
    _P, _G = P, G
    jobs = [(s, c, v) for s in sets for c in CLASSES for v in ("base", "r3")]
    with mp.get_context("fork").Pool(NPROC) as pool:
        out = dict(zip(jobs, pool.map(job, jobs)))
    FL = {(s, c): out[(s, c, "base")] for s in sets for c in CLASSES}
    base = {s: {c: ap_from(*FL[(s, c)]) for c in CLASSES} for s in sets}
    res["base_ap"] = base
    print("base AP (kit matching, 3D, R40, [0.5, 0.7]):", json.dumps(base), flush=True)

    # option 0 AP check
    rs = {}
    for s in sets:
        for c in CLASSES:
            r = ap_from(*out[(s, c, "r3")])
            b = base[s][c]
            diff = [None if (x is None or y is None) else x - y for x, y in zip(r, b)]
            rs.setdefault(s, {})[c] = dict(ap=r, diff=diff)
            print(f"option 0 {s} {c}: AP {r} vs {b} | difference {diff} | equal at full precision: {r == b} | at 6 dp: "
                  f"{[None if x is None else round(x, 6) for x in r] == [None if x is None else round(x, 6) for x in b]}", flush=True)
    res["option0"]["ap"] = rs

    # option 1: score floors per class
    bpb = {c: {f: [len(compact(r3(d)).encode()) + 1 for d in v if d["class"] == c] for f, v in test.items()} for c in CLASSES}
    o1 = {}
    for c in CLASSES:
        n_c = sum(1 for v in test.values() for d in v if d["class"] == c)
        o1[c] = {"boxes": n_c, "bytes": sum(sum(x) for x in bpb[c].values())}
        for fl in FLOORS:
            drop = [(d, b) for f, v in test.items() for d, b in zip([d for d in v if d["class"] == c], bpb[c][f]) if d["score"] < fl]
            row = {"dropped": len(drop), "bytes_saved": sum(b for _, b in drop)}
            for s in sets:
                ap = ap_from(*FL[(s, c)], floor=fl)
                row[s] = [None if (x is None or y is None) else x - y for x, y in zip(ap, base[s][c])]
            v5 = [row[s][0] for s in sets if row[s][0] is not None]
            row["board_cost"] = -BOARD_AP * float(np.mean(v5)) if v5 else 0.0
            o1[c][fl] = row
            print(f"option 1 {c} floor {fl}: drops {len(drop):,} of {n_c:,} test boxes, saves {row['bytes_saved']:,} B | AP change val "
                  f"{row['val']} fresh {row['fresh']} | board cost {row['board_cost']:.5f}", flush=True)
    res["option1"] = o1

    # option 2: thinning
    pc = TRACKS["lumpi"]["pc_range"]
    frames = [_load(p) for p in syn_files]
    real = [SC.load_bin(p) for p in sorted((REF / "realism_reference").glob("*.bin"))]
    c1 = R1.CFG["lumpi"]
    rep = json.loads((Path(c1["pool"]) / "realism_report.json").read_text())
    check = R1.real_frames("lumpi", c1, "check")
    if rep["zeros"] == "drop":
        check = [nonzero(f) for f in check]
    o2 = {}
    for q in [1.0] + QS:
        fr = frames if q == 1.0 else thin(frames, q)
        kit = kit_realism(fr, real)
        ps = [r["score"] for r in R1.per_seed(fr, check, pc)]
        o2[q] = {"bytes": int(sum(f.astype(np.float32).nbytes for f in fr)), "kit_seed1234": kit, "check_seeds": ps, "check_mean": float(np.mean(ps)),
                 "max_file": int(max(f.astype(np.float32).nbytes for f in fr))}
        if q == 1.0 and round(kit, 6) != 0.941798:
            sys.exit(f"STOP kit realism of v54's frames {kit} is not 0.941798")
        print(f"option 2 keep {q}: synthetic {o2[q]['bytes']:,} B | kit realism (seed 1234) {kit:.6f} | realism_check 5 seeds "
              f"{[round(x, 4) for x in ps]} mean {np.mean(ps):.4f}", flush=True)
    for q in QS:
        dk, dm = o2[q]["kit_seed1234"] - o2[1.0]["kit_seed1234"], o2[q]["check_mean"] - o2[1.0]["check_mean"]
        o2[q].update(d_kit=dk, d_check_mean=dm, board_cost_kit=-0.4 * dk, board_cost_check=-0.4 * dm)
    res["option2"] = o2

    # option 3: cheapest combination under TARGET
    opts = {c: [(None, 0, 0.0)] + [(fl, o1[c][fl]["bytes_saved"], o1[c][fl]["board_cost"]) for fl in FLOORS] for c in CLASSES}
    best = None
    for q in QS + [1.0]:
        budget = TARGET - decl - o2[q]["bytes"] - c_r3
        rc = max(0.0, o2[q].get("board_cost_check", 0.0))
        for combo in itertools.product(*[opts[c] for c in CLASSES]):
            saved = sum(x[1] for x in combo)
            if saved < -budget:
                continue
            cost = sum(x[2] for x in combo) + rc
            if best is None or cost < best[0] - 1e-12:
                best = (cost, q, {c: x[0] for c, x in zip(CLASSES, combo)})
    if best is None:
        print("no combination reaches the target", flush=True)
        res["option3"] = None
    else:
        cost, q, fl = best
        pred = {f: [r3(d) for d in v if fl[d["class"]] is None or d["score"] >= fl[d["class"]]] for f, v in test.items()}
        pb = len(compact(pred).encode())
        tot = pb + o2[q]["bytes"] + decl
        res["option3"] = dict(keep=q, floors=fl, predictions=pb, synthetic=o2[q]["bytes"], total=tot, est_board_cost=cost,
                              det_cost=cost - max(0.0, o2[q].get("board_cost_check", 0.0)))
        print(f"cheapest under {TARGET:,} B: thinning keep {q}, floors {fl} | predictions.json {pb:,} + synthetic {o2[q]['bytes']:,} + declaration "
              f"{decl} = {tot:,} B | estimated board cost {cost:.5f}", flush=True)
    Path("logs/size_check_lumpi.json").write_text(json.dumps(res, indent=1, default=str))
    print("wrote logs/size_check_lumpi.json", flush=True)


if __name__ == "__main__":
    main()
