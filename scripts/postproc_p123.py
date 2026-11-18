"""Post-processing experiments P1-P3 (and P4). CPU only, on
existing outputs, like for like (same members and chain, only the rule changes); one try each, no values searched. Fixed before any reading:
  P1 "R7" (ProbEn): per input box, member probabilities q_i (refiners' predicted IoU, clipped to [1e-6, 1 - 1e-6]; members without a refinement left
     out): score = prod q / (prod q + prod (1 - q)); box from R0. Read against R0 and against R4 on the same members (must pass both).
  P2 "R8" (KDE box): the member box with the highest Gaussian kernel density of centre distances (bandwidth 0.25 m for Person, pedestrian, Bicycle),
     all members, no gate; score from R4. Read against R4.
  Averages: LUMPI Person (lumpi_v51's five members, then the v37 size change and length rule), LUMPI Bicycle (L11b, one refiner), V2X pedestrian
     (v2x_v25's X6b and negatives model, then rules X1 and X3).
  P3 static background: R1's static real maps (0.2 m voxels occupied in at least half the allowed fit frames: V2X rfit_i1, LUMPI realism_fit; through
     the forbidden-frame guard). A Person / pedestrian box (and Bicycle on LUMPI) with at least 80 % of its points in background voxels gets score
     x 0.3; boxes without points are untouched. On v2x_v25's pedestrian chain and lumpi_v53's Person (v51 chain, R0) and Bicycle (R4) chains.
  Gates: Person, pedestrian: +0.003 on both halves, IoU 0.7 not down, val then fresh (V2X fresh: +0.003 overall and on half 2, IoU 0.7 not down).
     Bicycle: the same plus labels-only up. A pass writes the best unflagged file (lumpi_v53 / v2x_v25) with only that class changed.
  P4 range-normalised point test: medians of the points inside Person / pedestrian boxes per 5 m plan-range bin, from the
     walker-inserted synthetic frames (output/plan_r3/<track>, thinned to real-scan density; synthetic labels only). A box with fewer points than a
     quarter of its bin's median gets score x 0.3 (bins without synthetic labels: untouched). LUMPI: after lumpi_v53's Person chain; V2X: in place of
     rule X1 in v2x_v25's pedestrian chain (X3 kept), read like for like against v2x_v25.
Run one experiment at a time (Mac queue: scripts/postproc_queue.sh):
    python scripts/postproc_p123.py lumpi P1 logs/fresh_ids.txt
    python scripts/postproc_p123.py v2x P3
V2X P1 and P2 were read on Nov 8 (3090 CPU, before the move to the Mac): both failed val; not read again. V2X P3 val passed there (0.3339); the
Mac run must rebuild that val reading exactly before its fresh reading."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ensemble_rules as E

_combine = E.combine
BW = 0.25


def combine(ds, rule, gate=E.GATE):
    if rule not in ("R7", "R8"):
        return _combine(ds, rule, gate)
    d0 = ds[0]
    rd = [d for d in ds if "ref1" in d]
    if not rd:
        return E.keep(d0)
    if rule == "R7":
        q = np.clip([max(d["iou1"], 0.0) for d in rd], 1e-6, 1 - 1e-6)
        a, b = np.prod(q), np.prod(1 - q)
        out = _combine(ds, "R0", gate)
        return {"box": out["box"], "score": float(a / (a + b)), "class": d0["class"]}
    c = np.array([d["ref1"][:2] for d in rd], float)
    dens = np.exp(-((c[:, None] - c[None]) ** 2).sum(-1) / (2 * BW ** 2)).sum(1)
    out = _combine(ds, "R4", gate)
    return {"box": [float(x) for x in rd[int(np.argmax(dens))]["ref1"]], "score": out["score"], "class": d0["class"]}


E.combine = combine   # ensemble_rules.fuse and every reader that calls E.combine now know R7 and R8


def inbox_frac(pts, boxes, keys):
    """Share of each box's points whose 0.2 m voxel is in the static map (None when the box holds no point)."""
    out = [None] * len(boxes)
    if not len(boxes) or not len(pts):
        return out
    tree = cKDTree(pts[:, :2])
    bg = np.isin(enc(np.floor(pts / 0.2).astype(np.int64)), keys)   # once per frame (Nov 13 speed-up; same values as the per-box isin)
    for i, b in enumerate(boxes):
        x, y, z, l, w, h, t = b[:7]
        idx = tree.query_ball_point([x, y], np.hypot(l, w) / 2 + 1e-3)
        if not idx:
            continue
        p = pts[idx] - np.array([x, y, z])
        c, s = np.cos(-t), np.sin(-t)
        u, v = c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]
        m = (np.abs(u) <= l / 2) & (np.abs(v) <= w / 2) & (np.abs(p[:, 2]) <= h / 2)
        if m.any():
            out[i] = float(bg[idx][m].mean())
    return out


def inbox_count(pts, boxes):
    out = np.zeros(len(boxes), int)
    if not len(boxes) or not len(pts):
        return out
    tree = cKDTree(pts[:, :2])
    for i, b in enumerate(boxes):
        x, y, z, l, w, h, t = b[:7]
        idx = tree.query_ball_point([x, y], np.hypot(l, w) / 2 + 1e-3)
        if idx:
            p = pts[idx] - np.array([x, y, z])
            c, s_ = np.cos(-t), np.sin(-t)
            u, v = c * p[:, 0] - s_ * p[:, 1], s_ * p[:, 0] + c * p[:, 1]
            out[i] = int(((np.abs(u) <= l / 2) & (np.abs(v) <= w / 2) & (np.abs(p[:, 2]) <= h / 2)).sum())
    return out


def p4_medians(track, cls):
    r3 = Path(f"output/plan_r3/{track}")
    lab = json.loads((r3 / "labels.json").read_text())
    cnt, rng = [], []
    for f, v in lab.items():
        b = [x[0] for x in v if x[1] == cls]
        if not b:
            continue
        pts = np.load(r3 / "points" / f"{f}.npy")[:, :3]
        pts = pts[np.abs(pts).sum(1) > 0]
        cnt += list(inbox_count(pts, b)); rng += [float(np.hypot(x[0], x[1])) for x in b]
    cnt, bins = np.array(cnt), (np.array(rng) // 5).astype(int)
    med = {int(k): float(np.median(cnt[bins == k])) for k in np.unique(bins)}
    print("P4 medians of points inside synthetic", cls, "boxes per 5 m bin:", {f"{5 * k}-{5 * k + 5} m": (round(v, 1), int((bins == k).sum())) for k, v in med.items()}, flush=True)
    return med


def p4(pred, cls, load, med):
    out, n, k = {}, 0, 0
    for f, v in pred.items():
        idx = [i for i, d in enumerate(v) if d["class"] == cls]
        c = inbox_count(load(f), [v[i]["box"] for i in idx]) if idx else []
        o = list(v)
        for i, m in zip(idx, c):
            d = v[i]; n += 1
            b = int(np.hypot(d["box"][0], d["box"][1]) // 5)
            if b in med and m < 0.25 * med[b]:
                o[i] = {**d, "score": d["score"] * 0.3}; k += 1
        out[f] = o
    print("P4", cls, "boxes", n, "down-weighted", k, "(%.3f)" % (k / max(n, 1)), flush=True)
    return out


enc = lambda k: ((k[:, 0] + 2000) * 4000 + (k[:, 1] + 2000)) * 4000 + (k[:, 2] + 2000)


def static_keys(track):
    import realism_r1 as R1
    c = R1.CFG[track]
    fit = [f for f in R1.real_frames(track, c, "fit")]
    from utl.realism import nonzero
    m = R1.static_map([nonzero(f) for f in fit])   # mean point per kept voxel; its voxel key is the kept voxel
    return np.unique(enc(np.floor(m / 0.2).astype(np.int64)))


def filt(pred, cls, load, keys):
    out, n, k = {}, 0, 0
    for f, v in pred.items():
        bx = [d for d in v if d["class"] == cls]
        fr = inbox_frac(load(f), [d["box"] for d in bx], keys) if bx else []
        it = iter(fr)
        o = []
        for d in v:
            if d["class"] == cls:
                x = next(it)
                n += 1
                if x is not None and x >= 0.8:
                    d = {**d, "score": d["score"] * 0.3}
                    k += 1
            o.append(d)
        out[f] = o
    print("P3", cls, "boxes", n, "down-weighted", k, "(%.3f)" % (k / max(n, 1)), flush=True)
    return out


def jdump(o):
    return json.dumps(o, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


# ------------------------------------------------------------------ LUMPI
def lumpi(exp, fr):
    import lumpi_member_read as M
    import lumpi_static as S
    import lumpi_v37_rules as R
    import route_read_lumpi as RL
    mref = json.loads(M.REFS.read_text())
    rref = json.loads(RL.REFJ.read_text())
    vid = S.ids_of("val")
    sets = {"val": (vid, S.labels(vid)), "fresh": (fr, S.labels(fr))}
    v53 = json.loads((M.O / "lumpi_bicycle_r4_test.json").read_text())

    def person(s, rule):
        p = RL.person(M.paths(s, None), rule)
        ids = sets[s][0] if s != "test" else sorted(v53)
        return {f: p.get(f, []) for f in ids}

    def bike(s, rule):
        return RL.Chain(s, sets[s][0] if s != "test" else sorted(v53)).cls("Bicycle", rule)

    def row(c, p, s):
        ids, gt = sets[s]
        r = R.row(p, gt, ids, c)
        if c == "Bicycle":
            r["labels_only"] = RL.thin(p, gt, ids)
        return r

    def gate(c, r, ref):
        return RL.gate(c, r, ref, (ref.get("labels_only"), r.get("labels_only")) if c == "Bicycle" else None)

    def write(c, pred, name):
        assert set(pred) == set(v53), "class writer: the new class covers other frames than lumpi_v53"
        (M.O / name).write_text(json.dumps({f: [d for d in v53[f] if d["class"] != c] + [RL.keep(d) for d in pred.get(f, []) if d["class"] == c] for f in v53}))
        print("wrote", name, ": lumpi_v53 with only", c, "changed", flush=True)

    refs = {"Person": {s: mref[s] for s in ("val", "fresh")}, "Bicycle": {s: rref[s]["Bicycle"] for s in ("val", "fresh")}}
    build = {"Person": person, "Bicycle": bike}
    res = {}
    if exp in ("P1", "P2"):
        rule, against = {"P1": ("R7", ["R0", "R4"]), "P2": ("R8", ["R4"])}[exp]
        for c in ("Person", "Bicycle"):
            r = row(c, build[c]("val", rule), "val")
            gs = {a: gate(c, r, refs[c]["val"][a]) for a in against}
            print(exp, c, "val", rule, json.dumps(r), "| " + " | ".join("vs %s %+.4f %s" % (a, r["all"][0] - refs[c]["val"][a]["all"][0], "PASS" if g else "fail") for a, g in gs.items()), flush=True)
            e = {"val": r, "val_gate": gs}
            if all(gs.values()):
                rf = row(c, build[c]("fresh", rule), "fresh")
                gf = {a: gate(c, rf, refs[c]["fresh"][a]) for a in against}
                print(exp, c, "fresh", rule, json.dumps(rf), "| " + " | ".join("vs %s %+.4f %s" % (a, rf["all"][0] - refs[c]["fresh"][a]["all"][0], "PASS" if g else "fail") for a, g in gf.items()), flush=True)
                e.update(fresh=rf, fresh_gate=gf)
                if all(gf.values()):
                    write(c, build[c]("test", rule), f"lumpi_{exp.lower()}_{c.lower()}_test.json")
            res[c] = e
    else:
        load = {"val": lambda f: np.load(f"data/pcdet/lumpi_real_50k/points/{f}.npy")[:, :3],
                "fresh": lambda f: np.load(f"data/pcdet/lumpi_real_fresh50k/points/{f}.npy")[:, :3],
                "test": lambda f: np.fromfile(f"data/kit/lumpi/detection_test_frames/{f}.bin", np.float32).reshape(-1, 3)}
        base_rule = {"Person": "R0", "Bicycle": "R4"}   # lumpi_v53's own chains
        if exp == "P3":
            keys = static_keys("lumpi")
            print("P3 static map voxels", len(keys), flush=True)
            classes, fx = ("Person", "Bicycle"), (lambda p, c, s: filt(p, c, load[s], keys))
        else:
            med = p4_medians("lumpi", "Person")
            classes, fx = ("Person",), (lambda p, c, s: p4(p, c, load[s], med))
        for c in classes:
            b0 = build[c]("val", base_rule[c])
            ref = row(c, b0, "val")
            assert abs(ref["all"][0] - refs[c]["val"][base_rule[c]]["all"][0]) < 1e-9, "base does not rebuild lumpi_v53"
            r = row(c, fx(b0, c, "val"), "val")
            g = gate(c, r, ref)
            print(exp, c, "val", json.dumps(r), "vs lumpi_v53 %+.4f" % (r["all"][0] - ref["all"][0]), "| gate", "PASS" if g else "fail", flush=True)
            e = {"val": r, "val_gate": g}
            if exp == "P3" and c == "Person" and r["all"][0] != 0.4841:
                sys.exit("STOP P3 Person val does not rebuild the Nov 8 reading (0.4841)")
            if g:
                bf = build[c]("fresh", base_rule[c])
                fb = fx(bf, c, "fresh")
                if exp == "P3":
                    # Integrity fix (Nov 8, before any LUMPI P3 fresh reading): R1's LUMPI map is built from realism_fit, and the
                    # fresh set is realism_fit + realism_check, so the gated fresh reading uses only the 160 realism_check frames; halves are the
                    # first and second 80 of them in frame order. The all-320 fresh reading is printed for information.
                    chk = sorted(l.strip() for l in open("data/pcdet/lumpi_real/ImageSets/realism_check.txt") if l.strip())
                    assert set(chk) <= set(fr) and not set(chk) & set(l.strip() for l in open("data/pcdet/lumpi_real/ImageSets/realism_fit.txt") if l.strip())
                    gc = S.labels(chk)

                    def row_chk(pp):
                        only = {f: [d for d in pp[f] if d["class"] == c] for f in chk}
                        rr = {}
                        for h, x in (("all", chk), ("h1", chk[:80]), ("h2", chk[80:])):
                            a = S.ap(only, gc, x, [c])
                            rr[h] = [a[0.5][c], a[0.7][c]]
                        if c == "Bicycle":
                            rr["labels_only"] = RL.thin(pp, gc, chk)
                        return rr
                    ref_f, rf = row_chk(bf), row_chk(fb)
                    info = (row(c, bf, "fresh"), row(c, fb, "fresh"))
                    print(exp, c, "fresh, all 320 frames (information; half are map frames)", json.dumps(info[1]), "vs lumpi_v53", json.dumps(info[0]), "%+.4f" % (info[1]["all"][0] - info[0]["all"][0]), flush=True)
                    e["fresh_all320_info"] = {"new": info[1], "ref": info[0]}
                else:
                    ref_f, rf = row(c, bf, "fresh"), row(c, fb, "fresh")
                gf = gate(c, rf, ref_f)
                print(exp, c, "fresh" + (" (160 check frames, gated)" if exp == "P3" else ""), json.dumps(rf), "vs lumpi_v53", json.dumps(ref_f), "%+.4f" % (rf["all"][0] - ref_f["all"][0]), "| gate", "PASS" if gf else "fail", flush=True)
                e.update(fresh=rf, fresh_ref=ref_f, fresh_gate=gf)
                if gf:
                    t = {f: [d for d in v if d["class"] == c] for f, v in v53.items()}
                    write(c, fx(t, c, "test"), f"lumpi_{exp.lower()}_{c.lower()}_test.json")
            res[c] = e
    Path(f"logs/postproc_{exp}_lumpi.json").write_text(jdump(res))


# ------------------------------------------------------------------ V2X
def v2x(exp):
    import v2x_plan as P
    import v2x_v20_rules as VR
    from v2x_neg_s2_read import fresh_score
    assert exp in ("P3", "P4"), "V2X P1 and P2 were read on Nov 8 (both failed val); not read again"

    def chain(s, kind, rules):
        p = E.fuse(E.load(P.v25_paths("pedestrian", s), "pedestrian"), "R0")
        return VR.apply(p, VR.features(kind, p, sorted(p)), rules)

    def g_val(r, ref):
        return bool(all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1])

    def g_fr(a, b):
        return bool(round(1e4 * (a["all"][0] - b["all"][0])) >= 30 and round(1e4 * (a["h2"][0] - b["h2"][0])) >= 30 and a["all"][1] >= b["all"][1])

    nz = lambda p: p[np.abs(p).sum(1) > 0]
    load = {"val_i1": lambda f: nz(np.load(f"data/pcdet/v2x_real_50k/points/{f}.npy")[:, :3]),
            "fresh": lambda f: nz(np.load(f"data/pcdet/v2x_fresh_50k/points/{f}.npy")[:, :3]),
            "test": lambda f: nz(np.fromfile(f"data/kit/v2x/detection_test_frames/{f}.bin", np.float32).reshape(-1, 3))}
    if exp == "P3":
        keys = static_keys("v2x")
        print("P3 static map voxels", len(keys), flush=True)
        make = lambda s, kind: filt(chain(s, kind, ["X1", "X3"]), "pedestrian", load[s], keys)
    else:
        med = p4_medians("v2x", "pedestrian")
        make = lambda s, kind: p4(chain(s, kind, ["X3"]), "pedestrian", load[s], med)   # P4 in place of X1
    ref = P.score(chain("val_i1", "val", ["X1", "X3"]), "pedestrian")
    assert ref["all"][0] == P.V25["pedestrian"], "base does not rebuild v2x_v25 pedestrian"
    r = P.score(make("val_i1", "val"), "pedestrian")
    g = g_val(r, ref)
    print(exp, "pedestrian val_i1", json.dumps(r), "vs v2x_v25 %+.4f" % (r["all"][0] - ref["all"][0]), "| gate", "PASS" if g else "fail", flush=True)
    if exp == "P3" and r["all"][0] != 0.3339:
        sys.exit("STOP P3 val_i1 does not rebuild the Nov 8 reading (0.3339)")
    e = {"val": r, "val_gate": g}
    if g:
        b, a = fresh_score(chain("fresh", "fresh", ["X1", "X3"])), fresh_score(make("fresh", "fresh"))
        gf = g_fr(a, b)
        print(exp, "pedestrian fresh", json.dumps(a), "vs v2x_v25", json.dumps(b), "%+.4f" % (a["all"][0] - b["all"][0]), "| gate", "PASS" if gf else "fail", flush=True)
        e.update(fresh=a, fresh_ref=b, fresh_gate=gf)
        if gf:
            t = json.loads((P.E / "v25_test.json").read_text())
            x = make("test", "test")
            assert set(x) == set(t), "class writer: the new class covers other frames than the base file"
            canon = lambda v: sorted(json.dumps([round(u, 5) for u in d["box"]] + [round(d["score"], 6), d["class"]]) for d in v)
            b0 = chain("test", "test", ["X1", "X3"])   # rebuild equals base: v25's pedestrian chain gives v25_test's pedestrians back
            ok = all(canon([E.keep(d) for d in b0.get(f, [])]) == canon([E.keep(d) for d in t[f] if d["class"] == "pedestrian"]) for f in t)
            print("test: v2x_v25 pedestrian chain rebuilt equals v25_test.json:", ok, flush=True)
            if not ok:
                sys.exit("STOP the base pedestrians do not rebuild on test")
            (P.E / f"v2x_{exp.lower()}_test.json").write_text(json.dumps({f: sorted([d for d in t[f] if d["class"] != "pedestrian"] + [E.keep(d) for d in x.get(f, [])], key=lambda d: -d["score"]) for f in t}))
            print(f"wrote v2x_{exp.lower()}_test.json: v2x_v25 with only pedestrians changed", flush=True)
    Path(f"logs/postproc_{exp}_v2x.json").write_text(jdump({"pedestrian": e}))


if __name__ == "__main__":
    lumpi(sys.argv[2], [l.strip() for l in open(sys.argv[3]) if l.strip()]) if sys.argv[1] == "lumpi" else v2x(sys.argv[2])
