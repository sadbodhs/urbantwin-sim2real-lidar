"""LUMPI test B: Van length rule on lumpi_v41. Van boxes shorter than 4.7 m get score x 0.3, once at most, on boxes scoring >= 0.05 that no earlier rule has lowered (v37 Van h < 2.0 m, v39 V5 npts < 15). No boxes added or moved; other classes and realism frames exactly lumpi_v41.
Val-informed: 4.7 m sits between the median length of Van boxes on Car labels (4.66 m) and on Van labels (5.09 m), from val. One val reading, one fresh reading if val passes; no other threshold, no retry.
Van gate, on val then fresh: up on both halves, half 2 at least +0.003, windows 2 and 3 each up, IoU 0.7 not down. Writes output/lumpi/ens_x/lumpi_vanlen_test.json if both pass.
    python scripts/lumpi_van_len.py logs/fresh_ids.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_static as S
import lumpi_v37_rules as R

R.RULES["VL"] = ("Van", lambda x: x["l"] < 4.7)
keep = R.keep


def gate(r, ref):
    return (r["h1"][0] > ref["h1"][0] and round(1e4 * (r["h2"][0] - ref["h2"][0])) >= 30 and r["windows"][2] > ref["windows"][2]
            and r["windows"][3] > ref["windows"][3] and r["all"][1] >= ref["all"][1])


def read(tag, p, feats, gt, ids, expect=None):
    base = R.apply(p, feats, ["V5"])
    ref = R.row(base, gt, ids, "Van")
    print(tag, "lumpi_v41 Van", json.dumps(ref), "expected", expect, flush=True)
    if expect is not None and ref["all"][0] != expect:
        print("STOP base does not match", flush=True)
        sys.exit(0)
    new = R.apply(p, feats, ["V5", "VL"])
    r = R.row(new, gt, ids, "Van")
    hit = sum(x["l"] < 4.7 and not x["n"] < 15 and p[f][i]["class"] == "Van" for f in ids for i, x in feats[f])
    hi3 = sum(x["l"] < 4.7 and not x["n"] < 15 and p[f][i]["class"] == "Van" and p[f][i]["score"] >= 0.3 for f in ids for i, x in feats[f])
    g = gate(r, ref)
    print(tag, "Van length rule", json.dumps(r), "boxes lowered", hit, "confident", hi3, "PASS" if g else "fail", flush=True)
    return g


if __name__ == "__main__":
    ids = S.ids_of("val")
    gt = S.labels(ids)
    p = {f: [R.rule37(R.grow(keep(d))) for d in v] for f, v in json.loads((R.O / "lumpi_v33_plow_val.json").read_text()).items()}
    p = {f: [d for d in p[f] if d["class"] == "Van"] for f in ids}
    feats = R.features("val", p, ids)
    if not read("val", p, feats, gt, ids, 0.1708):
        print("kept none", flush=True)
        sys.exit(0)
    del p, feats, gt
    fr = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    gf = S.labels(fr)
    a = json.loads((R.L / "v29_fresh_ref.json").read_text())
    pf = {f: [R.rule37(keep(d)) for d in a[f] if d["class"] == "Van"] for f in fr}
    del a
    ff = R.features("fresh", pf, fr)
    if not read("fresh", pf, ff, gf, fr):
        print("kept none", flush=True)
        sys.exit(0)
    t37 = json.loads((R.O / "lumpi_v37_test.json").read_text())
    t41 = json.loads((R.O / "lumpi_v41_test.json").read_text())
    ft = R.features("test", t37, sorted(t37))
    canon = lambda v: sorted(json.dumps(d, sort_keys=True) for d in v if d["class"] == "Van")
    v39 = R.apply(t37, ft, ["V5"])
    assert all(canon(v39[f]) == canon(t41[f]) for f in t41), "v37 test with V5 is not lumpi_v41's Van"
    new = R.apply(t37, ft, ["V5", "VL"])
    out = {f: [d for d in t41[f] if d["class"] != "Van"] + [d for d in new[f] if d["class"] == "Van"] for f in t41}
    (R.O / "lumpi_vanlen_test.json").write_text(json.dumps(out))
    n = sum(a["score"] != b["score"] for f in t41 for a, b in zip(sorted((d for d in v39[f] if d["class"] == "Van"), key=lambda d: d["box"]), sorted((d for d in new[f] if d["class"] == "Van"), key=lambda d: d["box"])))
    print("wrote lumpi_vanlen_test.json, Van boxes lowered on test", n, flush=True)
