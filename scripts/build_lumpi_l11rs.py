"""Task A: the next LUMPI zip from the L11-RS (c) verdict.
Mac CPU, no new readings or settings. Base lumpi_v54 (= output/lumpi/ens_x/lumpi_p3_both_test.json, the LUMPI main line). Each class that passed
its gate in scripts/l11rs_read.py (logs/l11rs_verdict_vrcnn_lumpi_rs.json; the usual class gate, Van by the Nov 8 Van gate) is replaced by exactly
the configuration its verdict read; every other class stays v54 box for box (the synthetic frames are checked after packaging).
Swappable: only classes whose reading was made on v54's own chain for that class. v54 = lumpi_v53 + P3 on Person and Bicycle, so
  Car, Truck, Van: v54's chain = lumpi_v53's (lumpi_v41 rules), the chain the reading used -> swapped;
  Person, Bicycle: read without P3 (lumpi_v51's five / Bicycle R4 on lumpi_v53) -> NOT swapped, logged only.
Sources: Car / Truck / Van (route arm) = output/lumpi/ens_x/route_vrcnn_lumpi_rs_test.json; Van (own arm) = l11rs_vrcnn_lumpi_rs_test.json.
Checks (STOP on any miss): each swapped class rebuilt on val (320) and fresh (realism_check, 160) equals its logged reading (AP@0.5 and AP@0.7 R40);
on test equals its source file and its chain rebuilt, box for box; every other class equals v54 box for box.
Writes output/lumpi/ens_x/lumpi_l11rs_next_test.json and logs/build_lumpi_l11rs.json (classes, readings, references); nothing when nothing swaps.
    python scripts/build_lumpi_l11rs.py vrcnn_lumpi_rs"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import l11rs_read as LR   # sets route_read_lumpi's tag / refiner entries for the model (argv[1]) and the 40-frame Van windows on realism_check
import lumpi_static as S
import lumpi_v37_rules as R
import route_read_lumpi as RR
from utl import guard

M = LR.M
O = RR.O
SWAP = ("Car", "Truck", "Van")
MC = M   # the model whose lanes the chains read (a dry test points it at an older model)
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}
canon = lambda v: sorted(json.dumps([round(x, 5) for x in d["box"]] + [round(d["score"], 6), d["class"]]) for d in v)
OUT = O / "lumpi_l11rs_next_test.json"
LOG = Path("logs/build_lumpi_l11rs.json")
VERDICT, READ, ROUTE = Path(f"logs/l11rs_verdict_{M}.json"), Path(f"logs/l11rs_read_{M}.json"), Path(f"logs/route_{M}_read.json")
REFF = Path("logs/route_refs_lumpi_chk.json")
SRC = {"own": O / f"l11rs_{M}_test.json", "route": O / f"route_{M}_test.json"}


def main():
    passed = json.loads(VERDICT.read_text())["passed"]
    rd = json.loads(READ.read_text())
    rr = json.loads(ROUTE.read_text())
    print("verdict, classes passed (val and fresh):", passed or "none", flush=True)
    res = {"model": M, "passed": passed, "swapped": {}, "not_swapped": {}}
    for c in passed:
        if c not in SWAP:
            res["not_swapped"][c] = "reading made without P3 (v54 has P3 on Person and Bicycle): not on v54's own chain for this class"
            print("NOT SWAPPED", c, ":", res["not_swapped"][c], flush=True)
    swap = [c for c in passed if c in SWAP]
    if not swap:
        res["built"] = False
        LOG.write_text(json.dumps(res, indent=1))
        print("nothing to swap: no file built", flush=True)
        return
    for i in LR.CHK:
        guard.check("lumpi", i)
    vid = S.ids_of("val")
    sets = {"val": (vid, S.labels(vid)), "fresh": (LR.CHK, S.labels(LR.CHK))}
    refv = json.loads(RR.REFJ.read_text())["val"]
    reff = json.loads(REFF.read_text())["fresh"]
    v54 = json.loads((O / "lumpi_p3_both_test.json").read_text())
    tid = sorted(v54)
    ok = True
    src = {}
    for c in swap:
        own = c == "Van" and rd.get("van_arm") == "own"
        if own:
            want = {"val": rd["van_val"]["own"], "fresh": rd["van_fresh"]}
            build = lambda s, ids: LR.own_van(s, ids)
            f = SRC["own"]
            k = "R0"
        else:
            k = rr["file_classes"][c]
            want = {"val": rr["val"][c]["rows"][k], "fresh": rr["fresh"][c]["rows"][k]}
            build = lambda s, ids, c=c, k=k: RR.Chain(s, ids, MC).cls(c, k)
            f = SRC["route"]
        arm = "own lane (L11-RS's 4-view kit boxes, one vote per view) through lumpi_v41's Van chain" if own else f"route lane, rule {k}"
        e = {"arm": arm, "source": f.name}
        for s, (ids, gt) in sets.items():
            r = R.row(build(s, ids), gt, ids, c)
            same = r["all"] == want[s]["all"]
            ok &= same
            ref = (refv if s == "val" else reff)[c]["R0"]
            e[s] = {"new": r, "ref_v53_v54": ref}
            print(f"{s} {c} ({arm}): rebuilt {r['all']} vs logged reading {want[s]['all']} | equal: {same} | v54 (= v53) {c} {ref['all']} "
                  f"({r['all'][0] - ref['all'][0]:+.4f}; halves {r['h1'][0] - ref['h1'][0]:+.4f} / {r['h2'][0] - ref['h2'][0]:+.4f})", flush=True)
        t = json.loads(f.read_text())
        assert sorted(t) == tid, "test frames differ"
        tb = build("test", tid)
        same = all(canon([d for d in t[x] if d["class"] == c]) == canon([keep(d) for d in tb.get(x, [])]) for x in tid)
        print(f"test {c}: {f.name} equals its chain rebuilt, box for box: {same}", flush=True)
        ok &= same
        src[c] = {x: [keep(d) for d in t[x] if d["class"] == c] for x in tid}
        res["swapped"][c] = e
    out = {x: sorted([keep(d) for d in v54[x] if d["class"] not in src] + [d for c in src for d in src[c][x]], key=lambda d: -d["score"]) for x in tid}
    for c in sorted({d["class"] for v in v54.values() for d in v} | set(src)):
        b = src[c] if c in src else {x: [keep(d) for d in v54[x] if d["class"] == c] for x in tid}
        same = all(canon([d for d in out[x] if d["class"] == c]) == canon(b[x]) for x in tid)
        print(f"test {c}: equals {'its source' if c in src else 'lumpi_v54'} box for box: {same}", flush=True)
        ok &= same
    n0, n1 = sum(map(len, v54.values())), sum(map(len, out.values()))
    print(f"boxes v54 {n0} -> new {n1}", flush=True)
    res.update(boxes=[n0, n1])
    if not ok:
        res["built"] = False
        LOG.write_text(json.dumps(res, indent=1, default=str))
        sys.exit("STOP a class does not equal its source")
    OUT.write_text(json.dumps(out))
    res["built"] = True
    LOG.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", OUT, "| swapped", list(src), flush=True)


if __name__ == "__main__":
    main()
