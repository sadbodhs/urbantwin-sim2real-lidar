"""L11-RS reading. Mac CPU, one try, fixed before any reading.
The block-1 LUMPI seed route, every class, like for like on lumpi_v53, with two differences fixed in advance:
  - Fresh = realism_check only (160 frames; the size statistics come from realism_fit). Fresh references (same chains without L11-RS) are computed
    on those frames first (logs/lumpi_member_refs_chk.json, logs/route_refs_lumpi_chk.json); val references are the logged ones.
  - One pre-set extra arm for Van only (A5 style): L11-RS's own Van boxes (its 4-view TTA kit predictions, one vote per view) as the only Van
    source, through lumpi_v41's Van chain (v37 rule, T5 / V5 with the v29 Trucks). The arm with the higher val Van AP of the two (route lane vs
    own) is the Van candidate: Nov 8 Van gate on val, then fresh once. The other arm is not read on fresh.
Person: scripts/lumpi_member_read.py's reading (L11-RS's second stage joins lumpi_v51's five), with the realism_check references.
Car, Truck, Bicycle (and Van by the route arm): scripts/route_read_lumpi.py's reading with model tag zQR (block-1 refiner seed: Person route lane skipped).
Van gate on realism_check: R.row's Van windows are 80 frames, which gives only 2 windows on 160 frames; here they are 40 frames (4 windows, 2 per
half), so the Nov 8 gate's "windows 2 and 3 up" keeps its form (flagged; val unchanged, 320 frames, 80-frame windows).
Every realism_check id goes through the forbidden-frame guard. A pass writes output/lumpi/ens_x/route_<model>_test.json (route classes, on the
member file or lumpi_v53) and, if the own Van arm passes, output/lumpi/ens_x/l11rs_<model>_test.json (that file, or the base, with Van from the own arm).
Verdict (logs/l11rs_verdict_<model>.json): the classes that passed val and fresh (Person by the member reading, the route classes, Van by the
own arm). "passed" empty = no class passed (Car, Van by both arms, Truck, Bicycle and Person each failed at val or fresh): for (c) that triggers
option (b) (scripts/l11rs_wait.sh, scripts/l11rs_b_insert.py). Models: vrcnn_lumpi_rs (c), vrcnn_lumpi_rs_b (b), vrcnn_lumpi_g / vrcnn_lumpi_gb
(L11-G on (c) / (b); each class also printed against its L11-RS for information).
    python scripts/l11rs_read.py [vrcnn_lumpi_rs | vrcnn_lumpi_rs_b | vrcnn_lumpi_g | vrcnn_lumpi_gb]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_member_read as LM
import lumpi_static as S
import lumpi_v37_rules as R
import route_read_lumpi as RR
from utl import guard

M = sys.argv[1] if len(sys.argv) > 1 else "vrcnn_lumpi_rs"
TAGS = {"vrcnn_lumpi_rs_wa": "zQW", "vrcnn_lumpi_rs": "zQR", "vrcnn_lumpi_rs_b": "zQB", "vrcnn_lumpi_g": "zQG", "vrcnn_lumpi_gb": "zQH"}   # route_lanes_lumpi.sh tags
RR.TAG[M] = TAGS[M]
RR.REFINER.add(M)
# L11-G is comparison only, never shipped.
# For vrcnn_lumpi_g / vrcnn_lumpi_gb every test file this reading would write (route_/lumpi_/l11rs_<model>_test.json, from this script and the
# member / route readers it calls) goes to output/lumpi/comparison_only/ instead of output/lumpi/ens_x/, so no builder can pick it up; val and
# fresh readings are unchanged. The verdict file says "comparison_only": true.
COMP = M in ("vrcnn_lumpi_g", "vrcnn_lumpi_gb")
if COMP:
    _QD = Path("output/lumpi/comparison_only")
    _QD.mkdir(parents=True, exist_ok=True)
    _wt = Path.write_text

    def _comp_write(self, data, *a, **k):
        if self.name.endswith("_test.json") and M in self.name:
            self = _QD / self.name
            print("comparison only, not for submission: test file written to", self, flush=True)
        return _wt(self, data, *a, **k)

    Path.write_text = _comp_write
    print(f"{M}: comparison only, not for submission; test files go to {_QD}", flush=True)
INFO = {"vrcnn_lumpi_rs_wa": "vrcnn_lumpi_rs", "vrcnn_lumpi_g": "vrcnn_lumpi_rs", "vrcnn_lumpi_gb": "vrcnn_lumpi_rs_b"}.get(M)   # L11-G: each class vs its L11-RS, information
O = RR.O
CHK = sorted(l.strip() for l in open("data/pcdet/lumpi_real/ImageSets/realism_check.txt") if l.strip())
VAN_W_CHK = 40
_row = R.row


def row(p, gt, ids, c):
    r = _row(p, gt, ids, c)
    if c == "Van" and len(ids) < 320:   # realism_check: 4 windows of 40 frames (flagged), see the docstring
        only = {f: [d for d in p[f] if d["class"] == c] for f in ids}
        r["windows"] = [S.ap(only, gt, x, [c])[0.5][c] for x in [sorted(ids)[i:i + VAN_W_CHK] for i in range(0, len(ids), VAN_W_CHK)]]
    return r


R.row = row


def own_van(split, ids):
    """L11-RS's own Van boxes as the only Van source, through lumpi_v41's Van chain (as scripts/lumpi_van_combos.py van())."""
    src = {"val": "output/lumpi/route_in", "test": "output/lumpi/route_in", "fresh": "fresh_ws/output/lumpi/route_in"}[split]
    raw = json.loads(Path(f"{src}/{M}_{'test' if split == 'test' else 'val'}.json").read_text())
    va = {f: [R.rule37(RR.keep(d)) for d in raw.get(f, []) if d["class"] == "Van"] for f in ids}
    tr = RR.Chain(split, ids).pre("Truck", False)
    tv = {f: tr[f] + va[f] for f in ids}
    tv = R.apply(tv, R.features(split, tv, ids), ["T5", "V5"])
    return {f: [d for d in tv[f] if d["class"] == "Van"] for f in ids}


def refs_chk():
    for i in CHK:
        guard.check("lumpi", i)
    gt = S.labels(CHK)
    lm = json.loads(LM.REFS.read_text())
    out = {"val": lm["val"], "fresh": {k: LM.row("fresh", CHK, gt, k, None) for k in LM.RULES}}
    for k in LM.RULES:
        print("reference (Person, lumpi_v51's five) realism_check", k, json.dumps(out["fresh"][k]), flush=True)
    LM.REFS = Path("logs/lumpi_member_refs_chk.json")
    LM.REFS.write_text(json.dumps(out, indent=1))
    rr = json.loads(RR.REFJ.read_text())
    ch = RR.Chain("fresh", CHK)
    fresh = {}
    for c in ("Car", "Truck", "Bicycle", "Van"):
        for rule in RR.RULES[c]:
            p = ch.cls(c, rule)
            r = R.row(p, gt, CHK, c)
            if c == "Bicycle":
                r["labels_only"] = RR.thin(p, gt, CHK)
            fresh.setdefault(c, {})[rule] = r
            print("reference (no new model) realism_check", c, rule, json.dumps(r), flush=True)
    RR.REFJ = Path("logs/route_refs_lumpi_chk.json")
    RR.REFJ.write_text(RR.dumps({"val": rr["val"], "fresh": fresh}))
    print("wrote", LM.REFS, RR.REFJ, flush=True)
    return out, {"val": rr["val"], "fresh": fresh}


def main():
    _, ref = refs_chk()
    print("== Person (member reading, fresh = realism_check)", flush=True)
    LM.read(M, CHK)
    print("== Van arms on val", flush=True)
    ids = S.ids_of("val")
    gt = S.labels(ids)
    rv = {"route": R.row(RR.Chain("val", ids, M).cls("Van"), gt, ids, "Van"), "own": R.row(own_van("val", ids), gt, ids, "Van")}
    for a, r in rv.items():
        print("val Van arm", a, json.dumps(r), "vs lumpi_v53 Van %+.4f" % (r["all"][0] - ref["val"]["Van"]["R0"]["all"][0]), flush=True)
    best = max(("route", "own"), key=lambda a: (rv[a]["all"][0], a == "route"))
    print("Van candidate (higher val AP):", best, flush=True)
    res = {"van_val": rv, "van_arm": best}
    if best == "own":
        RR.ONLY[M] = ["Car", "Truck", "Bicycle"]   # the route reading without Van; Van by the own arm below
    print("== route reading (fresh = realism_check)", flush=True)
    RR.read(CHK, M)
    if best == "own":
        g = RR.van_gate(rv["own"], ref["val"]["Van"]["R0"])
        print("val Van own arm | Nov 8 Van gate", "PASS" if g else "fail", flush=True)
        res["van_val_gate"] = g
        if g:
            gf = S.labels(CHK)
            a = R.row(own_van("fresh", CHK), gf, CHK, "Van")
            b = ref["fresh"]["Van"]["R0"]
            g2 = RR.van_gate(a, b)
            print("fresh (realism_check) Van own arm", json.dumps(a), "vs lumpi_v53 Van", json.dumps(b), "%+.4f" % (a["all"][0] - b["all"][0]),
                  "| gate", "PASS" if g2 else "fail", flush=True)
            res.update(van_fresh=a, van_fresh_ref=b, van_fresh_gate=g2)
            if g2:
                rf = O / f"route_{M}_test.json"
                own = O / f"lumpi_{M}_test.json"
                base = rf if rf.exists() else own if own.exists() else O / "lumpi_bicycle_r4_test.json"
                t = json.loads(base.read_text())
                tid = sorted(t)
                v = own_van("test", tid)
                out = {f: sorted([RR.keep(d) for d in t[f] if d["class"] != "Van"] + [RR.keep(d) for d in v.get(f, [])], key=lambda d: -d["score"]) for f in tid}
                canon = lambda x: sorted(json.dumps([round(u, 5) for u in d["box"]] + [round(d["score"], 6), d["class"]]) for d in x)
                ok = all(canon([d for d in out[f] if d["class"] != "Van"]) == canon([RR.keep(d) for d in t[f] if d["class"] != "Van"]) for f in tid)
                print("test: classes other than Van equal", base.name, ":", ok, flush=True)
                if not ok:
                    sys.exit("STOP classes other than Van changed")
                (O / f"l11rs_{M}_test.json").write_text(json.dumps(out))
                print(f"wrote l11rs_{M}_test.json:", base.name, "with Van from the model's own lane", flush=True)
    passed = []
    lm = json.loads(Path(f"logs/lumpi_member_{M}.json").read_text())
    if lm.get("val_gate") and lm.get("fresh_gate"):
        passed.append("Person")
    rr = json.loads(Path(f"logs/route_{M}_read.json").read_text())
    passed += [c for c in rr.get("file_classes", {}) if c not in passed]
    if res.get("van_fresh_gate"):
        passed.append("Van")
    res["passed"] = passed
    print("classes passed (val and fresh):", passed or "none", flush=True)
    if INFO and Path(f"logs/route_{INFO}_read.json").exists():
        o = json.loads(Path(f"logs/route_{INFO}_read.json").read_text())
        for c, e in rr["val"].items():
            if c in o.get("val", {}):
                print("information: val", c, "R0", M, e["rows"]["R0"]["all"], "|", INFO, o["val"][c]["rows"]["R0"]["all"], flush=True)
        if Path(f"logs/lumpi_member_{INFO}.json").exists():
            om = json.loads(Path(f"logs/lumpi_member_{INFO}.json").read_text())
            print("information: val Person", lm["chosen"], M, lm["val"]["new"][lm["chosen"]]["all"], "|", INFO, om["chosen"], om["val"]["new"][om["chosen"]]["all"], flush=True)
        if Path(f"logs/l11rs_read_{INFO}.json").exists():
            ov = json.loads(Path(f"logs/l11rs_read_{INFO}.json").read_text())
            print("information: val Van arms", {a: r["all"] for a, r in rv.items()}, "|", INFO, {a: r["all"] for a, r in ov["van_val"].items()}, flush=True)
    Path(f"logs/l11rs_read_{M}.json").write_text(RR.dumps(res))
    Path(f"logs/l11rs_verdict_{M}.json").write_text(json.dumps({"model": M, "passed": passed, **({"comparison_only": True, "note": "not for submission"} if COMP else {})}))


if __name__ == "__main__":
    main()
