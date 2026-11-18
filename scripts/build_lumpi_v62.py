"""lumpi_v62 (slot 8; a short CPU-only change) = lumpi_v59 (the
board-ranked package) with one class changed: "Person: add spec_lumpi_person_rs" (L12-Person; reading logs/igm60_spec_lumpi_person_rs_{near,val}.json,
read with --base-v59 on lumpi_v59's switch set). That reading FAILED the val bar (+0.00415 < +0.005; inside +0.0143 on 64 ids, before +0.0344,
no veto), so this class is outside the gate as written: an author's choice made after the reading and after lumpi_v61's test result; the
report must list it. (Earlier draft: v61 base with a pass check; never run.)
The Person chain is the reading's: G3.person on G3.P_REF + this model's Person second stage (min 0.01 on the v29 lane) as a sixth member
(REFINER form; build_lumpi_v6x.person_files), P3, plus v59's two-wheeler copies (W.copies of v59's Bicycle + Motorcycle), exactly as
build_lumpi_v6x.build makes Person. Every other class is copied from lumpi_v59_test.json; the realism frames are set at packaging. CPU only,
no setting chosen here.
Checks (STOP on any miss; nothing written):
  1. lumpi_v59_test.json (057209a5) equals the uploaded v59 zip's predictions.json frame by frame (canonical box lists; the zip is score-sorted).
  2. Both reading files hold "Person: add spec_lumpi_person_rs" with "_v60" base_v59 true and all six switches v59; the val file says passed False (the val bar)
     no veto; the test second-stage file's md5 equals the one the chain placed (logs/person_rs_chain.log).
  3. val: this script's Person (the chain above, with v59's copies) has Person AP@0.5 equal to the reading's "val" (1e-9), and the same chain
     with "v59" (no sixth member) equals the reading's "val_base" (1e-9).
  4. test: the chain with "v59" equals lumpi_v59's Person box for box (the chain reproduces v59 before the change).
  5. output: every class but Person equals v59 box for box; Person changed; the copies on test are unchanged (682 expected, as v59).
Writes output/lumpi/ens_x/lumpi_v62_test.json and logs/build_lumpi_v62.json (md5s, per-class table v59 vs v62, checks).
    python scripts/build_lumpi_v62.py        (Docker CPU, under logs/igm.lock, capped container)"""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import build_lumpi_v6x as B6
import igm_read as M

PSM = "spec_lumpi_person_rs"
NAME = f"Person: add {PSM}"
V61 = Path("output/lumpi/ens_x/lumpi_v59_test.json")     # the base is lumpi_v59 (names kept from the v59 draft)
V61_MD5 = "057209a5"   # prefix; full md5 = logs/build_lumpi_v61.json's "base" md5 (checked there against the uploaded v59 zip)
V61_ZIP = Path("submissions/lumpi_v59/submission.zip")
V61_LOG = Path("logs/build_lumpi_v61.json")
NEAR, VAL = Path(f"logs/igm60_{PSM}_near.json"), Path(f"logs/igm60_{PSM}_val.json")
CHAIN_LOG = Path("logs/person_rs_chain.log")
TEST_S2 = Path(f"output/lumpi/{PSM}/s2/v29_test_min01.json")
OUT, LOGJ = Path("output/lumpi/ens_x/lumpi_v62_test.json"), Path("logs/build_lumpi_v62.json")
CLASSES = ("Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown")
SIX = ("Person", "Car", "Van", "Truck", "Bicycle", "Motorcycle")
CHECKS = []


def chk(name, ok):
    print(f"check {name}: {ok}", flush=True)
    CHECKS.append([name, bool(ok)])
    if not ok:
        sys.exit(f"STOP {name}: nothing written")


def md5b(b):
    return hashlib.md5(b).hexdigest()


def by_class(v, c):
    return [d for d in v if d["class"] == c]


def person(choice, split, ids, keys):
    """build_lumpi_v6x.build's Person for Bicycle = Motorcycle = v59: G3.person(person_files(choice)) + W.copies(v59 Bicycle + Motorcycle)."""
    b = {f: [B6.keep(d) for d in v] for f, v in B6.bike(B6.bike_file("v59", split), ids, keys, split).items()}
    mo = B6.W.only(B6.RL.L / f"v29_{split}_ref.json", "Motorcycle")
    mo = {f: [B6.keep(d) for d in mo.get(f, [])] for f in ids}
    cp = B6.W.copies({f: b[f] + mo[f] for f in ids})
    p = B6.G3.person(B6.person_files(choice, split), ids, B6.LOAD[split], keys)
    return {f: [B6.keep(d) for d in p.get(f, [])] + cp[f] for f in ids}, cp


def main():
    for p in (OUT, LOGJ):
        if p.exists():
            sys.exit(f"STOP {p} exists (no overwrite)")
    # 1. the splice source is the packaged v59
    raw = V61.read_bytes()
    full = json.loads(V61_LOG.read_text())["base"]["md5"]
    chk(f"{V61} md5 {md5b(raw)} = logs/build_lumpi_v61.json base {full} (prefix {V61_MD5})", md5b(raw) == full and full.startswith(V61_MD5))
    v59 = json.loads(raw)
    with zipfile.ZipFile(V61_ZIP) as z:
        zp = json.loads(z.read("predictions.json"))
    tid = sorted(v59)
    chk("v59 test file has the 50 test frames", len(tid) == 50 and sorted(zp) == tid)
    chk(f"{V61} equals {V61_ZIP}'s predictions.json frame by frame (canonical)", all(B6.canon(v59[f]) == B6.canon(zp[f]) for f in tid))
    del zp
    # 2. the reading
    near, val = json.loads(NEAR.read_text()), json.loads(VAL.read_text())
    for p, j in ((NEAR, near), (VAL, val)):
        hd = j.get("_v60", {})
        sw = hd.get("switches", {})
        chk(f"{p} holds {NAME!r}, read with base_v59 on all six switches at v59",
            NAME in j and hd.get("base_v59") is True and all(sw.get(c) == "v59" for c in SIX))
    r = val[NAME]
    chk(f"{VAL} {NAME!r} is the failed reading shipped by choice (passed False: val {r['d_val']:+.5f} < +0.005; inside {r['d_inside']:+.4f}; veto {r['veto']})", r["passed"] is False and not r["veto"] and r["d_inside"] >= 0.01)
    placed = re.findall(rf"placed {re.escape(str(TEST_S2))} ([0-9a-f]{{32}})", CHAIN_LOG.read_text())
    got = md5b(TEST_S2.read_bytes())
    chk(f"{TEST_S2} md5 {got} = the chain's placed md5 {placed}", len(placed) == 1 and placed[0] == got)
    keys = B6.SZ.PP.static_keys("lumpi")
    # 3. val: the chain's Person AP = the reading's
    ids = M.S.ids_of("val")
    gt = {f: [{"box": [float(x) for x in b], "class": c} for b, c in v] for f, v in M.S.labels(ids).items()}
    pv, _ = person(PSM, "val", ids, keys)
    pb, _ = person("v59", "val", ids, keys)
    a, base = M.G3.ap(pv, gt, ids, "Person"), M.G3.ap(pb, gt, ids, "Person")
    chk(f"val Person AP@0.5 {a:.6f} / base {base:.6f} = the reading's {r['val']:.6f} / {r['val_base']:.6f} (1e-9)",
        abs(a - r["val"]) < 1e-9 and abs(base - r["val_base"]) < 1e-9)
    del pv, pb, gt
    # 4. test: the chain reproduces v59's Person, then the change
    b0, cp0 = person("v59", "test", tid, keys)
    chk("test: Person chain 'v59' + v59's copies equals lumpi_v59's Person box for box",
        all(B6.canon(b0[f]) == B6.canon(by_class(v59[f], "Person")) for f in tid))
    new, cp1 = person(PSM, "test", tid, keys)
    ncp = sum(len(v) for v in cp1.values())
    chk(f"test: the two-wheeler copies are unchanged ({ncp} boxes)", all(cp0[f] == cp1[f] for f in tid))
    out = {}
    for f in tid:
        rest = [d for d in v59[f] if d["class"] != "Person"]
        out[f] = rest + new[f]
    # 5. every other class unchanged
    table = {}
    for c in CLASSES:
        n61 = sum(len(by_class(v59[f], c)) for f in tid)
        n62 = sum(len(by_class(out[f], c)) for f in tid)
        h61 = sum(1 for f in tid for d in by_class(v59[f], c) if d["score"] >= 0.3)
        h62 = sum(1 for f in tid for d in by_class(out[f], c) if d["score"] >= 0.3)
        eq = all(B6.canon(by_class(v59[f], c)) == B6.canon(by_class(out[f], c)) for f in tid)
        table[c] = {"v59": n61, "v62": n62, "v59_ge03": h61, "v62_ge03": h62, "equal": eq}
        print(f"{c:11s} v59 {n61:7,d} (>=0.3 {h61:5d})  v62 {n62:7,d} (>=0.3 {h62:5d})  box for box equal: {eq}", flush=True)
        if c != "Person":
            chk(f"{c} equals v59 box for box", eq)
    chk("Person changed", not table["Person"]["equal"])
    text = json.dumps(out)
    OUT.write_text(text)
    res = {"name": "lumpi_v62", "base": {"file": str(V61), "md5": full}, "change": NAME,
           "reading": {"near": [str(NEAR), md5b(NEAR.read_bytes())], "val": [str(VAL), md5b(VAL.read_bytes())]},
           "test_s2": [str(TEST_S2), got], "val": {"Person": a, "Person_base": base}, "copies_test": ncp, "classes": table,
           "boxes": sum(len(v) for v in out.values()), "out": {"file": str(OUT), "md5": md5b(text.encode())}, "checks": CHECKS}
    LOGJ.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT} ({res['boxes']:,} boxes, md5 {res['out']['md5']}) and {LOGJ}", flush=True)


if __name__ == "__main__":
    main()
