"""lumpi_v59 = lumpi_v58 + lumpi_v57's two-wheeler -> Person copies, built to upload instead of v58 as a manual override of
M3's fixed gate on the test-window half (near-test Person AP@0.5 before -0.0182, inside +0.1000, both +0.0714).
FLAGGED, no upload from this script, no P3-free twin.
  Car: exactly lumpi_v58's (L11-RS route lane). Person: lumpi_v58's Person (= lumpi_v54's, after P3) plus the copies by v57's rule exactly
  (lumpi_copies_routes.copies: every Bicycle or Motorcycle box with score >= 0.3 also becomes a Person box, same box, score x 0.3, added after P3).
  Every other class exactly lumpi_v58's. The realism frames are v58's (checked after packaging).
Checks (STOP on any miss): Car equals v58's box for box; Person equals v57's Person box for box (v57 = v54 + the same copies; v54's and v58's
Person, Bicycle and Motorcycle are identical, checked); every other class equals v58's box for box; copies count = two-wheelers >= 0.3.
Readings carried by construction (each class equals its source file): Car = v58's (logs/build_lumpi_l11rs.json), Person = v57's
(logs/m1_late_person.json "whole", copies_on).
Writes output/lumpi/ens_x/lumpi_v59_test.json and logs/build_lumpi_v59.json.
    python scripts/build_lumpi_v59.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_copies_routes as W

O = Path("output/lumpi/ens_x")
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}
canon = lambda v: sorted(json.dumps([round(x, 6) for x in d["box"]] + [round(d["score"], 8), d["class"]]) for d in v)
ok = True


def chk(name, c):
    global ok
    ok &= bool(c)
    print(f"{name}: {bool(c)}", flush=True)


def cls(p, f, c):
    return [d for d in p[f] if d["class"] == c]


def main():
    v58 = json.loads((O / "lumpi_v58_test.json").read_text())
    v57 = json.loads((O / "lumpi_v57_test.json").read_text())
    v54 = json.loads((O / "lumpi_p3_both_test.json").read_text())
    chk("same 50 test frames in v54, v57, v58", set(v58) == set(v57) == set(v54) and len(v58) == 50)
    for c in ("Person", "Bicycle", "Motorcycle"):
        chk(f"v58 {c} equals v54 {c} box for box", all(canon(cls(v58, f, c)) == canon(cls(v54, f, c)) for f in v58))
    tw = {f: [keep(d) for d in v if d["class"] in W.TW] for f, v in v58.items()}
    cp = W.copies(tw)
    n = sum(map(len, cp.values()))
    src = sum(1 for v in tw.values() for d in v if d["score"] >= 0.3)
    chk(f"copies {n} = two-wheelers >= 0.3 ({src}), every copy score < 0.3", n == src and all(d["score"] < 0.3 * 1.0000001 for v in cp.values() for d in v))
    v59 = {f: [keep(d) for d in v] + cp[f] for f, v in v58.items()}
    classes = sorted({d["class"] for v in v58.values() for d in v} | {"Person"})
    for c in classes:
        if c == "Person":
            chk("Person equals lumpi_v57's Person box for box (v54 Person + the copies)", all(canon(cls(v59, f, c)) == canon(cls(v57, f, c)) for f in v58))
        else:
            chk(f"{c} equals lumpi_v58's box for box", all(canon(cls(v59, f, c)) == canon(cls(v58, f, c)) for f in v58))
    chk("Car differs from v54's (v58's L11-RS Car kept)", any(canon(cls(v59, f, "Car")) != canon(cls(v54, f, "Car")) for f in v58))
    nb = (sum(map(len, v58.values())), sum(map(len, v59.values())))
    print(f"boxes v58 {nb[0]:,} -> v59 {nb[1]:,} (+{n} copies, {n / 50:.1f} per frame)", flush=True)
    if not ok:
        sys.exit("STOP a class does not equal its source")
    b = json.loads(Path("logs/build_lumpi_l11rs.json").read_text())["swapped"]["Car"]
    m1 = json.loads(Path("logs/m1_late_person.json").read_text())["whole"]
    read = {"Car": {"val": b["val"]["new"]["all"], "fresh": b["fresh"]["new"]["all"], "val_v54": b["val"]["ref_v53_v54"]["all"],
                    "fresh_v54": b["fresh"]["ref_v53_v54"]["all"]},
            "Person": {"val": m1["copies_on"]["val"], "fresh": m1["copies_on"]["fresh"], "val_v54": m1["base"]["val"], "fresh_v54": m1["base"]["fresh"]}}
    for c, r in read.items():
        print(f"{c} (carried): val {r['val_v54'][0]} -> {r['val'][0]}, realism_check 160 {r['fresh_v54'][0]} -> {r['fresh'][0]}", flush=True)
    (O / "lumpi_v59_test.json").write_text(json.dumps(v59))
    Path("logs/build_lumpi_v59.json").write_text(json.dumps({"copies": n, "boxes": nb, "readings": read}, indent=1))
    print("wrote", O / "lumpi_v59_test.json", flush=True)


if __name__ == "__main__":
    main()
