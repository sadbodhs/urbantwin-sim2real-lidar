"""M2, lumpi_v57 = lumpi_v54 + two-wheeler copies. FLAGGED,
no upload from this script. Base lumpi_v54, not v55: v54 is the LUMPI main line
(v55's v47 Car fell on test, 0.6558 -> 0.6535, test-informed). CPU only, no new settings: v46's rule exactly (scripts/lumpi_copies_routes.copies): every Bicycle or
Motorcycle box of the file with score >= 0.3 also becomes a Person box, same box, score x 0.3, no size change, no length rule. The copies are taken
from v54's own (final, P3-filtered) Bicycle and its Motorcycle boxes and added after P3, so P3 does not touch them. Every other class equals v54
box for box; Person = v54's Person plus the copies, nothing else. v54 and v55 differ only in Car, so their Person, Bicycle and
Motorcycle (and the copies) are the same.
Readings (information, one each): Person on val (320) and realism_check (160) for v54 and v57, from scripts/m1_late_person.py's rebuilt v54/v55 Person
(base) and base + copies (logs/m1_late_person.json, "whole"); since every other class is identical box for box, whole-file mAP over the 7 classes
moves by exactly the Person change / 7.
Writes output/lumpi/ens_x/lumpi_v57_test.json.
    python scripts/build_lumpi_v57.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_copies_routes as W

O = Path("output/lumpi/ens_x")
keep = lambda d: {k: d[k] for k in ("box", "score", "class")}
canon = lambda v: sorted(json.dumps([round(x, 6) for x in d["box"]] + [round(d["score"], 8), d["class"]]) for d in v)


def main():
    v55 = json.loads((O / "lumpi_p3_both_test.json").read_text())   # lumpi_v54 (name kept from the first version)
    tw = {f: [keep(d) for d in v if d["class"] in W.TW] for f, v in v55.items()}
    cp = W.copies(tw)
    n = sum(map(len, cp.values()))
    v57 = {f: [keep(d) for d in v] + cp[f] for f, v in v55.items()}
    ok = set(v57) == set(v55)
    classes = sorted({d["class"] for v in v55.values() for d in v})
    for c in classes:
        if c == "Person":
            same = all(canon([d for d in v57[f] if d["class"] == c]) == canon([d for d in v55[f] if d["class"] == c] + cp[f]) for f in v55)
            print(f"test Person = v54 Person + {n} copies, box for box: {same}", flush=True)
        else:
            same = all(canon([d for d in v57[f] if d["class"] == c]) == canon([d for d in v55[f] if d["class"] == c]) for f in v55)
            print(f"test {c} equals v54 box for box: {same}", flush=True)
        ok &= same
    src = sum(1 for v in tw.values() for d in v if d["score"] >= 0.3)
    ok &= src == n and all(d["score"] < 0.3 * 1.0000001 for v in cp.values() for d in v)
    print(f"two-wheelers >= 0.3 on test: {src}, copies {n} ({n / len(v55):.1f} per frame); boxes v54 {sum(map(len, v55.values()))} -> v57 "
          f"{sum(map(len, v57.values()))}", flush=True)
    if not ok:
        sys.exit("STOP a class does not equal its source")
    m1 = json.loads(Path("logs/m1_late_person.json").read_text())["whole"]
    for s, name in (("val", "val 320"), ("fresh", "realism_check 160")):
        b, c = m1["base"][s], m1["copies_on"][s]
        print(f"{name}: Person v54 {b[0]} / IoU 0.7 {b[1]} -> v57 {c[0]} / {c[1]} ({c[0] - b[0]:+.4f}); whole-file mAP change {(c[0] - b[0]) / 7:+.4f}",
              flush=True)
    (O / "lumpi_v57_test.json").write_text(json.dumps(v57))
    print("wrote", O / "lumpi_v57_test.json", flush=True)


if __name__ == "__main__":
    main()
