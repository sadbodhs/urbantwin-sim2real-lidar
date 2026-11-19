"""v2x_v35 (slot 8, a short CPU-only change): v2x_v34 with the pedestrian
class from the DSVT-Pillar joint lane's cached second stages (output/v2x/dsvt_v2x_j/s2/pedestrian_{x6b,neg}_<s>_ref.json, R6, P3), then P-Z
(z + 0.03 m). The DSVT-Pillar pedestrian reading (logs/dsvt_v2x_j_read.json, vs v2x_v31) FAILED the gate (val_i1 h2 +0.0011, AP@0.7 -0.0001), so this
class is outside the gate as written, our own choice made after that reading; the report must list it. Vehicle and truck stay v34's box for box
(no truck option is built here). CPU only; nothing is tuned here.
Checks (STOP on any miss; nothing written):
  1. v2x_v34_test.json md5 = 55a20719 (= predictions.json of the uploaded v2x_v34d zip); P3 static map has 16,542 voxels.
  2. val_i1 / fresh: the unshifted chain's pedestrian rows (all / h1 / h2, AP@0.5 and AP@0.7, 4 dp) equal dsvt_v2x_j_read.json's pedestrian "new" rows.
  3. val_i1 / fresh: v34's pedestrian (v31's chain + P-Z) rebuilt here equals v2x_pz_read.json's pedestrian "new" rows (the comparison base).
  4. test: v34's pedestrian rebuilt (B.ped + pz_shift) equals v2x_v34's pedestrian box for box.
  5. output: vehicle and truck equal v34 box for box; pedestrian changed.
Prints (information, the reading of this exact package's pedestrian vs v34's): the shifted chain's rows on val_i1 / fresh against v34's, with the
standard gate's conditions (val_i1 +0.003 on all / h1 / h2 and AP@0.7 not down; fresh +0.003 on all / h2 and AP@0.7 not down).
Writes output/v2x/ens/v2x_v35_test.json and logs/build_v2x_v35.json (rows, checks, md5s, per-class box table vs v34).
    python scripts/build_v2x_v35.py        (Docker CPU)"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import build_v2x_v32 as V

V34 = V.P.E / "v2x_v34_test.json"
V34_MD5 = "55a2071985af722bf8154e7a10b2809c"
OUT, LOGJ = V.P.E / "v2x_v35_test.json", Path("logs/build_v2x_v35.json")
DJ = V.DJ
CHECKS = []


def chk(name, ok):
    print(f"check {name}: {ok}", flush=True)
    CHECKS.append([name, bool(ok)])
    if not ok:
        sys.exit(f"STOP {name}: nothing written")


def r4(row):
    return {h: [round(float(x), 4) for x in row[h]] for h in ("all", "h1", "h2")}


def dsvt_ped(s, keys, dz):
    p = V.ped_r6([DJ / f"pedestrian_x6b_{s}_ref.json", DJ / f"pedestrian_neg_{s}_ref.json"], s, keys)
    return V.pz_shift(p, dz) if dz else p


def gate(n, b, fresh):
    hs = ("all", "h2") if fresh else ("all", "h1", "h2")
    fl = [f"{'f' if fresh else 'v'}-{h}" for h in hs if round(1e4 * (n[h][0] - b[h][0])) < 30]
    if n["all"][1] < b["all"][1]:
        fl.append(f"{'f' if fresh else 'v'}-0.7")
    return fl


def main():
    for p in (OUT, LOGJ):
        if p.exists():
            sys.exit(f"STOP {p} exists (no overwrite)")
    raw = V34.read_bytes()
    chk(f"{V34} md5 {hashlib.md5(raw).hexdigest()} = {V34_MD5}", hashlib.md5(raw).hexdigest() == V34_MD5)
    keys = V.PP.static_keys("v2x")
    chk(f"P3 static map voxels {len(keys)} = {V.V31_KEYS}", len(keys) == V.V31_KEYS)
    dr = json.loads(Path("logs/dsvt_v2x_j_read.json").read_text())["pedestrian"]["new"]
    pzr = json.loads(Path("logs/v2x_pz_read.json").read_text())["pedestrian"]["new"]
    rows = {}
    fails = []
    for s in ("val_i1", "fresh"):
        r0 = r4(V.score("pedestrian", s, dsvt_ped(s, keys, 0.0)))
        chk(f"{s}: unshifted dsvt_j pedestrian rows = dsvt_v2x_j_read.json's new rows", r0 == r4(dr[s]))
        b = r4(V.score("pedestrian", s, V.pz_shift(V.B.ped(s, keys), V.PZ["dz"])))
        chk(f"{s}: v34's pedestrian (v31 chain + P-Z) rows = v2x_pz_read.json's new rows", b == r4(pzr[s]))
        n = r4(V.score("pedestrian", s, dsvt_ped(s, keys, V.PZ["dz"])))
        fl = gate(n, b, s == "fresh")
        fails += fl
        rows[s] = {"new": n, "v34": b, "fails": fl}
        d = {h: round(n[h][0] - b[h][0], 4) for h in ("all", "h1", "h2")}
        print(f"{s} pedestrian dsvt_j + P-Z vs v34: all {b['all'][0]:.4f} -> {n['all'][0]:.4f} ({d['all']:+.4f}; h1 {d['h1']:+.4f}, h2 {d['h2']:+.4f}; "
              f"AP@0.7 {b['all'][1]:.4f} -> {n['all'][1]:.4f}) | standard-gate conditions missed: {fl or 'none'}", flush=True)
    print(f"standard gate on this change vs v34 (information; the class is outside the gate by choice): {'PASS' if not fails else 'FAIL ' + str(fails)}", flush=True)
    v34 = json.loads(raw)
    b34 = V.pz_shift(V.B.ped("test", keys), V.PZ["dz"])
    chk("test: v34's pedestrian rebuilt (v31 chain + P-Z) equals v2x_v34's box for box",
        set(b34) == set(v34) and all(V.B.canon([d for d in b34[f] if d["class"] == "pedestrian"]) == V.B.canon([d for d in v34[f] if d["class"] == "pedestrian"]) for f in v34))
    newp = dsvt_ped("test", keys, V.PZ["dz"])
    chk("test frames equal", set(newp) == set(v34))
    out = {f: sorted([V.B.keep(d) for d in v34[f] if d["class"] != "pedestrian"] + [V.B.keep(d) for d in newp[f] if d["class"] == "pedestrian"],
                     key=lambda d: -d["score"]) for f in v34}
    table = {}
    for c in V.CL:
        eq = all(V.B.canon([d for d in out[f] if d["class"] == c]) == V.B.canon([d for d in v34[f] if d["class"] == c]) for f in v34)
        n35 = sum(d["class"] == c for x in out.values() for d in x)
        n34 = sum(d["class"] == c for x in v34.values() for d in x)
        h35 = sum(d["class"] == c and d["score"] >= 0.3 for x in out.values() for d in x)
        h34 = sum(d["class"] == c and d["score"] >= 0.3 for x in v34.values() for d in x)
        table[c] = {"v34": n34, "v35": n35, "v34_ge03": h34, "v35_ge03": h35, "equal": eq}
        print(f"{c:10s} v34 {n34:7,d} (>=0.3 {h34:5d})  v35 {n35:7,d} (>=0.3 {h35:5d})  box for box equal: {eq}", flush=True)
        if c != "pedestrian":
            chk(f"{c} equals v34 box for box", eq)
    chk("pedestrian changed", not table["pedestrian"]["equal"])
    text = json.dumps(out)
    OUT.write_text(text)
    res = {"name": "v2x_v35", "base": {"file": str(V34), "md5": V34_MD5}, "change": "pedestrian: dsvt_v2x_j second stages (R6, P3) + P-Z 0.03 m",
           "outside_gate": "dsvt_v2x_j pedestrian failed its reading (val_i1 h2, AP@0.7); shipped by the author's choice (slot 8)",
           "rows": rows, "classes": table, "boxes": sum(len(v) for v in out.values()), "out": {"file": str(OUT), "md5": hashlib.md5(text.encode()).hexdigest()},
           "checks": CHECKS}
    LOGJ.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT} ({res['boxes']:,} boxes, md5 {res['out']['md5']}) and {LOGJ}", flush=True)


if __name__ == "__main__":
    main()
