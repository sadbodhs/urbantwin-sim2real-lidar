"""lumpi_v51 with only Bicycle scored by combining rule R4: L11b's second stage on v51's Bicycle boxes
(>= 0.05), score s^0.29 x q^0.71 (scripts/ensemble_rules.py, a = 0.71 as fixed in the study spec), box as now. No new reading: first checks that
Bicycle rebuilds the logged readings (logs/route_refs_lumpi.json: val 0.5469, halves 0.6356 / 0.4742, labels-only 0.1879; fresh 0.5476,
0.6422 / 0.4720, labels-only 0.1877) and that the R0 chain on test equals v51's Bicycle. Writes output/lumpi/ens_x/lumpi_bicycle_r4_test.json.
    python scripts/build_lumpi_bicycle_r4.py logs/fresh_ids.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_static as S
import lumpi_v37_rules as R
import route_read_lumpi as RL

O = Path("output/lumpi/ens_x")
EXPECT = {"val": (0.5469, 0.6356, 0.4742, 0.1879), "fresh": (0.5476, 0.6422, 0.4720, 0.1877)}
fr = [l.strip() for l in open(sys.argv[1]) if l.strip()]
for s, ids in (("val", S.ids_of("val")), ("fresh", fr)):
    gt = S.labels(ids)
    p = RL.Chain(s, ids).cls("Bicycle", "R4")
    r = R.row(p, gt, ids, "Bicycle")
    got = (r["all"][0], r["h1"][0], r["h2"][0], RL.thin(p, gt, ids))
    print(s, "Bicycle R4", json.dumps(r), "labels-only", got[3], "expected", EXPECT[s], flush=True)
    if got != EXPECT[s]:
        sys.exit("STOP Bicycle does not rebuild the logged reading")
t = json.loads((O / "lumpi_pv_s2_test.json").read_text())
tid = sorted(t)
canon = lambda v: sorted(json.dumps([round(x, 5) for x in d["box"]] + [round(d["score"], 6), d["class"]]) for d in v)
ch = RL.Chain("test", tid)
ok = all(canon(ch.cls("Bicycle", "R0").get(f, [])) == canon([d for d in t[f] if d["class"] == "Bicycle"]) for f in tid)
print("test Bicycle R0 chain equals lumpi_v51:", ok, flush=True)
if not ok:
    sys.exit("STOP test base does not match")
b = ch.cls("Bicycle", "R4")
out = {f: [d for d in t[f] if d["class"] != "Bicycle"] + [RL.keep(d) for d in b.get(f, [])] for f in tid}
print("test Bicycle boxes", sum(d["class"] == "Bicycle" for v in t.values() for d in v), "->", sum(d["class"] == "Bicycle" for v in out.values() for d in v), flush=True)
(O / "lumpi_bicycle_r4_test.json").write_text(json.dumps(out))
print("wrote lumpi_bicycle_r4_test.json", flush=True)
