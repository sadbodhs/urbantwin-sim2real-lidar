"""Block 1 of the extra slots: one new seed of a lumpi_v51 Person refiner (L11 seed 3, L12 seed 2,
PV-RCNN++ seed 2) joins lumpi_v51's Person average as one more member. Fixed before any reading, the same reading as scripts/pnrcnn_read.py:
its second stage on the lumpi_v29 Person boxes (down to 0.01) is added to lumpi_v51's five (L11, L12, L11b second seed, PV-RCNN++, test A), read
with R0 / R4 / R6, each like for like against the same rule without the new member (references logged first), then the v37 size change and length rule.
Chosen rule: the largest val gain over its own reference (R0 on a tie). Person gate on val, then on fresh: +0.003 on both halves, IoU 0.7 not down.
A file only if both pass: R0 -> lumpi_v53 with this Person; R4 -> lumpi_v53 with this Person by R4, flagged; R6 -> lumpi_v52 with this Person by R6,
flagged as lumpi_v52 is. Members join lumpi_v51's average one at a time (not stacked). Writes output/lumpi/ens_x/lumpi_<name>_test.json.
    python scripts/lumpi_member_read.py refs logs/fresh_ids.txt
    python scripts/lumpi_member_read.py read vrcnn_lumpi_c_s2 logs/fresh_ids.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import lumpi_static as S
import lumpi_v37_rules as R
import route_read_lumpi as RL

O = Path("output/lumpi/ens_x")
RULES = ["R0", "R4", "R6"]
REFS = Path("logs/lumpi_member_refs.json")


def paths(s, new):
    base = [RL.L / f"v29_{s}_ref_min01.json", RL.C / f"v29_{s}_ref_min01.json", RL.Z / f"v29_{s}_ref_min01.json",
            RL.PVD / f"v29_{s}_min01.json", RL.N / f"v29_{s}_min01.json"]   # lumpi_v51's members, its order
    return base + ([Path(f"output/lumpi/{new}/s2/v29_{s}_min01.json")] if new else [])


def row(s, ids, gt, rule, new):
    p = RL.person(paths(s, new), rule)
    return R.row({f: p.get(f, []) for f in ids}, gt, ids, "Person")


def gate(r, ref):
    return bool(all(round(1e4 * (r[h][0] - ref[h][0])) >= 30 for h in ("h1", "h2")) and r["all"][1] >= ref["all"][1])


def refs(fr):
    out = {}
    for s, ids in (("val", S.ids_of("val")), ("fresh", fr)):
        gt = S.labels(ids)
        out[s] = {k: row(s, ids, gt, k, None) for k in RULES}
        for k in RULES:
            print("reference", s, k, "(lumpi_v51's five members)", json.dumps(out[s][k]), flush=True)
    if out["val"]["R0"]["all"][0] != 0.4366:
        sys.exit("STOP R0 without the new member does not rebuild lumpi_v51 Person 0.4366")
    REFS.write_text(json.dumps(out, indent=1))
    print("wrote", REFS, flush=True)


def read(new, fr):
    ref = json.loads(REFS.read_text())
    ids = S.ids_of("val")
    gt = S.labels(ids)
    print("members: L11, L12, L11b second seed, PV-RCNN++, test A +", new, "| rules", RULES, "| references from", REFS, flush=True)
    nv = {}
    for k in RULES:
        nv[k] = row("val", ids, gt, k, new)
        print("val", k, "with", new, json.dumps(nv[k]), "vs without %+.4f" % (nv[k]["all"][0] - ref["val"][k]["all"][0]), flush=True)
    k = max(RULES, key=lambda x: (nv[x]["all"][0] - ref["val"][x]["all"][0], -RULES.index(x)))
    gv = gate(nv[k], ref["val"][k])
    print("chosen", k, "| val gate", "PASS" if gv else "fail", flush=True)
    res = {"member": new, "val": {"ref": ref["val"], "new": nv}, "chosen": k, "val_gate": gv}
    if gv:
        a, b = row("fresh", fr, S.labels(fr), k, new), ref["fresh"][k]
        gf = gate(a, b)
        print("fresh", k, "with", json.dumps(a), "| without", json.dumps(b), "%+.4f" % (a["all"][0] - b["all"][0]), "| gate", "PASS" if gf else "fail", flush=True)
        res.update(fresh={"new": a, "ref": b}, fresh_gate=gf)
        if gf:
            base = {"R0": "lumpi_bicycle_r4_test.json", "R4": "lumpi_bicycle_r4_test.json", "R6": "lumpi_r6_test.json"}[k]
            t = json.loads((O / base).read_text())
            x = RL.person(paths("test", new), k)
            assert set(x) == set(t), "class writer: the new class covers other frames than the base file"
            canon = lambda v: sorted(json.dumps([round(u, 5) for u in d["box"]] + [round(d["score"], 6), d["class"]]) for d in v)
            if k in ("R0", "R6"):   # rebuild equals base: the five members by this rule give the base file's Person back
                b0 = RL.person(paths("test", None), k)
                ok = all(canon(b0.get(f, [])) == canon([d for d in t[f] if d["class"] == "Person"]) for f in t)
                print("test: Person by", k, "without", new, "equals", base, ":", ok, flush=True)
                if not ok:
                    sys.exit("STOP the base Person does not rebuild")
            (O / f"lumpi_{new}_test.json").write_text(json.dumps({f: [d for d in t[f] if d["class"] != "Person"] + x.get(f, []) for f in t}))
            print(f"wrote lumpi_{new}_test.json:", base, "with Person by", k, "and", new, "(flagged)" if k != "R0" else "", flush=True)
    Path(f"logs/lumpi_member_{new}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    fr_path = sys.argv[-1]
    fr = [l.strip() for l in open(fr_path) if l.strip()]
    refs(fr) if sys.argv[1] == "refs" else read(sys.argv[2], fr)
