"""lumpi_v39 = lumpi_v37 with two score rules from scripts/lumpi_v37_rules.py: T5 Truck npts < 100, V5 Van npts < 15; score x 0.3 once at most, on boxes scoring >= 0.05 that no v37 rule has lowered; no boxes added or moved.
Val vs v37: Truck 0.4561 -> 0.4632 (halves 0.2508 -> 0.2546, 0.5781 -> 0.5864), IoU 0.7 0.0368 -> 0.0374; Van 0.1672 -> 0.1708 (0.0010 flat, 0.2652 -> 0.2717; windows 2 and 3 up), IoU 0.7 0.0539 -> 0.0557. Fresh, T5 only: Truck 0.4347 -> 0.4440 (0.2149 -> 0.2205, 0.5609 -> 0.5726), IoU 0.7 0.0365 -> 0.0360.
Both miss a gate: T5 on fresh IoU 0.7, V5 on val half 1 (flat, about 15 real Vans). A single check; alternative package at most.
    python scripts/build_lumpi_v39.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import lumpi_v37_rules as R

if __name__ == "__main__":
    t = json.loads((R.O / "lumpi_v37_test.json").read_text())
    ft = R.features("test", t, sorted(t))
    out = R.apply(t, ft, ["T5", "V5"])
    (R.O / "lumpi_v39_test.json").write_text(json.dumps(out))
    for n in ("T5", "V5"):
        c, tf = R.RULES[n]
        print(n, "test boxes", sum(tf(x) and t[f][i]["class"] == c for f in t for i, x in ft[f]), "confident", sum(tf(x) and t[f][i]["class"] == c and t[f][i]["score"] >= 0.3 for f in t for i, x in ft[f]), flush=True)
