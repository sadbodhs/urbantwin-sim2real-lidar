"""lumpi_v41 = lumpi_v37 with Person from lumpi_v40 and Truck and Van from lumpi_v39; other classes and realism frames exactly v37. No new settings and no new readings: each part was read separately. Combines two parts that each missed a gate (v40 on fresh IoU 0.7; v39 Truck on fresh IoU 0.7, Van on val half 1). Alternative package only, built to prove the zip before a hidden upload.
    python scripts/build_lumpi_v41.py"""
import json
from pathlib import Path

O = Path("output/lumpi/ens_x")
a, b, c = (json.loads((O / f"lumpi_{v}_test.json").read_text()) for v in ("v37", "v40", "v39"))
SRC = {"Person": b, "Truck": c, "Van": c}
out = {f: [d for s in (a, b, c) for d in s[f] if SRC.get(d["class"], a) is s] for f in a}
(O / "lumpi_v41_test.json").write_text(json.dumps(out))
print("boxes", sum(len(v) for v in a.values()), "->", sum(len(v) for v in out.values()), flush=True)
