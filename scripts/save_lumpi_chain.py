"""Write a lane's final val boxes through the v24 chain: python scripts/save_lumpi_chain.py <lane> <out> [val|test]."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import lumpi_pole as P3
import lumpi_rescorer as RS
import lumpi_rider as R
import lumpi_static as S
import lumpi_truck_fix as T
from build_lumpi_v23 import truck_width_floor

T.BIG = ("Truck",)
lane, out = sys.argv[1], sys.argv[2]
split = sys.argv[3] if len(sys.argv) > 3 else "val"
pred = json.loads(Path(f"{lane}_{split}_v9.json").read_text())
pred = S.apply(pred, S.windows_of(split, sorted(pred)), ["Bicycle"], 0.1)
pred = truck_width_floor(T.fix(pred, split, "wh"))
pred = R.apply(pred, lambda f: T.pts(split, f))
pred = P3.apply(pred, lambda f: T.pts(split, f))
Path(out).write_text(json.dumps(RS.apply(pred, lambda f: T.pts(split, f))[0]))
print("wrote", out)
