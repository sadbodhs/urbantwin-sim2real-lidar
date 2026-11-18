"""Build lumpi_v23 = lumpi_v22 + Truck width floor (TS-A): every Truck box at least 2.95 m wide
(EU 2.55 m body plus mirrors), applied to v22's boxes (the floor comes after the Truck fix in the chain; v22's later
steps touch Person scores only, so the order does not matter). Val on the v20 chain: Truck 0.4063 ->
0.4115 (halves 0.1945 -> 0.2050, 0.5508 -> 0.5532), Truck @0.7 0.0253 -> 0.0264, other classes identical.
Writes output/lumpi/ens_l8/v23_test.json from output/lumpi/ens_l8/v22_test.json."""
import json
from pathlib import Path

TRUCK_MIN_W = 2.95
O = Path("output/lumpi/ens_l8")


def truck_width_floor(pred):
    return {f: [dict(d, box=list(d["box"][:4]) + [max(d["box"][4], TRUCK_MIN_W)] + list(d["box"][5:]))
                if d["class"] == "Truck" else d for d in v] for f, v in pred.items()}


if __name__ == "__main__":
    t22 = json.loads((O / "v22_test.json").read_text())
    t23 = truck_width_floor(t22)
    widened = sum(a["box"] != b["box"] for f in t22 for a, b in zip(t22[f], t23[f]))
    other = sum(a != b for f in t22 for a, b in zip(t22[f], t23[f]) if a["class"] != "Truck")
    assert other == 0
    (O / "v23_test.json").write_text(json.dumps(t23))
    n = sum(d["class"] == "Truck" for v in t22.values() for d in v)
    nc = sum(d["class"] == "Truck" and d["score"] >= 0.3 for v in t22.values() for d in v)
    wc = sum(a["box"] != b["box"] and a["score"] >= 0.3 for f in t22 for a, b in zip(t22[f], t23[f]))
    print(f"v23 test written: {widened} of {n} Truck boxes widened ({wc} of {nc} confident); other classes identical")
