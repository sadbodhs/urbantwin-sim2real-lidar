"""M3' step 3 (LUMPI only; it passed val with a = 0.3): retrain the same rescorer (same data, same seed), save it, and
check that it behaves alike on val and test: distribution of p over confident (>= 0.3) and all (>= 0.05) v20 Person
boxes, val vs test. Writes /work/r3/lumpi_rescorer.pkl and /work/r3/lumpi_test_multipliers.json
({frame: [[box rounded to 4 dp, multiplier], ...]} for v20 test Person boxes >= 0.05, multiplier = 0.3 + 0.7 p).
    python /work/r3_apply.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/pylib")
sys.path.insert(0, "/work")
import lumpi_next as N  # noqa: E402
import r3_train as R  # noqa: E402
from lumpi_next import T  # noqa: E402


def v20(split):
    p = N.chain(split)
    pole = N.measure(p, split, "Person", N.pole_ratio)
    return N.demote(p, {f: [x is not None and x >= 0.1 for x in v] for f, v in pole.items()})


def main():
    m, info = R.fit(R.synthetic("lumpi", "Person"))
    (R.W / "lumpi_rescorer.pkl").write_bytes(pickle.dumps(m))
    out = {"fit": info}
    mult = {}
    for split in ("val", "test"):
        p = v20(split)
        X = R.real_feats(p, lambda f: T.pts(split, f), "Person")
        ps, conf = [], []
        for f, v in p.items():
            idx = sorted(X[f])
            if not idx:
                continue
            pr = m.predict_proba(np.array([X[f][i] for i in idx], float))[:, 1]
            for i, q in zip(idx, pr):
                ps.append(q)
                if v[i]["score"] >= 0.3:
                    conf.append(q)
                if split == "test":
                    mult.setdefault(f, []).append([[round(float(x), 4) for x in v[i]["box"]], round(float(0.3 + 0.7 * q), 6)])
        q = lambda a: np.round(np.percentile(a, [10, 25, 50, 75, 90]), 3).tolist()
        out[split] = {"boxes": len(ps), "confident": len(conf), "p_all_pct": q(ps), "p_conf_pct": q(conf),
                      "conf_share_p_below_0.5": round(float(np.mean(np.array(conf) < 0.5)), 3)}
        print(split, json.dumps(out[split]), flush=True)
    (R.W / "lumpi_test_multipliers.json").write_text(json.dumps(mult))
    Path("/work/r3_apply.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
