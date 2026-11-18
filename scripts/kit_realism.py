"""Realism of a pool of 50 frames scored with the official kit's code:
the server's exact computation (seed 1234) and four more reference-side draws (seeds 1-4; the synthetic subsample stays the server's), against
the track's local check frames (V2X rcheck_i1, LUMPI realism_check; the same frames as scripts/realism_r1.py). Information; changes no reading.
    python scripts/kit_realism.py v2x output/v2x/realism_v2_i1 output/v2x/realism_r2"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import realism_r1 as R1
from utl import metrics
from utl.realism import _load, nonzero
from utl.tracks import TRACKS


def main(track, pools):
    c, pc = R1.CFG[track], TRACKS[track]["pc_range"]
    rep = json.loads((Path(c["pool"]) / "realism_report.json").read_text())
    check = R1.real_frames(track, c, "check")
    if rep["zeros"] == "drop":
        check = [nonzero(f) for f in check]
    out = {}
    for pool in pools:
        fr = [_load(Path(pool) / f"{k:04d}.bin") for k in range(1, 51)]
        rows = [metrics.kit_realism_scores(fr, check, pc, track)] + [metrics.kit_realism_scores(fr, check, pc, track, s) for s in (1, 2, 3, 4)]
        ours = metrics.realism_scores(fr, check, pc, 10000, seed=1234)
        out[pool] = dict(kit_server_seed=rows[0], kit_ref_seeds=rows[1:], ours_seed1234=ours)
        print(track, pool, "| kit (server computation) %.4f" % rows[0]["score"], {k: round(v, 4) for k, v in rows[0]["raw"].items()},
              "| kit, reference seeds 1-4: mean %.4f" % np.mean([r["score"] for r in rows[1:]]), "| ours seed 1234 %.4f" % ours["score"], flush=True)
    Path(f"logs/kit_realism_{track}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
