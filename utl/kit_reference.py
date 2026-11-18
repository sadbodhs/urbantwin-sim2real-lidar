"""Local reference set for the kit's local_eval.py: the kit config with 50 realism frames, the kit's (empty) detection
ground truth, and 50 allowed held-out real frames (never forbidden ones), cut to the range and thinned to 50,000
points like the official frames.

    python -m utl.kit_reference --track lumpi --kit data/kit/lumpi --real_dir data/pcdet/lumpi_real/points \
        --ids data/pcdet/lumpi_real/ImageSets/realism_check.txt --out data/kit_ref/lumpi
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import guard, metrics
from .tracks import TRACKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--kit", required=True)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kit, out = Path(a.kit), Path(a.out)
    (out / "realism_reference").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((kit / "reference_data_dummy" / "config.json").read_text())
    cfg["n_realism_frames"] = 50
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy(kit / "reference_data_dummy" / "detection_gt.json", out / "detection_gt.json")
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    ids = [ids[i] for i in np.linspace(0, len(ids) - 1, 50).round().astype(int)]
    pc = cfg["point_cloud_range"]
    for k, i in enumerate(ids, 1):
        guard.check(a.track, i)
        p = metrics.mask_range(np.load(Path(a.real_dir) / f"{i}.npy")[:, :3], pc)
        rng = np.random.default_rng([7, k])
        if len(p) > 50000:
            p = p[rng.choice(len(p), 50000, replace=False)]
        p.astype(np.float32).tofile(out / "realism_reference" / f"{k:04d}.bin")
    print(f"{len(ids)} reference frames -> {out}")


if __name__ == "__main__":
    main()
