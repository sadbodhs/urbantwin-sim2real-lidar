"""One-process, frame-parallel version of scripts/lumpi_lane_x.sh: the same fusion (four fixed lumpi_v14 members + SPEC)
and the same lumpi_v9 post-processing (class copies, Person column test, Truck length refine, Bicycle / Person size prior,
L9), run per frame in a process pool with the members loaded once, instead of eight single-core steps that each read and
write a 0.5 GB JSON. Frames are independent in every step, so the output equals the shell lane's (checked on v24's lane).
    python scripts/lumpi_lane_fast.py NAME 'SPEC' [--cpv ens_l7/cpvnp1.0_cpv] [--baseovr '{...}'] [--workers 14]
    -> output/lumpi/ens_x/NAME_{val,test}_v9.json"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from lumpi_l9_l10 import l9_frame, load_pts
from utl import column_test, length_refine, postproc, size_prior
from utl.ensemble import ensemble

CLS = ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown"]
COPIES = "Motorcycle>Bicycle Unknown>Car Van>Car Van>Truck:tail Bus>Truck:tail".split()
PRIOR = size_prior.parse(["Bicycle:1.78,0.84,1.78", "Person:0.73,0.66,1.76"])
FRAMES = {"val": Path("data/pcdet/lumpi_real_50k/points"), "test": Path("data/kit/lumpi/detection_test_frames")}
_S = {}


def frame(fid):
    preds, w, cw, split = _S["preds"], _S["w"], _S["cw"], _S["split"]
    ens = ensemble([{fid: p.get(fid, [])} for p in preds], w, class_weights=cw, one_per_member=_S.get("opm", False))[fid]
    dets = postproc.apply({fid: ens}, postproc.parse_copies(COPIES))[fid]
    f = next(FRAMES[split].glob(f"{fid}.*"))
    pts = np.load(f) if f.suffix == ".npy" else np.fromfile(f, np.float32).reshape(-1, 3)
    pts = pts[np.abs(pts[:, :3]).sum(1) > 0][:, :3]
    dets = column_test.apply(dets, pts, {"Person"}, 3, 0.1)
    dets = length_refine.refine(dets, pts, ("Truck",), gap=0.75, max_len=16.5)
    dets = size_prior.apply(dets, PRIOR, "max")
    return fid, l9_frame(dets, load_pts(split, fid), 0.6, 0.0, 0.3)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("spec")
    ap.add_argument("--cpv", default="ens_l7/cpvnp1.0_cpv")
    ap.add_argument("--baseovr", default="{}")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--one_per_member", action="store_true", help="fusion fix F1/F2 (opt-in): one vote per member per cluster")
    a = ap.parse_args()
    _S["opm"] = a.one_per_member
    spec, ovr = json.loads(a.spec), json.loads(a.baseovr)
    out_dir = Path("output/lumpi/ens_x")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = [1.0, 0.5, 1.0, 0.5]
    for split, F in [(s, {"val": "val50k", "test": "test"}[s]) for s in a.splits.split(",")]:
        t0 = time.time()
        paths = [f"output/lumpi/pp_lumpi_long/kit_mc/pred_{F}.json", f"output/lumpi/pp_lumpi_d50/kit_mc/pred_{F}.json",
                 f"output/lumpi/{a.cpv}_{split}.json", f"output/lumpi/pp_lumpi_long_all/kit_mc/pred_{F}.json"]
        for model, _, _ in spec:
            paths.append(f"output/lumpi/{model[5:]}_{split}.json" if model.startswith("file:")
                         else f"output/lumpi/{model}/kit_mc/pred_{F}.json")
        keep = [None] * 4 + [None if cls == "all" else set(cls) for _, _, cls in spec]  # classes with weight 0 are never used
        preds = []
        for path, k in zip(paths, keep):
            d = json.loads(Path(path).read_text())
            preds.append(d if k is None else {f: [x for x in v if x["class"] in k] for f, v in d.items()})
            del d
        _S.update(preds=preds, split=split,
                  w=base + [0.0] * len(spec),
                  cw={c: ovr.get(c, base) + [w if cls == "all" or c in cls else 0.0 for _, w, cls in spec] for c in CLS})
        fids = list(_S["preds"][0])
        with Pool(a.workers) as pool:
            res = dict(pool.map(frame, fids, chunksize=4))
        (out_dir / f"{a.name}_{split}_v9.json").write_text(json.dumps({f: res[f] for f in fids}))
        print(f"{a.name} {split}: {len(fids)} frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
