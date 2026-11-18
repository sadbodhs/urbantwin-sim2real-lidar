"""Convert raw datasets into the OpenPCDet layout used by SimDataset (see utl.pcdet_ext).

    python -m utl.prepare lumpi_real --lidar data/raw/lumpi/Measurement4 \
        --labels data/raw/lumpi/extracted/Label/Measurement4/Label.csv --out data/pcdet/lumpi_real
    python -m utl.prepare v2x_real --root data/raw/v2xreal/extracted --split val --out data/pcdet/v2x_real
    python -m utl.prepare ut --track lumpi --raw data/raw/ut_lumpi --out data/pcdet/ut_lumpi
    python -m utl.prepare infos --root data/pcdet/ut_lumpi --track lumpi --splits train val --gt_db

Real frames are only written when guard.allowed() says so; they are used for validation and for fitting the
realism calibration, never for training.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import guard, lumpi, metrics, ut, v2xreal
from .tracks import TRACKS

MARGIN = 2.0  # keep points slightly beyond the scored range (the model range is +-40.96 m)


def _crop(points, track):
    x0, y0, z0, x1, y1, z1 = TRACKS[track]["pc_range"]
    return metrics.mask_range(points, [x0 - MARGIN, y0 - MARGIN, z0 - 1, x1 + MARGIN, y1 + MARGIN, z1 + 1])


def _in_range(boxes, track):
    x0, y0, _, x1, y1, _ = TRACKS[track]["pc_range"]
    return (boxes[:, 0] >= x0) & (boxes[:, 0] <= x1) & (boxes[:, 1] >= y0) & (boxes[:, 1] <= y1)


def write_labels(path, boxes, names):
    with open(path, "w") as f:
        for b, n in zip(boxes, names):
            f.write(" ".join(f"{v:.4f}" for v in b[:7]) + f" {n}\n")


def write_split(out, name, ids):
    (out / "ImageSets").mkdir(parents=True, exist_ok=True)
    (out / "ImageSets" / f"{name}.txt").write_text("\n".join(ids) + "\n")


# ------------------------------------------------------------------------------------------------ LUMPI real
_LZ = None


def _lumpi_init(zip_dir):
    global _LZ
    _LZ = lumpi.LidarZip(zip_dir)


def _lumpi_one(args):
    idx, rows, out = args
    fid = f"{idx:06d}"
    guard.check("lumpi", fid)
    pts = _crop(_LZ.read(idx), "lumpi")
    np.save(out / "points" / f"{fid}.npy", pts)
    rows = rows if rows is not None else np.zeros((0, 8), np.float32)
    rows = rows[_in_range(rows, "lumpi")]
    write_labels(out / "labels" / f"{fid}.txt", rows, [lumpi.CLASSES[int(c)] for c in rows[:, 7]])
    return fid


def lumpi_real(a):
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    labels = lumpi.load_labels(a.labels)
    n = len(lumpi.LidarZip(a.lidar))
    val = lumpi.val_frames(n, stride=20, max_index=6400)               # 320 detection-validation frames
    fit = [i + 10 for i in val[::2]]                                    # realism calibration frames
    check = [i + 10 for i in val[1::2]]                                 # disjoint realism check frames
    ids = sorted(set(val) | set(fit) | set(check))
    with ProcessPoolExecutor(a.workers, initializer=_lumpi_init, initargs=(a.lidar,)) as ex:
        list(ex.map(_lumpi_one, [(i, labels.get(i), out) for i in ids], chunksize=4))
    write_split(out, "val", [f"{i:06d}" for i in val])
    write_split(out, "realism_fit", [f"{i:06d}" for i in fit])
    write_split(out, "realism_check", [f"{i:06d}" for i in check])
    print(f"LUMPI real: {len(val)} val, {len(fit)} realism-fit, {len(check)} realism-check frames -> {out}")


def lumpi_test(a):
    """Stand-in for the starting kit's detection_test_frames/: the 50 test-id frames of Measurement 4 as float32
    Nx3 .bin, points only. Labels are never read for these frames; they are only ever passed to utl.infer.
    Replace with the official files once Codabench is reachable (and check they are identical)."""
    from .tracks import test_frame_ids
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lz = lumpi.LidarZip(a.lidar)
    for fid in test_frame_ids("lumpi"):
        lz.read(int(fid)).astype(np.float32).tofile(out / f"{fid}.bin")
    print(f"wrote {len(test_frame_ids('lumpi'))} test frames -> {out}")


# ------------------------------------------------------------------------------------------------ V2X-Real
def v2x_real(a):
    """All infrastructure frames of a split, every `stride`-th, with the 3 challenge super-classes.

    The challenge ids are '<scene>_<frame>' over a scene numbering we cannot reproduce without the starting kit,
    so these frames cannot be checked against forbidden_frames.txt yet; they are used for validation only and
    must be re-checked (by matching against the released test frames) once the kit is available.
    """
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    ids = []
    for k, (scen, ag, stem, b, y) in enumerate(v2xreal.frames(a.root, a.split)):
        if int(stem) % a.stride:
            continue
        pts, boxes, names = v2xreal.read_frame(b, y)
        m = _in_range(boxes, "v2x") if len(boxes) else np.zeros(0, bool)
        fid = f"{a.split}_{scen}_{ag.replace('-', 'i')}_{stem}"
        np.save(out / "points" / f"{fid}.npy", _crop(pts[:, :3], "v2x"))
        write_labels(out / "labels" / f"{fid}.txt", boxes[m], [n for n, mm in zip(names, m) if mm])
        ids.append(fid)
    write_split(out, a.split, ids)
    print(f"V2X-Real {a.split}: {len(ids)} frames -> {out}")


# ------------------------------------------------------------------------------------------------ UrbanTwin
_ZF = None


def _ut_init(raw):
    global _ZF
    _ZF = ut.open_zip(raw)


def _ut_one(args):
    fid, track, out, scales = args
    pts = _crop(ut.read_points(_ZF, fid), track)
    boxes, raw_names = ut.read_labels(_ZF, fid)
    if scales:
        from . import objfix
        pts, boxes = objfix.rescale(pts, boxes, np.array(raw_names), scales)
    cmap = ut.CLASS_MAP[track]
    keep = [i for i, n in enumerate(raw_names) if cmap.get(n)]
    boxes, names = boxes[keep], [cmap[raw_names[i]] for i in keep]
    m = _in_range(boxes, track) if len(boxes) else np.zeros(0, bool)
    np.save(out / "points" / f"{fid}.npy", pts)
    write_labels(out / "labels" / f"{fid}.txt", boxes[m], [n for n, k in zip(names, m) if k])
    return fid, len(pts), sorted(set(raw_names))


def raw_class_stats(zf, ids, every=40):
    """Median box size per raw synthetic class over every `every`-th frame -> {class: {'n', 'lwh'}}."""
    sizes = {}
    for fid in ids[::every]:
        boxes, names = ut.read_labels(zf, fid)
        for b, n in zip(boxes, names):
            sizes.setdefault(n, []).append(b[3:6])
    return {n: {"n": len(v), "lwh": np.median(v, 0).tolist()} for n, v in sizes.items()}


def ut_synthetic(a):
    """All synthetic frames; the last `val_frac` of the sequence (contiguous) is the synthetic validation split.
    With --objfix, objects are first rescaled per raw class to generic sizes (utl.objfix), before class mapping."""
    import json
    out = Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    zf = ut.open_zip(a.raw)
    ids = ut.frame_ids(zf)[:: a.stride]
    scales = None
    if a.objfix:
        from . import objfix
        raw_stats = raw_class_stats(zf, ids)
        scales = objfix.class_scales(raw_stats)
        (out / "objfix.json").write_text(json.dumps({"raw_stats": raw_stats, "scales": scales}, indent=1))
        print("class scales (l, w, h):", scales)
    with ProcessPoolExecutor(a.workers, initializer=_ut_init, initargs=(a.raw,)) as ex:
        res = list(ex.map(_ut_one, [(i, a.track, out, scales) for i in ids], chunksize=8))
    n_val = int(len(ids) * a.val_frac)
    write_split(out, "train", ids[: len(ids) - n_val - a.gap])
    write_split(out, "val", ids[len(ids) - n_val:])
    raw_classes = sorted({c for r in res for c in r[2]})
    print(f"UT {a.track}: {len(ids)} frames, mean {np.mean([r[1] for r in res]):.0f} pts in range, "
          f"raw classes {raw_classes} -> {out}")


# ------------------------------------------------------------------------------------------------ UT-LUMPI fix
def _fix_one(args):
    fid, src, out, scales, seed, rider = args
    from . import objfix, synth_ped
    rng = np.random.default_rng([seed, int(fid)])
    pts = np.load(src / "points" / f"{fid}.npy")
    rows = [l.split() for l in (src / "labels" / f"{fid}.txt").read_text().splitlines() if l.strip()]
    boxes = np.array([[float(v) for v in r[:7]] for r in rows], np.float32).reshape(-1, 7)
    names = np.array([r[7] for r in rows])
    pts, boxes = objfix.rescale(pts, boxes, names, scales)
    pts, boxes = synth_ped.add_riders(pts, boxes, names, rng, body=rider)
    np.save(out / "points" / f"{fid}.npy", pts.astype(np.float32))
    write_labels(out / "labels" / f"{fid}.txt", boxes, names)


def ut_fix(a):
    """UT-LUMPI with generic-size objects and ray-cast riders (see utl.objfix, utl.synth_ped; --rider v2 = rider_v2)."""
    import json
    import shutil
    from . import objfix
    src, out = Path(a.src), Path(a.out)
    (out / "points").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "ImageSets", out / "ImageSets", dirs_exist_ok=True)
    scales = objfix.class_scales(json.loads((src / "stats.json").read_text()))
    print("class scales (l, w, h):", scales)
    ids = sorted({l.strip() for sp in ("train", "val") for l in open(src / "ImageSets" / f"{sp}.txt") if l.strip()})
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(_fix_one, [(i, src, out, scales, a.seed, a.rider) for i in ids], chunksize=16))
    (out / "objfix.json").write_text(json.dumps({"scales": scales, "seed": a.seed, "rider": a.rider}, indent=1))
    print(f"fixed {len(ids)} frames -> {out}")


# ------------------------------------------------------------------------------------------------ infos
def infos(a):
    from .pcdet_ext import write_gt_database, write_infos
    import json
    classes = TRACKS[a.track]["classes"]
    stats = {}
    for sp in a.splits:
        inf = write_infos(a.root, sp, classes)
        if sp == "train":
            boxes = np.concatenate([i["annos"]["gt_boxes_lidar"] for i in inf] or [np.zeros((0, 7))])
            names = np.concatenate([i["annos"]["name"] for i in inf] or [np.zeros(0, str)])
            for c in classes:
                b = boxes[names == c]
                if len(b):
                    stats[c] = dict(n=int(len(b)), lwh=np.median(b[:, 3:6], 0).round(3).tolist(),
                                    bottom_z=float(np.median(b[:, 2] - b[:, 5] / 2).round(3)))
            if a.gt_db:
                stats["_gt_db"] = write_gt_database(a.root, inf, every=a.db_every, classes=classes)
        print(f"{sp}: {len(inf)} frames")
    (Path(a.root) / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("lumpi_real")
    p.add_argument("--lidar", required=True, help="directory with Measurement4 lidar.zip.NNN")
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("lumpi_test")
    p.add_argument("--lidar", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("v2x_real")
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sp.add_parser("ut")
    p.add_argument("--track", required=True, choices=list(TRACKS))
    p.add_argument("--raw", required=True, help="directory with the <name>.zip.NNN parts")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--gap", type=int, default=50, help="frames dropped between train and val (temporal leakage)")
    p.add_argument("--objfix", action="store_true", help="rescale objects to generic class sizes (utl.objfix)")
    p.add_argument("--workers", type=int, default=12)
    p = sp.add_parser("ut_fix")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--rider", default="cylinder", choices=["cylinder", "v2"])
    p = sp.add_parser("infos")
    p.add_argument("--root", required=True)
    p.add_argument("--track", required=True, choices=list(TRACKS))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--gt_db", action="store_true")
    p.add_argument("--db_every", type=int, default=2)
    a = ap.parse_args()
    {"lumpi_real": lumpi_real, "lumpi_test": lumpi_test, "v2x_real": v2x_real, "ut": ut_synthetic, "ut_fix": ut_fix, "infos": infos}[a.cmd](a)


if __name__ == "__main__":
    main()
