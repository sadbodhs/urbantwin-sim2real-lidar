"""L11-RS data: data/pcdet/ut_lumpi_rs = ut_lumpi_fix with the
objects of Car, Van, Truck, Bus and Bicycle resized by per-class, per-axis quantile mapping onto the realism_fit label sizes (real labels used only
as box-size statistics, which the organisers allow; no real point or box enters the training set). Classes with fewer than 30 realism_fit labels
keep objfix (left as in ut_lumpi_fix); Person and Motorcycle unchanged.
  - Synthetic distribution per class and axis: every object of the class in ut_lumpi_fix (train and val, 9950 frames).
  - An object at quantile q of its synthetic class gets the realism_fit size at quantile q (numpy linear quantile). Synthetic sizes come from a few
    vehicle models, so many objects share a size: inside a group of equal sizes, q = F(<v) + u x (F(<=v) - F(<v)), with one u per object drawn
    from a fixed seed (SEED, frame, box index) and the same u on all three axes (larger objects stay larger on every axis).
  - Points inside the box and the box are scaled about the bottom centre with utl.objfix.rescale (objfix's own code).
  - Every realism_fit id goes through the forbidden-frame guard.
Options (default = the recipe above): --keep CLS[,CLS] leaves classes as objfix; --real_min_len CLS=M[,...] drops realism_fit labels shorter than M m
from that class's statistics; --stats_only prints the mapping without writing. Writes ut_lumpi_rs/{points,labels} (from ut_lumpi_fix) and
ut_lumpi_rs/rs_mapping.json.
    python scripts/make_ut_lumpi_rs.py [--workers 8] [--stats_only] [--keep Van] [--real_min_len Van=3.0]"""
import argparse
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

SRC, OUT = Path("data/pcdet/ut_lumpi_fix"), Path("data/pcdet/ut_lumpi_rs")
REAL, FIT = Path("data/pcdet/lumpi_real/labels"), Path("data/pcdet/lumpi_real/ImageSets/realism_fit.txt")
CLASSES = ["Car", "Van", "Truck", "Bus", "Bicycle"]
MIN_REAL = 30
SEED = 20261109
Q = [0.1, 0.5, 0.9]
_M = {}


def read(p):
    rows = [l.split() for l in p.read_text().splitlines() if l.strip()]
    return np.array([[float(x) for x in r[:7]] for r in rows], np.float32).reshape(-1, 7), [r[7] for r in rows]


def u_of(fid, k):
    return np.random.default_rng([SEED, zlib.crc32(fid.encode()), k]).random()


def target(c, size, u):
    """Real size at the object's synthetic quantile, per axis (same u on all axes)."""
    m = _M[c]
    out = []
    for a in range(3):
        s = m["syn"][a]
        lo, hi = np.searchsorted(s, size[a], "left") / len(s), np.searchsorted(s, size[a], "right") / len(s)
        out.append(float(np.quantile(m["real"][a], min(lo + u * (hi - lo), 1.0))))
    return np.array(out)


def one(fid):
    boxes, names = read(SRC / "labels" / f"{fid}.txt")
    tags, scales, sizes = [], {}, []
    for k, (b, n) in enumerate(zip(boxes, names)):
        if n in _M:
            t = target(n, np.round(b[3:6].astype(np.float64), 4), u_of(fid, k))
            tags.append(f"B{k}"); scales[f"B{k}"] = tuple(t / b[3:6])
            sizes.append((n, list(map(float, b[3:6])), list(map(float, t))))
        else:
            tags.append("-")
    if not _M.get("_write"):
        return fid, sizes
    from utl import objfix
    from utl.prepare import write_labels
    pts = np.load(SRC / "points" / f"{fid}.npy")
    if scales:
        pts, boxes = objfix.rescale(pts, boxes, np.array(tags), scales)
    np.save(OUT / "points" / f"{fid}.npy", pts)
    write_labels(OUT / "labels" / f"{fid}.txt", boxes, names)
    return fid, sizes


def main():
    from utl import guard
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stats_only", action="store_true")
    ap.add_argument("--keep", default="")
    ap.add_argument("--real_min_len", default="")
    a = ap.parse_args()
    keep = {c for c in a.keep.split(",") if c}
    floor = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.real_min_len.split(",") if kv}
    ids = sorted(p.stem for p in (SRC / "labels").glob("*.txt"))
    syn = {c: [] for c in CLASSES}
    for f in ids:
        b, n = read(SRC / "labels" / f"{f}.txt")
        for bb, nn in zip(b, n):
            if nn in syn:
                syn[nn].append(np.round(bb[3:6].astype(np.float64), 4))
    real = {c: [] for c in CLASSES}
    fit = [l.strip() for l in open(FIT) if l.strip()]
    for i in fit:
        guard.check("lumpi", i)
        b, n = read(REAL / f"{i}.txt")
        for bb, nn in zip(b, n):
            if nn in real and bb[3] >= floor.get(nn, 0.0):
                real[nn].append(bb[3:6].astype(np.float64))
    log = {"seed": SEED, "realism_fit_frames": len(fit), "keep": sorted(keep), "real_min_len": floor, "classes": {}}
    for c in CLASSES:
        s, r = np.array(syn[c]), np.array(real[c]).reshape(-1, 3)
        e = {"synthetic_objects": len(s), "realism_fit_labels": len(r)}
        if c in keep or len(r) < MIN_REAL:
            e["mapped"] = False
            e["reason"] = "kept as objfix: " + ("--keep" if c in keep else f"{len(r)} realism_fit labels < {MIN_REAL}")
        else:
            e["mapped"] = True
            _M[c] = {"syn": [np.sort(s[:, k]) for k in range(3)], "real": [r[:, k] for k in range(3)]}
            e["real_q10_50_90"] = np.round(np.quantile(r, Q, axis=0), 3).tolist()
        e["synthetic_q10_50_90"] = np.round(np.quantile(s, Q, axis=0), 3).tolist()
        log["classes"][c] = e
    _M["_write"] = not a.stats_only
    if not a.stats_only:
        for d in ("points", "labels"):
            assert (OUT / d).is_dir() and len(list((OUT / d).iterdir())) == len(ids), f"{OUT / d} is not the full copy of ut_lumpi_fix"
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, ids, chunksize=32))
    for c, e in log["classes"].items():
        if e["mapped"]:
            t = np.array([x[2] for _, v in res for x in v if x[0] == c])
            e["out_q10_50_90"] = np.round(np.quantile(t, Q, axis=0), 3).tolist()
            e["out_share_len_ge_9m"] = round(float((t[:, 0] >= 9).mean()), 4)
            e["out_share_h_ge_2m"] = round(float((t[:, 2] >= 2).mean()), 4)
        print(c, json.dumps(e), flush=True)
    if not a.stats_only:
        (OUT / "rs_mapping.json").write_text(json.dumps(log, indent=1))
        print("wrote", OUT, "frames", len(ids), flush=True)


if __name__ == "__main__":
    main()
