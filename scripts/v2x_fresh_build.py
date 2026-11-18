"""V2X fresh set, built after the organisers allowed the V2X-Real train split (minus the 100 forbidden frames) for validation and model selection.
Scoring only: no real frame enters training. Fixed before any reading:
  frames      agent -1 only, every 10th frame (stem % 10 == 0), all train-split folders with a -1 agent (train1-4.zip, read without extraction).
  excluded    the 100 forbidden frames (organisers' table data/raw/v2xreal/forbidden_frames_mapping.csv); every frame within 20 frames of one in
              the same folder; every frame whose points copy a forbidden frame (share of 200 sampled forbidden points found exactly in the
              frame above 0.9, as scripts/v2x_frame_exact.py).
  labels      utl/v2xreal.py, 3 super-classes, boxes in the track range; points cut as utl/prepare.py (data/pcdet/v2x_fresh), then the
              50,000-point thinning of utl/thin.py (data/pcdet/v2x_fresh_50k, run separately).
  halves      first and second half of the sorted frame list.
The frame list and every excluded frame with its reason are written before any point or label file.
    python scripts/v2x_fresh_build.py"""
import csv
import io
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, ".")
from utl import prepare, v2xreal

RAW = Path("data/raw/v2xreal")
OUT = Path("data/pcdet/v2x_fresh")
ZIPS = ["train1", "train2", "train3", "train4"]
BUFFER, SHARE = 20, 0.9


def h(a):
    b = np.ascontiguousarray(a[:, :3].astype(np.float32)).view(np.uint32).reshape(-1, 3).astype(np.uint64)
    return b[:, 0] * np.uint64(11400714819323198485) ^ b[:, 1] * np.uint64(14029467366897019727) ^ b[:, 2] * np.uint64(1609587929392839161)


def nonzero(a):
    return a[np.abs(a[:, :3]).sum(1) > 0]


def load(zf, name):
    return np.frombuffer(zf.read(name), np.float32).reshape(-1, 4)


def frame(zf, scen, stem):
    """utl.v2xreal.read_frame from bytes in the zip: points and boxes in the agent LiDAR frame, 3 super-classes."""
    pts = load(zf, f"{scen}/-1/{stem}.bin")
    boxes, names = np.zeros((0, 7), np.float32), []
    meta = yaml.safe_load(io.BytesIO(zf.read(f"{scen}/-1/{stem}.yaml")))
    w2l = np.linalg.inv(v2xreal.x_to_world(meta["lidar_pose"]))
    out = []
    for obj in (meta.get("vehicles") or {}).values():
        cls = v2xreal.TO_SUPER.get(obj.get("obj_type"))
        if cls is None:
            continue
        loc = np.asarray(obj["location"], float) + np.asarray(obj.get("center", [0, 0, 0]), float)
        T = w2l @ v2xreal.x_to_world([*loc, *obj["angle"]])
        fwd = T[:3, :3] @ np.array([1.0, 0, 0])
        out.append([*T[:3, 3], *(np.asarray(obj["extent"], float) * 2), np.arctan2(fwd[1], fwd[0])])
        names.append(cls)
    if out:
        boxes = np.asarray(out, np.float32)
    return pts, boxes, names


def main():
    forb = list(csv.DictReader(open(RAW / "forbidden_frames_mapping.csv")))
    assert len(forb) == 100
    fset = {(r["scenario_folder"], int(r["frame"])) for r in forb}
    where, cands = {}, []
    for z in ZIPS:
        zf = zipfile.ZipFile(RAW / f"{z}.zip")
        for n in zf.namelist():
            if "/-1/" in n and n.endswith(".bin"):
                scen, stem = n.split("/-1/")[0], n.rsplit("/", 1)[1][:-4]
                where[(scen, int(stem))] = (z, stem)
    folders = sorted({s for s, _ in where})
    print("train -1 folders", len(folders), "frames", len(where), flush=True)
    assert all(k in where for k in fset), "a forbidden frame is not in the train zips"
    excl = []
    for (scen, i), (z, stem) in sorted(where.items()):
        if i % 10:
            continue
        near = [j for s, j in fset if s == scen and abs(j - i) <= BUFFER]
        if (scen, i) in fset:
            excl.append((scen, stem, "forbidden"))
        elif near:
            excl.append((scen, stem, f"within {BUFFER} of forbidden {min(near, key=lambda j: abs(j - i)):06d}"))
        else:
            cands.append((scen, stem, z))
    zfs = {z: zipfile.ZipFile(RAW / f"{z}.zip") for z in ZIPS}
    fh = {}
    for scen, i in sorted(fset):
        z, stem = where[(scen, i)]
        a = nonzero(load(zfs[z], f"{scen}/-1/{stem}.bin"))
        fh[(scen, stem)] = h(a[np.linspace(0, len(a) - 1, 200).astype(int)])
    keep, top = [], []
    for scen, stem, z in cands:
        r = np.sort(h(nonzero(load(zfs[z], f"{scen}/-1/{stem}.bin"))))
        best = max((float((r[np.minimum(np.searchsorted(r, s), len(r) - 1)] == s).mean()), k) for k, s in fh.items())
        top.append((best[0], scen, stem, best[1]))
        if best[0] > SHARE:
            excl.append((scen, stem, "point copy of forbidden %s/%s, share %.3f" % (best[1][0], best[1][1], best[0])))
        else:
            keep.append((scen, stem, z))
    top.sort(reverse=True)
    print("highest shared-point share against any forbidden frame:", [(round(t[0], 3), t[1], t[2]) for t in top[:3]], flush=True)
    ids = sorted(f"train_{scen}_i1_{stem}" for scen, stem, _ in keep)
    (OUT / "ImageSets").mkdir(parents=True, exist_ok=True)
    for name, x in (("fresh", ids), ("val", ids), ("fresh_h1", ids[: len(ids) // 2]), ("fresh_h2", ids[len(ids) // 2:])):
        (OUT / "ImageSets" / f"{name}.txt").write_text("\n".join(x) + "\n")
    with open(OUT / "excluded.tsv", "w") as f:
        f.write("folder\tframe\treason\n")
        for e in sorted(excl):
            f.write("\t".join(e) + "\n")
    reasons = {}
    for e in excl:
        k = e[2].split(" ")[0]
        reasons[k] = reasons.get(k, 0) + 1
    print("candidates (stem % 10 == 0)", len(cands) + sum(1 for e in excl if not e[2].startswith("point")), "excluded", len(excl), reasons, "kept", len(ids),
          "halves", len(ids) // 2, len(ids) - len(ids) // 2, flush=True)
    print("frame list and exclusions written; now points and labels", flush=True)
    (OUT / "points").mkdir(exist_ok=True)
    (OUT / "labels").mkdir(exist_ok=True)
    with ProcessPoolExecutor(8) as ex:
        n = list(ex.map(write_one, keep, chunksize=8))
    print("written", len(n), "frames, labels per frame", round(float(np.mean(n)), 1), flush=True)


def write_one(item):
    scen, stem, z = item
    zf = zipfile.ZipFile(RAW / f"{z}.zip")
    pts, boxes, names = frame(zf, scen, stem)
    m = prepare._in_range(boxes, "v2x") if len(boxes) else np.zeros(0, bool)
    fid = f"train_{scen}_i1_{stem}"
    np.save(OUT / "points" / f"{fid}.npy", prepare._crop(pts[:, :3], "v2x"))
    prepare.write_labels(OUT / "labels" / f"{fid}.txt", boxes[m], [n for n, mm in zip(names, m) if mm])
    return int(m.sum())


if __name__ == "__main__":
    main()
