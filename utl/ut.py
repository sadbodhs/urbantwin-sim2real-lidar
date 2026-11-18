"""UrbanTwin synthetic datasets (UT-LUMPI, UT-V2X-Real-IC; Harvard Dataverse, CC0).

Each is one stored (uncompressed) zip cut into raw byte chunks <name>.zip.001 ... .NNN, with
<Seq>/pcd/<frame>.npy (float32 Nx3), <Seq>/lbl/<frame>.txt ("x y z l w h heading Class track_id") and
<Seq>/ins_seg/<frame>.label. The chunks are read in place as one seekable stream, so no 50 GB concatenation
or extraction is needed.
"""
import io
import zipfile
from pathlib import Path

import numpy as np

# synthetic class -> track class (None = not used for training). UT-V2X has no walking pedestrians; its riders
# map to V2X-Real's "pedestrian" super-class, as in V2X-Real's own SUPER_CLASS_MAP.
CLASS_MAP = {
    "lumpi": {"Person": "Person", "Pedestrian": "Person", "Car": "Car", "Bicycle": "Bicycle", "Cyclist": "Bicycle",
              "BicycleRider": "Bicycle", "Motorcycle": "Motorcycle", "MotorcyleRider": "Motorcycle",
              "Bus": "Bus", "Truck": "Truck", "Van": "Van", "Unknown": "Unknown"},
    "v2x": {"Car": "vehicle", "Person": "pedestrian", "Pedestrian": "pedestrian", "Bicycle": "pedestrian",
            "Cyclist": "pedestrian", "BicycleRider": "pedestrian", "Motorcycle": "pedestrian",
            "MotorcyleRider": "pedestrian", "Truck": "truck", "Van": "truck", "Bus": "truck",
            "Unknown": None},
}


class MultiPartFile(io.RawIOBase):
    def __init__(self, parts):
        self.parts = [Path(p) for p in parts]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.size = int(self.starts[-1])
        self.pos = 0
        self.fhs = [None] * len(self.parts)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.size:
            k = int(np.searchsorted(self.starts, self.pos, side="right") - 1)
            if self.fhs[k] is None:
                self.fhs[k] = open(self.parts[k], "rb")
            fh = self.fhs[k]
            fh.seek(self.pos - int(self.starts[k]))
            chunk = fh.read(min(n - done, int(self.starts[k + 1]) - self.pos))
            b[done:done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
        return done


def open_zip(raw_dir, prefix="*"):
    parts = sorted(p for p in Path(raw_dir).glob(f"{prefix}.zip.[0-9][0-9][0-9]"))
    if not parts:
        raise FileNotFoundError(f"no .zip.NNN parts in {raw_dir}")
    zf = zipfile.ZipFile(io.BufferedReader(MultiPartFile(parts), 1 << 20))
    zf.seq = zf.namelist()[0].split("/")[0]
    return zf


def frame_ids(zf):
    return sorted(Path(n).stem for n in zf.namelist() if "/pcd/" in n and n.endswith(".npy"))


def _member(zf, kind, fid, ext):
    return f"{zf.seq}/{kind}/{fid}{ext}"


def read_points(zf, fid):
    with zf.open(_member(zf, "pcd", fid, ".npy")) as f:
        return np.ascontiguousarray(np.load(io.BytesIO(f.read()))[:, :3], np.float32)


def read_labels(zf, fid):
    """-> (boxes (K,7) float32, raw class names list)"""
    try:
        txt = zf.read(_member(zf, "lbl", fid, ".txt")).decode()
    except KeyError:
        return np.zeros((0, 7), np.float32), []
    rows = [l.split() for l in txt.splitlines() if l.strip()]
    return np.array([[float(v) for v in r[:7]] for r in rows], np.float32).reshape(-1, 7), [r[7] for r in rows]
