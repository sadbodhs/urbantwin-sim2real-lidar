"""Real LUMPI (Measurement 4) reader: merged multi-sensor .ply frames and the track-CSV labels.

Label CSV columns: time, object id, BEV rect (4), score, class_id, visibility, x, y, z, length, width, height,
heading, [optional shape values]. Frame index = floor(time * 10). class_id follows the track class order
(0 Person, 1 Car, 2 Bicycle, 3 Motorcycle, 4 Bus, 5 Truck, 6 Van, 7 Unknown). Boxes are already in the
OpenPCDet convention (centre, full extents, heading about +Z from +X).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import guard
from .tracks import TRACKS

CLASSES = TRACKS["lumpi"]["classes"]


def read_ply(src, fields=("x", "y", "z")):
    """src: a path or a binary file object."""
    from plyfile import PlyData
    v = PlyData.read(src if hasattr(src, "read") else str(src))["vertex"]
    return np.stack([np.asarray(v[f], np.float32) for f in fields], 1)


class LidarZip:
    """Measurement<N>/lidar.zip.NNN (raw byte chunks of one ZIP64 archive) read in place."""

    def __init__(self, zip_dir):
        from .ut import open_zip
        self.zf = open_zip(zip_dir, prefix="lidar")
        self.members = {Path(n).stem: n for n in self.zf.namelist() if n.endswith(".ply")}

    def __len__(self):
        return len(self.members)

    def read(self, idx, fields=("x", "y", "z")):
        import io
        return read_ply(io.BytesIO(self.zf.read(self.members[f"{idx:06d}"])), fields)


def load_labels(csv_path):
    """-> {frame_index: float32 array (K, 8) = [x, y, z, l, w, h, heading, class_id]}"""
    out = defaultdict(list)
    with open(csv_path) as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            f = int(np.floor(float(row[0]) * 10 + 1e-6))
            out[f].append([float(row[i]) for i in (9, 10, 11, 12, 13, 14, 15)] + [float(row[7])])
    return {f: np.asarray(v, np.float32) for f, v in out.items()}


def frame_path(lidar_dir, idx):
    return Path(lidar_dir) / f"{idx:06d}.ply"


def val_frames(n_total, stride=20, max_index=6400):
    """Real validation frames: every `stride`-th frame below `max_index`, never near a forbidden frame.

    All forbidden frames lie in 6497..8109, so the default window keeps validation well away from them.
    """
    ids = [i for i in range(0, min(n_total, max_index), stride)]
    for i in ids:
        guard.check("lumpi", f"{i:06d}")
    return ids
