"""Forbidden-frame guard. Every real frame the code touches goes through `check` / `allowed`.

The rules forbid training, validating or tuning on the 100 frames in forbidden_frames.txt. Real LiDAR runs at
10 Hz, so a neighbouring frame is almost a copy of a forbidden one; frames within `BUFFER` of a forbidden frame
of the same sequence are refused too.
"""
import csv
import re
from functools import lru_cache
from pathlib import Path

from .tracks import TRACKS

BUFFER = 20  # frames (2 s at 10 Hz)
V2X_MAPPING = Path("data/raw/v2xreal/forbidden_frames_mapping.csv")   # the kit's forbidden ids -> V2X-Real scenario folders
_REPO = re.compile(r"^(?:train|val|test)_(.+)_i[12]_(\d+)$")       # repo ids, e.g. val_2023-03-17-16-03-02_11_1_i1_000000
_KIT = re.compile(r"^(\d+)_(\d+)$")                                 # kit ids, e.g. 12_000035 (scenario index, frame)


class ForbiddenFrame(RuntimeError):
    pass


@lru_cache(None)
def _v2x_folders():
    with open(V2X_MAPPING) as fh:
        return {r["scenario_index"]: r["scenario_folder"] for r in csv.DictReader(fh)}


def _key(track, frame_id):
    """Normalise to (sequence, index). LUMPI '006498' -> ('', 6498). V2X-Real: the scenario folder and frame, from repo ids
    ('val_2023-03-17-16-03-02_11_1_i1_000035' -> ('2023-03-17-16-03-02_11_1', 35); both infrastructure LiDARs share the folder's
    frame numbers) or kit ids through the kit's mapping ('12_000035' -> (folder of scenario 12, 35)). Any other V2X format raises
    (repo ids used to pass unchecked)."""
    s = str(frame_id).strip()
    if track == "lumpi":
        return "", int(s)
    m = _REPO.match(s)
    if m:
        return m.group(1), int(m.group(2))
    m = _KIT.match(s)
    if m:
        folders = _v2x_folders()
        if m.group(1) not in folders:
            raise ValueError(f"V2X kit id {s}: scenario {m.group(1)} is not in {V2X_MAPPING}")
        return folders[m.group(1)], int(m.group(2))
    raise ValueError(f"unknown V2X frame id format: {s!r}")


@lru_cache(None)
def forbidden(track):
    lines = (TRACKS[track]["kit"] / "forbidden_frames.txt").read_text().splitlines()
    return frozenset(_key(track, l) for l in lines if l.strip() and not l.startswith("#"))


def allowed(track, frame_id, buffer=BUFFER):
    seq, idx = _key(track, frame_id)
    return not any(s == seq and abs(i - idx) <= buffer for s, i in forbidden(track))


def check(track, frame_id, buffer=BUFFER):
    if not allowed(track, frame_id, buffer):
        raise ForbiddenFrame(f"{track} frame {frame_id} is (or is next to) a forbidden frame")
