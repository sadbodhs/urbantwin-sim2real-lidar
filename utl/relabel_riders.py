"""V2X training labels with person-sized rider boxes.

In UT-V2X the class 'pedestrian' holds only riders (BicycleRider, MotorcyleRider; no walkers), boxed together with
their two-wheeler (median 1.84 x 0.80 m), while real V2X-Real pedestrian boxes are person-sized. Every pedestrian box
longer than --max_len is shrunk to a person footprint (--size, about the rider's torso) at the same centre and
height; the bike's points stay in the frame as background. Points are linked, labels rewritten, splits copied.

    python -m utl.relabel_riders --src data/pcdet/ut_v2x --out data/pcdet/ut_v2x_ped
"""
import argparse
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=float, default=1.2)
    ap.add_argument("--size", type=float, nargs=2, default=[0.75, 0.75])
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if not (out / "points").exists():
        os.symlink(os.path.relpath(src / "points", out), out / "points")
    shutil.copytree(src / "ImageSets", out / "ImageSets", dirs_exist_ok=True)
    n_box = n_changed = 0
    for f in sorted((src / "labels").glob("*.txt")):
        rows = []
        for line in f.read_text().splitlines():
            r = line.split()
            if not r:
                continue
            if r[7] == "pedestrian":
                n_box += 1
                if float(r[3]) > a.max_len:
                    r[3], r[4] = f"{a.size[0]:.3f}", f"{a.size[1]:.3f}"
                    n_changed += 1
            rows.append(" ".join(r))
        (out / "labels" / f.name).write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"{n_changed} of {n_box} pedestrian boxes shrunk to {a.size} -> {out}")


if __name__ == "__main__":
    main()
