"""M3' step 1: walker-inserted synthetic val frames for the learned rescorer (synthetic data only).
For each of the 1,000 synthetic val frames per track: insert walkers with the training code and the walker models'
settings (utl.synth_ped.insert: N 5-25, limbs, groups 0.4, occlusion, area mix 0.5; LUMPI five-sensor rig, V2X Ouster
rigs), then thin uniformly to real-scan density (LUMPI 50,000 points as the kit frames; V2X 35,000 returns, the
non-zero share of a 50k real frame). Labels: the frame's synthetic boxes plus the inserted walkers.
Writes /work/r3/<track>/points/<id>.npy (float32 x, y, z) and /work/r3/<track>/labels.json. Seeded per frame.
    python /work/r3_make.py
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from utl.synth_ped import insert, lumpi_rig  # noqa: E402

W = Path("/work/r3")
TRACKS = {"lumpi": dict(src="ut_lumpi_fix", name="Person", rig="lumpi", n_keep=50000),
          "v2x": dict(src="ut_v2x", name="pedestrian", rig=None, n_keep=35000)}


def one(args):
    track, fid, boxes, names = args
    t = TRACKS[track]
    rng = np.random.default_rng(int(fid) * 7 + (1 if track == "lumpi" else 2))
    pts = np.load(W / "src" / t["src"] / f"{fid}.npy")[:, :3].astype(np.float32)
    rig = lumpi_rig() if t["rig"] == "lumpi" else None
    pts, boxes, names = insert(pts, np.asarray(boxes, np.float64).reshape(-1, 7), list(names), rng, (5, 25),
                               name=t["name"], rig=rig, body="limbs", group_prob=0.4, occlude=True, area_mix=0.5)
    pts = pts[:, :3]
    if len(pts) > t["n_keep"]:
        pts = pts[rng.choice(len(pts), t["n_keep"], replace=False)]
    out = W / track / "points"
    np.save(out / f"{fid}.npy", pts.astype(np.float32))
    return fid, [[list(map(float, b)), str(n)] for b, n in zip(boxes, names)]


def main():
    for track, t in TRACKS.items():
        (W / track / "points").mkdir(parents=True, exist_ok=True)
        infos = pickle.load(open(f"data/pcdet/{t['src']}/infos_val.pkl", "rb"))
        tasks = [(track, i["point_cloud"]["lidar_idx"], i["annos"]["gt_boxes_lidar"].tolist(), list(i["annos"]["name"]))
                 for i in infos]
        with ProcessPoolExecutor(16) as ex:
            labels = dict(ex.map(one, tasks, chunksize=8))
        (W / track / "labels.json").write_text(json.dumps(labels))
        n_walk = sum(n == t["name"] for v in labels.values() for _, n in v)
        print(track, len(labels), "frames,", n_walk, "inserted", t["name"], flush=True)


if __name__ == "__main__":
    main()
