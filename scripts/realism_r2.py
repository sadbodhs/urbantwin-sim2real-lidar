"""Realism plan R2, learned refiner of the 50 realism frames.
This relaxes our own rule for realism only: unlabelled allowed real frames (fit split, through the forbidden-frame guard) train this refiner.
The detector stays synthetic-only; no real labels are read; check frames only score. Settings fixed before any reading:
  Start pool: R1's pool if R1 passed on this track, else the uploaded pool.
  Model: per-point displacement. Per point, its 16 nearest neighbours in the same frame give 16 rows [point xyz / 40, neighbour offset (m)]
         -> shared MLP 64-128-128 (ReLU) -> max over the 16 -> MLP 128-64-3 -> tanh x 1.0 m. Every output point is a twin point moved <= 1 m.
  Training (unpaired): twin frames of the pool's source set except its 50 pool frames, prepared as the pool was (R1's transform when R1 passed,
         range mask, utl.realism_emit's calibration as one set; no zero rows) against the fit frames (no-return rows removed, range mask).
         Loss per pair: Sinkhorn divergence (geomloss, p = 1, blur 0.05 m) between a 10,000-point sample of the displaced twin frame and a
         10,000-point sample of a fit frame, + 0.1 x mean displacement length. Adam, LR 0.001, batch 4 frame pairs, 2,000 steps, seed 0.
  Smoke test: the mean loss of steps 181-200 is below that of steps 1-20; otherwise stop (exit 3), no fixes by trial.
  Apply: every non-zero point of the 50 start-pool frames; zero rows kept as they were. Score (Docker, "score" mode) as R1, 5 seeds, same gate
         against the start pool: realism +0.003 or more on every seed, no other term down by more than 0.001.
  Information only: share of refined points within 2 cm of any fit-frame point, and the same share for the check frames.
One try per track. Writes output/<track>/realism_r2/{0001..0050.bin, model.pt, r2_train.json, r2_report.json}.
    .venv-mac/bin/python scripts/realism_r2.py train v2x      (Apple GPU)
    python scripts/realism_r2.py score v2x                    (Docker CPU)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import realism_r1 as R1
from utl import metrics
from utl import realism_sweep as rs
from utl.realism import _load, nonzero
from utl.tracks import TRACKS

K, NS, STEPS, BATCH = 16, 10000, 2000, 4


def start_pool(track):
    r1 = Path(f"output/{track}/realism_r1/r1_report.json")
    rep1 = json.loads(r1.read_text())
    return (Path(f"output/{track}/realism_r1"), rep1) if rep1["gate"] else (Path(R1.CFG[track]["pool"]), rep1)


def features(f, q_idx, tree):
    """f (N, 3) frame, q_idx query points -> (Q, K, 6) rows [p / 40, neighbour - p]."""
    _, nn = tree.query(f[q_idx], K + 1)
    p = f[q_idx][:, None, :]
    off = f[nn[:, 1:]] - p
    return np.concatenate([np.broadcast_to(p / 40.0, off.shape), off], -1).astype(np.float32)


def train(track):
    import torch
    import torch.nn as nn
    from geomloss import SamplesLoss
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    c, pc = R1.CFG[track], TRACKS[track]["pc_range"]
    pool_dir, rep1 = start_pool(track)
    print(track, "start pool", pool_dir, "| R1 gate", rep1["gate"], "| device", dev, flush=True)
    rep = json.loads((Path(c["pool"]) / "realism_report.json").read_text())
    fit = [nonzero(f) for f in R1.real_frames(track, c, "fit")]
    rs._init(dict(pc=pc, fit=fit))
    srcs = sorted(p.name for p in Path(c["syn"]).glob("*.npy"))
    tr_names = [n for n in srcs if n not in set(rep["synthetic_sources"])]
    R, t = (np.array(rep1["R"]), np.array(rep1["t"])) if rep1["gate"] else (np.eye(3), np.zeros(3))
    raw = [nonzero(_load(Path(c["syn"]) / n)).astype(np.float64) for n in tr_names]
    frames = rs.calibrate([metrics.mask_range((f @ R.T + t).astype(np.float32), pc) for f in raw], fit,
                          rep["cfg"]["voxels"], rep["cfg"]["align"], rep["cfg"]["jitter"])
    frames = [nonzero(f) for f in frames]
    trees = [cKDTree(f) for f in frames]
    print("training frames", len(frames), "(source set %d minus the 50 pool frames) | fit frames %d" % (len(srcs), len(fit)), flush=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.b = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, x):          # (B, Q, K, 6) -> (B, Q, 3)
            return torch.tanh(self.b(self.a(x).max(-2).values)) * 1.0

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    S = SamplesLoss("sinkhorn", p=1, blur=0.05, backend="tensorized")
    hist, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        tot = 0.0
        opt.zero_grad()
        for _ in range(BATCH):
            i = int(rng.integers(len(frames)))
            f = frames[i]
            q = rng.choice(len(f), NS, replace=len(f) < NS)
            x = torch.from_numpy(features(f, q, trees[i])).to(dev)
            p = torch.from_numpy(f[q].astype(np.float32)).to(dev)
            y = fit[int(rng.integers(len(fit)))]
            y = torch.from_numpy(y[rng.choice(len(y), NS, replace=len(y) < NS)].astype(np.float32)).to(dev)
            d = net(x[None])[0]
            loss = (S(p + d, y) + 0.1 * d.norm(dim=1).mean()) / BATCH
            loss.backward()
            tot += float(loss)
        opt.step()
        hist.append(tot)
        if step % 50 == 0 or step <= 3:
            print("step %d loss %.4f | %.0f s" % (step, tot, time.time() - t0), flush=True)
        if step == 200:
            a, b = float(np.mean(hist[:20])), float(np.mean(hist[180:200]))
            ok = np.isfinite(b) and b < a
            print("SMOKE", "PASS" if ok else "FAIL", "steps 1-20 %.4f, steps 181-200 %.4f" % (a, b), flush=True)
            if not ok:
                sys.exit(3)
    out = Path(f"output/{track}/realism_r2")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "model.pt")
    net.eval()
    disp = []
    for k in range(1, 51):
        g = _load(pool_dir / f"{k:04d}.bin")
        nz = np.where(np.abs(g).sum(1) > 0)[0]
        f = g[nz]
        tree = cKDTree(f)
        ds = []
        with torch.no_grad():
            for s in range(0, len(f), 20000):
                q = np.arange(s, min(s + 20000, len(f)))
                ds.append(net(torch.from_numpy(features(f, q, tree)).to(dev)[None])[0].cpu().numpy())
        d = np.concatenate(ds)
        h = g.copy()
        h[nz] = f + d
        h.astype(np.float32).tofile(out / f"{k:04d}.bin")
        disp.append(float(np.linalg.norm(d, axis=1).mean()))
    (out / "r2_train.json").write_text(json.dumps(dict(track=track, start_pool=str(pool_dir), r1_gate=rep1["gate"], training_frames=len(frames),
        loss=hist, mean_displacement_per_frame=disp), indent=1))
    print("applied to the 50 start-pool frames | mean displacement %.3f m | wrote %s" % (np.mean(disp), out), flush=True)


def score(track):
    c, pc = R1.CFG[track], TRACKS[track]["pc_range"]
    pool_dir, _ = start_pool(track)
    rep = json.loads((Path(c["pool"]) / "realism_report.json").read_text())
    check = R1.real_frames(track, c, "check")
    if rep["zeros"] == "drop":
        check = [nonzero(f) for f in check]
    fit = [nonzero(f) for f in R1.real_frames(track, c, "fit")]
    out = Path(f"output/{track}/realism_r2")
    st = [_load(pool_dir / f"{k:04d}.bin") for k in range(1, 51)]
    new = [_load(out / f"{k:04d}.bin") for k in range(1, 51)]
    a, b = R1.per_seed(st, check, pc), R1.per_seed(new, check, pc)
    gain = [y["score"] - x["score"] for x, y in zip(a, b)]
    terms = {k: float(np.mean([y["norm"][k] - x["norm"][k] for x, y in zip(a, b)])) for k in a[0]["norm"]}
    for s, x, y in zip(R1.SEEDS, a, b):
        print("seed", s, "start pool %.4f -> R2 %.4f (%+.4f) | EMD %.3f -> %.3f" % (x["score"], y["score"], y["score"] - x["score"], x["raw"]["EMD"], y["raw"]["EMD"]), flush=True)
    mm = (R1.median_move(st, check, pc), R1.median_move(new, check, pc))
    g = bool(min(gain) >= 0.003 and all(v >= -0.001 for k, v in terms.items() if k != "EMD"))
    tree = cKDTree(np.concatenate(fit))
    rng = np.random.default_rng(0)
    share = lambda fr: float((tree.query((lambda p: p[rng.choice(len(p), min(len(p), 100000), replace=False)])(np.concatenate([nonzero(f) for f in fr])))[0] <= 0.02).mean())
    info = dict(refined=share(new), start_pool=share(st), check=share(check))
    print("mean realism %.4f -> %.4f | term changes" % (np.mean([x["score"] for x in a]), np.mean([y["score"] for y in b])),
          {k: round(v, 4) for k, v in terms.items()}, "| median transport distance (seed 1234) %.3f -> %.3f m" % mm, "| gate", "PASS" if g else "fail", flush=True)
    print("information: share of points within 2 cm of a fit-frame point:", {k: round(v, 4) for k, v in info.items()}, flush=True)
    (out / "r2_report.json").write_text(json.dumps(dict(track=track, start_pool=str(pool_dir), start=a, r2=b, gain_per_seed=gain, term_change=terms,
        median_move=mm, gate=g, within_2cm=info), indent=1))


if __name__ == "__main__":
    {"train": train, "score": score}[sys.argv[1]](sys.argv[2])
