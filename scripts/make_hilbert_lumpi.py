"""Hilbert-curve templates for Voxel Mamba (made for the LUMPI grid; the V2X config uses the same files), with the repo's own code:
convert_to_index from tools/hilbert_curves/create_hilbert_curve_template.py (vendored unchanged in utl/ext3p/voxel_mamba/). The repo's __main__ builds
every (z, y, x) of a 2^bit cube as a Python list (z outer, x inner), converts it and keeps the first 2^bit * 2^bit * z_max entries. Each row is
converted on its own, so here the same rows (z < z_max only, same order, built with torch.arange) go through the same function; checked against the
repo's own construction for rank 7 (exact). CPU (the repo uses its input's device).
z_max per rank (must exceed the largest z index + block shift the backbone looks up; vmamba_lumpi grid 250 x 250 x 40, sparse z 41):
  rank 9 (stride 1, z <= 40 + shift 1): 42 (repo: 41, for Waymo z 33)
  rank 8 (stride 2 in stage 2, z <= 9 + shift 3): 17 (as the repo)
  rank 7 (stride 4 in stage 3, z <= 2 + shift 5): 9 (as the repo)
x / y: 2^9 = 512 >= 250 + 1, 2^8 = 256 >= 125 + 3, 2^7 = 128 >= 63 + 5.
    python scripts/make_hilbert_lumpi.py   (image urbantwin-lidar:mamba)"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from utl.ext3p.voxel_mamba.create_hilbert_curve_template import convert_to_index

OUT = Path("data/hilbert_lumpi")
ZMAX = {9: 42, 8: 17, 7: 9}


def locs(bit, zmax):
    n = 2 ** bit
    z, y, x = torch.meshgrid(torch.arange(zmax), torch.arange(n), torch.arange(n), indexing="ij")
    return torch.stack([z.reshape(-1), y.reshape(-1), x.reshape(-1)], 1)


# check against the repo's own construction (rank 7: full cube as a Python list, then truncated)
n7 = 2 ** 7
ref = convert_to_index(torch.tensor([(z, y, x) for z in range(n7) for y in range(n7) for x in range(n7)]), 3, 7)[: n7 * n7 * ZMAX[7]]
new = convert_to_index(locs(7, ZMAX[7]), 3, 7)
assert torch.equal(ref, new), "rows built with arange do not give the repo's template"
print("rank 7: arange rows give the repo's template exactly (%d entries)" % len(new), flush=True)
OUT.mkdir(parents=True, exist_ok=True)
for bit, zmax in ZMAX.items():
    t = convert_to_index(locs(bit, zmax), 3, bit)
    torch.save(t, OUT / f"curve_template_3d_rank_{bit}.pth")
    print("rank", bit, "z_max", zmax, "entries", len(t), "->", OUT / f"curve_template_3d_rank_{bit}.pth", flush=True)
