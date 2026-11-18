"""Build submission.zip: synthetic/0001..0050.bin (float32 Nx3, <= 1.5 MB), predictions.json, declaration.txt.

    python -m utl.package --track lumpi --pred output/lumpi/pred_test.json --synthetic output/lumpi/realism \
        --out submissions/lumpi_v1/submission.zip --name "Sadbodh Sharma"
"""
import argparse
import datetime
import json
import zipfile
from pathlib import Path

import numpy as np

from .tracks import DECLARATION, TRACKS, test_frame_ids

MAX_BYTES = 1_500_000


def validate(track, preds, syn_files):
    ids = test_frame_ids(track)
    missing = [i for i in ids if i not in preds]
    extra = [k for k in preds if k not in set(ids)]
    assert not missing, f"predictions missing frames {missing[:5]}"
    assert not extra, f"predictions has unknown frames {extra[:5]}"
    classes = {c.lower() for c in TRACKS[track]["classes"]}
    for f, dets in preds.items():
        for d in dets:
            assert len(d["box"]) == 7 and all(np.isfinite(d["box"])), (f, d)
            assert 0.0 <= d["score"] <= 1.0, (f, d)
            assert str(d["class"]).lower() in classes, (f, d)
    assert len(syn_files) == 50, f"need exactly 50 synthetic frames, got {len(syn_files)}"
    for p in syn_files:
        a = np.fromfile(p, np.float32)
        assert p.stat().st_size <= MAX_BYTES, f"{p} is {p.stat().st_size} bytes"
        assert a.size % 3 == 0 and np.isfinite(a).all(), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--pred", required=True)
    ap.add_argument("--synthetic", required=True, help="directory with 0001.bin .. 0050.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="name written under the declaration")
    a = ap.parse_args()

    preds = json.loads(Path(a.pred).read_text())
    # highest score first in every frame, in case the official evaluator keeps only the first boxes of a frame
    preds = {f: sorted(d, key=lambda x: -x["score"]) for f, d in preds.items()}
    syn = sorted(Path(a.synthetic).glob("*.bin"))
    validate(a.track, preds, syn)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    decl = f"{DECLARATION}\n\nName: {a.name}\nDate: {datetime.date.today().isoformat()}\n"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i, p in enumerate(syn, 1):
            z.write(p, f"synthetic/{i:04d}.bin")
        z.writestr("predictions.json", json.dumps(preds))
        z.writestr("declaration.txt", decl)
    n = sum(len(v) for v in preds.values())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(preds)} frames, {n} boxes, {len(syn)} synthetic frames")


if __name__ == "__main__":
    main()
