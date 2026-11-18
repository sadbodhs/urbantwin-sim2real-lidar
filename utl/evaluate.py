"""Score predictions.json against a labelled frame set (OpenPCDet layout) with the re-implemented challenge metric.

    python -m utl.evaluate --track lumpi --pred output/lumpi/pred_val.json --root data/pcdet/lumpi_real --split val \
        [--realism output/lumpi/realism_v1/realism_report.json] [--out output/lumpi/eval_val.json]
"""
import argparse
import json
from pathlib import Path

from . import metrics
from .tracks import TRACKS


def load_gt(root, split, classes):
    root = Path(root)
    ids = [l.strip() for l in (root / "ImageSets" / f"{split}.txt").read_text().splitlines() if l.strip()]
    gt = {}
    for i in ids:
        rows = [l.split() for l in (root / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
        gt[i] = [([float(v) for v in r[:7]], r[7]) for r in rows if r[7] in classes]
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--pred", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ids", help="optional file restricting the evaluated frames")
    ap.add_argument("--realism", help="realism_report.json of the synthetic frames, for the local combined score")
    ap.add_argument("--out")
    a = ap.parse_args()
    classes = TRACKS[a.track]["classes"]
    gt = load_gt(a.root, a.split, classes)
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        gt = {k: v for k, v in gt.items() if k in keep}
    pred = json.loads(Path(a.pred).read_text())
    missing = [k for k in gt if k not in pred]
    assert not missing, f"{len(missing)} evaluated frames have no predictions, e.g. {missing[:3]}"
    res = metrics.detection_scores(pred, gt, classes)
    out = {"frames": len(gt), "detection": res}
    for thr, r in res.items():
        print(f"IoU {thr}: mAP40 {r['mAP40']:.4f}  mAP11 {r['mAP11']:.4f}   ({len(gt)} frames)")
        for c, v in r["per_class"].items():
            if v["n_gt"]:
                print(f"    {c:<11s} AP40 {v['ap40']:.4f}  n_gt {v['n_gt']}")
    if a.realism:
        rs = json.loads(Path(a.realism).read_text())["calibrated"]["score"]
        out["combined"] = metrics.combined(res[0.5]["mAP40"], rs)
        print("local combined:", {k: round(v, 4) for k, v in out["combined"].items()})
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
