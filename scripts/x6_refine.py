"""X6c: Voxel R-CNN's second stage (X6, vrcnn_v2x) refines our own final V2X boxes instead of its own proposals.
A forward hook on the dense head replaces its RoIs with our boxes (OpenPCDet 233f849: RoIHeadTemplate.proposal_layer
returns early when batch_dict has rois), a hook on the RoI head keeps the refined boxes and sigmoid(cls), the
predicted 3D IoU (CLS_SCORE_TYPE roi_iou). No NMS: one output per input box. 4-view TTA (x / y / xy flips), boxes
averaged over views, heading from the unflipped view. Pass 2 refines the pass-1 boxes again. Boxes below --min_score
are copied unchanged. Each refined detection gains ref1 / iou1 / ref2 / iou2.
--rot (plan W): extra yaw views of each flip view, as in utl.infer (e.g. --rot 22.5 -22.5 gives
12 views); boxes averaged over the views that count, heading from the unflipped, unrotated view. Without --rot nothing changes.
A rotated view counts for a box only if all 8 corners of the box (the RoI fed in) lie inside POINT_CLOUD_RANGE in that view's frame
(fix Nov 8: near the range corners a rotated view crops the points and clamps the RoI); the 4 flip views at yaw 0
always count. Box and predicted IoU are averaged over the counted views only; the share of boxes with fewer counted views is logged.
    scripts/docker_run.sh python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x.yaml \
        --ckpt output/pcdet/vrcnn_v2x/default/ckpt/checkpoint_epoch_20.pth --frames data/pcdet/v2x_real_50k/points \
        --pred output/v2x/ens/v12_val_i1.json --out output/v2x/vrcnn_v2x/x6c/v12_val_i1_ref.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from utl import ground
from utl.infer import FrameDataset, corners_inside, unflip, unrotate
from utl.pcdet_ext import register

FLIPS = [(False, False), (True, False), (False, True), (True, True)]
CHUNK = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin_dims", type=int, default=3)
    ap.add_argument("--min_score", type=float, default=0.05)
    ap.add_argument("--ground_ref")
    ap.add_argument("--only_class", help="keep and refine only the boxes of this class")
    ap.add_argument("--as_class", help="run those boxes through this class's refine outputs; the class field stays")
    ap.add_argument("--rot", nargs="*", type=float, default=[], help="extra yaw views in degrees, each with the 4 flips (plan W)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (identity checks)")
    a = ap.parse_args()

    register()
    cfg_from_yaml_file(a.cfg, cfg)
    logger = common_utils.create_logger()
    pred = json.loads(Path(a.pred).read_text())
    if a.only_class:
        pred = {f: [d for d in v if d["class"] == a.only_class] for f, v in pred.items()}
    files = sorted(p for p in Path(a.frames).iterdir() if p.suffix in (".npy", ".bin") and p.stem in pred)
    if a.limit:
        files = files[:a.limit]
    else:   # fail loudly on missing frames: every frame of --pred must be refined and written
        miss = set(pred) - {p.stem for p in files}
        assert not miss, f"{len(miss)} frames of {a.pred} are not in {a.frames}, e.g. {sorted(miss)[:3]}"
        assert files, f"no frames of {a.pred} in {a.frames}"
    views = [(fl, np.radians(r)) for r in [0.0] + a.rot for fl in FLIPS]
    ground_ref = np.array(json.loads(Path(a.ground_ref).read_text())["abc"]) if a.ground_ref else None
    ds = FrameDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, files, a.bin_dims, logger, ground_ref)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    model.load_params_from_file(filename=a.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()
    label = {c: k + 1 for k, c in enumerate(cfg.CLASS_NAMES)}

    state = {"short": [0, 0], "boxes": [0, 0]}   # per pass: boxes with fewer counted views than len(views), boxes refined
    pcr = cfg.DATA_CONFIG.POINT_CLOUD_RANGE

    def set_rois(module, inputs, out):
        r = state["rois"]
        out["rois"] = r["rois"]
        out["roi_scores"] = r["scores"]
        out["roi_labels"] = r["labels"]
        out["has_class_labels"] = True
        return out

    def keep_refined(module, inputs, out):
        state["box"] = out["batch_box_preds"][0, :, :7].detach().cpu().numpy()
        state["iou"] = torch.sigmoid(out["batch_cls_preds"][0].reshape(-1)).detach().cpu().numpy()
        return out

    model.dense_head.register_forward_hook(set_rois)
    model.roi_head.register_forward_hook(keep_refined)

    def refine(i, boxes, scores, labels, pas=0):
        """boxes in the sensor frame -> (refined boxes in the sensor frame, predicted IoU), TTA-averaged."""
        views_b, views_i, views_w = [], [], []
        for fl, yaw in views:
            batch = ds.collate_batch([ds.__getitem__(i, fl, yaw)] if yaw else [ds.__getitem__(i, fl)])  # sets ds.align[i] when ground_ref is given
            b = boxes.copy()
            if ground_ref is not None:
                R, t = ds.align[i]
                b[:, :3] = b[:, :3] @ R.T + t
                head = np.stack([np.cos(b[:, 6]), np.sin(b[:, 6]), np.zeros(len(b))], 1) @ R.T
                b[:, 6] = np.arctan2(head[:, 1], head[:, 0])
            b = unflip(unrotate(b, -yaw), fl)  # into the view: rotate by yaw, then flip (a flip is its own inverse)
            views_w.append(corners_inside(b, pcr) if yaw else np.ones(len(b), bool))
            out_b, out_i = [], []
            for s in range(0, len(b), CHUNK):
                load_data_to_gpu(batch)
                sl = slice(s, s + CHUNK)
                state["rois"] = {
                    "rois": torch.from_numpy(b[sl]).float().cuda()[None],
                    "scores": torch.from_numpy(scores[sl]).float().cuda()[None],
                    "labels": torch.from_numpy(labels[sl]).long().cuda()[None]}
                bb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
                model(bb)
                out_b.append(state["box"])
                out_i.append(state["iou"])
            rb = unrotate(unflip(np.concatenate(out_b), fl), yaw)
            if ground_ref is not None:
                rb = ground.boxes_back(rb, *ds.align[i])
            views_b.append(rb)
            views_i.append(np.concatenate(out_i))
        vb = np.stack(views_b)
        if not a.rot:
            mean = vb.mean(0)
            mean[:, 6] = vb[0, :, 6]
            return mean, np.mean(views_i, 0)
        w = np.stack(views_w).astype(np.float64)          # (views, boxes); view 0 always counts, so w.sum(0) >= 4
        state["short"][pas] += int((w.sum(0) < len(views)).sum())
        state["boxes"][pas] += w.shape[1]
        mean = (vb * w[:, :, None]).sum(0) / w.sum(0)[:, None]
        mean[:, 6] = vb[0, :, 6]
        return mean, (np.stack(views_i) * w).sum(0) / w.sum(0)

    out = {}
    with torch.no_grad():
        for i, p in enumerate(files):
            dets = [dict(d) for d in pred[p.stem]]
            idx = [k for k, d in enumerate(dets) if d["score"] >= a.min_score and d["class"] in label]
            if idx:
                boxes = np.array([dets[k]["box"][:7] for k in idx], np.float64)
                scores = np.array([dets[k]["score"] for k in idx], np.float32)
                labels = np.array([label[a.as_class or dets[k]["class"]] for k in idx], np.int64)
                r1, i1 = refine(i, boxes, scores, labels)
                r2, i2 = refine(i, r1, scores, labels, 1)
                for n, k in enumerate(idx):
                    dets[k].update(ref1=[float(v) for v in r1[n]], iou1=float(i1[n]),
                                   ref2=[float(v) for v in r2[n]], iou2=float(i2[n]))
            out[p.stem] = dets
            if i % 20 == 0:
                logger.info(f"{i}/{len(files)} {p.stem}: refined {len(idx)} of {len(dets)}")
    if a.rot:
        for pas in (0, 1):
            n, k = state["boxes"][pas], state["short"][pas]
            logger.info(f"pass {pas + 1}: boxes with fewer than {len(views)} counted views: {k} of {n} ({100.0 * k / max(n, 1):.2f}%)")
    if not a.limit:
        assert set(out) == set(pred), "refined output does not cover every input frame"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    logger.info(f"wrote {a.out}")


if __name__ == "__main__":
    main()
