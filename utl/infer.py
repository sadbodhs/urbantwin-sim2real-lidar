"""Run a trained OpenPCDet model on a directory of frames and write submission-style predictions.json.

    scripts/docker_run.sh python -m utl.infer --cfg configs/pcdet/pp_lumpi.yaml --ckpt output/.../checkpoint_epoch_40.pth \
        --frames data/pcdet/lumpi_real_val/points --out output/lumpi/pred_val.json [--tta] [--ids ids.txt]

Frames may be .npy (Nx>=3) or .bin (float32 Nx3, or Nx4 when --bin_dims 4). TTA averages the original, x-flip,
y-flip and xy-flip views by weighted box fusion; --rot adds yaw rotations of those views (e.g. --rot 22.5 -22.5).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from . import ground, metrics
from .fusion import wbf
from .pcdet_ext import register


class FrameDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, files, bin_dims=3, logger=None, ground_ref=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=Path("."), logger=logger)
        self.files, self.bin_dims = files, bin_dims
        self.ground_ref, self.align = ground_ref, {}

    def __len__(self):
        return len(self.files)

    def load(self, i):
        p = self.files[i]
        pts = np.load(p) if p.suffix == ".npy" else np.fromfile(p, np.float32).reshape(-1, self.bin_dims)
        pts = pts[:, :3].astype(np.float32)
        return pts[np.abs(pts).sum(1) > 0]  # V2X-Real encodes no-returns as (0, 0, 0)

    def alignment(self, i, pts):
        """(R, t) mapping this frame's ground plane onto the training data's (see utl.ground), cached per frame."""
        if i not in self.align:
            self.align[i] = ground.alignment(ground.fit_plane(pts), self.ground_ref)
        return self.align[i]

    def __getitem__(self, i, flip=(False, False), yaw=0.0):
        pts = self.load(i)
        if self.ground_ref is not None:
            pts = ground.apply_points(pts, *self.alignment(i, pts)).astype(np.float32)
        if yaw:
            c, s = np.cos(yaw), np.sin(yaw)
            pts[:, :2] = np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], 1).astype(np.float32)
        if flip[0]:
            pts[:, 1] = -pts[:, 1]
        if flip[1]:
            pts[:, 0] = -pts[:, 0]
        return self.prepare_data({"points": pts, "frame_id": self.files[i].stem})


def unflip(boxes, flip):
    b = boxes.copy()
    if flip[0]:  # mirrored across the x axis (y -> -y)
        b[:, 1] = -b[:, 1]
        b[:, 6] = -b[:, 6]
    if flip[1]:  # mirrored across the y axis (x -> -x)
        b[:, 0] = -b[:, 0]
        b[:, 6] = np.pi - b[:, 6]
    return b


def corners_inside(boxes, pc_range):
    """Per box: True when all 8 corners lie inside POINT_CLOUD_RANGE (x0, y0, z0, x1, y1, z1). Rotated views (--rot) count for a
    box only when this holds in that view's frame (fix Nov 8): near the range corners a rotated view crops the
    box's points and clamps its RoI, so it must not be averaged in."""
    x0, y0, z0, x1, y1, z1 = pc_range
    b = np.asarray(boxes, np.float64).reshape(-1, 7)
    c, s = np.abs(np.cos(b[:, 6])), np.abs(np.sin(b[:, 6]))
    ex, ey, ez = c * b[:, 3] / 2 + s * b[:, 4] / 2, s * b[:, 3] / 2 + c * b[:, 4] / 2, b[:, 5] / 2
    return ((b[:, 0] - ex >= x0) & (b[:, 0] + ex <= x1) & (b[:, 1] - ey >= y0) & (b[:, 1] + ey <= y1)
            & (b[:, 2] - ez >= z0) & (b[:, 2] + ez <= z1))


def to_view(boxes, flip, yaw, align=None):
    """Sensor-frame boxes -> the frame of view (flip, yaw): ground alignment if any, rotate by yaw, then flip."""
    b = np.array(boxes, np.float64).reshape(-1, 7)
    if align is not None:
        R, t = align
        b[:, :3] = b[:, :3] @ R.T + t
        head = np.stack([np.cos(b[:, 6]), np.sin(b[:, 6]), np.zeros(len(b))], 1) @ R.T
        b[:, 6] = np.arctan2(head[:, 1], head[:, 0])
    return unflip(unrotate(b, -yaw), flip)


def unrotate(boxes, yaw):
    """Undo a yaw rotation of the points: boxes back to the original frame."""
    if not yaw:
        return boxes
    b = boxes.copy()
    c, s = np.cos(-yaw), np.sin(-yaw)
    b[:, 0], b[:, 1] = c * boxes[:, 0] - s * boxes[:, 1], s * boxes[:, 0] + c * boxes[:, 1]
    b[:, 6] = boxes[:, 6] - yaw
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ids", help="optional file with frame ids (stems) to run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin_dims", type=int, default=3)
    ap.add_argument("--score_thresh", type=float, default=0.01)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--multi_class_nms", action="store_true",
                    help="NMS per class (every class keeps its own boxes) instead of class-agnostic NMS on argmax labels")
    ap.add_argument("--nms_thresh", type=float, help="override the BEV NMS IoU threshold of the config")
    ap.add_argument("--max_dets", type=int, help="override NMS_POST_MAXSIZE (per class with --multi_class_nms)")
    ap.add_argument("--ground_ref", help="JSON with the training data's ground plane (python -m utl.ground); "
                    "each frame is aligned to it before detection and boxes are mapped back")
    ap.add_argument("--rot", nargs="*", type=float, default=[], help="extra yaw views in degrees, each with the --tta flips")
    ap.add_argument("--one_per_view", action="store_true",
                    help="fusion fix F2 (opt-in, Nov 8): each view adds only its best box per cluster to the fused score")
    ap.add_argument("--save_views", action="store_true", help="also write each view's boxes (sensor frame) to <out stem>_views.json")
    a = ap.parse_args()

    register()
    cfg_from_yaml_file(a.cfg, cfg)
    # anchor heads read MODEL.POST_PROCESSING; CenterPoint's CenterHead keeps its own score threshold and NMS
    head_pp = cfg.MODEL.DENSE_HEAD.get("POST_PROCESSING")
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = a.score_thresh
    if head_pp is not None:
        head_pp.SCORE_THRESH = a.score_thresh
    nms = (head_pp or cfg.MODEL.POST_PROCESSING).NMS_CONFIG
    anchor_head = cfg.MODEL.DENSE_HEAD.NAME.startswith("Anchor")
    if a.multi_class_nms and not anchor_head:
        raise SystemExit("--multi_class_nms applies to anchor heads only (CenterHead already runs NMS per head)")
    if a.multi_class_nms:
        nms.MULTI_CLASSES_NMS = True
    if a.nms_thresh is not None:
        nms.NMS_THRESH = a.nms_thresh
    if a.max_dets is not None:
        nms.NMS_POST_MAXSIZE = a.max_dets
    logger = common_utils.create_logger()
    files = sorted(p for p in Path(a.frames).iterdir() if p.suffix in (".npy", ".bin"))
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        files = [p for p in files if p.stem in keep]
        miss = keep - {p.stem for p in files}
        assert not miss, f"--ids: {len(miss)} ids have no frame in {a.frames}, e.g. {sorted(miss)[:3]}"   # fail loudly
    ground_ref = np.array(json.loads(Path(a.ground_ref).read_text())["abc"]) if a.ground_ref else None
    ds = FrameDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, files, a.bin_dims, logger, ground_ref)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    model.load_params_from_file(filename=a.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()
    if a.multi_class_nms:
        # OpenPCDet 233f849 maps single-head scores to labels with arange(1, num_class), one class short, and asserts;
        # hand the head's scores over as a one-element multi-head list with the full mapping 1..C instead
        n_cls = len(cfg.CLASS_NAMES)

        def as_multihead(module, inputs, out):
            if not isinstance(out["batch_cls_preds"], list):
                out["batch_cls_preds"] = [out["batch_cls_preds"]]
                out["multihead_label_mapping"] = [torch.arange(1, n_cls + 1, device=out["batch_box_preds"].device)]
            return out
        model.dense_head.register_forward_hook(as_multihead)
    x0, y0, _, x1, y1, _ = pcr = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    flips = [(False, False), (True, False), (False, True), (True, True)] if a.tta else [(False, False)]
    views = [(fl, np.radians(r)) for r in [0.0] + a.rot for fl in flips]
    out, per_view = {}, {}
    # G3 (DSVT NaN fix): for configs with IOU_REG_LOSS true (DSVT, Voxel Mamba) every view drops
    # boxes with a non-finite value or any size (l, w, h) above 50 m before fusion or writing (their training head can leave background size logits
    # near exp overflow). Other configs: no change.
    g3 = bool(cfg.MODEL.DENSE_HEAD.get("IOU_REG_LOSS", False)) if "DENSE_HEAD" in cfg.MODEL else False
    g3_drop, g3_max = 0, 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            per_class = {c: [] for c in cfg.CLASS_NAMES}
            vrec = []
            for fl, yaw in views:
                batch = ds.collate_batch([ds.__getitem__(i, fl, yaw)])
                load_data_to_gpu(batch)
                pd = model(batch)[0][0]
                raw = pd["pred_boxes"].cpu().numpy()
                s, l = pd["pred_scores"].cpu().numpy(), pd["pred_labels"].cpu().numpy()
                if g3:
                    ok = np.isfinite(raw).all(1) & np.isfinite(s) & (raw[:, 3:6] <= 50.0).all(1)
                    g3_drop += int((~ok).sum())
                    raw, s, l = raw[ok], s[ok], l[ok]
                    if len(raw):
                        g3_max = max(g3_max, float(raw[:, 3:6].max()))
                if yaw:   # a rotated view counts for a box only if the box lies wholly inside the range in that view
                    m_in = corners_inside(raw, pcr)
                    raw, s, l = raw[m_in], s[m_in], l[m_in]
                b = unrotate(unflip(raw, fl), yaw)
                if ground_ref is not None:
                    b = ground.boxes_back(b, *ds.align[i])
                for k, c in enumerate(cfg.CLASS_NAMES):
                    m = l == k + 1
                    per_class[c].append((b[m], s[m]))
                if a.save_views:
                    vrec.append([{"box": [float(v) for v in bb], "score": float(ss), "class": cfg.CLASS_NAMES[int(ll) - 1]} for bb, ss, ll in zip(b, s, l)])
            dets = []
            if a.rot:   # score = sum / the number of views that count for the fused box (flip views at yaw 0 always count)
                al = ds.align[i] if ground_ref is not None else None
                nv = lambda bb: len(flips) + sum(int(corners_inside(to_view(bb, fl, yaw, al), pcr)[0]) for fl, yaw in views if yaw)
            for c, vs in per_class.items():
                b, s = wbf(vs, n_views=nv if a.rot else len(views), one_per_view=a.one_per_view) if len(views) > 1 else vs[0]
                for bb, ss in zip(b, s):
                    if x0 <= bb[0] <= x1 and y0 <= bb[1] <= y1:
                        dets.append({"box": [float(v) for v in bb], "score": float(ss), "class": c})
            out[files[i].stem] = sorted(dets, key=lambda d: -d["score"])  # an evaluator may cap boxes per frame
            if a.save_views:
                per_view[files[i].stem] = {"views": [[bool(fl[0]), bool(fl[1]), float(np.degrees(yaw))] for fl, yaw in views], "boxes": vrec}
            if i % 50 == 0:
                logger.info(f"{i}/{len(ds)} {files[i].stem}: {len(dets)} boxes")
    if g3:
        logger.info(f"G3 (IOU_REG_LOSS config): dropped {g3_drop} non-finite or > 50 m boxes over all views of {a.out}; max size kept {g3_max:.2f} m")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    logger.info(f"wrote {a.out}")
    if a.save_views:
        vp = Path(a.out).with_name(Path(a.out).stem + "_views.json")
        vp.write_text(json.dumps(per_view))
        logger.info(f"wrote {vp} (per-view boxes in the sensor frame, before fusion)")


if __name__ == "__main__":
    main()
