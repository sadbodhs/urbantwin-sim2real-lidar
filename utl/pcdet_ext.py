"""OpenPCDet glue: a CustomDataset variant with Sim2Real point augmentations and the challenge metric.

Dataset layout (written by utl.prepare): <root>/points/<id>.npy (float32 Nx3), <root>/labels/<id>.txt
("x y z dx dy dz heading Class"), <root>/ImageSets/{train,val}.txt, <root>/infos_{split}.pkl,
<root>/gt_database/*.bin + <root>/dbinfos_train.pkl.
"""
import copy
import os
import pickle
from pathlib import Path

import numpy as np

from pcdet.datasets.custom.custom_dataset import CustomDataset
from pcdet.utils import common_utils

from . import metrics


def sim2real_augment(points, boxes, cfg, rng):
    """Sensor-gap augmentations on synthetic frames (train only).

    DROP_RANGE      keep a random fraction of all points (real scans are sparser and have ray losses)
    OBJ_DROP_PROB   with this probability per object, remove a random 0-60% of its interior points
    JITTER_STD      Gaussian xyz noise (m)
    Z_SHIFT         random global height offset applied to points and boxes (sensor-height mismatch)
    PED_INSERT      {PROB, N: [lo, hi], NAME, RIG}: ray-cast procedural walkers (utl.synth_ped), applied first;
                    RIG 'lumpi' = the five LUMPI sensors, otherwise a random V2X-Real infrastructure Ouster. Optional
                    (defaults keep the single cylinder walkers): BODY 'limbs', GROUP_PROB, OCCLUDE, AREA_MIX
                    (see utl.synth_ped.insert)
    CLUTTER         {PROB, N: [lo, hi]}: unlabelled ray-cast street furniture at random free places, after the walkers
                    (utl.synth_ped.clutter); LUMPI rig, or with RIG 'v2x' a random V2X-Real infrastructure Ouster
    """
    if cfg.get("OBJ_DROP_PROB", 0) > 0 and len(boxes):
        from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
        import torch
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, :3].astype(np.float32)), torch.from_numpy(boxes[:, :7].astype(np.float32))).numpy()
        keep = np.ones(len(points), bool)
        for i in np.where(rng.random(len(boxes)) < cfg.OBJ_DROP_PROB)[0]:
            idx = np.where(inside[i] > 0)[0]
            keep[idx[rng.random(len(idx)) < rng.uniform(0, 0.6)]] = False
        points = points[keep]
    if "DROP_RANGE" in cfg:
        lo, hi = cfg.DROP_RANGE
        points = points[rng.random(len(points)) < rng.uniform(lo, hi)]
    if cfg.get("JITTER_STD", 0) > 0:
        points = points.copy()
        points[:, :3] += rng.normal(0, cfg.JITTER_STD, (len(points), 3)).astype(points.dtype)
    if cfg.get("Z_SHIFT", 0) > 0:
        dz = rng.uniform(-cfg.Z_SHIFT, cfg.Z_SHIFT)
        points = points.copy()
        points[:, 2] += dz
        boxes = boxes.copy()
        boxes[:, 2] += dz
    return points, boxes


def stretch_objects(points, boxes, names, cfg, rng):
    """STRETCH {NAME, PROB, LEN: [lo, hi], HSCALE}: with probability PROB per NAME object of the scene, stretch it about
    its centre to a length ~ U(lo, hi) (height x U(1, HSCALE), bottom kept; utl.stretch_db.stretch). Skipped when the
    longer box would overlap another box. Scene points inside the new box are replaced by the stretched object."""
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    import torch
    from .stretch_db import stretch
    cand = np.where((names == cfg.NAME) & (rng.random(len(names)) < cfg.PROB))[0]
    if not len(cand):
        return points, boxes
    boxes = boxes.copy()
    for i in cand:
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, :3].astype(np.float32)), torch.from_numpy(boxes[i:i + 1, :7].astype(np.float32))).numpy()[0] > 0
        obj, new = stretch(points[inside, :3] - boxes[i, :3], boxes[i, :7].astype(np.float64),
                           rng.uniform(*cfg.LEN), rng.uniform(1.0, cfg.HSCALE))
        others = np.delete(boxes[:, :7], i, 0).astype(np.float32)
        if len(others) and iou3d_nms_utils.boxes_bev_iou_cpu(new[None], others).max() > 0:
            continue
        cover = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, :3].astype(np.float32)), torch.from_numpy(new[None])).numpy()[0] > 0
        obj = np.concatenate([obj + new[:3], np.zeros((len(obj), points.shape[1] - 3), points.dtype)], 1)
        points = np.concatenate([points[~(cover | inside)], obj.astype(points.dtype)])
        boxes[i, :7] = new
    return points, boxes


def ray_emulate(points, beams, n_az):
    """Keep the nearest return per (beam, azimuth step) ray of a sensor at the origin with the given beam elevations
    (deg, evenly spaced) and n_az azimuth steps; points outside every beam's half-spacing band are dropped."""
    xyz = points[:, :3].astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    ok = r > 1e-6
    e = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    step = beams[1] - beams[0]
    b = np.round((e - beams[0]) / step).astype(np.int64)
    ok &= (b >= 0) & (b < len(beams)) & (np.abs(e - beams[0] - b * step) <= step / 2)
    a = np.floor((np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi) / (2 * np.pi / n_az)).astype(np.int64) % n_az
    idx = np.where(ok)[0]
    ray = b[idx] * n_az + a[idx]
    order = np.lexsort((r[idx], ray))
    first = order[np.r_[True, np.diff(ray[order]) > 0]] if len(order) else order
    return points[np.sort(idx[first])]


class SimDataset(CustomDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        if "MAP_CLASS_TO_KITTI" not in dataset_cfg:
            dataset_cfg.MAP_CLASS_TO_KITTI = {}
        super().__init__(dataset_cfg, class_names, training=training, root_path=root_path, logger=logger)
        self.s2r = dataset_cfg.get("SIM2REAL", {})
        self._rng, self._rng_pid = None, None
        if training and any(k in self.s2r for k in ("TARGET_POINTS", "TWO_WHEEL_NEG", "SIZE_FLOOR", "RAY_EMUL")):
            self._wrap_augmentor()

    @property
    def rng(self):
        """One generator per data-loader worker (a generator created before the fork would repeat in every worker). Seeded from torch's
        per-worker seed (Nov 8, for runs started from then on): with --fix_random_seed that seed is fixed
        by (SEED, worker id, epoch), so the augmentation draws repeat run to run; before, the generator was unseeded."""
        if self._rng_pid != os.getpid():
            import torch
            self._rng, self._rng_pid = np.random.default_rng(torch.initial_seed() % (2 ** 63)), os.getpid()
        return self._rng

    def _wrap_augmentor(self):
        """Steps run after OpenPCDet's augmentor, so pasted GT-database objects get them too (train only):
        TWO_WHEEL_NEG {NAMES, MAX_H}  drop the labels of NAMES boxes lower than MAX_H m: riderless (parked) two-wheelers
                                      become unlabelled negatives (ridden UT-LUMPI bicycles are >= 1.6 m, riderless 1.26 m)
        SIZE_FLOOR {Class: [l, w, h]} grow boxes of a class to at least this public reference size, bottom kept
        RAY_EMUL {BEAMS: [lo, hi, n], N_AZ}  datasheet ray pattern of a sensor at the origin: n beams evenly over
                                      lo..hi deg, N_AZ azimuth steps, nearest return per ray (others removed)
        TARGET_POINTS [lo, hi]        keep a uniform random subset with N ~ log-uniform(lo, hi) points inside the
                                      track range. The official LUMPI test frames are cut to the range and uniformly
                                      thinned to 50,000 points, about a quarter of a full frame."""
        s2r = self.s2r
        x0, y0, z0, x1, y1, z1 = self.point_cloud_range
        forward = self.data_augmentor.forward

        def post(data_dict):
            data_dict = forward(data_dict)
            if "TWO_WHEEL_NEG" in s2r and len(data_dict.get("gt_boxes", [])):
                c = s2r.TWO_WHEEL_NEG
                keep = ~(np.isin(data_dict["gt_names"], list(c.NAMES)) & (data_dict["gt_boxes"][:, 5] < c.MAX_H))
                data_dict["gt_boxes"], data_dict["gt_names"] = data_dict["gt_boxes"][keep], data_dict["gt_names"][keep]
            if "SIZE_FLOOR" in s2r and len(data_dict.get("gt_boxes", [])):
                b = data_dict["gt_boxes"].copy()
                for name, lwh in s2r.SIZE_FLOOR.items():
                    m = data_dict["gt_names"] == name
                    if m.any():
                        bottom = b[m, 2] - b[m, 5] / 2
                        b[m, 3:6] = np.maximum(b[m, 3:6], np.asarray(lwh, b.dtype))
                        b[m, 2] = bottom + b[m, 5] / 2
                data_dict["gt_boxes"] = b
            if "RAY_EMUL" in s2r:
                lo, hi, n = s2r.RAY_EMUL.BEAMS
                data_dict["points"] = ray_emulate(data_dict["points"], np.linspace(lo, hi, int(n)), s2r.RAY_EMUL.N_AZ)
            if "TARGET_POINTS" in s2r:
                lo, hi = s2r.TARGET_POINTS
                pts = data_dict["points"]
                inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
                          & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)).sum()
                n = np.exp(self.rng.uniform(np.log(lo), np.log(hi)))
                if inside > n:
                    data_dict["points"] = pts[self.rng.random(len(pts)) < n / inside]
            return data_dict
        self.data_augmentor.forward = post

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)
        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info["point_cloud"]["lidar_idx"]
        points = self.get_lidar(sample_idx)
        input_dict = {"frame_id": sample_idx, "points": points}
        if "annos" in info:
            names, boxes = info["annos"]["name"], info["annos"]["gt_boxes_lidar"]
            if self.training and "DROP_SCENE_LABELS" in self.s2r and len(boxes):   # VG-1f (Nov 13): the scene's
                d = self.s2r.DROP_SCENE_LABELS   # own NAME boxes longer than MIN_L (V2X riders) are renamed to NAME_unlabelled: their points stay,
                rider = (np.asarray(names) == d.NAME) & (boxes[:, 3] > d.MIN_L)   # walkers, clutter and gt_sampling still avoid the boxes, and
                names = np.array([f"{n}_unlabelled" if r else n for n, r in zip(names, rider)])   # pcdet's class filter drops them after gt_sampling
            if self.training and "STRETCH" in self.s2r:
                points, boxes = stretch_objects(points, boxes, names, self.s2r.STRETCH, self.rng)
            if self.training and "PED_INSERT" in self.s2r and self.rng.random() < self.s2r.PED_INSERT.PROB:
                from .synth_ped import V2X_RIGS, insert, lumpi_rig
                rig = lumpi_rig() if self.s2r.PED_INSERT.get("RIG") == "lumpi" else None
                if self.s2r.PED_INSERT.get("RIG") in V2X_RIGS:   # R6 (Nov 11): one fixed V2X rig; the rig draw insert() would make
                    self.rng.choice(list(V2X_RIGS))              # is still drawn and discarded, so every later random draw is unchanged
                    rig = V2X_RIGS[self.s2r.PED_INSERT.RIG]
                pi = self.s2r.PED_INSERT
                points, boxes, names = insert(points, boxes, names, self.rng, tuple(pi.N), name=pi.NAME, rig=rig,
                                              body=pi.get("BODY", "cylinder"), group_prob=pi.get("GROUP_PROB", 0.0),
                                              occlude=pi.get("OCCLUDE", False), area_mix=pi.get("AREA_MIX", 0.0))
            if self.training and "PERSON_BOX_MAP" in self.s2r and len(boxes):   # L11-G G4 (utl.real_place); only configs that ask for it
                from .real_place import person_box_map
                boxes = person_box_map(boxes, names, self.s2r.PERSON_BOX_MAP, self.root_path)
            if self.training and "CLUTTER" in self.s2r and self.rng.random() < self.s2r.CLUTTER.PROB:
                from .synth_ped import V2X_RIGS, clutter, lumpi_rig
                v2x = self.s2r.CLUTTER.get("RIG") == "v2x"
                rig = V2X_RIGS[self.rng.choice(list(V2X_RIGS))] if v2x else lumpi_rig()
                points = clutter(points, boxes, self.rng, tuple(self.s2r.CLUTTER.N), rig=rig)
            if self.training and self.s2r:
                points, boxes = sim2real_augment(points, boxes, self.s2r, self.rng)
            input_dict.update(points=points, gt_names=names, gt_boxes=boxes)
        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos, class_names, **kwargs):
        if "annos" not in self.custom_infos[0]:
            return "No ground-truth boxes for evaluation", {}
        gts = {i["point_cloud"]["lidar_idx"]: [(b.tolist(), n) for b, n in zip(i["annos"]["gt_boxes_lidar"], i["annos"]["name"])]
               for i in self.custom_infos}
        preds = {d["frame_id"]: [{"box": b.tolist(), "score": float(s), "class": n}
                                 for b, s, n in zip(d["boxes_lidar"], d["score"], d["name"])] for d in det_annos}
        res = metrics.detection_scores(preds, gts, class_names)
        lines, flat = [], {}
        for thr, r in res.items():
            lines.append(f"IoU {thr}: mAP40 {r['mAP40']:.4f}  mAP11 {r['mAP11']:.4f}")
            flat[f"mAP40_{thr}"] = r["mAP40"]
            for c, v in r["per_class"].items():
                if v["n_gt"]:
                    lines.append(f"   {c:<12s} AP40 {v['ap40']:.4f}  (n_gt {v['n_gt']})")
                    flat[f"{c}_AP40_{thr}"] = v["ap40"]
        return "\n".join(lines), flat


def write_infos(root, split, class_names):
    root = Path(root)
    ids = [l.strip() for l in (root / "ImageSets" / f"{split}.txt").read_text().splitlines() if l.strip()]
    infos = []
    for i in ids:
        lines = [l.split() for l in (root / "labels" / f"{i}.txt").read_text().splitlines() if l.strip()]
        lines = [l for l in lines if l[-1] in class_names]
        boxes = np.array([[float(v) for v in l[:7]] for l in lines], np.float32).reshape(-1, 7)
        infos.append({"point_cloud": {"num_features": 3, "lidar_idx": i},
                      "annos": {"name": np.array([l[-1] for l in lines]), "gt_boxes_lidar": boxes}})
    with open(root / f"infos_{split}.pkl", "wb") as f:
        pickle.dump(infos, f)
    return infos


def write_gt_database(root, infos, every=1, classes=()):
    """Per-object point crops for GT sampling (from every `every`-th training frame). Every class gets a key,
    possibly empty (OpenPCDet's sampler indexes the database by all class names)."""
    import torch
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    root = Path(root)
    (root / "gt_database").mkdir(exist_ok=True)
    db = {c: [] for c in classes}
    for info in infos[::every]:
        idx = info["point_cloud"]["lidar_idx"]
        boxes, names = info["annos"]["gt_boxes_lidar"], info["annos"]["name"]
        if not len(boxes):
            continue
        pts = np.load(root / "points" / f"{idx}.npy")
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(pts[:, :3]), torch.from_numpy(boxes)).numpy()
        for k in range(len(boxes)):
            gp = pts[inside[k] > 0].copy()
            gp[:, :3] -= boxes[k, :3]
            rel = f"gt_database/{idx}_{names[k]}_{k}.bin"
            gp.tofile(root / rel)
            db.setdefault(names[k], []).append({"name": names[k], "path": rel, "gt_idx": k, "image_idx": idx,
                                                "box3d_lidar": boxes[k], "num_points_in_gt": len(gp), "difficulty": 0})
    with open(root / "dbinfos_train.pkl", "wb") as f:
        pickle.dump(db, f)
    return {k: len(v) for k, v in db.items()}


def per_class_rcnn_head():
    """VoxelRCNNHead with class-specific refine outputs (plan items X6b / L11). OpenPCDet builds num_class score and
    num_class x 7 box outputs when ROI_HEAD.CLASS_AGNOSTIC is False, but its losses and box decoding assume one output
    per RoI. This head keeps those outputs and picks, for every RoI, the column of its own class (roi_labels,
    1-indexed; padded RoIs with label 0 take column 0 and carry no loss), so loss, decoding and post-processing run
    unchanged on (N, 1) scores and (N, 7) boxes. X6 (class-agnostic) learned vehicles only: one shared output that
    vehicles dominate."""
    import torch
    from torch import nn
    from pcdet.models.roi_heads.voxelrcnn_head import VoxelRCNNHead

    class _PickClass(nn.Module):
        def __init__(self, layer, width, owner):
            super().__init__()
            self.layer, self.width = layer, width
            self._owner = [owner]  # a list, so the owner is not registered as a submodule

        def forward(self, x):
            y = self.layer(x)
            idx = (self._owner[0].cur_labels.reshape(-1).long() - 1).clamp(min=0)
            y = y.view(y.shape[0], -1, self.width)
            return y[torch.arange(y.shape[0], device=y.device), idx]

    class VoxelRCNNHeadPerClass(VoxelRCNNHead):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # num_class >= 1 (was > 1): class specialists S1-S4 have one class; with one class the per-class layers pick index 0 and have the
            # stock layer shape. Compatibility fix (Nov 8) for single-class models; multi-class unchanged.
            assert not self.model_cfg.CLASS_AGNOSTIC and self.num_class >= 1
            self.cls_pred_layer = _PickClass(self.cls_pred_layer, 1, self)
            self.reg_pred_layer = _PickClass(self.reg_pred_layer, self.box_coder.code_size, self)

        def clean_rois(self, batch_dict):
            # L11 (epoch 12) and L11b (epoch 5) died on a device-side assert after 5-30k clean iterations, at the first sync
            # after the RoI pooling. Likely a proposal outside the voxel grid or with an absurd size (dim is an exp) indexing
            # outside the volume: keep every proposal finite, its centre inside the grid and its size sane.
            # Without a gradient: the centre head's proposals carry a graph, and in-place edits recorded on it were never
            # freed (X6b and L11b grew 21 MB per iteration to an out-of-memory stop). Sampling drops the graph anyway.
            with torch.no_grad():
                r = torch.nan_to_num(batch_dict["rois"].detach(), nan=0.0, posinf=0.0, neginf=0.0)
                pc = self.point_cloud_range
                for k in range(3):
                    r[..., k] = r[..., k].clamp(float(pc[k]), float(pc[k + 3]))
                r[..., 3:6] = r[..., 3:6].clamp(0.05, 25.0)
            batch_dict["rois"] = r

        def assign_targets(self, batch_dict):
            self.clean_rois(batch_dict)
            return super().assign_targets(batch_dict)

        def roi_grid_pool(self, batch_dict):
            self.clean_rois(batch_dict)
            self.cur_labels = batch_dict["roi_labels"]  # set after target assignment in training
            return super().roi_grid_pool(batch_dict)

        def get_box_cls_layer_loss(self, forward_ret_dict):
            # a rare batch gives a non-finite IoU target. The stock BCE passes every target to the kernel, whose device-side
            # assert rejects anything outside [0, 1], including the -1 meant as ignored, so bad targets are left out here.
            # utl.train skips a step whose loss is not finite.
            cfg = self.model_cfg.LOSS_CONFIG
            if cfg.CLS_LOSS != "BinaryCrossEntropy":
                return super().get_box_cls_layer_loss(forward_ret_dict)
            flat = torch.nan_to_num(forward_ret_dict["rcnn_cls"].view(-1), nan=0.0, posinf=20.0, neginf=-20.0)
            lab = forward_ret_dict["rcnn_cls_labels"].view(-1).float()
            valid = torch.isfinite(lab) & (lab >= 0) & (lab <= 1)
            target = torch.where(valid, lab, torch.zeros_like(lab))
            per = torch.nn.functional.binary_cross_entropy(torch.sigmoid(flat), target, reduction="none")
            loss = (per * valid.float()).sum() / torch.clamp(valid.float().sum(), min=1.0)
            loss = loss * cfg.LOSS_WEIGHTS["rcnn_cls_weight"]
            return loss, {"rcnn_loss_cls": loss.item()}

    return VoxelRCNNHeadPerClass


def per_class_pv_head():
    """PVRCNNHead with the same class-specific refine outputs, proposal guard and classification-loss guard as per_class_rcnn_head. Its layers take and return (N, C, 1)."""
    import torch
    from torch import nn
    from pcdet.models.roi_heads.pvrcnn_head import PVRCNNHead
    voxel = per_class_rcnn_head()

    class _PickClass(nn.Module):
        def __init__(self, layer, width, owner):
            super().__init__()
            self.layer, self.width = layer, width
            self._owner = [owner]

        def forward(self, x):
            y = self.layer(x)
            idx = (self._owner[0].cur_labels.reshape(-1).long() - 1).clamp(min=0)
            y = y.view(y.shape[0], -1, self.width)
            return y[torch.arange(y.shape[0], device=y.device), idx].unsqueeze(-1)

    class PVRCNNHeadPerClass(PVRCNNHead):
        clean_rois = voxel.clean_rois
        get_box_cls_layer_loss = voxel.get_box_cls_layer_loss

        def __init__(self, input_channels, model_cfg, num_class=1, point_cloud_range=None, **kwargs):
            super().__init__(input_channels, model_cfg, num_class=num_class, **kwargs)
            assert not self.model_cfg.CLASS_AGNOSTIC and self.num_class > 1
            self.point_cloud_range = point_cloud_range
            self.cls_layers = _PickClass(self.cls_layers, 1, self)
            self.reg_layers = _PickClass(self.reg_layers, self.box_coder.code_size, self)

        def proposal_layer(self, batch_dict, nms_config):
            out = super().proposal_layer(batch_dict, nms_config)
            self.clean_rois(batch_dict)
            return out

        def assign_targets(self, batch_dict):
            self.clean_rois(batch_dict)
            return super().assign_targets(batch_dict)

        def roi_grid_pool(self, batch_dict):
            self.clean_rois(batch_dict)
            self.cur_labels = batch_dict["roi_labels"]
            return super().roi_grid_pool(batch_dict)

    return PVRCNNHeadPerClass


def xyz_set_abstraction():
    """VoxelSetAbstraction for xyz-only points: its raw-point branch needs at least one feature beyond x, y, z, so height z is passed as that feature while this module runs."""
    import torch
    from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import VoxelSetAbstraction

    class VoxelSetAbstractionXYZ(VoxelSetAbstraction):
        def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None, num_rawpoint_features=None, **kwargs):
            super().__init__(model_cfg, voxel_size, point_cloud_range, num_bev_features=num_bev_features, num_rawpoint_features=num_rawpoint_features + 1, **kwargs)

        def forward(self, batch_dict):
            pts = batch_dict["points"]
            batch_dict["points"] = torch.cat([pts, pts[:, 3:4]], dim=1)
            out = super().forward(batch_dict)
            out["points"] = pts
            return out

    return VoxelSetAbstractionXYZ


def fix_center_head_targets():
    """OpenPCDet's CenterHead.assign_targets relabels gt_boxes[..., -1] to head-local classes (1, 2, ...) IN PLACE
    (temp_box = cur_gt_boxes[idx] is a view). Later heads then read corrupted labels (with LUMPI's head order the
    Person head also trained on cars, vans and buses) and a two-stage RoI head downstream sees only classes 1-2 (X6 /
    L11: the refine stage never learned the other classes). Give it a copy instead."""
    from pcdet.models.dense_heads.center_head import CenterHead
    if getattr(CenterHead.assign_targets, "_ut_copy", False):
        return
    orig = CenterHead.assign_targets

    def assign_targets(self, gt_boxes, *args, **kwargs):
        return orig(self, gt_boxes.clone(), *args, **kwargs)

    assign_targets._ut_copy = True
    CenterHead.assign_targets = assign_targets


def register():
    import pcdet.datasets as D
    import pcdet.models.roi_heads as H
    D.__all__["SimDataset"] = SimDataset
    H.__all__["VoxelRCNNHeadPerClass"] = per_class_rcnn_head()
    H.__all__["PVRCNNHeadPerClass"] = per_class_pv_head()
    import pcdet.models.backbones_3d.pfe as PFE
    PFE.__all__["VoxelSetAbstractionXYZ"] = xyz_set_abstraction()
    fix_center_head_targets()
    from utl.real_place import register_real_place   # L11-G G1b sampler, used only by configs with gt_sampling REAL_PLACEMENT
    register_real_place()
    from utl.ext3p.register import register_ext  # FSHNet / SAFDNet; only on the urbantwin-lidar:scatter image
    register_ext()
    return common_utils
