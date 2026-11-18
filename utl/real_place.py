"""L11-G training-time options G1 and G4. Active
only when a config asks for them; every other config is unchanged.
  G1b  gt_sampling with REAL_PLACEMENT: {STATS: <json under DATA_PATH>, SEED: 0, TRIES: 10}: each sampled database object is moved to a spot drawn
       like a real object of its class: a 1 m cell from the per-class histogram of realism_fit label positions (by count), a uniform point in the
       cell, the heading of a real label in that cell (the class's own headings if the cell has fewer than 3), z = the scene's ground there
       (utl.synth_ped.ground_grid, 10th-percentile z per 1 m cell) + h / 2. Its points are turned by the heading change about the box centre and
       moved with it. Usual collision check: no BEV overlap with the scene's boxes or the objects placed before; up to TRIES spots, else not placed.
       Random draws: numpy generator seeded by (SEED, the worker's torch seed), as SimDataset's own generator.
  G4   SIM2REAL.PERSON_BOX_MAP: {STATS: <json under DATA_PATH>}: Person training boxes (the ray-cast walkers of PED_INSERT) get l, w, h mapped by
       per-axis quantile from the synthetic walker box sizes to the realism_fit Person label sizes; box only (points unchanged), bottom kept.
       Optional NAME (default "Person") selects the class: V2X VG-1f uses NAME 'pedestrian' with scripts/make_v2x_g4_stats.py's file.
The statistics file is written by scripts/make_l11g_stats.py (real labels only as statistics; no real point or box enters training)."""
import json
import os
from pathlib import Path

import numpy as np

_STATS = {}


def stats(root, rel):
    p = Path(rel) if Path(rel).is_absolute() else Path(root) / rel
    if str(p) not in _STATS:
        _STATS[str(p)] = json.loads(p.read_text())
    return _STATS[str(p)]


def person_box_map(boxes, names, cfg, root, name="Person"):
    name = cfg.get("NAME", name)   # VG-1f (Nov 13): V2X walkers are "pedestrian"; configs without NAME (L11-G LUMPI) keep "Person"
    m = np.asarray(names) == name
    if not m.any():
        return boxes
    st = stats(root, cfg["STATS"])["person"]
    b = boxes.copy()
    bottom = b[m, 2] - b[m, 5] / 2
    for a in range(3):
        syn = np.asarray(st["syn"][a])
        q = np.interp(b[m, 3 + a], syn, (np.arange(len(syn)) + 0.5) / len(syn))
        b[m, 3 + a] = np.quantile(np.asarray(st["real"][a]), q).astype(b.dtype)
    b[m, 2] = bottom + b[m, 5] / 2
    return b


class Placer:
    def __init__(self, st):
        self.cell = st["cell"]
        self.cls = {}
        for c, e in st["classes"].items():
            cells = np.asarray(e["cells"], np.float64).reshape(-1, 3)   # i, j, count
            self.cls[c] = dict(ij=cells[:, :2].astype(np.int64), p=cells[:, 2] / cells[:, 2].sum(),
                               hd={tuple(map(int, k.split(","))): np.asarray(v) for k, v in e["cell_headings"].items()},
                               all=np.asarray(e["headings"]))

    def draw(self, c, rng):
        e = self.cls[c]
        k = rng.choice(len(e["p"]), p=e["p"])
        i, j = e["ij"][k]
        x, y = (i + rng.random()) * self.cell, (j + rng.random()) * self.cell
        h = e["hd"].get((int(i), int(j)))
        yaw = float(rng.choice(h if h is not None and len(h) >= 3 else e["all"]))
        return x, y, yaw


def sampler_class():
    import copy
    import SharedArray
    from pcdet.datasets.augmentor.database_sampler import DataBaseSampler
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    from pcdet.utils import box_utils
    from utl.synth_ped import ground_grid

    class RealPlaceSampler(DataBaseSampler):
        def __init__(self, root_path, sampler_cfg, class_names, logger=None):
            super().__init__(root_path, sampler_cfg, class_names, logger)
            rp = sampler_cfg.REAL_PLACEMENT
            self.placer = Placer(stats(root_path, rp["STATS"])["place"])
            self.seed, self.tries = int(rp.get("SEED", 0)), int(rp.get("TRIES", 10))
            self._r, self._pid = None, None

        @property
        def rng(self):
            if self._pid != os.getpid():
                import torch
                self._r, self._pid = np.random.default_rng([self.seed, torch.initial_seed() % (2 ** 32)]), os.getpid()
            return self._r

        def __call__(self, data_dict):
            gt_boxes = data_dict["gt_boxes"]
            gt_names = data_dict["gt_names"].astype(str)
            existed = gt_boxes[:, :7].astype(np.float32)
            g, cell, lim = ground_grid(data_dict["points"])
            placed, boxes = [], []
            for class_name, sample_group in self.sample_groups.items():
                if self.limit_whole_scene:
                    sample_group["sample_num"] = str(int(self.sample_class_num[class_name]) - np.sum(class_name == gt_names))
                if int(sample_group["sample_num"]) <= 0 or class_name not in self.placer.cls:
                    continue
                for info in self.sample_with_fixed_number(class_name, sample_group):
                    b = info["box3d_lidar"].astype(np.float32)
                    for _ in range(self.tries):
                        x, y, yaw = self.placer.draw(class_name, self.rng)
                        i, j = int((x + lim) / cell), int((y + lim) / cell)
                        zg = g[i, j] if 0 <= i < g.shape[0] and 0 <= j < g.shape[1] else np.nan
                        if np.isnan(zg):
                            continue
                        nb = b.copy()
                        nb[0], nb[1], nb[2], nb[6] = x, y, zg + b[5] / 2, yaw
                        if len(existed) and iou3d_nms_utils.boxes_bev_iou_cpu(nb[None, :7], existed).max() > 0:
                            continue
                        existed = np.concatenate([existed, nb[None, :7]])
                        placed.append((info, nb, float(yaw - b[6])))
                        boxes.append(nb)
                        break
            if placed:
                data_dict = self._add(data_dict, np.stack(boxes), placed)
            data_dict.pop("gt_boxes_mask")
            return data_dict

        def _add(self, data_dict, sampled_gt_boxes, placed):
            """OpenPCDet's add_sampled_boxes_to_scene (no road plane, no image) with the objects' points turned and moved to their new boxes."""
            m = data_dict["gt_boxes_mask"]
            gt_boxes, gt_names, points = data_dict["gt_boxes"][m], data_dict["gt_names"][m], data_dict["points"]
            gdb = None
            if self.use_shared_memory:
                gdb = SharedArray.attach(f"shm://{self.gt_database_data_key}")
                gdb.setflags(write=0)
            objs = []
            for info, nb, d in placed:
                if self.use_shared_memory:
                    s, e = info["global_data_offset"]
                    op = copy.deepcopy(gdb[s:e])
                else:
                    fp = self.root_path / info["path"]
                    op = np.fromfile(str(fp), dtype=np.float32).reshape([-1, self.sampler_cfg.NUM_POINT_FEATURES])
                    if op.shape[0] != info["num_points_in_gt"]:
                        op = np.fromfile(str(fp), dtype=np.float64).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)
                assert op.shape[0] == info["num_points_in_gt"]
                c, s = np.cos(d), np.sin(d)
                x, y = op[:, 0].copy(), op[:, 1].copy()
                op[:, 0], op[:, 1] = c * x - s * y, s * x + c * y
                op[:, :3] += nb[:3].astype(op.dtype)
                objs.append(op.astype(np.float32))
            obj_points = np.concatenate(objs, axis=0)
            names = np.array([info["name"] for info, _, _ in placed])
            large = box_utils.enlarge_box3d(sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH)
            points = box_utils.remove_points_in_boxes3d(points, large)
            data_dict["points"] = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)
            data_dict["gt_names"] = np.concatenate([gt_names, names], axis=0)
            data_dict["gt_boxes"] = np.concatenate([gt_boxes, sampled_gt_boxes[:, :gt_boxes.shape[-1]]], axis=0)
            return data_dict

    return RealPlaceSampler


def register_real_place():
    from pcdet.datasets.augmentor import data_augmentor as DA
    if getattr(DA.DataAugmentor, "_real_place", False):
        return
    orig = DA.DataAugmentor.gt_sampling

    def gt_sampling(self, config=None):
        if config is not None and config.get("REAL_PLACEMENT"):
            return sampler_class()(root_path=self.root_path, sampler_cfg=config, class_names=self.class_names, logger=self.logger)
        return orig(self, config)
    DA.DataAugmentor.gt_sampling = gt_sampling
    DA.DataAugmentor._real_place = True
