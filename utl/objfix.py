"""Fix the synthetic objects of UT-LUMPI before training: rescale each class to generic real-world sizes and seat
riders on the riderless two-wheelers.

UT-LUMPI's vehicle models are about 20-25 % smaller than real road users (points inside UT cars span
3.30 x 1.45 x 1.13 m), while the scene itself matches the real intersection at scale 0.997. Each object's
interior points and its box are scaled about the box's bottom centre by target / synthetic-median per axis.
Targets are published generic class sizes (l, w, h in m), not LUMPI labels:
  nuScenes mean box sizes, as used for the MMDetection3D nuScenes anchors (car, truck, bus, motorcycle, bicycle);
  VW Transporter T6 panel van (4.90 x 1.90 x 1.99) for Van. Riders are added afterwards (utl.synth_ped).
UT-V2X has the same problem in height (synthetic cars 4.79 x 2.01 x 1.52 m); it is fixed per raw class during
conversion (`utl.prepare ut --objfix`), before Car / Van / Truck / Bus are merged into the track classes.
"""
import numpy as np

TARGETS = {
    "Car": (4.61, 1.95, 1.72),
    "Truck": (6.74, 2.46, 2.73),
    "Bus": (11.19, 2.94, 3.47),
    "Motorcycle": (2.11, 0.77, 1.47),
    "Bicycle": (1.68, 0.60, 1.27),
    "Van": (4.90, 1.90, 1.99),
}


def class_scales(stats):
    """stats: {class: {'lwh': [l, w, h]}} of the synthetic training set -> {class: (sl, sw, sh)}."""
    return {c: tuple(np.round(np.asarray(t) / np.asarray(stats[c]["lwh"]), 4)) for c, t in TARGETS.items() if c in stats}


def rescale(points, boxes, names, scales):
    import torch
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    if not len(boxes):
        return points, boxes
    inside = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, :3].copy()), torch.from_numpy(boxes[:, :7].astype(np.float32))).numpy()
    points, boxes = points.copy(), boxes.copy()
    for k, n in enumerate(names):
        if n not in scales:
            continue
        s = np.asarray(scales[n], np.float32)
        x, y, z, l, w, h, yaw = boxes[k]
        zb = z - h / 2
        idx = np.where(inside[k] > 0)[0]
        c, si = np.cos(yaw), np.sin(yaw)
        d = points[idx, :2] - (x, y)
        u, v = d[:, 0] * c + d[:, 1] * si, -d[:, 0] * si + d[:, 1] * c
        u, v = u * s[0], v * s[1]
        points[idx, 0] = x + u * c - v * si
        points[idx, 1] = y + u * si + v * c
        points[idx, 2] = zb + (points[idx, 2] - zb) * s[2]
        boxes[k, 3:6] = (l * s[0], w * s[1], h * s[2])
        boxes[k, 2] = zb + boxes[k, 5] / 2
    return points, boxes
