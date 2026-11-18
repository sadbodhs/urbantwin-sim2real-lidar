"""Real V2X-Real reader (OPV2V layout): <split>/<scenario>/<agent>/<frame>.bin|.yaml; infrastructure agents are
negative ids. Points are float32 Nx4 in the agent's LiDAR frame. YAML objects are world-frame: location (m),
extent (half sizes), angle [roll, yaw, pitch] (deg); lidar_pose = [x, y, z, roll, yaw, pitch].
Boxes are moved into the agent LiDAR frame the way OpenCOOD does it and mapped to the 3 challenge super-classes.
"""
from pathlib import Path

import numpy as np
import yaml

SUPER = {"vehicle": ["LongVehicle", "Car", "PoliceCar"],
         "pedestrian": ["Child", "RoadWorker", "Pedestrian", "Scooter", "ScooterRider", "Motorcycle",
                        "MotorcyleRider", "BicycleRider"],
         "truck": ["Truck", "Van", "TrashCan", "ConcreteTruck", "Bus"]}
TO_SUPER = {c: s for s, cs in SUPER.items() for c in cs}


def x_to_world(pose):
    x, y, z, roll, yaw, pitch = pose
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    m = np.eye(4)
    m[0] = cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x
    m[1] = sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y
    m[2] = sp, -cp * sr, cp * cr, z
    return m


def read_frame(bin_path, yaml_path=None):
    pts = np.fromfile(bin_path, np.float32).reshape(-1, 4)
    boxes, names = np.zeros((0, 7), np.float32), []
    if yaml_path is not None and Path(yaml_path).exists():
        meta = yaml.safe_load(open(yaml_path))
        w2l = np.linalg.inv(x_to_world(meta["lidar_pose"]))
        out = []
        for obj in (meta.get("vehicles") or {}).values():
            cls = TO_SUPER.get(obj.get("obj_type"))
            if cls is None:
                continue
            loc = np.asarray(obj["location"], float) + np.asarray(obj.get("center", [0, 0, 0]), float)
            o2w = x_to_world([*loc, *obj["angle"]])
            T = w2l @ o2w
            fwd = T[:3, :3] @ np.array([1.0, 0, 0])
            ext = np.asarray(obj["extent"], float) * 2
            out.append([*T[:3, 3], *ext, np.arctan2(fwd[1], fwd[0])])
            names.append(cls)
        if out:
            boxes = np.asarray(out, np.float32)
    return pts, boxes, names


def frames(root, split, agents=("-1", "-2")):
    """Yield (scenario, agent, frame_stem, bin_path, yaml_path) for infrastructure agents, sorted."""
    base = Path(root) / split
    for scen in sorted(p for p in base.iterdir() if p.is_dir()):
        for ag in agents:
            d = scen / ag
            if not d.is_dir():
                continue
            for b in sorted(d.glob("*.bin")):
                yield scen.name, ag, b.stem, b, b.with_suffix(".yaml")
