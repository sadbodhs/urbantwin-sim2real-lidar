"""Procedural humans ray-cast into synthetic frames: walking pedestrians (neither UrbanTwin set has walkers) and
riders on the riderless UT-LUMPI bicycles / motorcycles.

A human is an elliptic cylinder (legs + torso) and a spherical head, or (BODY 'limbs') two legs, a torso and two arms
as capped cylinders plus the head, in a walking pose. It is scanned by every LiDAR of a rig (origin, beam elevations,
azimuth steps) with range noise, and the scene points it occludes from each sensor are removed. Optional: walkers in
groups (side by side or standing clusters), walkers occluding each other, and placement over the whole detection
square. Rigs use public specifications and, for LUMPI, one real frame:
  V2X-Real infrastructure: Ouster OS1-64 (64 beams, +-16.6 deg) or OS1-128 (128 beams, +-22.5 deg) at the frame
    origin, 1024 azimuth steps (datasheets);
  LUMPI: the five Measurement-4 sensors of configs/lumpi_rig.json (beam tables and resolutions from LUMPI's
    meta.json, origins solved from the per-point range field of frame 0).
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

V2X_RIGS = {"os1_64": [dict(origin=[0, 0, 0], elev_deg=np.linspace(-16.6, 16.6, 64).tolist(), n_az=1024)],
            "os1_128": [dict(origin=[0, 0, 0], elev_deg=np.linspace(-22.5, 22.5, 128).tolist(), n_az=1024)]}


@lru_cache(None)
def lumpi_rig():
    return json.loads((Path(__file__).resolve().parent.parent / "configs/lumpi_rig.json").read_text())


def ground_grid(points, cell=1.0, lim=42.0):
    """Robust local ground height per cell (10th-percentile z of cells with >= 3 points), NaN elsewhere."""
    n = int(2 * lim / cell)
    ij = np.floor((points[:, :2] + lim) / cell).astype(np.int64)
    m = (ij >= 0).all(1) & (ij < n).all(1)
    ij, z = ij[m], points[m, 2]
    key = ij[:, 0] * n + ij[:, 1]
    order = np.lexsort((z, key))
    key, z = key[order], z[order]
    uk, start, cnt = np.unique(key, return_index=True, return_counts=True)
    g = np.full(n * n, np.nan, np.float32)
    ok = cnt >= 3
    g[uk[ok]] = z[start[ok] + (cnt[ok] * 0.1).astype(np.int64)]
    return g.reshape(n, n), cell, lim


def _cyl_hits(dirs, cyl, o):
    """Nearest positive hit distance of unit rays from origin o with a capped vertical elliptic cylinder
    {c, a, b, yaw, z0, z1}; inf where missed. Rays may enter through the side or the top / bottom cap."""
    cx, cy = cyl["c"][0] - o[0], cyl["c"][1] - o[1]
    z0, z1 = cyl["z0"] - o[2], cyl["z1"] - o[2]
    a, b = cyl["a"], cyl["b"]
    c, s = np.cos(-cyl["yaw"]), np.sin(-cyl["yaw"])
    lx0, ly0 = c * -cx - s * -cy, s * -cx + c * -cy
    dx = c * dirs[:, 0] - s * dirs[:, 1]
    dy = s * dirs[:, 0] + c * dirs[:, 1]
    A = (dx / a) ** 2 + (dy / b) ** 2
    B = 2 * (lx0 * dx / a ** 2 + ly0 * dy / b ** 2)
    C = (lx0 / a) ** 2 + (ly0 / b) ** 2 - 1
    disc = B * B - 4 * A * C
    sq = np.sqrt(np.clip(disc, 0, None))
    t0, t1 = (-B - sq) / (2 * A + 1e-12), (-B + sq) / (2 * A + 1e-12)
    dz = dirs[:, 2]
    flat = np.abs(dz) < 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        za, zb = z0 / np.where(flat, 1, dz), z1 / np.where(flat, 1, dz)
    inside = (z0 <= 0) & (z1 >= 0)
    lo = np.where(flat, -np.inf if inside else np.inf, np.minimum(za, zb))
    hi = np.where(flat, np.inf if inside else -np.inf, np.maximum(za, zb))
    enter, leave = np.maximum(t0, lo), np.minimum(t1, hi)
    return np.where((disc >= 0) & (enter <= leave) & (enter > 0), enter, np.inf)


def _ray_hits(dirs, body, o):
    """Nearest positive hit distance of unit rays from origin o with the body; inf where missed."""
    if "cyls" in body:
        t = np.full(len(dirs), np.inf)
        for cyl in body["cyls"]:
            t = np.minimum(t, _cyl_hits(dirs, cyl, o))
        hc = np.asarray(body["head_c"]) - o
        hb = -2 * dirs @ hc
        hd = hb * hb - 4 * (hc @ hc - body["head_r"] ** 2)
        th = np.where(hd >= 0, (-hb - np.sqrt(np.clip(hd, 0, None))) / 2, np.inf)
        return np.minimum(t, np.where(th > 0, th, np.inf))
    cx, cy0 = body["c"][0] - o[0], body["c"][1] - o[1]
    z0, z1 = body["z0"] - o[2], body["z1"] - o[2]
    hc = np.asarray(body["head_c"]) - o
    a, b = body["a"], body["b"]
    c, s = np.cos(-body["yaw"]), np.sin(-body["yaw"])
    lx0, ly0 = c * -cx - s * -cy0, s * -cx + c * -cy0
    dx = c * dirs[:, 0] - s * dirs[:, 1]
    dy = s * dirs[:, 0] + c * dirs[:, 1]
    A = (dx / a) ** 2 + (dy / b) ** 2
    B = 2 * (lx0 * dx / a ** 2 + ly0 * dy / b ** 2)
    C = (lx0 / a) ** 2 + (ly0 / b) ** 2 - 1
    disc = B * B - 4 * A * C
    t = np.where(disc >= 0, (-B - np.sqrt(np.clip(disc, 0, None))) / (2 * A + 1e-12), np.inf)
    with np.errstate(invalid="ignore"):
        z = t * dirs[:, 2]
        t = np.where((t > 0) & (z >= z0) & (z <= z1), t, np.inf)
    hb = -2 * dirs @ hc
    hd = hb * hb - 4 * (hc @ hc - body["head_r"] ** 2)
    th = np.where(hd >= 0, (-hb - np.sqrt(np.clip(hd, 0, None))) / 2, np.inf)
    return np.minimum(t, np.where(th > 0, th, np.inf))


def polar(points, rig):
    """Per-sensor (range, azimuth, elevation) of all scene points, computed once per frame."""
    out = []
    for sen in rig:
        v = points[:, :3] - np.asarray(sen["origin"], np.float32)
        r = np.linalg.norm(v, axis=1)
        out.append((v, r, np.arctan2(v[:, 1], v[:, 0]), np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-6), -1, 1))))
    return out


def scan(points, body, rig, rng, keep, pol=None, sensor_ids=False):
    """Ray-cast `body` from every sensor of `rig`; clear occluded scene points in `keep`. -> hit points (M,3), and with
    sensor_ids the index of the sensor that saw each hit."""
    hits, sid = [], []
    rad = body.get("rad", 0.45)
    top = body["head_c"][2] + body["head_r"]
    pol = pol if pol is not None else polar(points, rig)
    for k, (sen, (v, r, az_p, el_p)) in enumerate(zip(rig, pol)):
        o = np.asarray(sen["origin"], np.float64)
        el_r = np.radians(np.asarray(sen["elev_deg"]))
        daz = 2 * np.pi / sen["n_az"]
        rel = np.asarray(body["c"]) - o[:2]
        rr = np.hypot(*rel)
        if rr < 1.0:
            continue
        th = np.arctan2(rel[1], rel[0])
        half = np.arctan2(rad, rr)
        e_lo = np.arctan2(body["z0"] - o[2] - 0.05, rr + rad)
        e_hi = np.arctan2(top - o[2] + 0.05, max(rr - rad, 0.5))
        el = el_r[(el_r >= e_lo) & (el_r <= e_hi)]
        az = np.arange(np.floor((th - half) / daz), np.ceil((th + half) / daz) + 1) * daz
        if not len(el) or not len(az):
            continue
        AZ, EL = np.meshgrid(az, el)
        dirs = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
        t = _ray_hits(dirs, body, o)
        ok = np.isfinite(t)
        if ok.any():
            hits.append(o + dirs[ok] * (t[ok] + rng.normal(0, 0.02, ok.sum()))[:, None])
            sid.append(np.full(ok.sum(), k))
        # occlusion of scene points behind the body, seen from this sensor
        sel = (np.abs(((az_p - th + np.pi) % (2 * np.pi)) - np.pi) < half) & (el_p >= e_lo) & (el_p <= e_hi)
        sel &= r > rr - body.get("near", 0.3)
        if sel.any():
            idx = np.where(sel)[0]
            occl = np.isfinite(_ray_hits(v[idx] / r[idx, None], body, o))
            keep[idx[occl]] = False
    pts = np.concatenate(hits).astype(np.float32) if hits else np.zeros((0, 3), np.float32)
    if sensor_ids:
        return pts, (np.concatenate(sid) if sid else np.zeros(0, np.int64))
    return pts


def human(rng, px, py, zg, yaw, h=None, z_start=None):
    h = h if h is not None else rng.uniform(1.5, 1.9)
    head_r = rng.uniform(0.09, 0.12)
    return dict(c=(px, py), a=rng.uniform(0.17, 0.26), b=rng.uniform(0.11, 0.17), yaw=yaw,
                z0=zg if z_start is None else z_start, z1=zg + h - 2 * head_r,
                head_c=np.array([px, py, zg + h - head_r]), head_r=head_r, h=h)


def walker(rng, px, py, zg, yaw, standing=False):
    """BODY 'limbs': two legs, a torso and two arms (capped vertical cylinders) plus the head of human(), in a walking
    pose: feet 0.09-0.12 m either side of the centre line and a stride of 0-0.75 m along the heading, arms swung
    0.15-0.25 m opposite the legs (standing: feet together, arms down). Heights as human(); a / b are drawn as in
    human() and only size the box label, which is grown to contain the limbs."""
    h = rng.uniform(1.5, 1.9)
    head_r = rng.uniform(0.09, 0.12)
    a, b = rng.uniform(0.17, 0.26), rng.uniform(0.11, 0.17)
    neck = zg + h - 2 * head_r
    hip = zg + rng.uniform(0.48, 0.52) * h
    leg_r, foot = rng.uniform(0.06, 0.08), rng.uniform(0.09, 0.12)
    half_w, half_d = rng.uniform(0.17, 0.23), rng.uniform(0.10, 0.14)
    arm_r, arm_len = rng.uniform(0.04, 0.05), rng.uniform(0.55, 0.65)
    stride = 0.0 if standing else rng.uniform(0, 0.75)
    swing = 0.0 if standing else rng.uniform(0.15, 0.25)
    side = 1 if rng.random() < 0.5 else -1          # which leg is in front
    arm_v, sh = half_w + arm_r + 0.01, neck - 0.03
    # (along the heading, to the left, half-depth, half-width, bottom, top)
    parts = [(side * stride / 2, foot, leg_r, leg_r, zg, hip), (-side * stride / 2, -foot, leg_r, leg_r, zg, hip),
             (0.0, 0.0, half_d, half_w, hip, neck),
             (-side * swing, arm_v, arm_r, arm_r, sh - arm_len, sh), (side * swing, -arm_v, arm_r, arm_r, sh - arm_len, sh)]
    c, s = np.cos(yaw), np.sin(yaw)
    cyls = [dict(c=(px + c * u - s * v, py + s * u + c * v), a=pa, b=pb, yaw=yaw, z0=z0, z1=z1)
            for u, v, pa, pb, z0, z1 in parts]
    rad = max(np.hypot(u, v) + max(pa, pb) for u, v, pa, pb, _, _ in parts) + 0.02
    ext = (max(abs(u) + pa for u, _, pa, _, _, _ in parts), max(abs(v) + pb for _, v, _, pb, _, _ in parts))
    return dict(c=(px, py), a=a, b=b, yaw=yaw, z0=zg, z1=neck, head_c=np.array([px, py, zg + h - head_r]),
                head_r=head_r, h=h, cyls=cyls, rad=rad, near=rad, ext=ext)


_RING = np.linspace(0, 2 * np.pi, 16, endpoint=False)


def _footprint(body):
    if "cyls" in body:
        return [(cy["c"], cy["a"], cy["b"], cy["yaw"]) for cy in body["cyls"]]
    return [(body["c"], body["a"], body["b"], body["yaw"])]


def touch(b1, b2, gap=0.02):
    """True if the ground footprints of two bodies overlap (all their parts span the same heights)."""
    if np.hypot(b1["c"][0] - b2["c"][0], b1["c"][1] - b2["c"][1]) > b1.get("rad", 0.45) + b2.get("rad", 0.45) + gap:
        return False
    for fa, fb in ((_footprint(b1), _footprint(b2)), (_footprint(b2), _footprint(b1))):
        for c, a, b, yaw in fa:
            u, v = a * np.cos(_RING), b * np.sin(_RING)
            P = np.vstack([np.stack([c[0] + np.cos(yaw) * u - np.sin(yaw) * v,
                                     c[1] + np.sin(yaw) * u + np.cos(yaw) * v], 1), c])
            for c2, a2, b2, yaw2 in fb:
                d = P - np.asarray(c2)
                lx = np.cos(yaw2) * d[:, 0] + np.sin(yaw2) * d[:, 1]
                ly = -np.sin(yaw2) * d[:, 0] + np.cos(yaw2) * d[:, 1]
                if ((lx / (a2 + gap)) ** 2 + (ly / (b2 + gap)) ** 2 <= 1).any():
                    return True
    return False


def group(rng, k, px, py, yaw, make):
    """Up to k bodies around (px, py): a walking group side by side (60%; shared heading +-0.2 rad, centre spacing
    0.55-0.9 m) or a standing cluster (40%; each member 0.55-0.9 m from another, within 1.2 m of the first, random
    yaw). make(x, y, yaw, standing) -> body or None (no ground / not free). Members that would touch are left out."""
    walking = rng.random() < 0.6
    first = make(px, py, yaw, not walking)
    out = [first] if first is not None else []
    ux, uy = -np.sin(yaw), np.cos(yaw)              # to the left of the heading
    lat = [0.0]
    for _ in range(k - 1):
        for _try in range(10):
            if walking:
                v = (max(lat) if rng.random() < 0.5 else min(lat))
                v += np.sign(v or rng.choice([-1, 1])) * rng.uniform(0.55, 0.9)
                x, y, w = px + ux * v, py + uy * v, yaw + rng.uniform(-0.2, 0.2)
            else:
                m = out[rng.integers(len(out))] if out else None
                ang, d = rng.uniform(-np.pi, np.pi), rng.uniform(0.55, 0.9)
                x, y = (m["c"][0] if m else px) + d * np.cos(ang), (m["c"][1] if m else py) + d * np.sin(ang)
                w = rng.uniform(-np.pi, np.pi)
                if np.hypot(x - px, y - py) > 1.2:
                    continue
            b = make(x, y, w, not walking)
            if b is not None and not any(touch(b, o) for o in out):
                out.append(b)
                if walking:
                    lat.append(v)
                break
    return out


def occlude_walkers(walk, rig):
    """walk: list of [points, sensor ids, body]. Drop each walker's hits whose ray from its sensor reaches another
    walker first (bodies are scanned one at a time, so without this a group comes out complete)."""
    origins = [np.asarray(sen["origin"], np.float64) for sen in rig]
    for i, (pts, sid, b) in enumerate(walk):
        if not len(pts):
            continue
        drop = np.zeros(len(pts), bool)
        for k, o in enumerate(origins):
            m = np.where(sid == k)[0]
            if not len(m):
                continue
            v = pts[m].astype(np.float64) - o
            d = np.linalg.norm(v, axis=1)
            dirs = v / d[:, None]
            ri, ai = np.hypot(b["c"][0] - o[0], b["c"][1] - o[1]), np.arctan2(b["c"][1] - o[1], b["c"][0] - o[0])
            hit = np.zeros(len(m), bool)
            for j, (_, _, bj) in enumerate(walk):
                rj = np.hypot(bj["c"][0] - o[0], bj["c"][1] - o[1])
                if j == i or rj - bj.get("rad", 0.45) > ri + b.get("rad", 0.45):
                    continue
                aj = np.arctan2(bj["c"][1] - o[1], bj["c"][0] - o[0])
                if abs((aj - ai + np.pi) % (2 * np.pi) - np.pi) > (np.arctan2(bj.get("rad", 0.45), max(rj, 0.5))
                                                                  + np.arctan2(b.get("rad", 0.45), max(ri, 0.5))):
                    continue
                hit |= _ray_hits(dirs, bj, o) < d - 0.05
            drop[m[hit]] = True
        walk[i][0], walk[i][1] = pts[~drop], sid[~drop]


def _pad(p, dim):
    return np.pad(p, ((0, 0), (0, dim - 3)))


def insert(points, boxes, names, rng, n_range=(2, 12), name="pedestrian", max_r=38.0, rig=None, body="cylinder",
           group_prob=0.0, occlude=False, area_mix=0.0, lim_xy=40.0):
    """Add n_range[0]..n_range[1] walkers at free ground positions. Returns new (points, boxes, names).
    body 'limbs' = walker(); group_prob = chance that an insertion is a group of 2 (60%), 3 (25%) or 4-5 (15%), members
    counting toward the total; occlude = drop walker points hidden behind another walker; area_mix = share of
    insertions placed uniformly over |x|, |y| <= lim_xy (r >= 4 m) instead of r-uniform in 4..max_r m.
    The defaults reproduce the original single cylinder walkers draw for draw."""
    rig = rig if rig is not None else V2X_RIGS[rng.choice(list(V2X_RIGS))]
    g, cell, lim = ground_grid(points)
    keep = np.ones(len(points), bool)
    pol = polar(points, rig)
    new_boxes, walk = [], []

    def ground(px, py):
        i, j = int((px + lim) / cell), int((py + lim) / cell)
        return g[i, j] if 0 <= i < g.shape[0] and 0 <= j < g.shape[1] else np.nan

    def free(px, py):
        if len(boxes) and (np.hypot(boxes[:, 0] - px, boxes[:, 1] - py) < 1.0 + boxes[:, 3:5].max(1) / 2).any():
            return False
        return not (new_boxes and (np.hypot(*(np.array(new_boxes)[:, :2] - [px, py]).T) < 1.0).any())

    def make(px, py, yaw, standing=False):
        zg = ground(px, py)
        if np.isnan(zg) or not free(px, py):
            return None
        return walker(rng, px, py, zg, yaw, standing)

    todo = int(rng.integers(n_range[0], n_range[1] + 1))
    while todo > 0:
        k = 1
        if group_prob and rng.random() < group_prob:
            k = int(min(todo, rng.choice([2, 3, 4, 5], p=[0.6, 0.25, 0.075, 0.075])))
        todo -= k
        for _try in range(10):
            if area_mix and rng.random() < area_mix:
                px, py = rng.uniform(-lim_xy, lim_xy, 2)
                if np.hypot(px, py) < 4:
                    continue
            else:
                r, th = rng.uniform(4, max_r), rng.uniform(-np.pi, np.pi)
                px, py = r * np.cos(th), r * np.sin(th)
            zg = ground(px, py)
            if np.isnan(zg) or not free(px, py):
                continue
            break
        else:
            continue
        yaw = rng.uniform(-np.pi, np.pi)
        if k > 1:
            members = group(rng, k, px, py, yaw, make if body == "limbs" else
                            lambda x, y, w, st: None if np.isnan(ground(x, y)) or not free(x, y)
                            else human(rng, x, y, ground(x, y), w))
        else:
            members = [walker(rng, px, py, zg, yaw) if body == "limbs" else human(rng, px, py, zg, yaw)]
        placed = []
        for bd in members:
            pts, sid = scan(points, bd, rig, rng, keep, pol, sensor_ids=True)
            if not len(pts):
                continue
            # annotation-style box: footprint a little larger than the body (arm swing), about 0.5-0.9 m
            l = max(2 * bd["a"] + rng.uniform(0.15, 0.35), 0.5)
            w = max(2 * bd["b"] + rng.uniform(0.3, 0.5), 0.5)
            if "ext" in bd:
                l, w = max(l, 2 * bd["ext"][0] + 0.1), max(w, 2 * bd["ext"][1] + 0.1)
            placed.append([bd["c"][0], bd["c"][1], bd["z0"] + bd["h"] / 2, l, w, bd["h"], bd["yaw"]])
            walk.append([pts, sid, bd])
        new_boxes += placed
    if occlude and len(walk) > 1:
        occlude_walkers(walk, rig)
        kept = [i for i, (p, _, _) in enumerate(walk) if len(p)]
        walk, new_boxes = [walk[i] for i in kept], [new_boxes[i] for i in kept]
    if not new_boxes:
        return points, boxes, names
    pts = np.concatenate([points[keep]] + [_pad(p, points.shape[1]) for p, _, _ in walk])
    boxes = np.concatenate([boxes.reshape(-1, 7), np.asarray(new_boxes, np.float32)])
    names = np.concatenate([np.asarray(names), np.array([name] * len(new_boxes))])
    return pts, boxes, names


def furniture(rng, px, py, zg):
    """One unlabelled roadside object from generic public sizes, in the body format of scan(). Types and shares fixed in advance, never from real frames: lamp or signal pole 25%, sign post with a plate 20%, bollard 15%, tree 15%, bin or cabinet 15%, upright panel 10%."""
    kind = rng.choice(6, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
    yaw = rng.uniform(-np.pi, np.pi)
    cyl = lambda a, b, z0, z1: dict(c=[px, py], a=float(a), b=float(b), yaw=yaw, z0=float(z0), z1=float(z1))
    head_r = 0.02
    if kind == 0:  # lamp or signal pole
        r, h = rng.uniform(0.05, 0.15), rng.uniform(3.0, 9.0)
        cyls = [cyl(r, r, zg, zg + h)]
    elif kind == 1:  # sign post with a plate at the top
        r, h = rng.uniform(0.03, 0.06), rng.uniform(2.0, 3.5)
        cyls = [cyl(r, r, zg, zg + h), cyl(rng.uniform(0.25, 0.45), 0.02, zg + h - rng.uniform(0.5, 0.9), zg + h)]
    elif kind == 2:  # bollard or short post
        r, h = rng.uniform(0.05, 0.10), rng.uniform(0.6, 1.3)
        cyls = [cyl(r, r, zg, zg + h)]
    elif kind == 3:  # tree: trunk and crown
        r, h = rng.uniform(0.10, 0.30), rng.uniform(2.0, 4.0)
        cyls = [cyl(r, r, zg, zg + h)]
        head_r = rng.uniform(1.0, 2.2)
    elif kind == 4:  # bin, hydrant or utility cabinet
        h = rng.uniform(0.8, 1.7)
        cyls = [cyl(rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.5), zg, zg + h)]
    else:  # upright panel, fence or hoarding piece
        h = rng.uniform(1.0, 2.0)
        cyls = [cyl(rng.uniform(0.5, 1.5), rng.uniform(0.02, 0.05), zg, zg + h)]
    top = max(c["z1"] for c in cyls)
    rad = max(max(c["a"], c["b"]) for c in cyls) + 0.05
    if kind == 3:
        rad = head_r + 0.05
    return dict(cyls=cyls, c=[px, py], a=cyls[0]["a"], b=cyls[0]["b"], yaw=yaw, z0=float(zg), h=float(top - zg), head_c=[px, py, float(top + (head_r * 0.7 if kind == 3 else 0))], head_r=float(head_r), rad=float(rad), near=0.3)


def clutter(points, boxes, rng, n_range=(10, 30), rig=None, lim_xy=40.0):
    """Ray-cast n_range[0]..n_range[1] unlabelled objects at random free ground positions, so that the detector sees person-sized clutter as background. Placement is uniform; it uses no real frame. Occluded scene points are removed. -> new points."""
    g, cell, lim = ground_grid(points)
    keep = np.ones(len(points), bool)
    pol = polar(points, rig)
    placed, hits = [], []
    sensors = np.array([sen["origin"][:2] for sen in rig])

    def free(px, py):
        # real sensors are mounted with a clear view: nothing within 8 m of one, which would hide a large share of the scene
        if (np.hypot(sensors[:, 0] - px, sensors[:, 1] - py) < 8.0).any():
            return False
        if len(boxes) and (np.hypot(boxes[:, 0] - px, boxes[:, 1] - py) < 1.5 + boxes[:, 3:5].max(1) / 2).any():
            return False
        return not (placed and (np.hypot(*(np.array(placed) - [px, py]).T) < 2.0).any())

    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        for _try in range(10):
            px, py = rng.uniform(-lim_xy, lim_xy, 2)
            i, j = int((px + lim) / cell), int((py + lim) / cell)
            if np.hypot(px, py) < 4 or not (0 <= i < g.shape[0] and 0 <= j < g.shape[1]) or np.isnan(g[i, j]) or not free(px, py):
                continue
            pts = scan(points, furniture(rng, px, py, g[i, j]), rig, rng, keep, pol)
            placed.append([px, py])
            if len(pts):
                hits.append(pts)
            break
    return np.concatenate([points[keep]] + [_pad(p, points.shape[1]) for p in hits])


def rider_v2(rng, px, py, zb, yaw, bike_l, motor=False):
    """A seated rider of capped vertical cylinders (as walker()), for BODY 'v2' in add_riders: hip on the saddle, a
    torso leaning forward as a staircase of three vertical segments, legs on both sides down to the pedals (one high,
    one low; thigh and shin as two segments) and arms forward to the handlebars (upper arm and forearm), plus the head.
    Public body and bike dimensions: city-bike saddle 0.90-1.05 m (motorcycle 0.75-0.90) and 0.20-0.35 m behind the bike centre,
    torso 0.58-0.68 m (adult seated height 0.85-0.95 m, seat to head top) leaning 10-30 deg (upright city / trekking bikes, as at LUMPI's urban intersection; a 20-50 deg
    sport posture put the head 0.2 m below real riders; motorcycle 0-20), head top 1.65-1.95 m, pedals 0.35-0.50 / 0.10-0.20 m, feet 0.12-0.15 m either side,
    handlebars 0.35-0.50 m ahead of the centre at 0.95-1.10 m and 0.25-0.32 m either side. px, py = bike centre,
    zb = bike bottom. Returns the body for scan() plus 'parts' = [(u, v, half_u, half_v, z0, z1)] in the bike frame."""
    for _ in range(20):
        zh = zb + (rng.uniform(0.75, 0.90) if motor else rng.uniform(0.90, 1.05))
        uh = -rng.uniform(0.20, 0.35)
        lt = rng.uniform(0.58, 0.68)
        lean = np.radians(rng.uniform(0, 20) if motor else rng.uniform(10, 30))
        head_r = rng.uniform(0.09, 0.12)
        neck = 0.06
        ut, zt = uh + lt * np.sin(lean), zh + lt * np.cos(lean)
        hc_u, hc_z = ut + (neck + head_r) * np.sin(lean), zt + (neck + head_r) * np.cos(lean)
        if 1.65 <= hc_z + head_r - zb <= 1.95:
            break
    half_d, half_w = rng.uniform(0.10, 0.14), rng.uniform(0.17, 0.23)
    leg_r, arm_r = rng.uniform(0.06, 0.08), rng.uniform(0.04, 0.05)
    parts = []
    for k in range(3):                                   # torso staircase
        za, zc = zh + k * (zt - zh) / 3, zh + (k + 1) * (zt - zh) / 3
        parts.append((uh + (k + 0.5) / 3 * (ut - uh), 0.0, half_d, half_w, za, zc))
    feet = rng.uniform(0.12, 0.15)
    high = 1 if rng.random() < 0.5 else -1
    for side in (1, -1):                                 # legs: thigh and shin
        pz = zb + (rng.uniform(0.35, 0.50) if side == high else rng.uniform(0.10, 0.20))
        pu = rng.uniform(-0.10, 0.10)
        ku, kz = (uh + pu) / 2 + 0.10, (zh + pz) / 2 + 0.05
        parts.append(((uh + ku) / 2, side * feet, leg_r, leg_r, kz, zh))
        parts.append(((ku + pu) / 2, side * feet, leg_r, leg_r, pz, kz))
    bu, bz, bv = rng.uniform(0.35, 0.50), zb + rng.uniform(0.95, 1.10), rng.uniform(0.25, 0.32)
    su, sz = ut - 0.03, zt - 0.05
    for side in (1, -1):                                 # arms: upper arm and forearm
        eu, ez = (su + bu) / 2, (sz + bz) / 2
        sv = side * (half_w + arm_r + 0.01)
        parts.append(((su + eu) / 2, (sv + side * bv) / 2, arm_r, arm_r, min(ez, sz), max(ez, sz)))
        parts.append(((eu + bu) / 2, side * bv, arm_r, arm_r, min(bz, ez), max(bz, ez)))
    c, s = np.cos(yaw), np.sin(yaw)
    cyls = [dict(c=(px + c * u - s * v, py + s * u + c * v), a=pa, b=pb, yaw=yaw, z0=z0, z1=z1)
            for u, v, pa, pb, z0, z1 in parts]
    head = np.array([px + c * hc_u, py + s * hc_u, hc_z])
    rad = max(max(np.hypot(u, v) + max(pa, pb) for u, v, pa, pb, _, _ in parts), np.hypot(hc_u, 0) + head_r) + 0.02
    return dict(c=(px, py), a=half_d, b=half_w, yaw=yaw, z0=min(p[4] for p in parts), z1=zt, head_c=head,
                head_r=head_r, h=hc_z + head_r - zb, cyls=cyls, rad=rad, near=rad,
                parts=parts + [(hc_u, 0.0, head_r, head_r, hc_z - head_r, hc_z + head_r)])


def add_riders(points, boxes, names, rng, classes=("Bicycle", "Motorcycle"), prob=0.8, rig=None, body="cylinder"):
    """Seat a ray-cast rider on riderless two-wheelers; the box grows to the rider's head. body 'v2' uses rider_v2()
    (legs to the pedals, leaning torso, arms to the handlebars) and grows the box to contain bike and rider + 0.05 m."""
    rig = rig if rig is not None else lumpi_rig()
    keep = np.ones(len(points), bool)
    boxes = boxes.copy()
    extra = []
    pol = polar(points, rig) if np.isin(names, classes).any() else None
    for k in np.where(np.isin(names, classes))[0]:
        if rng.random() > prob:
            continue
        x, y, z, l, w, h, yaw = boxes[k]
        zb = z - h / 2
        if body == "v2":
            bd = rider_v2(rng, x, y, zb, yaw, l, motor=names[k] == "Motorcycle")
            pts = scan(points, bd, rig, rng, keep, pol)
            if len(pts):
                extra.append(pts)
                P = np.array(bd["parts"])
                u0, u1 = min(-l / 2, (P[:, 0] - P[:, 2]).min()) - 0.05, max(l / 2, (P[:, 0] + P[:, 2]).max()) + 0.05
                v0, v1 = min(-w / 2, (P[:, 1] - P[:, 3]).min()) - 0.05, max(w / 2, (P[:, 1] + P[:, 3]).max()) + 0.05
                top = max(h, bd["h"]) + 0.05
                uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
                boxes[k] = [x + np.cos(yaw) * uc - np.sin(yaw) * vc, y + np.sin(yaw) * uc + np.cos(yaw) * vc,
                            zb + top / 2, u1 - u0, v1 - v0, top, yaw]
            continue
        top = rng.uniform(1.6, 1.9)
        back = rng.uniform(0.0, 0.2) * l           # riders sit slightly behind the box centre
        px, py = x - back * np.cos(yaw), y - back * np.sin(yaw)
        body = human(rng, px, py, zb, yaw, h=top, z_start=zb + rng.uniform(0.55, 0.8))
        body["b"] = rng.uniform(0.15, 0.22)        # torso across the bike, legs to both sides
        pts = scan(points, body, rig, rng, keep, pol)
        if len(pts):
            extra.append(pts)
            boxes[k, 5] = max(h, top)
            boxes[k, 2] = zb + boxes[k, 5] / 2
            boxes[k, 4] = max(w, 2 * body["b"] + 0.2)
    if not extra:
        return points, boxes
    return np.concatenate([points[keep]] + [_pad(p, points.shape[1]) for p in extra]), boxes
