"""Test-split rebuild of the V2X-Real entry v2x_v35d from the released checkpoints, one CPU stage at a time.

Runs from the root of a copy of the release tree (or --root), in urbantwin-realism:cpu or in the GPU image with UT_NOGPU=1.
Every stage calls the shipped functions in scripts/ and utl/. The wrappers here restrict the work to the 50 test frames,
leave out the split-count and md5 locks of the original scripts, and compare each output with the md5 the original run recorded.

    python tools/rebuild_v2x_test.py members-check
    python tools/rebuild_v2x_test.py lanes [--workers 6] [--parts dsvt q7]
    python tools/rebuild_v2x_test.py p3-ids [--out data/pcdet/v2x_real/ImageSets/rfit_i1.txt]
    python tools/rebuild_v2x_test.py final --v27 anchors/v2x_v27_predictions.json --p3 off --submitted sub/predictions.json
    python tools/rebuild_v2x_test.py final --v27 anchors/v2x_v27_predictions.json --p3 ids data/pcdet/v2x_real/ImageSets/rfit_i1.txt
    python tools/rebuild_v2x_test.py compare --submitted sub/predictions.json

Every subcommand takes --dry-run (print what it reads and writes, then stop) and --root.
The GPU stages (member inference, second-stage refines) are run by scripts/rebuild_v2x_test.sh, which also runs these subcommands.
"""
import argparse
import builtins
import contextlib
import gc
import hashlib
import importlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

KIT_FRAMES = Path("data/kit/v2x/detection_test_frames")
KIT_IDS = Path("configs/starter_kit_mirror/v2x_real/detection_test_frame_ids.json")
REPORTS = Path("output/v2x/rebuild")
OUT_FINAL = Path("output/v2x/ens/v2x_v35_rebuild_test.json")
OUT_P3SUB = Path("output/v2x/ens/v2x_v35_rebuild_test_p3sub.json")
RFIT = Path("data/pcdet/v2x_real/ImageSets/rfit_i1.txt")
REAL_POINTS = Path("data/pcdet/v2x_real/points")

# Member test predictions read by the lanes, with the md5 the original run recorded for each (full values from the records named).
MEMBER_FILES = [
    ("pp_v2x", "output/v2x/sweep_in/pp_test.json", "8d263643437794978b3d1c1271d1120f", "logs/dsvt_v2x_j_lanes.json member_md5"),
    ("cp_v2x", "output/v2x/sweep_in/cp_test.json", "10ba1d1b4400612171fc20d82056de36", "logs/dsvt_v2x_j_lanes.json member_md5"),
    ("pp_v2x_ped", "output/v2x/sweep_in/ped_test.json", "eb56bcd9c0fceb9b9d11f253149f0fcb", "logs/dsvt_v2x_j_lanes.json member_md5"),
    ("pp_v2x_all", "output/v2x/sweep_in/ppall_test.json", "25645e7ae4b7b5544679eb50b79b0db2", "logs/dsvt_v2x_j_lanes.json member_md5"),
    ("pp_v2x_walk", "output/v2x/pp_v2x_walk/kit_mc/pred_test.json", "2bf9a2e5607282d38932736725520ffb", "logs/dsvt_v2x_j_lanes.json member_md5"),
    ("vrcnn_v2x", "output/v2x/vrcnn_v2x/kit/pred_test.json", "4187333862c28fdab0662daef1040c17", "logs/dsvt_v2x_j_lanes.json member_md5"),
    ("vrcnn_v2x_b", "output/v2x/vrcnn_v2x_b/kit/pred_test.json", "2c0a6ad41304a21040e98042d0b811d5", "logs/dsvt_v2x_j_lanes.json member_md5"),
    ("fshnet_v2x", "output/v2x/fshnet_v2x/kit/pred_test.json", "fa728ad136cf0e2897a28f8ee40c4574", "logs/dsvt_v2x_j_lanes.json kit_md5"),
    ("safdnet_v2x", "output/v2x/safdnet_v2x/kit/pred_test.json", "06d883937e45590f7c975b2e5b598d43", "logs/dsvt_v2x_j_lanes.json kit_md5"),
    ("dsvt_v2x", "output/v2x/dsvt_v2x/kit/pred_test.json", "61116c40909029d3de2b11d6c8c67101", "logs/dsvt_v2x_predict.md5"),
    ("vmamba_v2x", "output/v2x/vmamba_v2x/kit/pred_test.json", "89a5cfe86842123da632a27c45ab0046", "logs/vmamba_v2x_pred.md5"),
    ("pv_v2x_n", "output/v2x/pv_v2x_n/kit/pred_test.json", "8cda9537f8aed4f4b17bd6c85d60d115", "logs/pv_v2x_n_pred.md5"),
]
# sweep_in copies and the kit file each was copied from (the copy was a manual step of the original run)
SWEEP_SRC = {"output/v2x/sweep_in/pp_test.json": "output/v2x/pp_v2x/kit_mc/pred_test.json",
             "output/v2x/sweep_in/cp_test.json": "output/v2x/cp_v2x/kit/pred_test.json",
             "output/v2x/sweep_in/ped_test.json": "output/v2x/pp_v2x_ped/kit_mc/pred_test.json",
             "output/v2x/sweep_in/ppall_test.json": "output/v2x/pp_v2x_all/kit_mc/pred_test.json"}

# Test lanes: file, original md5, boxes, boxes with score >= 0.3 (logs/dsvt_v2x_j_lanes.json, logs/q7_dsvt_vmamba_pvn_v2x_j_lanes.json,
# logs/{vmamba_v2x,pv_v2x_n}_j_lanes.md5 and the counts in those members' lane reports). The v18 trucks were never a file of their own.
LANES = {
    "fsh_saf_v2x vehicle": ("output/v2x/fsh_saf_v2x/s2_in/vehicle_test.json", "c5ad44c6e3d022915f37d5340f222767", 95434, 387),
    "fshnet_v2x pedestrian": ("output/v2x/fshnet_v2x/s2_in/pedestrian_test.json", "96320b41d208b68f89f7d2484afeba36", 52008, 1302),
    "v18 truck": (None, None, 12818, 215),
    "dsvt_v2x_j vehicle": ("output/v2x/dsvt_v2x_j/s2_in/vehicle_test.json", "0a2a248bd87af921f3fdfcd741ee19dd", 99290, 386),
    "dsvt_v2x_j pedestrian": ("output/v2x/dsvt_v2x_j/s2_in/pedestrian_test.json", "391a0e518bec383ebd5db18603762273", 62695, 1320),
    "dsvt_v2x_j truck": ("output/v2x/dsvt_v2x_j/s2_in/truck_test.json", "367f213ebd3f7a7755ade8bbe8c37134", 16755, 195),
    "vmamba_v2x_j vehicle": ("output/v2x/vmamba_v2x_j/s2_in/vehicle_test.json", "7af187cdf3df87efe7da2f08c0ee7c6f", 99191, 385),
    "pv_v2x_n_j vehicle": ("output/v2x/pv_v2x_n_j/s2_in/vehicle_test.json", "ff9c056d3d7a11a2c6c6bb386ebc97b7", 108963, 404),
    "q7_dsvt_vmamba_pvn_v2x_j vehicle": ("output/v2x/q7_dsvt_vmamba_pvn_v2x_j/s2_in/vehicle_test.json", "6a6c9fe91fd87031dcf661223f95c469",
                                         116299, 399),
}
# (class, kind) of each lane module's work() -> lane label; the second element says whether this part writes the file
DSVT_KINDS = {("vehicle", "off"): ("fsh_saf_v2x vehicle", True), ("pedestrian", "off"): ("fshnet_v2x pedestrian", True),
              ("truck", "off"): ("v18 truck", False), ("vehicle", "new"): ("dsvt_v2x_j vehicle", True),
              ("pedestrian", "new"): ("dsvt_v2x_j pedestrian", True), ("truck", "new"): ("dsvt_v2x_j truck", True)}
Q7_MEMBERS = ("dsvt_v2x", "vmamba_v2x", "pv_v2x_n")
Q7_KINDS = {("vehicle", "off"): ("fsh_saf_v2x vehicle", False), ("vehicle", "one:dsvt_v2x"): ("dsvt_v2x_j vehicle", False),
            ("vehicle", "one:vmamba_v2x"): ("vmamba_v2x_j vehicle", True), ("vehicle", "one:pv_v2x_n"): ("pv_v2x_n_j vehicle", True),
            ("vehicle", "new"): ("q7_dsvt_vmamba_pvn_v2x_j vehicle", True)}

# Second-stage test outputs read by the final build (logs/q7_dsvt_vmamba_pvn_v2x_j_refine.md5, logs/dsvt_v2x_j_refine.md5)
REFINED = {"output/v2x/q7_dsvt_vmamba_pvn_v2x_j/s2/vehicle_x6_test_ref.json": "f0d938fc9bb1ee18395c17fb512365b9",
           "output/v2x/dsvt_v2x_j/s2/pedestrian_x6b_test_ref.json": "6ea33fe5b1db3514c9c53fd601a62714",
           "output/v2x/dsvt_v2x_j/s2/pedestrian_neg_test_ref.json": "add1048f7eeb424aa1bec15ae102fb19"}
V27_MD5 = "fc5b57f32f7be0486fdd69a3efb487ee"      # predictions.json of the uploaded v2x_v27 zip (trucks and frame order)
V35_MD5 = "be6122b96cf619280f260c1e34244d24"      # predictions.json of the ranked v2x_v35d zip = output/v2x/ens/v2x_v35_test.json
V35_COUNTS = {"vehicle": (116299, 393), "pedestrian": (62695, 1736), "truck": (12818, 248)}   # logs/build_v2x_v35.json (boxes, >= 0.3)
P3_VOXELS = 16542                                  # logs/build_v2x_v31.log, checked by build_v2x_v32 / v35
P3_TEST = (62695, 20895)                           # logs/build_v2x_v35.log: test pedestrian boxes, of which P3 lowered
RFIT_COUNT, RFIT_CLIPS = 128, 9
# frame lists read at import time by scripts/v2x_plan.py and scripts/v2x_fresh_read.py; the test split never uses them
SPLIT_LISTS = ("data/pcdet/v2x_real/ImageSets/val_i1.txt", "data/pcdet/v2x_fresh/ImageSets/fresh.txt")


# ------------------------------------------------------------------ helpers
def stop(msg):
    sys.exit("STOP " + msg)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def md5_text(t):
    return hashlib.md5(t.encode()).hexdigest()


def table(header, rows):
    w = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: "  ".join(str(x).ljust(n) for x, n in zip(r, w))  # noqa: E731
    print(line(header))
    print("  ".join("-" * n for n in w))
    for r in rows:
        print(line(r))


def stats(lane):
    return sum(len(v) for v in lane.values()), sum(d["score"] >= 0.3 for v in lane.values() for d in v)


def save_report(name, rep):
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / f"{name}.json"
    p.write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"report: {p}")


def plan(a, reads, writes):
    """Print the files a stage reads and writes; True when this is a dry run (the caller stops there)."""
    if not a.dry_run:
        return False
    print(f"dry run: {a.cmd}")
    for role, items in (("reads", reads), ("writes", writes)):
        for p in items:
            print(f"  {role:6s} {p}  ({'present' if Path(p).exists() else 'absent'})")
    return True


@contextlib.contextmanager
def split_lists_optional():
    """While the shipped modules are imported, a missing val_i1 / fresh frame list reads as an empty file."""
    real_open = builtins.open
    missing = {str(Path(p)) for p in SPLIT_LISTS if not Path(p).exists()}

    def _open(file, *args, **kw):
        if isinstance(file, (str, os.PathLike)) and str(Path(file)) in missing:
            return io.StringIO("")
        return real_open(file, *args, **kw)

    builtins.open = _open
    try:
        yield sorted(missing)
    finally:
        builtins.open = real_open


_NOTED = set()


def shipped(*names):
    """Import the shipped modules from scripts/ (and utl/) by name."""
    with split_lists_optional() as miss:
        mods = [importlib.import_module(n) for n in names]
    for m in miss:
        if m not in _NOTED:
            _NOTED.add(m)
            print(f"note: {m} is absent; it read as empty while scripts/ was imported (the test split does not use it)")
    return mods


def ensure_fork():
    """The lanes and rules use multiprocessing with the Linux default (fork), as in the original runs."""
    if mp.get_start_method() != "fork" and "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)
        print("note: multiprocessing start method set to fork")


def test_frames():
    """The 50 test frame ids, sorted (the lanes' frame order)."""
    ids = sorted(p.stem for p in KIT_FRAMES.glob("*.bin")) if KIT_FRAMES.is_dir() else []
    if KIT_IDS.exists():
        kit = json.loads(KIT_IDS.read_text())
        if ids and set(ids) != set(kit):
            stop(f"{KIT_FRAMES} and {KIT_IDS} hold different frame ids")
        ids = ids or sorted(kit)
    if len(ids) != 50:
        stop(f"expected the 50 test frames in {KIT_FRAMES} (or {KIT_IDS}), found {len(ids)}")
    return ids


def recorded_md5():
    """{path: md5} from the shipped records: logs/*.md5 and the md5 tables inside logs/*_lanes.json."""
    rec = {}
    logs = Path("logs")
    for p in sorted(logs.glob("*.md5")):
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                rec.setdefault(parts[1], parts[0])
    for p in sorted(logs.glob("*_lanes.json")):
        try:
            r = json.loads(p.read_text())
        except ValueError:
            continue
        for key in ("member_md5", "kit_md5", "v31_lane_md5", "member_pred_md5", "dsvt_v2x_md5"):
            for f, m in (r.get(key) or {}).items():
                rec.setdefault(f, m)
    return rec


# ------------------------------------------------------------------ members-check
def cmd_members(a):
    files = [f for _, f, _, _ in MEMBER_FILES]
    if plan(a, files + list(SWEEP_SRC.values()) + ["logs/"], []):
        return
    rec = recorded_md5()
    frames = None
    try:
        frames = set(test_frames())
    except SystemExit as e:
        print(f"note: frame coverage not checked ({e})")
    rows, out, notes = [], [], []
    for model, f, orig, src in MEMBER_FILES:
        if rec.get(f) and rec[f] != orig:
            notes.append(f"{f}: the record in logs/ says {rec[f]}, this table {orig}")
        p = Path(f)
        got = md5_file(p) if p.exists() else "missing"
        same = "-" if got == "missing" else ("yes" if got == orig else "NO")
        rows.append((model, f, got, orig, same))
        e = dict(model=model, file=f, rebuilt=got, original=orig, record=src, same=same == "yes")
        if p.exists() and frames is not None:
            keys = set(json.loads(p.read_text()))
            e["frames_ok"] = keys == frames
            if keys != frames:
                notes.append(f"{f}: covers {len(keys & frames)} of the 50 test frames and {len(keys - frames)} others")
        out.append(e)
    table(("model", "file", "rebuilt md5", "original md5", "same?"), rows)
    for dst, srcf in SWEEP_SRC.items():
        d, s = Path(dst), Path(srcf)
        if d.exists() and s.exists() and md5_file(d) != md5_file(s):
            notes.append(f"{dst} is not a copy of {srcf}")
        if not d.exists() and s.exists():
            notes.append(f"{dst} missing; copy it: mkdir -p {d.parent} && cp {srcf} {dst}")
    n_same = sum(r[4] == "yes" for r in rows)
    n_miss = sum(r[4] == "-" for r in rows)
    print(f"\n{n_same} of {len(rows)} member files equal the original run's bytes; {n_miss} missing")
    for n in notes:
        print("note:", n)
    save_report("members", {"members": out, "notes": notes})
    if a.strict and n_same != len(rows):
        sys.exit(1)


# ------------------------------------------------------------------ lanes
def run_chunks(mod, kinds, frames, workers):
    """The member loading and frame-chunked pool of mod.run_split, on the test split with the given frames; mod.work does the chains."""
    paths = mod.member_paths("test")

    def rd(p):
        m = json.loads(Path(p).read_text())
        return {f: m.get(f, []) for f in frames}

    mod._D.clear()
    mod._D["frames"] = list(frames)
    mod._D["base5"] = [rd(p) for p in paths["base5"]]
    for key, p in paths.items():
        if key != "base5":
            mod._D[key] = rd(p)
    t0 = time.time()
    if workers <= 0:
        parts = [mod.work(("test", 0, 1, kinds))]
    else:
        n = min(len(frames), 4 * workers)
        gc.collect()
        gc.freeze()
        try:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
                parts = list(ex.map(mod.work, [("test", k, n, kinds) for k in range(n)]))
        finally:
            gc.unfreeze()
    out = {}
    for c in mod.CL:
        for t in kinds:
            got = {}
            for _, res, _ in parts:
                got.update(res[(c, t)])
            assert set(got) == set(frames), (c, t)
            out[(c, t)] = {f: got[f] for f in frames}
    mod._D.clear()
    gc.collect()
    return out, round(time.time() - t0, 1)


def cmd_lanes(a):
    parts = [p for p in ("dsvt", "q7") if p in a.parts]
    writes = [LANES[lab][0] for part, kinds in (("dsvt", DSVT_KINDS), ("q7", Q7_KINDS)) if part in parts
              for lab, w in kinds.values() if w]
    reads = [f for _, f, _, _ in MEMBER_FILES if parts != ["dsvt"] or f.split("/")[2] not in ("vmamba_v2x", "pv_v2x_n")]
    if plan(a, reads + [str(KIT_FRAMES)], writes):
        return
    frames = test_frames()
    miss = [f for f in reads if not Path(f).exists()]
    if miss:
        stop(f"member files missing: {miss}")
    ex = [p for p in writes if Path(p).exists()]
    if ex and not a.force:
        stop(f"{ex} exist (pass --force to overwrite)")
    diff = [f for _, f, orig, _ in MEMBER_FILES if f in reads and md5_file(f) != orig]
    print(f"member inputs: {len(reads) - len(diff)} of {len(reads)} equal the original run's bytes" + (f"; differ: {diff}" if diff else ""))
    ensure_fork()
    rows, rep, texts = [], {"frames": frames, "workers": a.workers, "member_inputs_differ": diff, "lanes": []}, {}

    def record(part, label, lane, write):
        path, orig, ob, oc = LANES[label]
        text = json.dumps(lane)
        m = md5_text(text)
        b, c = stats(lane)
        if write:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        agree = None
        if label in texts:
            agree = texts[label] == text
            print(f"{part}: {label} equals the dsvt part's lane byte for byte: {agree}")
        texts.setdefault(label, text)
        rows.append((part, label, path or "(not a file)", m, orig or "no record", "-" if orig is None else ("yes" if m == orig else "NO"),
                     f"{b}/{ob}", f"{c}/{oc}"))
        rep["lanes"].append(dict(part=part, lane=label, file=path, written=bool(write), rebuilt=m, original=orig, same=m == orig,
                                 boxes=[b, ob], confident=[c, oc], equals_dsvt_part=agree))

    if "dsvt" in parts:
        (DL,) = shipped("dsvt_v2x_lanes")
        kinds = ["off", "new"]
        out, secs = run_chunks(DL, kinds, frames, a.workers)
        print(f"dsvt_v2x_lanes chains {kinds} on {len(frames)} test frames: {secs} s")
        for (c, t), (label, w) in DSVT_KINDS.items():
            record("dsvt", label, out[(c, t)], w)
        rep["dsvt_seconds"] = secs
        del out
    if "q7" in parts:
        (QL,) = shipped("q7_v2x_lanes")
        QL.configure(list(Q7_MEMBERS))
        kinds = ["off"] + [f"one:{m}" for m in QL.MEMBERS] + ["new"]
        out, secs = run_chunks(QL, kinds, frames, a.workers)
        print(f"q7_v2x_lanes ({QL.LANES}) chains {kinds} on {len(frames)} test frames: {secs} s")
        for (c, t), (label, w) in Q7_KINDS.items():
            record("q7", label, out[(c, t)], w)
        rep["q7_seconds"] = secs
        del out
    print()
    table(("part", "lane", "file", "rebuilt md5", "original md5", "same?", "boxes new/orig", ">=0.3 new/orig"), rows)
    save_report("lanes", rep)


# ------------------------------------------------------------------ p3-ids
def ids_from_raw(root):
    """test_<clip>_i1_<frame> for every agent -1 frame of the V2X-Real test split whose frame number is a multiple of 10."""
    try:
        from utl import v2xreal
        pairs = [(scen, stem) for scen, _, stem, _, _ in v2xreal.frames(root, "test", agents=("-1",))]
        how = "utl.v2xreal.frames"
    except ImportError:
        base = Path(root) / "test"
        pairs = [(s.name, b.stem) for s in sorted(p for p in base.iterdir() if p.is_dir()) if (s / "-1").is_dir()
                 for b in sorted((s / "-1").glob("*.bin"))]
        how = "directory listing (utl.v2xreal needs PyYAML)"
    return [f"test_{scen}_i1_{stem}" for scen, stem in pairs if int(stem) % 10 == 0], how


def ids_from_list(path):
    """The same rule applied to an ImageSets list written by utl.prepare v2x_real --split test."""
    out = []
    for line in open(path):
        i = line.strip()
        if i.startswith("test_") and "_i1_" in i and int(i.rsplit("_", 1)[1]) % 10 == 0:
            out.append(i)
    return out


def cmd_p3_ids(a):
    raw_test, tl, out = Path(a.raw) / "test", Path(a.test_list), Path(a.out)
    if plan(a, [str(raw_test), str(tl), str(REAL_POINTS)], [str(out)]):
        return
    src = {}
    if raw_test.is_dir():
        ids, how = ids_from_raw(a.raw)
        src[f"{raw_test} ({how})"] = ids
    if tl.exists():
        src[str(tl)] = ids_from_list(tl)
    if not src:
        stop(f"neither {raw_test} (the extracted V2X-Real test split) nor {tl} is present")
    sets = list(src.items())
    for name, ids in sets:
        print(f"{name}: {len(ids)} ids")
    if len(sets) == 2 and set(sets[0][1]) != set(sets[1][1]):
        stop(f"the two sources disagree: {len(set(sets[0][1]) - set(sets[1][1]))} / {len(set(sets[1][1]) - set(sets[0][1]))} ids only in one")
    ids = sorted(set(sets[0][1]))
    per = Counter(i[len("test_"):].rsplit("_i1_", 1)[0] for i in ids)
    table(("clip", "frames", "first", "last"),
          [(c, n, min(i for i in ids if f"_{c}_i1_" in i)[-6:], max(i for i in ids if f"_{c}_i1_" in i)[-6:]) for c, n in sorted(per.items())])
    checks = {"128 frames": len(ids) == RFIT_COUNT, "9 clips": len(per) == RFIT_CLIPS,
              "clip 2023-04-04-15-47-17_19_0: 19 frames ending at 000180":
                  per.get("2023-04-04-15-47-17_19_0") == 19 and "test_2023-04-04-15-47-17_19_0_i1_000180" in ids}
    try:
        from utl import guard
        bad = [i for i in ids if not guard.allowed("v2x", i)]
        checks["every id passes utl.guard"] = not bad
        if bad:
            print("guard refuses:", bad[:5])
    except (OSError, ValueError) as e:
        print(f"note: utl.guard not run ({e})")
    have = sum((REAL_POINTS / f"{i}.npy").exists() for i in ids)
    for k, v in checks.items():
        print(f"check {k}: {v}")
    print(f"{have} of {len(ids)} frames have {REAL_POINTS}/<id>.npy" + ("" if have == len(ids) else
          "; write them with: UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare v2x_real --root data/raw/v2xreal/extracted "
          "--split test --stride 10 --out data/pcdet/v2x_real"))
    text = "".join(i + "\n" for i in ids)
    if out.exists() and out.read_text() != text and not a.force:
        stop(f"{out} exists with other content (pass --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(ids)} ids, md5 {md5_text(text)})")
    save_report("p3_ids", {"sources": {k: len(v) for k, v in src.items()}, "clips": dict(per), "checks": checks, "points_present": have,
                           "out": str(out), "md5": md5_text(text)})


# ------------------------------------------------------------------ final
def parse_p3(p3):
    if p3 == ["off"]:
        return "off", None
    if len(p3) == 2 and p3[0] == "ids":
        return "ids", Path(p3[1])
    stop(f"--p3 takes 'off' or 'ids FILE', not {' '.join(p3)}")


def assemble(v27, veh, ped, keep):
    """Per frame in v27's frame order: v27's trucks in file order, the vehicles in build order, the pedestrians in chain order, one stable
    sort by descending score. The v29, v30, v31, v34 and v35 builders' assembly lines compose to exactly this."""
    return {f: sorted([keep(d) for d in v27[f] if d["class"] == "truck"] + [keep(d) for d in veh[f] if d["class"] == "vehicle"]
                      + [keep(d) for d in ped[f] if d["class"] == "pedestrian"], key=lambda d: -d["score"]) for f in v27}


def p3_from_submission(ped0, sub, radius, tol):
    """Pair each rebuilt pedestrian box (P3 not applied) with a submitted pedestrian box (same box first, else the nearest unused centre
    within radius, highest rebuilt score first) and classify the score ratio submitted / rebuilt: x0.3 = lowered by P3 in the submission."""
    import numpy as np
    n = Counter()
    lowered, rows = {}, {}
    for f, v in ped0.items():
        R = [i for i, d in enumerate(v) if d["class"] == "pedestrian"]
        S = [d for d in sub.get(f, []) if d["class"] == "pedestrian"]
        by_box = defaultdict(list)
        for j, d in enumerate(S):
            by_box[tuple(d["box"])].append(j)
        used = np.zeros(len(S), bool)
        pair = {}
        for i in R:
            for j in by_box.get(tuple(v[i]["box"]), ()):
                if not used[j]:
                    used[j] = True
                    pair[i] = (j, "same box")
                    break
        rest = sorted((i for i in R if i not in pair), key=lambda i: -v[i]["score"])
        if rest and len(S):
            centres = np.array([d["box"][:3] for d in S], float)
            for i in rest:
                dist = np.linalg.norm(centres - np.asarray(v[i]["box"][:3], float), axis=1)
                dist[used] = np.inf
                j = int(np.argmin(dist))
                if dist[j] <= radius:
                    used[j] = True
                    pair[i] = (j, "near")
        n["submitted pedestrians left unpaired"] += int((~used).sum())
        for i in R:
            if i not in pair:
                n["rebuilt pedestrians unpaired"] += 1
                continue
            j, how = pair[i]
            n[f"paired ({how})"] += 1
            s0, s1 = v[i]["score"], S[j]["score"]
            r = s1 / s0 if s0 > 0 else float("nan")
            if s1 == s0 * 0.3 or abs(r - 0.3) <= tol * 0.3:
                k = "x0.3 (P3 in the submission)"
                lowered.setdefault(f, []).append(i)
                rows.setdefault(f, []).append([i, v[i]["box"], s0, s1])
            elif s1 == s0 or abs(r - 1.0) <= tol:
                k = "x1 (P3 did not act)"
            elif abs(r - 1 / 0.3) <= tol / 0.3:
                k = "x3.33 (lowered here by another rule, not in the submission)"
            else:
                k = "other ratio"
            n[k] += 1
            if s1 == s0 or s1 == s0 * 0.3:
                n["score bit-exact up to P3"] += 1
    return n, lowered, rows


def transfer(ped0, lowered):
    """ped0 with the score of each box in lowered multiplied by 0.3, the same expression postproc_p123.filt uses."""
    out = {}
    for f, v in ped0.items():
        idx = set(lowered.get(f, ()))
        out[f] = [{**d, "score": d["score"] * 0.3} if i in idx else d for i, d in enumerate(v)]
    return out


def cmd_final(a):
    mode, p3file = parse_p3(a.p3)
    outs = [OUT_FINAL] + ([OUT_P3SUB] if mode == "off" and a.submitted else [])
    reads = [a.v27] + list(REFINED) + ([a.submitted] if a.submitted else []) + ([str(p3file), str(REAL_POINTS)] if p3file else [])
    if plan(a, reads + [str(KIT_FRAMES)], [str(p) for p in outs]):
        return
    if mode == "off" and not a.submitted:
        stop("--p3 off reports P3's boxes from the submitted predictions: pass --submitted")
    if os.environ.get("UT_REALISM_FIT"):
        stop("UT_REALISM_FIT is set: it changes P3's static map; unset it")
    if p3file and not p3file.exists():
        stop(f"{p3file} not found (python tools/rebuild_v2x_test.py p3-ids writes it)")
    raw27 = Path(a.v27).read_bytes()
    m27 = hashlib.md5(raw27).hexdigest()
    print(f"v27 anchor {a.v27}: md5 {m27} vs {V27_MD5}: {m27 == V27_MD5}")
    if m27 != V27_MD5:
        stop("the truck anchor must be the v27 zip's predictions.json")
    rows = []
    for p, orig in REFINED.items():
        if not Path(p).exists():
            stop(f"{p} missing (GPU refine step)")
        got = md5_file(p)
        rows.append((p, got, orig, "yes" if got == orig else "NO"))
    table(("second-stage file", "rebuilt md5", "original md5", "same?"), rows)
    ex = [str(p) for p in outs if p.exists()]
    if ex and not a.force:
        stop(f"{ex} exist (pass --force to overwrite)")
    ensure_fork()
    import numpy as np
    V35, V = shipped("build_v2x_v35", "build_v2x_v32")
    empty = np.zeros(0, np.int64)
    rep = {"p3": mode, "p3_ids": str(p3file) if p3file else None, "refined": {r[0]: {"rebuilt": r[1], "original": r[2]} for r in rows}}
    if mode == "ids":
        (R1,) = shipped("realism_r1")
        R1.CFG["v2x"]["fit"] = str(p3file)
        keys = V.PP.static_keys("v2x")
        print(f"P3 static map voxels {len(keys)} vs the original {P3_VOXELS} (build_v2x_v32.V31_KEYS {V.V31_KEYS}): {len(keys) == P3_VOXELS}")
        rep["p3_voxels"] = [int(len(keys)), P3_VOXELS]
    else:
        keys = empty
        print("P3 off: the pedestrian chain runs with an empty static map, so postproc_p123.filt lowers no box")
    veh = V.BUILD["vehicle"]["q7_dsvt_vmamba_pvn"]("test", keys)
    ped = V35.dsvt_ped("test", keys, V.PZ["dz"])
    ped0 = V35.dsvt_ped("test", empty, V.PZ["dz"]) if mode == "ids" else ped
    v27 = json.loads(raw27)
    if not set(veh) == set(ped) == set(v27):
        stop("vehicle, pedestrian and v27 frames differ")
    if mode == "ids":
        low = sum(d["score"] != d0["score"] for f in ped for d, d0 in zip(ped[f], ped0[f]))
        tot = sum(d["class"] == "pedestrian" for v in ped.values() for d in v)
        print(f"P3 on test: {low} of {tot} pedestrian boxes lowered vs the original {P3_TEST[1]} of {P3_TEST[0]}")
        rep["p3_lowered"] = [low, tot, P3_TEST[1], P3_TEST[0]]
    out = assemble(v27, veh, ped, V.B.keep)
    text = json.dumps(out)
    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_FINAL.write_text(text)
    m = md5_text(text)
    print(f"wrote {OUT_FINAL}: md5 {m} vs v2x_v35 {V35_MD5}: {'SAME' if m == V35_MD5 else 'different'}")
    crow = []
    for c in V.CL:
        b = sum(d["class"] == c for x in out.values() for d in x)
        h = sum(d["class"] == c and d["score"] >= 0.3 for x in out.values() for d in x)
        crow.append((c, f"{b}/{V35_COUNTS[c][0]}", f"{h}/{V35_COUNTS[c][1]}"))
    table(("class", "boxes new/v35", ">=0.3 new/v35"), crow)
    rep["out"] = {"file": str(OUT_FINAL), "md5": m, "same": m == V35_MD5}
    if a.submitted:
        sub = json.loads(Path(a.submitted).read_text())
        if set(sub) != set(out):
            stop(f"{a.submitted} holds other frames")
        cls = {}
        for c in V.CL:
            bad = [f for f in sub if V.B.canon([d for d in out[f] if d["class"] == c]) != V.B.canon([d for d in sub[f] if d["class"] == c])]
            cls[c] = len(bad)
            print(f"{c}: box for box equal to the submission (builders' precision: box 6 dp, score 8 dp) in {len(sub) - len(bad)} of {len(sub)} frames")
        rep["class_frames_differing"] = cls
        n, lowered, lrows = p3_from_submission(ped0, sub, a.match_radius, a.ratio_tol)
        print(f"\nP3 read from the submission (rebuilt pedestrians before P3 vs submitted; ratio tolerance {a.ratio_tol}, "
              f"pairing radius {a.match_radius} m):")
        table(("pairs and score ratios", "boxes"), sorted(n.items()))
        k = sum(len(v) for v in lowered.values())
        print(f"boxes the submission's P3 lowered: {k} (the original run lowered {P3_TEST[1]} of {P3_TEST[0]})")
        rep["p3_from_submission"] = {"counts": dict(n), "lowered": k, "boxes": lrows}
        if mode == "ids":
            mine = {(f, i) for f in ped for i, (d, d0) in enumerate(zip(ped[f], ped0[f])) if d["score"] != d0["score"]}
            theirs = {(f, i) for f, v in lowered.items() for i in v}
            agree = {"both": len(mine & theirs), "rebuilt only": len(mine - theirs), "submission only": len(theirs - mine)}
            print(f"P3 decisions, rebuilt map vs the submission: {agree}")
            rep["p3_agreement"] = agree
        else:
            out_t = assemble(v27, veh, transfer(ped0, lowered), V.B.keep)
            text_t = json.dumps(out_t)
            OUT_P3SUB.write_text(text_t)
            mt = md5_text(text_t)
            print(f"wrote {OUT_P3SUB} (P3 taken from the submission): md5 {mt} vs v2x_v35 {V35_MD5}: {'SAME' if mt == V35_MD5 else 'different'}")
            rep["out_p3sub"] = {"file": str(OUT_P3SUB), "md5": mt, "same": mt == V35_MD5}
    save_report("final", rep)


# ------------------------------------------------------------------ compare
def load_compare():
    for p in (Path(__file__).resolve().parent / "compare_predictions.py", Path("tools/compare_predictions.py")):
        if p.exists():
            spec = importlib.util.spec_from_file_location("compare_predictions", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    stop("tools/compare_predictions.py not found")


def packager_md5(path):
    preds = json.loads(Path(path).read_text())
    preds = {f: sorted(d, key=lambda x: -x["score"]) for f, d in preds.items()}   # utl/package.py: highest score first, stable
    return md5_text(json.dumps(preds))


def p3sub_current():
    """True when OUT_P3SUB was written by the latest final run (its md5 is in output/v2x/rebuild/final.json)."""
    rp = REPORTS / "final.json"
    if not (OUT_P3SUB.exists() and rp.exists()):
        return False
    rec = json.loads(rp.read_text()).get("out_p3sub") or {}
    return rec.get("md5") == md5_file(OUT_P3SUB)


def cmd_compare(a):
    news = a.new or [str(OUT_FINAL)] + ([str(OUT_P3SUB)] if p3sub_current() else [])
    if plan(a, [a.submitted] + news, []):
        return
    CP = load_compare()
    res = {}
    for n in news:
        if not Path(n).exists():
            stop(f"{n} not found")
        print(f"\n=== {n}")
        saved = sys.argv
        sys.argv = ["compare_predictions.py", "--ref", a.submitted, "--new", n, "--min_score", str(a.min_score), "--radius", str(a.radius)]
        try:
            CP.main()
        finally:
            sys.argv = saved
        m = packager_md5(n)
        print(f"\npackager serialisation (score-sorted, json.dumps) md5 {m} vs the submitted {V35_MD5}: {'SAME' if m == V35_MD5 else 'different'}")
        res[n] = {"packager_md5": m, "same": m == V35_MD5}
    save_report("compare", res)


# ------------------------------------------------------------------ main
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".", help="root of the copy of the release tree (default: the current directory)")
    common.add_argument("--dry-run", action="store_true", help="print the files the stage reads and writes, then stop")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("members-check", parents=[common], help="md5 of each member test prediction vs the original run")
    s.add_argument("--strict", action="store_true", help="exit 1 unless every member file equals the original bytes")
    s = sub.add_parser("lanes", parents=[common], help="test-split lanes: DSVT joint (all classes) and Q7 joint vehicle")
    s.add_argument("--workers", type=int, default=6, help="frame-chunk processes, as the original runs (0: one process)")
    s.add_argument("--parts", nargs="+", choices=("dsvt", "q7"), default=["dsvt", "q7"])
    s.add_argument("--force", action="store_true", help="overwrite lane files that exist")
    s = sub.add_parser("p3-ids", parents=[common], help="regenerate the rfit_i1 list (P3's 128 frames) from the V2X-Real test split")
    s.add_argument("--raw", default="data/raw/v2xreal/extracted", help="extracted V2X-Real (with test/<clip>/-1/*.bin)")
    s.add_argument("--test-list", default="data/pcdet/v2x_real/ImageSets/test.txt", help="ImageSets list from utl.prepare --split test")
    s.add_argument("--out", default=str(RFIT))
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("final", parents=[common], help="final test predictions from the refined lanes, the v27 trucks and the rules")
    s.add_argument("--v27", required=True, help="predictions.json of the v2x_v27 zip (md5 fc5b57f3...)")
    s.add_argument("--p3", nargs="+", required=True, metavar="off|ids FILE", help="'off', or 'ids FILE' with the rfit_i1 frame list")
    s.add_argument("--submitted", help="predictions.json of the ranked v2x_v35d zip (needed with --p3 off)")
    s.add_argument("--ratio-tol", type=float, default=0.01, help="relative tolerance of the score ratio classes")
    s.add_argument("--match-radius", type=float, default=0.1, help="pairing radius (m) for boxes that are not identical")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("compare", parents=[common], help="tools/compare_predictions.py vs the submission, and the packager md5")
    s.add_argument("--submitted", required=True)
    s.add_argument("--new", nargs="+", help=f"rebuilt files (default {OUT_FINAL}, and {OUT_P3SUB} when present)")
    s.add_argument("--min_score", type=float, default=0.05)
    s.add_argument("--radius", type=float, default=0.5)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    root = Path(a.root).resolve()
    if not (root / "scripts" / "build_v2x_v35.py").exists() or not (root / "utl").is_dir():
        stop(f"{root} is not the root of the V2X release tree (scripts/build_v2x_v35.py, utl/)")
    os.chdir(root)
    for p in (str(root), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    {"members-check": cmd_members, "lanes": cmd_lanes, "p3-ids": cmd_p3_ids, "final": cmd_final, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
