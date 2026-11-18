# vmamba_v2x copy of scripts/dsvt_v2x_lanes.py (845bdf81; queue-line spec in logs/vmamba_v2x_queue_lines.txt): only LANES, N and PRED_MD5
# changed; every check, chain and weight is dsvt_v2x's (the member at weight 1). Messages that say "dsvt_v2x" mean the member (vmamba_v2x).
"""dsvt_v2x joint lanes. v2x_v31's three stage-1 lane inputs with dsvt_v2x (trained on an A100 GPU, epoch 20; utl.infer --tta, default
score 0.01, no --max_dets) added as ONE more fusion member at weight 1, as every new V2X member before it (route_v2x_lanes.py: FSHNet / SAFDNet /
VoxelNeXt at weight 1). Nothing else changes: same members, weights, chains, frames and file format.
  vehicle     v31's lane output/v2x/fsh_saf_v2x/s2_in/vehicle_<s>.json = route_v2x_lanes_joint.veh(1, 1): x7 members and weights (pp 1, cp 1.5,
              rider 0.5, all-frames 0.5, walker 1.5, X6 3) + FSHNet 1 + SAFDNet 1, x2_fuse.chain vehicle (WBF 0.55, height refine); + dsvt_v2x 1.
  pedestrian  v31's lane output/v2x/fshnet_v2x/s2_in/pedestrian_<s>.json = route_v2x_lanes.py fshnet_v2x ped(1): x6bf members and weights (pp 1,
              cp 1, rider 0.5, all-frames 0, walker 2, X6b 2) + FSHNet 1, x2_fuse.chain pedestrian, ped_shape pole + short 1.0, then tall 2.3; + dsvt_v2x 1.
  truck       v31's lane = the v18 trucks output/v2x/ens/v18_<s>.json (class truck) = route_v2x_lanes.trk(0): pp 1 + all-frames pp 1, WBF IoU 0.4,
              height refine (pct 99.9), width >= 2.55 m; + dsvt_v2x 1 (route_v2x_lanes.trk(1)). Truck is read as a single-member
              reading [v31 truck base + dsvt_v2x], X6b + P.size (the pooled FSHNet + SAFDNet + DSVT truck lane is skipped).
Members exactly as scripts/v2x2_lanes.py (= route_v2x_lanes*.py): val_i1 / test output/v2x/sweep_in/{pp,cp,ped,ppall}_<sfx>.json +
output/v2x/pp_v2x_walk/kit_mc/pred_<sfx>.json, fresh route_v2x_lanes.FRESH_MEMBERS; X6 output/v2x/vrcnn_v2x/kit, X6b output/v2x/vrcnn_v2x_b/kit; FSHNet /
SAFDNet output/v2x/{fshnet_v2x,safdnet_v2x}/kit (v31's capped files); dsvt_v2x output/v2x/dsvt_v2x/kit/pred_<sfx>.json; frames sorted(v18_<s>.json).
STOP checks, all before anything is written:
  (3) v31's 6 lane files, the 3 v18 files and the 6 FSHNet / SAFDNet kit files equal their md5s below; the 3 dsvt_v2x kit files equal
      logs/dsvt_v2x_predict.md5 (the A100 predict job's record; exactly those 3 paths); every other member file's md5 is recorded in the report.
  (1) lane-off: this code WITHOUT the dsvt_v2x member rebuilds v31's 9 lanes (val_i1 / test / fresh x vehicle / pedestrian / truck) box for box
      (parsed equality, no tolerance; byte equality printed). Truck against the truck boxes of v18_<s>.json: the same boxes bit for bit (multiset)
      and the same score sequence per frame, so only boxes of equal score may trade places (v18's older writer orders ties differently;
      dry run Nov 14: 2 tied boxes at score 0.0082, below x6_refine's 0.05, in 1 of 146 val_i1 frames); the in-file-order result is recorded.
Frame-parallel as v2x2_lanes.py (every chain step works on one frame); the lane-off check proves the chunked run equals v31's lanes on every frame.
Writes, only if every check passes: output/v2x/dsvt_v2x_j/s2_in/{vehicle,pedestrian,truck}_<s>.json, logs/dsvt_v2x_j_lanes.md5 (md5sum format, the 9
new lanes), logs/dsvt_v2x_j_lanes.json (checks, md5s, box counts, times), logs/dsvt_v2x_j_lanes.done. No overwrite (refuses if any exists).
Test modes (write nothing but --report): --lane-off-only (only the lane-off chains; no dsvt_v2x file needed), --dry-run (all chains).
    UT_NOGPU=1 OMP_NUM_THREADS=1 docker run ... python scripts/dsvt_v2x_lanes.py --workers 6 2>&1 | tee logs/dsvt_v2x_j_lanes.log      (3090 CPU)"""
import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import ped_shape as PS
from route_v2x_lanes import E, FRESH_MEMBERS, WP, WV, keep, only, width
from utl import height_refine
from utl.ensemble import ensemble
from x2_fuse import chain

LANES = "vmamba_v2x_j"
SPLITS = ("test", "val_i1", "fresh")          # test first: the shortest lane-off check fails first
SFX = {"val_i1": "val50k", "test": "test", "fresh": "fresh50k"}
A, B, N = "fshnet_v2x", "safdnet_v2x", "vmamba_v2x"
CL = ("vehicle", "pedestrian", "truck")
LANE = {"vehicle": "output/v2x/fsh_saf_v2x/s2_in/vehicle_{s}.json", "pedestrian": "output/v2x/fshnet_v2x/s2_in/pedestrian_{s}.json",
        "truck": "output/v2x/ens/v18_{s}.json"}
# v31's lanes (vehicle / pedestrian: = scripts/v2x2_lanes.py LANE_MD5; truck: the v18 files X6b refined for v31's trucks, output/v2x/vrcnn_v2x_b/x6bc)
LANE_MD5 = {"output/v2x/fsh_saf_v2x/s2_in/vehicle_val_i1.json": "bc36abccdbfb978857bdc5e46919be01",
            "output/v2x/fsh_saf_v2x/s2_in/vehicle_test.json": "c5ad44c6e3d022915f37d5340f222767",
            "output/v2x/fsh_saf_v2x/s2_in/vehicle_fresh.json": "04d991191cf1dd622e823ddba70b8632",
            "output/v2x/fshnet_v2x/s2_in/pedestrian_val_i1.json": "8d860619437d76e66fb4e9b220d5d112",
            "output/v2x/fshnet_v2x/s2_in/pedestrian_test.json": "96320b41d208b68f89f7d2484afeba36",
            "output/v2x/fshnet_v2x/s2_in/pedestrian_fresh.json": "b5358fe9c8c70280f82de04acb216db6",
            "output/v2x/ens/v18_val_i1.json": "9035c7aa14068d58a834f8cd85ade6f3",
            "output/v2x/ens/v18_test.json": "ddaba482383f258752d6b41e4e2bd6e9",
            "output/v2x/ens/v18_fresh.json": "099b1a4b288b25534db0cb39f488b1c0"}
# v31's capped kit files (= scripts/v2x2_lanes.py KIT_MD5)
KIT_MD5 = {"output/v2x/fshnet_v2x/kit/pred_val50k.json": "e7cd475b690b6317918e870c7aab452c",
           "output/v2x/fshnet_v2x/kit/pred_test.json": "fa728ad136cf0e2897a28f8ee40c4574",
           "output/v2x/fshnet_v2x/kit/pred_fresh50k.json": "3090ad3b32eee49b1cce8452caa532e8",
           "output/v2x/safdnet_v2x/kit/pred_val50k.json": "4d28204379b20e0cf5f4adc73d1a919a",
           "output/v2x/safdnet_v2x/kit/pred_test.json": "06d883937e45590f7c975b2e5b598d43",
           "output/v2x/safdnet_v2x/kit/pred_fresh50k.json": "269e51c9eb539ae27da59449d14bcc31"}
PRED_MD5 = Path("logs/vmamba_v2x_pred.md5")
OUT = Path(f"output/v2x/{LANES}/s2_in")
LOGJ, LOGM, DONE = Path(f"logs/{LANES}_lanes.json"), Path(f"logs/{LANES}_lanes.md5"), Path(f"logs/{LANES}_lanes.done")
_D = {}


def stop(msg, rep=None, path=None):
    if rep is not None and path:
        rep["stopped"] = msg
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(rep, indent=1))
    sys.exit("STOP " + msg)


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def pred_md5():
    """{path: md5} from logs/dsvt_v2x_predict.md5, checked against the files; exactly the 3 dsvt_v2x kit paths."""
    want = {f"output/v2x/{N}/kit/pred_{x}.json" for x in SFX.values()}
    rec = dict(reversed(l.split()) for l in PRED_MD5.read_text().splitlines() if l.strip())
    if set(rec) != want:
        sys.exit(f"STOP {PRED_MD5} lists {sorted(rec)}, not the 3 dsvt_v2x kit files")
    for p, m in sorted(rec.items()):
        got = md5f(p)
        print(f"dsvt_v2x {p}: md5 {got} vs {PRED_MD5} {m}: {got == m}", flush=True)
        if got != m:
            sys.exit(f"STOP {p} md5 differs from {PRED_MD5}")
    return rec


def member_paths(split):
    """Exactly the paths route_v2x_lanes.py / route_v2x_lanes_joint.py read for this split, plus dsvt_v2x."""
    sfx = SFX[split]
    if split == "fresh":
        base5 = [f"output/v2x/{m}/pred_{sfx}.json" for m in FRESH_MEMBERS]
    else:
        base5 = [f"output/v2x/sweep_in/{n}_{sfx}.json" for n in ("pp", "cp", "ped", "ppall")] + [f"output/v2x/pp_v2x_walk/kit_mc/pred_{sfx}.json"]
    p = {"base5": base5, "x6": f"output/v2x/vrcnn_v2x/kit/pred_{sfx}.json", "x6b": f"output/v2x/vrcnn_v2x_b/kit/pred_{sfx}.json",
         "fsh": f"output/v2x/{A}/kit/pred_{sfx}.json", "saf": f"output/v2x/{B}/kit/pred_{sfx}.json", "dsvt": f"output/v2x/{N}/kit/pred_{sfx}.json"}
    flat = p["base5"] + [v for k, v in p.items() if k != "base5"]
    assert not any("/kit300" in x for x in flat), "kit300 / kit300s01 must never be read"
    return p


def load_pts(split, f):
    if split == "test":
        a = np.fromfile(Path("data/kit/v2x/detection_test_frames") / f"{f}.bin", np.float32).reshape(-1, 3)
    else:
        root = "v2x_real_50k" if split == "val_i1" else "v2x_fresh_50k"
        a = np.load(Path("data/pcdet") / root / "points" / f"{f}.npy")[:, :3]
    return a[np.abs(a).sum(1) > 0]


def veh(D, t, pts, frames):     # route_v2x_lanes_joint.veh(1.0, 1.0) [+ dsvt_v2x 1]
    w, m = WV + [1.0, 1.0], D["base5"] + [D["x6"], D["fsh"], D["saf"]]
    if t == "new":
        w, m = w + [1.0], m + [D["dsvt"]]
    return chain("vehicle", w, m, pts)


def ped(D, t, pts, frames):     # route_v2x_lanes.ped(1.0) with model fshnet_v2x [+ dsvt_v2x 1]
    w, m = WP + [1.0], D["base5"] + [D["x6b"], D["fsh"]]
    if t == "new":
        w, m = w + [1.0], m + [D["dsvt"]]
    p = chain("pedestrian", w, m, pts)
    p = PS.apply(p, lambda f: pts[f], "pedestrian", pole=True, short=1.0)
    return PS.apply(p, lambda f: pts[f], "pedestrian", tall=2.3)


def trk(D, t, pts, frames):     # route_v2x_lanes.trk(0) = the v18 trucks; trk(1) with dsvt_v2x as the model
    ms, ws = [only(D["base5"][0], "truck"), only(D["base5"][3], "truck")], [1.0, 1.0]
    if t == "new":
        ms, ws = ms + [only(D["dsvt"], "truck")], ws + [1.0]
    e = ensemble(ms, ws, iou_thr=0.4)
    return {f: width(height_refine.refine(e.get(f, []), pts[f], pct=99.9)) for f in frames}


FN = {"vehicle": veh, "pedestrian": ped, "truck": trk}


def work(job):
    split, k, n, kinds = job
    t0 = time.time()
    chunk = _D["frames"][k::n]
    pts = {f: load_pts(split, f) for f in chunk}
    sub = lambda m: {f: m[f] for f in chunk}
    D = {key: ([sub(m) for m in v] if key == "base5" else sub(v)) for key, v in _D.items() if key != "frames"}
    res = {}
    for c in CL:
        for t in kinds:
            p = FN[c](D, t, pts, chunk)
            res[(c, t)] = {f: sorted([keep(d) for d in p.get(f, []) if d["class"] == c], key=lambda d: -d["score"]) for f in chunk}
    return k, res, time.time() - t0


def run_split(split, kinds, workers):
    t0 = time.time()
    paths = member_paths(split)
    frames = sorted(json.loads((E / f"v18_{split}.json").read_text()))
    rd = lambda p: (lambda m: {f: m.get(f, []) for f in frames})(json.loads(Path(p).read_text()))
    _D.clear()
    _D["frames"] = frames
    _D["base5"] = [rd(p) for p in paths["base5"]]
    for key in ["x6", "x6b", "fsh", "saf"] + (["dsvt"] if "new" in kinds else []):
        _D[key] = rd(paths[key])
    t_load = time.time() - t0
    gc.collect()
    gc.freeze()   # forked workers share the loaded members copy-on-write
    n = min(len(frames), 4 * workers)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        parts = list(ex.map(work, [(split, k, n, kinds) for k in range(n)]))
    gc.unfreeze()
    out = {}
    for c in CL:
        for t in kinds:
            got = {}
            for _, res, _ in parts:
                got.update(res[(c, t)])
            assert set(got) == set(frames)
            out[(c, t)] = {f: got[f] for f in frames}
    _D.clear()
    gc.collect()
    used = {k: (v if k == "base5" else [v]) for k, v in paths.items() if k != "dsvt" or "new" in kinds}
    return out, frames, dict(load_s=round(t_load, 1), chains_s=round(time.time() - t0 - t_load, 1), chunks=n,
                             chunk_s_max=round(max(p[2] for p in parts), 1), members=used)


def stats(lane):
    return dict(boxes=sum(len(v) for v in lane.values()), confident=sum(d["score"] >= 0.3 for v in lane.values() for d in v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--lane-off-only", action="store_true", help="test: only the lane-off chains; writes nothing but --report")
    ap.add_argument("--dry-run", action="store_true", help="test: all chains; writes nothing but --report")
    ap.add_argument("--report", help="test modes: json report path")
    a = ap.parse_args()
    test = a.lane_off_only or a.dry_run
    if not test and sorted(a.splits) != sorted(SPLITS):
        sys.exit("STOP a full run builds all three splits (the second stages and the reading need val_i1, test and fresh)")
    splits = [s for s in SPLITS if s in a.splits]
    kinds = ["off"] if a.lane_off_only else ["off", "new"]
    rpath = a.report if test else str(LOGJ)
    T0 = time.time()
    mode = "lane_off_only" if a.lane_off_only else "dry_run" if a.dry_run else "full"
    rep = {"experiment": "dsvt_v2x joint lanes: v2x_v31's vehicle (fsh_saf_v2x), pedestrian (fshnet_v2x) and truck (v18) lanes + dsvt_v2x at weight 1",
           "lanes": LANES, "mode": mode, "workers": a.workers, "splits": splits, "script_md5": md5f(__file__)}
    print(f"{LANES} lanes: mode {mode}, splits {splits}, workers {a.workers}, script md5 {rep['script_md5']}", flush=True)
    if not test:
        ex = [str(OUT / f"{c}_{s}.json") for c in CL for s in SPLITS if (OUT / f"{c}_{s}.json").exists()] + [str(p) for p in (LOGM, LOGJ, DONE) if p.exists()]
        if ex:
            sys.exit(f"STOP {ex} exist (no overwrite)")
    # (3) inputs
    if "new" in kinds:
        rep["dsvt_v2x_md5"] = pred_md5()
    for name, rec in (("v31 capped kit", KIT_MD5), ("v31 lane", LANE_MD5)):
        for p, m in rec.items():
            got = md5f(p)
            print(f"{name} {p}: md5 {got} vs {m}: {got == m}", flush=True)
            if got != m:
                stop(f"{p} md5 {got} differs from v31's {m}", rep, rpath)
    rep["kit_md5"], rep["v31_lane_md5"] = KIT_MD5, LANE_MD5
    rep["member_md5"] = {}
    for s in splits:
        mp_ = member_paths(s)
        for p in mp_["base5"] + [mp_["x6"], mp_["x6b"]]:
            rep["member_md5"][p] = md5f(p)
    rep["lane_off"], rep["new"], rep["time"] = {c: {} for c in CL}, {c: {} for c in CL}, {}
    texts = {}
    for s in splits:
        out, frames, tm = run_split(s, kinds, a.workers)
        rep["time"][s] = {k: v for k, v in tm.items() if k != "members"}
        rep.setdefault("members", {})[s] = tm["members"]
        print(f"{s}: {len(frames)} frames, load {tm['load_s']} s, chains {tm['chains_s']} s ({tm['chunks']} chunks, longest {tm['chunk_s_max']} s)", flush=True)
        for c in CL:   # (1) lane-off
            p = LANE[c].format(s=s)
            raw = Path(p).read_text()
            ref = json.loads(raw)
            mine = out[(c, "off")]
            extra = {}
            if c == "truck":   # v18 holds every class; its truck boxes (keep keys only). v18 was written by older code (Nov 3) whose order among
                # boxes of EQUAL score differs from this sort in a few frames, so truck is compared exactly but tie-order-free: per frame the same
                # boxes (every value bit-identical, as a multiset) AND the same score sequence (so only equal-score boxes can trade places).
                ref = {f: [keep(d) for d in ref[f] if d["class"] == "truck"] for f in ref}
                same_bytes = None
                canon = lambda v: sorted(json.dumps(d, sort_keys=True) for d in v)
                in_order = list(ref) == list(mine) and ref == mine
                ties = [f for f in frames if ref.get(f) != mine.get(f)]
                same = list(ref) == list(mine) and all(canon(ref[f]) == canon(mine[f]) and [d["score"] for d in ref[f]] == [d["score"] for d in mine[f]]
                                                       for f in frames)
                extra = dict(in_file_order=bool(in_order), frames_tie_order_differs=ties)
            else:
                same_bytes = json.dumps(mine) == raw
                same = list(ref) == list(mine) and ref == mine
            st = stats(mine)
            rep["lane_off"][c][s] = dict(file=p, box_for_box=bool(same), bytes=same_bytes, **extra, **st)
            print(f"lane-off {s} {c}: rebuilt without dsvt_v2x vs {p}{' (truck boxes, tie-order-free)' if c == 'truck' else ''}: box for box {same}"
                  f"{'' if same_bytes is None else f', bytes {same_bytes}'} ({st['boxes']} boxes, {st['confident']} with score >= 0.3)", flush=True)
            if extra and not extra["in_file_order"]:
                print(f"NOTE lane-off {s} truck: same boxes and score sequence; only equal-score boxes in a different order in {len(ties)} frame(s) {ties[:3]}", flush=True)
            if not same:
                bad = [f for f in frames if ref.get(f) != mine.get(f)]
                print(f"lane-off {s} {c}: {len(bad)} of {len(frames)} frames differ, e.g. {bad[:3]}", flush=True)
                for f in bad[:2]:
                    A_, B_ = ref.get(f, []), mine.get(f, [])
                    print(f"  {f}: {len(A_)} boxes in the file vs {len(B_)} rebuilt", flush=True)
                    if len(A_) == len(B_) and A_:
                        dmax = max(max(abs(x - y) for x, y in zip([u["score"]] + u["box"], [v["score"]] + v["box"])) for u, v in zip(A_, B_))
                        print(f"  {f}: max abs difference (score, box) {dmax:.3g}", flush=True)
                stop(f"lane-off {s} {c}: the rebuild does not reproduce v31's lane", rep, rpath)
            if same_bytes is False:
                print(f"NOTE lane-off {s} {c}: boxes equal, json text differs", flush=True)
            if "new" in kinds:
                new = out[(c, "new")]
                t = json.dumps(new)
                sn = stats(new)
                rep["new"][c][s] = dict(md5=hashlib.md5(t.encode()).hexdigest(), **sn)
                print(f"dsvt_v2x {s} {c}: {st['boxes']} -> {sn['boxes']} boxes, score >= 0.3: {st['confident']} -> {sn['confident']}", flush=True)
                if not a.dry_run:
                    texts[(c, s)] = t
        del out
        gc.collect()
    rep["lane_off_passed"] = True
    rep["time"]["total_s"] = round(time.time() - T0, 1)
    rep["maxrss_mb"] = {"main": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), "largest_worker": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)}
    print(f"lane-off passed on {splits} x {list(CL)}; total {rep['time']['total_s']} s; max RSS main {rep['maxrss_mb']['main']} MB, largest worker {rep['maxrss_mb']['largest_worker']} MB", flush=True)
    if test:
        if a.report:
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text(json.dumps(rep, indent=1))
            print("wrote", a.report, flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in SPLITS:
        for c in CL:
            p = OUT / f"{c}_{s}.json"
            p.write_text(texts[(c, s)])
            m = md5f(p)
            assert m == rep["new"][c][s]["md5"], p
            lines.append(f"{m}  {p}")
            print("wrote", p, m, flush=True)
    LOGM.write_text("\n".join(lines) + "\n")
    LOGJ.write_text(json.dumps(rep, indent=1))
    DONE.write_text("ok\n")
    print(f"wrote {LOGM}, {LOGJ}, {DONE}", flush=True)


if __name__ == "__main__":
    main()
