# UrbanTwin Sim2Real LiDAR Challenge, LUMPI track: 3rd place

This folder describes the code, configs and build steps behind entry **`lumpi_v62d`** (team **sadbodh**, Sadbodh Sharma) in the
[UCF UrbanTwin Sim2Real LiDAR Challenge](https://urbanity-lab.github.io/bdc2026-urbantwin-sim2real/) at the IEEE BigData Cup 2026, LUMPI track.
- Every detector and second stage was trained only on the **UT-LUMPI** synthetic digital twin.
- The organisers ranked the entry **3rd of 44 teams**.

| Final-phase leaderboard, submission 1054262 | |
|---|---|
| Combined score | **0.5377** |
| 3D mAP, IoU 0.5 (R40) | 0.1885 |
| 3D mAP, IoU 0.7 (R40) | 0.0719 |
| BEV mAP, IoU 0.5 (R40) | 0.1988 |
| Realism: CD / MMD / EMD / FPD | 1.5639 / 0.000023 / 1.1773 / 0.1770 |

The trained weights and the submitted `submission.zip` are on Hugging Face: [Sadbodh/urbantwin-sim2real-lidar](https://huggingface.co/Sadbodh/urbantwin-sim2real-lidar), folder `lumpi/`.

The code is shared by both tracks and lives at the repository root ([top README](../README.md)); run every command below from the repository root.

## Method in brief

**1. Twin fixes.** The synthetic objects of UT-LUMPI are rescaled to published generic class sizes, and riders are seated on the riderless two-wheelers (`utl/objfix.py`, `utl/synth_ped.py`).
- Two further training sets, `ut_lumpi_rs` and `ut_lumpi_fix2_rs`, rescale Car, Van, Truck and Bicycle (see Real-data use, item 2).

**2. The v29 lane.** All members run with 4-view flip test-time augmentation (TTA). One fusion lane covers all eight classes:
- base members: PointPillars `pp_lumpi_long` (weight 1.0) and `pp_lumpi_d50` (0.5), CenterPoint `cpv_lumpi_long` without Person (1.0), and PointPillars on all frames `pp_lumpi_long_all` (0.5);
- per-class members: `pp_lumpi_br`, `pp_lumpi_walk`, `pp_lumpi_walk_all` and Voxel R-CNN `vrcnn_lumpi_fix` (L11);
- WBF at IoU 0.55, class copies, then the point tests: column, truck length, size prior, L9, static bicycle, truck width, rider and pole;
- a learned Person rescorer.

**3. Per class:**
- **Car:** a lane that adds L11-RS (`vrcnn_lumpi_rs`, trained on `ut_lumpi_rs`).
- **Van and Truck:** L11 second-stage boxes, then the height rule (h < 2.0 m → ×0.3) and the point-count rules T5 and V5.
- **Bicycle:** the L11b (`vrcnn_lumpi_b`) second stage, rule R4, then the static-background map P3.
- **Person:** six Voxel R-CNN / PV-RCNN++ second stages, merged by R0. They are L11, L12 (`vrcnn_lumpi_c`), `vrcnn_lumpi_b_s2`, `pv_lumpi`, `vrcnn_lumpi_fix_s2` and L12-Person (`spec_lumpi_person_rs`). After them come the size step (+0.044 m length, +0.023 m width, +0.10 m height), the length rule (l ≥ 0.85 m → ×0.3), P3 and the two-wheeler Person copies (×0.3).
- **Motorcycle, Bus and Unknown:** the v29 lane boxes.

**4. Synthetic frames.** The 50 frames are twin frames after realism calibration (`utl/realism_emit.py`; see Real-data use, item 4).

Each step, with its parameters, is in the build scripts listed under [Level 2](#level-2-rebuild-from-the-checkpoints). The "Real-data use" section below states every way real data entered the entry.

## Level 1: check the submitted package (CPU, minutes)

The submitted file can be checked byte for byte:

```bash
docker build -t urbantwin-realism:cpu -f docker/Dockerfile.realism .
# official starting kit (Codabench competition 16684, Files tab): copy detection_test_frame_ids.json into configs/starter_kit_mirror/lumpi/
# download lumpi/submission/submission.zip from the Hugging Face repo above
docker run --rm -v "$PWD":/workspace urbantwin-realism:cpu \
    python tools/verify_submission.py check --track lumpi --zip submission.zip
unzip submission.zip -d sub
docker run --rm -v "$PWD":/workspace urbantwin-realism:cpu \
    python tools/verify_submission.py repack --track lumpi --pred sub/predictions.json --synthetic sub/synthetic \
    --out repacked.zip --name "Sadbodh Sharma" --date 2026-11-15
```

Both commands print `ALL MATCH` for these three parts:

| Part | md5 |
|---|---|
| `predictions.json` | `500b2d308d4df3d2c3e47abb6f25ab30` |
| the 50 `synthetic/*.bin` frames, concatenated in order | `811425d61c2c793c61101278299a032e` |
| `declaration.txt` | `b694dd4eed70290e2828cb767d8473d0` |

- The build output `lumpi_v62_test.json` had md5 `446102cea120e86bc49adcf5cdabc02c`.
- The packager sorts each frame's boxes by score (a stable sort) and frame ids in order, and that gives `500b2d30`.
- Both files hold the same 390,275 boxes.

## Level 2: rebuild from the checkpoints

Level 2 rebuilds the entry from the released checkpoints:
- it runs every member, lane, second stage, rule and the packaging;
- it compares the result with the submitted `predictions.json` using `tools/compare_predictions.py`.

Level 2 has not been run end to end for this track. The V2X-Real folder ([v2x/README.md](../v2x/README.md)) has a test-split rebuild that was run and measured.

GPU steps do not give bit-identical floats on another GPU. Most models ran on an RTX 3090, and L12-Person was trained and run on an A100. So a rebuild is judged by box matches, not by md5 (see [Known differences](#known-differences)).

### Environment
- **GPU:** RTX 3090 (sm_86), Docker. `docker build -t urbantwin-lidar:latest .` gives PyTorch 2.4.1, CUDA 12.1, OpenPCDet commit `233f849`.
- **CPU:** `urbantwin-realism:cpu` (above) runs the lanes, rules and builds, with shapely.
  - The rescorer also needs scikit-learn 1.5.2 on `PYTHONPATH`. The scripts read it from `.pylib_sk152`, which you create with `pip install --target .pylib_sk152 scikit-learn==1.5.2` (Python 3.11, as in the image).
- **Runner:** `scripts/docker_run.sh` runs a command in the GPU image; set `UT_NOGPU=1` for CPU-only steps.
- **CPU form:** `docker run --rm -v $PWD:/workspace -w /workspace -e PYTHONPATH=.pylib_sk152:.:scripts urbantwin-realism:cpu python …`

### Data
None of the challenge data is included here. Download each set from its source:
- **UT-LUMPI twin:** [Harvard Dataverse, doi:10.7910/DVN/D9SSWD](https://doi.org/10.7910/DVN/D9SSWD), licence CC0 1.0. Put the `UCF-UT-LUMPI-Seq-1.zip.NNN` parts in `data/raw/ut_lumpi/`.
- **LUMPI Measurement 4 with labels:** [data.uni-hannover.de/dataset/lumpi](https://data.uni-hannover.de/dataset/lumpi), licence CC BY-NC 3.0. Put it in `data/raw/lumpi/`.
- **Official starting kit:** Codabench competition 16684, Files tab.
  - Place the kit at `data/kit/lumpi/`.
  - Copy `detection_test_frame_ids.json`, `forbidden_frames.txt` and `config.json` into `configs/starter_kit_mirror/lumpi/`.
  - `utl/guard.py` refuses any real frame listed in `forbidden_frames.txt` or within 20 frames of one.

**Prepare the training and evaluation sets:**
```bash
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare ut --track lumpi --raw data/raw/ut_lumpi --out data/pcdet/ut_lumpi
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare ut_fix --src data/pcdet/ut_lumpi --out data/pcdet/ut_lumpi_fix --seed 0
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare infos --root data/pcdet/ut_lumpi_fix --track lumpi --gt_db --db_every 3
UT_NOGPU=1 scripts/docker_run.sh python -m utl.stretch_db --root data/pcdet/ut_lumpi_fix
scripts/make_ut_lumpi_fix2.sh                         # ut_fix with the v2 rider model -> data/pcdet/ut_lumpi_fix2
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare lumpi_real --lidar data/raw/lumpi/Measurement4 \
    --labels data/raw/lumpi/extracted/Label/Measurement4/Label.csv --out data/pcdet/lumpi_real   # val, realism_fit and realism_check splits (Real-data use, items 1 and 4)
UT_NOGPU=1 scripts/docker_run.sh python -m utl.thin --track lumpi --src data/pcdet/lumpi_real --out data/pcdet/lumpi_real_50k
cp -R data/pcdet/ut_lumpi_fix data/pcdet/ut_lumpi_rs   # the script resizes this copy in place
scripts/make_ut_lumpi_rs.sh                           # ut_lumpi_rs (see Real-data use, item 2)
scripts/make_ut_lumpi_fix2_rs.sh                      # ut_lumpi_fix2_rs: the same sizes on ut_lumpi_fix2
```
- **Fresh split.** The fresh split (see Real-data use, item 1) has ids 10, 30, …, 6390. Write the list with `seq -f '%06g' 10 20 6390 > logs/fresh_ids.txt`, then thin them to 50,000 points like `lumpi_real_50k`, into `data/pcdet/lumpi_real_fresh50k`.
- **Synthetic frames.** `data/pcdet/ut_lumpi` also feeds the 50 synthetic frames.

### 1. Training
Each model trains with:
```bash
scripts/docker_run.sh python -m utl.train --cfg_file configs/pcdet/<model>.yaml --workers 7 --num_epochs_to_eval 1 --fix_random_seed
```
The train log of every model, with its full config dump, is in the Hugging Face repo (`lumpi/checkpoints/<model>/`).

| Model | Config | Training set | Epochs | Role in lumpi_v62d |
|---|---|---|---|---|
| PointPillars | `pp_lumpi_long` | ut_lumpi_fix | 20 | lane base, weight 1.0 |
| PointPillars, density-robust | `pp_lumpi_d50` | ut_lumpi_fix | 20 | lane base, 0.5 |
| CenterPoint (voxel) | `cpv_lumpi_long` | ut_lumpi_fix | 15 | lane base, 1.0 (Person removed) |
| PointPillars, all frames | `pp_lumpi_long_all` | ut_lumpi_fix | 20 | lane base, 0.5 |
| PointPillars | `pp_lumpi_br` | ut_lumpi_fix2 | 20 | lane member: Person, Car, Bicycle, Motorcycle, Bus, Van, Unknown |
| PointPillars, walkers | `pp_lumpi_walk` | ut_lumpi_fix | 20 | lane member: Truck |
| PointPillars, walkers, all frames | `pp_lumpi_walk_all` | ut_lumpi_fix | 20 | lane member: Person, Truck |
| Voxel R-CNN (L11) | `vrcnn_lumpi_fix` | ut_lumpi_fix | 15 | lane member: Person; second stage (Person, Van, Truck, Motorcycle) |
| Voxel R-CNN (L11-RS) | `vrcnn_lumpi_rs` | ut_lumpi_rs | 15 | Car lane member |
| Voxel R-CNN (L12) | `vrcnn_lumpi_c` | ut_lumpi_fix2 | 15 | Person second stage |
| Voxel R-CNN | `vrcnn_lumpi_b_s2` | ut_lumpi_fix2 | 15 | Person second stage |
| PV-RCNN++ | `pv_lumpi` | ut_lumpi_fix2 | 15 | Person second stage |
| Voxel R-CNN | `vrcnn_lumpi_fix_s2` | ut_lumpi_fix | 15 | Person second stage |
| Voxel R-CNN, Person only (L12-Person) | `spec_lumpi_person_rs` | ut_lumpi_fix2_rs | 15 | Person second stage |
| Voxel R-CNN (L11b) | `vrcnn_lumpi_b` | ut_lumpi_fix2 | 15 | Bicycle second stage |

### 2. Member inference (GPU)
Put the released checkpoints at `output/pcdet/<model>/default/ckpt/checkpoint_epoch_<N>.pth`.
```bash
P=output/pcdet
for m in pp_lumpi_long pp_lumpi_d50 pp_lumpi_long_all pp_lumpi_walk pp_lumpi_walk_all pp_lumpi_br; do
  scripts/kit_predict.sh lumpi configs/pcdet/$m.yaml $P/$m/default/ckpt/checkpoint_epoch_20.pth kit_mc --multi_class_nms --max_dets 100
done
scripts/kit_predict.sh lumpi configs/pcdet/cpv_lumpi_long.yaml $P/cpv_lumpi_long/default/ckpt/checkpoint_epoch_15.pth kit
scripts/kit_predict.sh lumpi configs/pcdet/vrcnn_lumpi_fix.yaml $P/vrcnn_lumpi_fix/default/ckpt/checkpoint_epoch_15.pth kit
scripts/kit_predict.sh lumpi configs/pcdet/vrcnn_lumpi_rs.yaml $P/vrcnn_lumpi_rs/default/ckpt/checkpoint_epoch_15.pth kit --one_per_view --save_views
scripts/fresh_check_predict.sh      # the same members on the fresh split
```
- `kit_predict.sh` runs `utl.infer --tta`, which uses 4 flip views fused by WBF. It writes `output/lumpi/<model>/<tag>/pred_test.json` and `pred_val50k.json`.

**CPU preparation:**
```bash
bash scripts/lumpi_l7_lane.sh 1.0 1                    # cpv_lumpi_long without Person -> output/lumpi/ens_l7/cpvnp1.0_cpv_{val,test}.json
mkdir -p output/lumpi/vrcnn_lumpi_fix/kit_mc output/lumpi/g3in output/lumpi/rescorer
cp output/lumpi/vrcnn_lumpi_fix/kit/pred_*.json output/lumpi/vrcnn_lumpi_fix/kit_mc/
cp output/lumpi/vrcnn_lumpi_rs/kit/pred_test.json output/lumpi/g3in/vrcnn_lumpi_rs_test.json
cp lumpi_rescorer.pkl output/lumpi/rescorer/        # from the Hugging Face repo (lumpi/rescorer/)
```
`rescorer/r3_*.py` hold the rescorer's training and apply code.

### 3. The v29 lane (CPU)
```bash
for s in val test fresh; do
  python scripts/lumpi_lane_fast.py zPK '[["pp_lumpi_br",1,["Person","Car","Bicycle","Motorcycle","Bus","Van","Unknown"]],["pp_lumpi_walk",1,["Truck"]],["pp_lumpi_walk_all",1,["Person","Truck"]],["vrcnn_lumpi_fix",1,["Person"]]]' --splits $s
  python scripts/save_lumpi_chain.py output/lumpi/ens_x/zPK output/lumpi/ens_x/lumpi_v29_$s.json $s
done
```

### 4. Second stages on the v29 lane (GPU)
- **Frames per split:**
  - val: `data/pcdet/lumpi_real_50k/points`;
  - test: `data/kit/lumpi/detection_test_frames`;
  - fresh: `data/pcdet/lumpi_real_fresh50k/points`.
- **Input:** `output/lumpi/ens_x/lumpi_v29_<split>.json`.
```bash
P=output/pcdet; E=output/lumpi/ens_x
ref() {  # model epoch outdir outname extra-args
  local m=$1 e=$2 o=output/lumpi/$1/$3 n=$4; shift 4; mkdir -p $o
  for s in val test fresh; do
    case $s in val) F=data/pcdet/lumpi_real_50k/points ;; test) F=data/kit/lumpi/detection_test_frames ;; fresh) F=data/pcdet/lumpi_real_fresh50k/points ;; esac
    scripts/docker_run.sh python scripts/x6_refine.py --cfg configs/pcdet/$m.yaml --ckpt $P/$m/default/ckpt/checkpoint_epoch_$e.pth \
        --frames $F --pred $E/lumpi_v29_$s.json --out $o/v29_${s}_$n.json "$@"
  done
}
ref vrcnn_lumpi_fix      15 l11c  ref                       # L
ref vrcnn_lumpi_fix      15 l11c  ref_min01 --min_score 0.01
ref vrcnn_lumpi_b        15 l11bc ref                       # B (Bicycle)
ref vrcnn_lumpi_c        15 l12c  ref_min01 --min_score 0.01
ref vrcnn_lumpi_b_s2     15 s2c   ref_min01 --min_score 0.01
ref vrcnn_lumpi_fix_s2   15 s2    min01     --min_score 0.01
ref pv_lumpi             15 s2p   min01     --min_score 0.01 --only_class Person
ref spec_lumpi_person_rs 15 s2    min01     --min_score 0.01 --only_class Person
```
- `scripts/x6_refine.py` uses 4 flip views and two passes.
- L12-Person's refine in the ranked entry ran on an A100.

### 5. Car lane, the v59 base and v62 (CPU)
```bash
python scripts/lumpi_lane_fast.py zG3IGc00 '[["pp_lumpi_br",1,["Person","Car","Bicycle","Motorcycle","Bus","Van","Unknown"]],["pp_lumpi_walk",1,["Truck"]],["pp_lumpi_walk_all",1,["Person","Truck"]],["vrcnn_lumpi_fix",1,["Person"]],["file:g3in/vrcnn_lumpi_rs",1,["Car"]]]' --baseovr '{}' --splits test --workers 6
python scripts/save_lumpi_chain.py output/lumpi/ens_x/zG3IGc00 output/lumpi/ens_x/zG3IGc00_chain_test.json test
```

**v59 as it was built.** These steps run in order on the val, fresh and test files, and each one stops unless its logged readings rebuild:
1. `lumpi_plow.py`
2. `lumpi_v34_hr.py`
3. `lumpi_v35_footprint.py logs/fresh_ids.txt`
4. `lumpi_v36_rules.py logs/fresh_ids.txt`
5. `build_lumpi_v39.py`
6. `build_lumpi_v40.py`
7. `build_lumpi_v41.py`
8. `lumpi_pv_s2_routes.py logs/fresh_ids.txt`
9. `build_lumpi_bicycle_r4.py logs/fresh_ids.txt`
10. `postproc_p123.py lumpi P3 logs/fresh_ids.txt`
11. `postproc_p3_time.py logs/fresh_ids.txt`
12. `build_lumpi_v57.py`
13. `route_lanes_lumpi.sh vrcnn_lumpi_rs`, then `l11rs_read.py vrcnn_lumpi_rs` and `build_lumpi_l11rs.py vrcnn_lumpi_rs`. Then `cp output/lumpi/ens_x/lumpi_l11rs_next_test.json output/lumpi/ens_x/lumpi_v58_test.json`. `route_lanes_lumpi.sh` ran across two machines: the lanes ran on a CPU host and the second stages on the GPU host over ssh (`UT_3090_ROOT`). On one machine, run its lane and refine commands directly.
14. `build_lumpi_v59.py`

**v62.** `python scripts/build_lumpi_v62.py` gives `output/lumpi/ens_x/lumpi_v62_test.json`. It is v59 with Person rebuilt from the six Person second stages.

### 6. Synthetic frames
```bash
python -m utl.realism_emit --track lumpi --syn data/pcdet/ut_lumpi/points --real_dir data/pcdet/lumpi_real/points \
    --real_fit data/pcdet/lumpi_real/ImageSets/realism_fit.txt --real_check check=data/pcdet/lumpi_real/ImageSets/realism_check.txt \
    --zeros drop --cfg '{"voxels": [[6, 6, 2]], "align": true, "jitter": 0.0, "pick": "match"}' --out output/lumpi/realism_v2
```
- The frames are written to `output/lumpi/realism_v2`.
- The density target and the check frames are described in Real-data use, item 4.
- The calibration seed is 0.

### 7. Package and compare
```bash
scripts/package_check.sh lumpi output/lumpi/ens_x/lumpi_v62_test.json output/lumpi/realism_v2 lumpi_v62
python tools/compare_predictions.py --ref sub/predictions.json --new output/lumpi/ens_x/lumpi_v62_test.json
```

### Known differences
- **Builders as used.** The build and lane scripts are the ones that made the upload; only comments and message strings were edited for this release. They lock the md5s of the original run's lane, second-stage and reading files (the records in `logs/`). Files from a rebuild have new md5s, so these checks stop the build until you replace the locked md5s with your own files' values.
- **GPU floats.** Re-running the L11 refine on the same GPU moved up to 8.2e-3 on 3 val frames. L12-Person ran on an A100. So a rebuilt `predictions.json` matches by box, not by md5.
- **Anchor files.** `build_lumpi_v62.py` checks the md5 of the v59 file (`057209a5`), which was not kept. The v59 upload's `predictions.json` is in the Hugging Face repo under `lumpi/submission/anchors/`. To start v62 from a rebuilt v59, remove that one check.
- **Rescorer.** The trained Person rescorer `lumpi_rescorer.pkl` (md5 `e60c3a289ba9221c58f8c1defbac121a`) is in the Hugging Face repo with the other weights.
- **Fixed inputs.** Motorcycle, Bus and Unknown come unchanged from the v29 lane. The Hugging Face repo also holds the v29 and v59 uploads' `predictions.json` under `lumpi/submission/anchors/`, so the build can start from them.
- **Commands not kept.** These steps match the files but their command lines were not kept: the `kit` → `kit_mc` copy of L11, the realism emit, and the fresh 50k thinning. The commands above are rebuilt from the scripts' defaults and the logged settings.

## Real-data use

The text below is the "Real-data use" section of the report we sent to the organisers, copied word for word.

No detector or second stage was trained on real frames or real labels; every model trains on the UT-LUMPI twin. Real data is the public LUMPI Measurement 4 with its labels. No frame in forbidden_frames.txt, and no frame within 20 frames of one, was used for anything except inference on the test frames.
1. **Evaluation and selection.** Labelled splits score every change against the gate: val (320 frames below frame 6400) and fresh (320 frames). A third split, NT, has 196 labelled frames: "before" (76 frames, 6401–6476) and "inside" (120 frames between 6765 and 8047). The inside frames are not forbidden, and each is at least 21 frames from a forbidden frame, but they lie within the span of the forbidden test frames. Three changes were chosen on the inside segment:
   - the two-wheeler Person copies, which are in our ranked package lumpi_v62d; we kept them by our own choice, although they failed the val and "before" conditions of the gate;
   - the Van member L12-Van, in lumpi_v61d, which was uploaded after lumpi_v60 and is not our ranked package;
   - the Person refiner L12-Person, in our ranked package lumpi_v62d, which we kept by our own choice after it failed the val condition of the gate.

2. **Box-size statistics.** The RS training data rescales synthetic Car, Van, Truck and Bicycle objects to size distributions taken from the labels of 160 real frames (realism_fit). Only the sizes are used; no real point or box enters training.
3. **Constants fitted on val labels.** These are the Person size step (+0.10 m height, +0.044 m length, +0.023 m width), the height and length rules, the point-count rules T5 and V5, and the static-Bicycle rule.
4. **Unlabelled real points.** The same 160 realism_fit frames, as points only, give:
   - the static-background map P3, which lowers Person and Bicycle scores ×0.3;
   - the targets of the realism calibration of our 50 synthetic frames.

   160 further unlabelled frames scored the calibration setting. Two learned point refiners for the realism frames were trained against the same 160 unlabelled frames and read on the other 160; neither passed, so neither is in the final package.

Validation, model selection, box-size statistics and realism calibration are on the organisers' list of allowed real-data uses. P3, the constants fitted on val labels and the choices made on the NT inside segment are not on that list as such; we list them here so that they can be judged.
