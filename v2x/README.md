# UrbanTwin Sim2Real LiDAR Challenge, V2X-Real track: joint 3rd place

This folder describes the code, configs and build steps behind entry **`v2x_v35d`** (team **sadbodh**, Sadbodh Sharma) in the
[UCF UrbanTwin Sim2Real LiDAR Challenge](https://urbanity-lab.github.io/bdc2026-urbantwin-sim2real/) at the IEEE BigData Cup 2026, V2X-Real track.
- Every detector and second stage was trained only on the **UT-V2X-Real-IC** synthetic digital twin.
- The organisers ranked the entry **joint 3rd of 46 teams**.

| Final-phase leaderboard, submission 1054261 | |
|---|---|
| Combined score | **0.8788** |
| 3D mAP, IoU 0.5 (R40) | 0.5825 |
| 3D mAP, IoU 0.7 (R40) | 0.2106 |
| BEV mAP, IoU 0.5 (R40) | 0.6384 |
| Realism: CD / MMD / EMD / FPD | 1.1696 / 0.000204 / 0.9755 / 0.4691 |

The trained weights and the submitted `submission.zip` are on Hugging Face: [Sadbodh/urbantwin-sim2real-lidar](https://huggingface.co/Sadbodh/urbantwin-sim2real-lidar), folder `v2x/`.

The code is shared by both tracks and lives at the repository root ([top README](../README.md)); run every command below from the repository root.

## Method in brief

**1. Ten detectors, all trained on the twin.** They are built with [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and its added models:
- PointPillars in four training variants (`pp_v2x`, `pp_v2x_ped` with rider boxes cut to person size, `pp_v2x_all` on all frames, `pp_v2x_walk` with walkers);
- CenterPoint (`cp_v2x`);
- FSHNet (`fshnet_v2x`) and SAFDNet (`safdnet_v2x`);
- DSVT-Pillar (`dsvt_v2x`) and Voxel Mamba (`vmamba_v2x`);
- PV-RCNN++ (`pv_v2x`).

**2. Per-class lanes.** Each class has its own set of members. All members run with 4-view flip test-time augmentation (TTA) and are fused by weighted box fusion (WBF).
- Vehicle lane: 11 members, WBF IoU 0.55, then a height refine.
- Pedestrian lane: 7 members, WBF IoU 0.55, then point tests (column, low-point, recentre, pole, short and tall).
- Truck lane: 2 members, WBF IoU 0.4, then a height refine and a minimum width.

**3. Second stages.** Voxel R-CNN models re-score and re-box each lane's output:
- X6 (`vrcnn_v2x`) for vehicles;
- X6b (`vrcnn_v2x_b`) and a negatives model trained with synthetic street furniture (`vrcnn_v2x_neg`) for pedestrians, combined by rule R6;
- X6b for trucks.

**4. Rules.**
- Score rules X1, X3 and X8.
- The vehicle geometry step V-G, with a roof lock.
- The truck size step.
- The static-background map P3 for pedestrians.
- The pedestrian height offset P-Z.

**5. Synthetic frames.** The 50 submitted frames are twin frames after realism calibration, refined by the point refiner R2.

Each step, with its parameters, is in the build scripts listed under [Level 2](#level-2-rebuild-from-the-checkpoints). The "Real-data use" section below states every way real data entered the entry.

## Level 1: check the submitted package (CPU, minutes)

The submitted file can be checked byte for byte:

```bash
docker build -t urbantwin-realism:cpu -f docker/Dockerfile.realism .
# official starting kit (Codabench competition 16685, Files tab): copy detection_test_frame_ids.json into configs/starter_kit_mirror/v2x_real/
# download v2x/submission/submission.zip from the Hugging Face repo above
docker run --rm -v "$PWD":/workspace urbantwin-realism:cpu \
    python tools/verify_submission.py check --track v2x --zip submission.zip
unzip submission.zip -d sub
docker run --rm -v "$PWD":/workspace urbantwin-realism:cpu \
    python tools/verify_submission.py repack --track v2x --pred sub/predictions.json --synthetic sub/synthetic \
    --out repacked.zip --name "Sadbodh Sharma" --date 2026-11-15
```

Both commands print `ALL MATCH` for these three parts:

| Part | md5 |
|---|---|
| `predictions.json` | `be6122b96cf619280f260c1e34244d24` |
| the 50 `synthetic/*.bin` frames, concatenated in order | `8a31aefaf1bfe96de6e4e6a04c471116` |
| `declaration.txt` | `b694dd4eed70290e2828cb767d8473d0` |

The repack step re-sorts each frame's boxes by score and writes the zip the way `utl/package.py` did for the upload.

## Level 2: rebuild from the checkpoints

Level 2 rebuilds the entry from the released checkpoints:
- it runs every member, lane, second stage, rule and the packaging;
- it compares the result with the submitted `predictions.json` using `tools/compare_predictions.py`.

GPU steps do not give bit-identical floats on another GPU. The last refines ran on an RTX 3090 (vehicles) and on an A100 (pedestrians). So a rebuild is judged by box matches, not by md5 (see [Known differences](#known-differences)).

### Test-split rebuild (run and measured)
The original lane and build scripts insist on the `val_i1` and fresh splits and on the md5s of the original run's files. `scripts/rebuild_v2x_test.sh` rebuilds the 50 test frames only. It calls the same functions through `tools/rebuild_v2x_test.py` and compares each stage with the md5 the original run recorded.

**Needs:**
- the images and the checkpoints (see [Environment](#environment) and [Member inference](#2-member-inference-gpu));
- the kit's test frames;
- the Hilbert templates (see [Data](#data));
- `anchors/v2x_v27_predictions.json` and `sub/predictions.json`, from the Hugging Face repo (`v2x/submission/anchors/` and `v2x/submission/submission.zip`);
- the frames of P3's static map (see Real-data use, item 3).

```bash
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare v2x_real --root data/raw/v2xreal/extracted --split test --stride 10 --out data/pcdet/v2x_real
docker run --rm -v "$PWD":/workspace -w /workspace urbantwin-realism:cpu python tools/rebuild_v2x_test.py p3-ids
scripts/rebuild_v2x_test.sh
```

**Measured on one RTX 3090 (17 Nov 2026).** The run takes about 32 minutes: member inference 14 minutes, the lanes 15 minutes on CPU, the second stages and the final file 2 minutes.

| Stage | Result |
|---|---|
| Member predictions, test | 8 of 12 are byte-identical to the original run: the four PointPillars, CenterPoint, both Voxel R-CNN and PV-RCNN++. FSHNet, SAFDNet, DSVT-Pillar and Voxel Mamba give the same boxes up to float differences. |
| Lanes and second stages | Different bytes, because they take those four members. Each lane's box count is within 2 of the original. |
| P3 static map | 16,542 voxels, the same as the original run. It lowers 20,896 of 62,694 test pedestrian boxes; the original lowered 20,895 of 62,695. |
| Final file | Different md5 (`c848ac91…`, against `be6122b9…`). |

**Box match against the submitted `predictions.json`** (`tools/compare_predictions.py`; a match is the nearest same-class box within 0.5 m):

| Class | Boxes ≥ 0.05 | Matched | Position diff p99 / max (m) | Mean score diff | Boxes ≥ 0.3 | Matched | Position diff max (m) | Score diff max |
|---|---|---|---|---|---|---|---|---|
| vehicle | 1,422 | 100% | 0.0008 / 0.0188 | 0.00000 | 393 | 100% | 0.0006 | 0.0000 |
| pedestrian | 4,805 | 99.96% (4,803) | 0.0014 / 0.2146 | 0.00009 | 1,736 | 100% | 0.0041 | 0.0028 |
| truck | 1,846 | 100% | 0 / 0 | 0 | 248 | 100% | 0 | 0 |

The trucks are the v27 upload's boxes, a fixed input, so they are identical by construction.

### Environment
- **GPU:** RTX 3090 (sm_86), Docker:
  - `docker build -t urbantwin-lidar:latest .` gives PyTorch 2.4.1, CUDA 12.1, OpenPCDet commit `233f849`.
  - `docker build -t urbantwin-lidar:scatter -f docker/Dockerfile.scatter .` adds torch-scatter 2.1.2 and einops 0.8.0. It is used for FSHNet, SAFDNet and DSVT.
  - `docker/Dockerfile.mamba` builds `urbantwin-lidar:mamba` for Voxel Mamba. Its build context needs two clones:
    - [Voxel-Mamba](https://github.com/gwenzhang/Voxel-Mamba) at `ad8172a`, which brings its bundled `mamba/` (mamba_ssm 1.1.4). the image was built from a clean clone at that commit, and the same commit's files are copied into `utl/ext3p/voxel_mamba`;
    - [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) at tag `v1.1.3.post1`.
- **CPU:** `urbantwin-realism:cpu` (above) runs the lanes, rules and builds.
- **Runners:** `scripts/docker_run.sh` runs a command in the GPU image. Set `UT_NOGPU=1` for CPU-only steps and `UT_IMAGE` to pick the image.

### Data
None of the challenge data is included here. Download each set from its source:
- **UT-V2X-Real-IC twin:** [Harvard Dataverse, doi:10.7910/DVN/N6N4UR](https://doi.org/10.7910/DVN/N6N4UR), licence CC0 1.0. Put the `UCF-UT-V2X-Real-Seq-1.zip.NNN` parts in `data/raw/ut_v2x/`.
- **V2X-Real:** [ucla-mobility/V2X-Real](https://github.com/ucla-mobility/V2X-Real), under its own licence. Extract it to `data/raw/v2xreal/extracted/`.
- **Official starting kit:** Codabench competition 16685, Files tab.
  - Place the kit at `data/kit/v2x/`; it holds `detection_test_frames/`, `forbidden_frames.txt` and the scoring program.
  - Copy `detection_test_frame_ids.json` and `forbidden_frames.txt` into `configs/starter_kit_mirror/v2x_real/`.
  - Put the organisers' `forbidden_frames_mapping.csv` in `data/raw/v2xreal/`.
  - `utl/guard.py` refuses any real frame listed there or within 20 frames of one.

**Prepare the training and evaluation sets:**
```bash
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare ut --track v2x --raw data/raw/ut_v2x --out data/pcdet/ut_v2x
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare infos --root data/pcdet/ut_v2x --track v2x --gt_db
UT_NOGPU=1 scripts/docker_run.sh python -m utl.relabel_riders --src data/pcdet/ut_v2x --out data/pcdet/ut_v2x_ped
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare infos --root data/pcdet/ut_v2x_ped --track v2x --gt_db
UT_NOGPU=1 scripts/docker_run.sh python -m utl.prepare v2x_real --root data/raw/v2xreal/extracted --split val --out data/pcdet/v2x_real
UT_NOGPU=1 scripts/docker_run.sh python -m utl.thin --track v2x --src data/pcdet/v2x_real --out data/pcdet/v2x_real_50k
UT_NOGPU=1 scripts/docker_run.sh python scripts/v2x_fresh_build.py     # fresh check split (see Real-data use, item 1)
UT_NOGPU=1 scripts/docker_run.sh python -m utl.thin --track v2x --src data/pcdet/v2x_fresh --out data/pcdet/v2x_fresh_50k
UT_NOGPU=1 UT_IMAGE=urbantwin-lidar:mamba scripts/docker_run.sh python scripts/make_hilbert_lumpi.py   # Voxel Mamba's Hilbert-curve templates
```
- **Hilbert templates.** `make_hilbert_lumpi.py` writes `data/hilbert_lumpi/curve_template_3d_rank_{7,8,9}.pth`, the curve templates `vmamba_v2x.yaml` reads. They hold no data. The files used for the entry have md5 `492c801d…`, `ccd4ca20…` and `51bf5600…` (ranks 7, 8 and 9).

### 1. Training
Each model trains with:
```bash
scripts/docker_run.sh python -m utl.train --cfg_file configs/pcdet/<model>.yaml --workers 7 --num_epochs_to_eval 1 --fix_random_seed
```
- FSHNet, SAFDNet and DSVT use `UT_IMAGE=urbantwin-lidar:scatter`; Voxel Mamba uses `UT_IMAGE=urbantwin-lidar:mamba`.
- The train log of every model, with its full config dump, is in the Hugging Face repo (`v2x/checkpoints/<model>/`).

| Model | Config | Training set | Epochs | Role in v2x_v35d |
|---|---|---|---|---|
| PointPillars | `pp_v2x` | ut_v2x | 30 | vehicle 1, pedestrian 1, truck 1 |
| CenterPoint | `cp_v2x` | ut_v2x | 30 | vehicle 1.5, pedestrian 1 |
| PointPillars, rider boxes at person size | `pp_v2x_ped` | ut_v2x_ped | 30 | vehicle 0.5, pedestrian 0.5 |
| PointPillars, all frames | `pp_v2x_all` | ut_v2x | 30 | vehicle 0.5, truck 1 |
| PointPillars, walkers | `pp_v2x_walk` | ut_v2x | 30 | vehicle 1.5, pedestrian 2 |
| FSHNet | `fshnet_v2x` | ut_v2x | 20 | vehicle 1, pedestrian 1 |
| SAFDNet | `safdnet_v2x` | ut_v2x | 20 | vehicle 1 |
| DSVT-Pillar | `dsvt_v2x` | ut_v2x | 20 | vehicle 1, pedestrian 1 |
| Voxel Mamba | `vmamba_v2x` | ut_v2x | 20 | vehicle 1 |
| PV-RCNN++ | `pv_v2x` | ut_v2x | 20 | vehicle 1 |
| Voxel R-CNN (X6) | `vrcnn_v2x` | ut_v2x | 20 | vehicle 3; vehicle second stage |
| Voxel R-CNN (X6b) | `vrcnn_v2x_b` | ut_v2x | 20 | pedestrian 2; pedestrian and truck second stage |
| Voxel R-CNN, street-furniture negatives | `vrcnn_v2x_neg` | ut_v2x | 20 | pedestrian second stage |

### 2. Member inference (GPU)
Put the released checkpoints where OpenPCDet writes them: `output/pcdet/<model>/default/ckpt/checkpoint_epoch_<N>.pth`.
```bash
P=output/pcdet
for m in pp_v2x pp_v2x_ped pp_v2x_all pp_v2x_walk; do
  scripts/kit_predict.sh v2x configs/pcdet/$m.yaml $P/$m/default/ckpt/checkpoint_epoch_30.pth kit_mc --multi_class_nms --max_dets 100
done
scripts/kit_predict.sh v2x configs/pcdet/cp_v2x.yaml $P/cp_v2x/default/ckpt/checkpoint_epoch_30.pth kit
for m in vrcnn_v2x vrcnn_v2x_b; do
  scripts/kit_predict.sh v2x configs/pcdet/$m.yaml $P/$m/default/ckpt/checkpoint_epoch_20.pth kit
done
for m in fshnet_v2x safdnet_v2x dsvt_v2x; do
  UT_IMAGE=urbantwin-lidar:scatter scripts/kit_predict.sh v2x configs/pcdet/$m.yaml $P/$m/default/ckpt/checkpoint_epoch_20.pth kit
done
UT_IMAGE=urbantwin-lidar:mamba scripts/kit_predict.sh v2x configs/pcdet/vmamba_v2x.yaml $P/vmamba_v2x/default/ckpt/checkpoint_epoch_20.pth kit
scripts/pv_v2x_n_pred.sh          # PV-RCNN++ (pv_v2x), written as output/v2x/pv_v2x_n/kit/
```
- `kit_predict.sh` runs `utl.infer --tta`, which uses 4 flip views fused by WBF. It writes `output/v2x/<model>/<tag>/pred_test.json` for the 50 test frames and `pred_val50k.json` for `val_i1`.
- Every member also needs `pred_fresh50k.json` on the fresh split:
  ```bash
  FR=data/pcdet/v2x_fresh_50k
  for me in pp_v2x:30:kit_mc pp_v2x_ped:30:kit_mc pp_v2x_all:30:kit_mc pp_v2x_walk:30:kit_mc cp_v2x:30:kit vrcnn_v2x:20:kit vrcnn_v2x_b:20:kit \
            fshnet_v2x:20:kit safdnet_v2x:20:kit dsvt_v2x:20:kit vmamba_v2x:20:kit; do
    IFS=: read m e t <<< "$me"
    case $m in fshnet_v2x|safdnet_v2x|dsvt_v2x) export UT_IMAGE=urbantwin-lidar:scatter ;; vmamba_v2x) export UT_IMAGE=urbantwin-lidar:mamba ;; *) unset UT_IMAGE ;; esac
    X=; [ $t = kit_mc ] && X="--multi_class_nms --max_dets 100"
    scripts/docker_run.sh python -m utl.infer --cfg configs/pcdet/$m.yaml --ckpt $P/$m/default/ckpt/checkpoint_epoch_$e.pth \
        --frames $FR/points --ids $FR/ImageSets/val.txt --out output/v2x/$m/$t/pred_fresh50k.json --tta $X
  done
  ```
  `scripts/pv_v2x_n_pred.sh` writes the fresh file for PV-RCNN++ itself.
- The lane scripts read `pp_v2x`, `cp_v2x`, `pp_v2x_ped` and `pp_v2x_all` from `output/v2x/sweep_in/{pp,cp,ped,ppall}_{val50k,test}.json`. Copy each model's `pred_val50k.json` and `pred_test.json` there.

### 3. Lanes and second stages
```bash
scripts/route_v2x.sh fshnet_v2x      # FSHNet lanes on val_i1, test and fresh, then the X6, X6b and negatives refines
scripts/saf_joint.sh                 # FSHNet + SAFDNet joint vehicle lane, then the X6, X6b and negatives refines
UT_NOGPU=1 OMP_NUM_THREADS=1 scripts/docker_run.sh python scripts/dsvt_v2x_lanes.py --workers 6
UT_NOGPU=1 OMP_NUM_THREADS=1 scripts/docker_run.sh python scripts/vmamba_v2x_lanes.py --workers 6
UT_NOGPU=1 OMP_NUM_THREADS=1 scripts/docker_run.sh python scripts/pv_v2x_n_lanes.py --workers 6
UT_NOGPU=1 OMP_NUM_THREADS=1 scripts/docker_run.sh python scripts/q7_v2x_lanes.py --members dsvt_v2x vmamba_v2x pv_v2x_n --workers 6
scripts/q7_v2x_refine.sh q7_dsvt_vmamba_pvn_v2x_j     # X6 on the final vehicle lane
# X6b and the negatives model on the DSVT pedestrian lane (the final pedestrian second stages):
P=output/pcdet; I=output/v2x/dsvt_v2x_j/s2_in; O=output/v2x/dsvt_v2x_j/s2; mkdir -p $O
for s in val_i1 test fresh; do
  case $s in val_i1) F=data/pcdet/v2x_real_50k/points ;; test) F=data/kit/v2x/detection_test_frames ;; fresh) F=data/pcdet/v2x_fresh_50k/points ;; esac
  scripts/docker_run.sh python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_b.yaml --ckpt $P/vrcnn_v2x_b/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/pedestrian_$s.json --out $O/pedestrian_x6b_${s}_ref.json --only_class pedestrian
  scripts/docker_run.sh python scripts/x6_refine.py --cfg configs/pcdet/vrcnn_v2x_neg.yaml --ckpt $P/vrcnn_v2x_neg/default/ckpt/checkpoint_epoch_20.pth \
      --frames $F --pred $I/pedestrian_$s.json --out $O/pedestrian_neg_${s}_ref.json --only_class pedestrian
done
scripts/x6bc_refine.sh               # X6b on the v18 file, which gives the truck second stage
```
- `scripts/x6_refine.py` runs every second stage with 4 flip views, two passes and `--min_score 0.05`, plus `--only_class` when one class is refined.
- The pedestrian refines of the ranked entry ran on an A100 with these commands.

### 4. Readings and builds (CPU)
Builds run in order. Each one checks the md5 of its anchor file and of the readings it relies on.
```bash
python scripts/v2x_vgeom_read.py; python scripts/dsvt_v2x_read.py; python scripts/v2x_pz_read.py; python scripts/q7_v2x_read.py
python scripts/build_v2x_v29.py         # from the v27 predictions.json (trucks and frame order)
python scripts/build_v2x_v30.py
python scripts/build_v2x_v3x.py --version v2x_v31 --vehicle vg --pedestrian v30
python scripts/build_v2x_v32.py --version v2x_v34 --vehicle q7_dsvt_vmamba_pvn --pedestrian pz --truck v31
python scripts/build_v2x_v35.py         # v34 with the pedestrian class from the DSVT lane's second stages + P-Z
```
The output is `output/v2x/ens/v2x_v35_test.json`.

### 5. Synthetic frames
- `utl.realism_emit` writes the calibrated twin pool `output/v2x/realism_v2_i1`.
- `scripts/realism_r2.py train v2x` trains the point refiner R2: seed 0, 2,000 steps (see Real-data use, item 3).
- R2 then refines the 50 pool frames into `output/v2x/realism_r2/0001.bin` to `0050.bin`.
- The exact calls are:
  ```bash
  python scripts/realism_r2.py train v2x                     # PyTorch; the ranked frames came from Apple MPS, seed 0, 2,000 steps
  docker run --rm -v "$PWD":/workspace urbantwin-realism:cpu python scripts/realism_r2.py score v2x
  ```
- The R2 weights were not kept, so a new run gives different frames. The submitted 50 frames are in `submission.zip`, and Level 1 checks them.

### 6. Package and compare
```bash
scripts/package_check.sh v2x output/v2x/ens/v2x_v35_test.json output/v2x/realism_r2 v2x_v35
python tools/compare_predictions.py --ref sub/predictions.json --new output/v2x/ens/v2x_v35_test.json
```

### Known differences
- **Updated code locks.** `scripts/pv_v2x_n_pred.sh` and `scripts/q7_v2x_refine.sh` check the md5 of the config, `utl/infer.py` and `scripts/x6_refine.py` before they run. Comments in those files were edited for this release, so the checked md5s were updated to the released files. Md5s quoted in comments and in `logs/` are the original run's.
- **Edited record.** After the run, the free-text fields of `logs/dsvt_v2x_j_read.json` were edited to remove internal notes. The md5 that `scripts/build_v2x_v32.py` checks for this file was updated to match; the numbers in the file are unchanged.
- **Builders as used.** The build and lane scripts are the ones that made the upload; only comments and message strings were edited for this release. They lock the md5s of the original run's lane, second-stage and reading files (the records in `logs/`). Files from a rebuild have new md5s, so these checks stop the build until you replace the locked md5s with your own files' values.
- **GPU floats.** The same X6 refine gave different bytes on an RTX 3090 and on an A100, with equal AP. DSVT's scatter operations use atomic adds. So a rebuilt `predictions.json` matches by box, not by md5; the test-split rebuild above measures how closely.
- **Anchor files.** The builders check the md5 of earlier versions:
  - v27, v30 and v34 are the `predictions.json` files inside those uploads, in the Hugging Face repo under `v2x/submission/anchors/`.
  - The v31 test file (`6a4360ac`) was not kept. To rebuild v34 from a fresh v31, remove that one md5 check.
- **Trucks.** They come unchanged from the v27 upload, which was built from the v18 lane and the X6b truck refine. The v8 to v18 chain behind them is not part of this repository. The builders take the trucks as a fixed input: the v27 `predictions.json` from `v2x/submission/anchors/` in the Hugging Face repo.
- **R2.** The trained R2 weights were not kept. Retraining with seed 0 on another device will not give the same 50 frames. The submitted frames are in `submission.zip`.
- **Local split lists.** The frame lists of `val_i1` (see Real-data use, item 1) and `rfit_i1` (see Real-data use, item 3) were not kept.
  - The build checks read both lists.
  - P3's static map is built from `rfit_i1`. `tools/rebuild_v2x_test.py p3-ids` rebuilds its ids (see Real-data use, item 3); the map built from them has 16,542 voxels, the same as the original run's.
- **Fresh split.** `scripts/v2x_fresh_build.py` cuts the fresh split (see Real-data use, item 1), and `utl.thin` writes its 50k-point copy `data/pcdet/v2x_fresh_50k`.

## Real-data use

The text below is the "Real-data use" section of the report we sent to the organisers, copied word for word.

No detector or second stage was trained on real frames or real labels; every model trains on the UT-V2X-Real-IC twin. Real V2X-Real data entered our entry in the ways listed below. No frame in forbidden_frames.txt, and no frame within 20 frames of one, was used for anything except inference on the 50 test frames.
1. **Evaluation and selection.** Labelled frames score every change against the gate: val_i1 (146 frames of the V2X-Real val split, infrastructure sensor OS1-64) and fresh (328 frames of the train split, same sensor). Earlier post-processing settings (fusion weights and IoU, the height-refine percentile, the pedestrian shifts and the choice of rule R6) were compared on 292 val frames of both infrastructure sensors.
2. **Constants fitted on val_i1 labels.**
   - The thresholds of score rules X1, X3 and X8.
   - The truck size step (+0.67 m length, +0.12 m bottom): medians of label minus box.
   - The vehicle geometry step V-G: length −0.125 m and width +0.067 m are medians of label minus box; height +0.18 m and z +0.06 m come from val_i1 height-error checks (vehicle height error −0.21 m and −0.15 m in two range bands; median bottom error +0.03 m). The V-G idea came from comparing our per-class board results with those of a public entry.
   - The pedestrian height offset P-Z (+0.03 m), chosen among nine settings on val_i1 and then read once on fresh.

3. **Unlabelled real points.** 128 frames of the V2X-Real test split (sensor OS1-64; not the challenge's test frames) were used as points only, with no labels read. They give:
   - the static-background map P3: voxels occupied in at least half of the frames; it lowers the score of a test pedestrian box ×0.3 when most of its points fall in the map;
   - the targets of the realism calibration of our 50 synthetic frames;
   - the training targets of the point refiner R2, which moves the points of those 50 frames.

   64 of these 128 frames come from four clips whose names share a timestamp and scenario prefix with clips that hold forbidden frames; we have not established whether they are the same recording.
4. **Read, not shipped.** A negatives model (neg-G) resized its synthetic walker boxes to the pedestrian label sizes of 109 frames of the same test-split set (1,749 labels); it failed the gate and is not in the final package.

Validation, model selection and realism calibration are on the organisers' list of allowed real-data uses. P3, R2's training on real points and the constants fitted on val_i1 are not on that list as such; we list them here so that they can be judged.
