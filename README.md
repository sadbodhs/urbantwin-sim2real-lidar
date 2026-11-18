# UrbanTwin Sim2Real LiDAR Challenge: our LUMPI and V2X-Real entries (3rd place on both tracks)

This repository holds the code, configs and build steps behind our ranked entries (team **sadbodh**, Sadbodh Sharma) in the
[UCF UrbanTwin Sim2Real LiDAR Challenge](https://urbanity-lab.github.io/bdc2026-urbantwin-sim2real/) at the IEEE BigData Cup 2026.
- Every detector and second stage was trained only on the synthetic digital twins: UT-LUMPI ([doi:10.7910/DVN/D9SSWD](https://doi.org/10.7910/DVN/D9SSWD)) and UT-V2X-Real-IC ([doi:10.7910/DVN/N6N4UR](https://doi.org/10.7910/DVN/N6N4UR)).
- Both tracks share one pipeline:
  1. per-class fusion lanes of single- and two-stage detectors;
  2. second stages applied to the fused boxes;
  3. fixed point-based score rules and box-geometry steps;
  4. synthetic frames calibrated to the recorded point density.

| Track | Final entry | Official place | Combined score | 3D mAP@0.5 | Realism | Details |
|---|---|---|---|---|---|---|
| LUMPI | `lumpi_v62d` | 3rd of 44 teams | **0.5377** | 0.1885 | 0.9402 | [lumpi/README.md](lumpi/README.md) |
| V2X-Real | `v2x_v35d` | 3rd of 46 teams | **0.8788** | 0.5825 | 0.9488 | [v2x/README.md](v2x/README.md) |

Scores are from the final leaderboards (Codabench competitions [16684](https://www.codabench.org/competitions/16684/) and [16685](https://www.codabench.org/competitions/16685/)). The method is described in our paper for the IEEE BigData 2026 Big Data Cup.

## Weights and submissions
The trained weights and the submitted files are on Hugging Face: [Sadbodh/urbantwin-sim2real-lidar](https://huggingface.co/Sadbodh/urbantwin-sim2real-lidar). It has one folder per track, `lumpi/` and `v2x/`, each holding:
- `checkpoints/<model>/`: the checkpoint, its config and its train log;
- `submission/submission.zip`: the file we uploaded;
- `submission/anchors/`: earlier uploads' `predictions.json`, which the builders take as fixed inputs;
- `lumpi/rescorer/`: the LUMPI Person rescorer, LUMPI only.

## Reproducing the entries
Each track README gives two levels:
1. **Level 1, exact (CPU, minutes).** `tools/verify_submission.py` checks the md5 of the submitted `predictions.json`, of the 50 synthetic frames and of `declaration.txt`. It also repacks them the way the upload was packed and checks the md5s again.
2. **Level 2, rebuild from the checkpoints (GPU).** Every command from the checkpoints to the zip, plus `tools/compare_predictions.py`, which matches a rebuild box by box against the submitted file.
   - GPU steps do not give bit-identical floats on other hardware, so this level is judged by box matches, not by md5.
   - For V2X-Real, `scripts/rebuild_v2x_test.sh` rebuilds the 50 test frames, and its measured result is in [v2x/README.md](v2x/README.md).

```bash
docker build -t urbantwin-realism:cpu -f docker/Dockerfile.realism .
# official starting kits (Codabench 16684 / 16685, Files tab): copy detection_test_frame_ids.json into
# configs/starter_kit_mirror/lumpi/ and configs/starter_kit_mirror/v2x_real/
# download lumpi/submission/submission.zip and v2x/submission/submission.zip from Hugging Face as lumpi_submission.zip and v2x_submission.zip
docker run --rm -v "$PWD":/workspace urbantwin-realism:cpu python tools/verify_submission.py check --track lumpi --zip lumpi_submission.zip
docker run --rm -v "$PWD":/workspace urbantwin-realism:cpu python tools/verify_submission.py check --track v2x --zip v2x_submission.zip
```

Both commands print `ALL MATCH`.

The challenge data is not redistributed here. Each track README links the official sources, and every real-data use is stated in that README's "Real-data use" section.

## Repository layout
| Path | What |
|---|---|
| `lumpi/README.md`, `v2x/README.md` | per-track method summary, results, reproduction steps, known differences and real-data use |
| `utl/` | data preparation (including the twin fixes), training and inference wrappers, fusion, realism metrics, frame guard, packaging |
| `utl/ext3p/` | FSHNet, SAFDNet and Voxel Mamba model files, copied from their repos, each with its LICENSE and SOURCE.txt |
| `scripts/` | lanes, second stages, rules, readings, builds, data-set builders and job scripts for both tracks |
| `rescorer/` | training and apply code of the LUMPI Person rescorer |
| `configs/pcdet/` | the model configs (15 LUMPI, 13 V2X-Real) |
| `configs/starter_kit_mirror/` | where the official kit's frame lists go (not redistributed) |
| `configs/lumpi_rig.json` | the LUMPI sensor rig for ray-casting walkers and riders, derived from LUMPI (see `configs/README.md`) |
| `docker/`, `Dockerfile` | GPU images (OpenPCDet 233f849, PyTorch 2.4.1, CUDA 12.1) and the CPU image |
| `logs/` | build records the builders read: readings, md5 lists and build logs |
| `tools/` | `verify_submission.py`, `compare_predictions.py` and `rebuild_v2x_test.py` |

Compute: models were trained on an RTX 3090 and on A100 GPUs; builds and readings run on CPU.

## Licence
- **Our code:** Apache License 2.0 (`LICENSE`, `NOTICE`).
- **Third-party code:** the files in `utl/ext3p/` keep their own Apache-2.0 licences. OpenPCDet, installed by the Dockerfile, is also Apache-2.0.
- **Weights:** the trained weights on Hugging Face are released under CC BY-NC 4.0.
- **Data:** the challenge datasets are not redistributed. Each keeps its own licence: UT-LUMPI and UT-V2X-Real-IC CC0 1.0, LUMPI CC BY-NC 3.0, V2X-Real under its own terms. `configs/lumpi_rig.json` holds values derived from LUMPI (see `NOTICE`).

## Citation
If you use this code, please cite the challenge's papers and datasets:
```bibtex
@article{shahbaz2026high,
  title={High-fidelity digital twins for bridging the sim2real gap in {LiDAR}-based {ITS} perception},
  author={Shahbaz, Muhammad and Agarwal, Shaurya},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={27},
  number={7},
  pages={8698--8709},
  year={2026},
  publisher={IEEE}
}

@article{shahbaz2026urbantwin,
  title={UrbanTwin: Synthetic Roadside LiDAR Datasets},
  author={Shahbaz, Muhammad and Agarwal, Shaurya},
  journal={IEEE Open Journal of Intelligent Transportation Systems},
  volume={7},
  pages={353--364},
  year={2026},
  publisher={IEEE}
}

@article{shahbaz2025urbantwin,
  title={UrbanTwin: Building High-Fidelity Digital Twins for Sim2Real LiDAR Perception and Evaluation},
  author={Shahbaz, Muhammad and Agarwal, Shaurya},
  journal={arXiv preprint arXiv:2509.02903},
  year={2025}
}
```

## Acknowledgements
- The UrbanTwin challenge and the UT-LUMPI and UT-V2X-Real-IC twins come from the Urbanity Lab, University of Central Florida.
- LUMPI comes from the Institute of Cartography and Geoinformatics, Leibniz University Hannover.
- V2X-Real comes from the UCLA Mobility Lab.
- Detector code: OpenPCDet, DSVT, FSHNet, SAFDNet (HEDNet), Voxel Mamba and PV-RCNN++.
