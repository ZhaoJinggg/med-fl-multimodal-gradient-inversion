# Gradient inversion on a jointly trained image-text model in federated learning

Code of an ICS-500 course project. A passive server in federated learning receives a client's update of an
image-text model and tries to recover the client's private data: first the report text, then the chest radiograph.

- **Model:** ResNet-18 (images) + GPT-2 small (text) + a linear head
- **Data:** IU X-Ray, 10 clients
- **Protocols:** FedAvg, FedProx, SCAFFOLD, FedBN
- **Attacks:** DAGER for the text; gradient matching with a BN prior and total variation for the images
- **Size:** 360 text attacks and 160 image attacks on 20 test victims (seeds 1-5, four clients each).
  Seed 0 is used for calibration only.

## Folders

| Folder | Content |
|---|---|
| `src/` | Pipeline of the main experiment |
| `src/calibration/` | Tools used only on the calibration seed |
| `tests/` | CPU tests |
| `scripts/main/` | Launch scripts of the main experiment |
| `scripts/calibration/` | Tuning, gates, and dry runs on seed 0 |
| `scripts/setup/` | Environment, data download, third-party code |
| `agent_notes/` | Work logs of single tasks |
| `data/` | Splits (tracked); images and reports (not tracked) |
| `pretrained/` | Weight caches (not tracked); `feature_norm_seed0.pt` (tracked) |
| `third_party/` | External code, fetched by `scripts/setup/get_third_party.sh` (not tracked) |
| `runs/`, `logs/` | Outputs and logs (not tracked) |

## `src/`: the main experiment

Run every file from the repository root: `python src/<file>.py`. Each file starts with a usage note.

| File | Purpose |
|---|---|
| `common.py` | Paths, data loading, the fusion model, parameter groups |
| `protocols.py` | The four protocols: local steps, the update, the server's replay |
| `prepare_data.py` | Image-text pairs, the fixed split, client partitions, victim batches |
| `view_score.py`, `view_classify.py` | Frontal or lateral view of every image |
| `feature_norm.py` | Fixed feature standardization constants |
| `train_fl.py` | Federated warm-up under the four protocols |
| `attack_text.py` | Text attack (DAGER on the GPT-2 part of the update) |
| `attack_image.py` | Image attack (gradient matching on the ResNet-18 and head part) |
| `score_text.py` | Text scores: ROUGE-L, zero point, mismatched control |
| `score_images.py` | Image scores: identification score, SSIM, PSNR, LPIPS, cross matrix |
| `analyze.py` | Floor tests, paired comparisons, decisions for the eight hypotheses |

## `src/calibration/`: seed 0 only

These files are not part of the main experiment. The diagnostics read the ground truth, so their output is not an
attack result.

| File | Purpose |
|---|---|
| `warmup.py` | First FedAvg warm-up script; `train_fl.py` reproduces it bit for bit |
| `text_baseline.py` | Guessing baseline of the text attack |
| `identify.py` | Early identification score, now part of `score_images.py` |
| `control_reference.py` | Control run with the reference code of Geiping et al. |
| `retrieval_check.py` | Does a reconstruction identify its own original among all images? |
| `cos_sensitivity.py`, `ssim_diag.py` | Diagnostics of the image attack |
| `label_diag.py`, `text_diag.py`, `text_diag2.py`, `text_diag3.py` | Diagnostics of the text attack |

## `tests/`

| File | Checks | Time on a CPU |
|---|---|---|
| `test_analyze.py` | Decision rules on synthetic data with known answers; needs no data files | 30 s |
| `test_protocols.py` | Uploads and the server's replay under every protocol | 2 min |
| `test_attack_text.py` | Protocol wiring of the text attack; text scoring | 10 min |
| `test_attack_image.py` | Protocol and text-condition wiring of the image attack | needs `--ref` |

```bash
python tests/test_analyze.py
python tests/test_protocols.py --out <scratch dir>
python tests/test_attack_text.py --out <scratch dir>
```

## `scripts/main/`

| Script | Purpose |
|---|---|
| `main_warmup.sh` | Warm-ups for seeds 1-5 under the four protocols |
| `main_text.sh` | The 360 text attacks; waits for each seed's warm-up |
| `main_image_stage1.sh` | The 160 image attacks; waits for the text attacks, repeats failed jobs once |
| `gpu_watchdog.sh` | Logs GPU temperature and power every second; stops the attacks at a limit |
| `sys_watch.sh` | Logs CPU temperature and frequency every second |
| `stop_jobs.sh` | Stops all job scripts and attack processes |

The shell scripts change into `~/jacob/mmgia` and call `src/` and `scripts/` relative to it. Edit that line to
run them elsewhere.

## Running the main experiment

```bash
# setup, once
bash scripts/setup/setup_env.sh
bash scripts/setup/get_third_party.sh
bash scripts/setup/dl_iuxray.sh
python src/prepare_data.py

conda activate mmgia
export HF_HOME=$PWD/pretrained/hf TORCH_HOME=$PWD/pretrained/torch

# warm-ups, then attacks
bash scripts/main/main_warmup.sh
bash scripts/main/main_text.sh
bash scripts/main/main_image_stage1.sh

# scores, then tests (full arguments: see the usage note of each file)
python src/score_text.py runs/main/text ... --out runs/scores/main_text_tidy.csv
python src/score_images.py ...
python src/analyze.py tidy-images runs/scores/main_images.csv --out runs/scores/main_image_tidy.csv
python src/analyze.py test runs/scores/main_text_tidy.csv runs/scores/main_image_tidy.csv \
       --cross runs/scores/main_images_cross.csv --out runs/analysis/main
```

## Outputs

| Path | Content |
|---|---|
| `runs/fl/` | Broadcast models of every seed and protocol |
| `runs/main/text/`, `runs/main/image/` | Raw attack results |
| `runs/scores/` | One row per attack: `main_text_tidy.csv`, `main_image_tidy.csv`, and the detailed image scores |
| `runs/analysis/main/` | `cells.csv`, `comparisons.csv`, `decisions.csv`, `per_seed.csv`, `anova.csv` |

## Notes

- All text files use LF line endings (`.gitattributes`), so the scripts run unchanged on Linux and in WSL.
- The files were reorganized on 2026-11-25. Before that date, the files of `src/calibration/` and `tests/` were in
  `src/`, and all scripts were in `scripts/`. The notes in `agent_notes/` use those earlier paths. The tag
  `pre-reorg-2026-11-25` marks the earlier layout.
