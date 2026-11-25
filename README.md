# mmgia — gradient inversion on a jointly trained image-text model in federated learning (ICS-500)

Design and results: `paper/main.tex` in the parent folder. Task list and run history: `progress.md` in the parent folder.

## Layout

    src/                    the pipeline of the main experiment (run from mmgia/: python src/<file>.py)
      common.py             paths, data loading, the fusion model, parameter groups
      protocols.py          FedAvg, FedProx, SCAFFOLD, FedBN: local steps, the update, the server's replay
      prepare_data.py       image-text pairs, the fixed split, client partitions, victim batches
      view_score.py, view_classify.py   frontal / lateral view of every IU X-Ray image
      feature_norm.py       fixed feature standardization constants (pretrained/feature_norm_seed0.pt)
      train_fl.py           federated warm-up under the four protocols
      attack_text.py        text attack (DAGER on the GPT-2 part of the update)
      attack_image.py       image attack (gradient matching, BN prior, total variation)
      score_text.py         tidy rows of the text results (ROUGE-L, zero point, mismatched control)
      score_images.py       identification score A and the other image metrics, cross matrix
      analyze.py            floor tests, paired comparisons, decisions for H1 to H6
    src/calibration/        tools used only on the calibration seed (seed 0): the first warm-up script, the
                            guessing baseline, the reference attack of Geiping et al., and diagnostics that
                            read the ground truth. Not part of the main experiment.
    tests/                  CPU tests (python tests/test_analyze.py; the others take --out <scratch dir>)
    scripts/main/           the main experiment: main_warmup.sh, main_text.sh, main_image_stage1.sh,
                            gpu_watchdog.sh, sys_watch.sh, stop_jobs.sh
    scripts/calibration/    tuning, gates and dry runs on seed 0
    scripts/setup/          environment, data download, third-party code
    agent_notes/            work logs of single tasks (they name files by their paths at the time: before
                            2026-11-25, src/calibration/* and tests/* were in src/, and all scripts in scripts/)
    data/splits/            pair list, labels, client partitions per seed (tracked)
    data/iuxray/            images and reports (not tracked), view-classifier outputs (tracked)
    pretrained/             HF_HOME and TORCH_HOME caches (not tracked), feature_norm_seed0.pt (tracked)
    third_party/            DAGER and the code of Geiping et al. (not tracked; scripts/setup/get_third_party.sh)
    runs/                   outputs (not tracked): runs/scores (tidy CSVs), runs/analysis/main (cells,
                            comparisons, decisions, per_seed, anova)
    logs/                   stdout logs (not tracked)

Environment: `conda activate mmgia; export HF_HOME=$PWD/pretrained/hf TORCH_HOME=$PWD/pretrained/torch`
Seeds: 0 = calibration only; test victims = 4 clients from each of seeds 1-5.

## Order of the main experiment

    bash scripts/main/main_warmup.sh         # seeds 1-5, four protocols
    bash scripts/main/main_text.sh           # 360 text attacks
    bash scripts/main/main_image_stage1.sh   # 160 image attacks
    python src/score_text.py ...             # -> runs/scores/main_text_tidy.csv
    python src/score_images.py ...           # -> runs/scores/main_images*.csv, main_image_tidy.csv
    python src/analyze.py test ...           # -> runs/analysis/main/

The shell scripts change into `~/jacob/mmgia` (the server path) and call `src/` and `scripts/` relative to it.

## Repository and sync

The main repository is the Windows folder `C:/Users/Jacob/Assignments/Couse Project ICS-500/mmgia`.
WSL2 (Ubuntu-22.04) keeps a clone in `~/mmgia` whose remote `windows` points at that folder.

    # in WSL2: get changes made on Windows
    cd ~/mmgia && git pull
    # in WSL2: send your commits to Windows (the Windows folder must have no uncommitted changes)
    git push

The server copy (`~/jacob/mmgia`) has no git history; files are copied there by hand. The experiments were run with
the flat layout (tag `pre-reorg-2026-11-25`); the server and the WSL clone still have that layout.

Tracked: code, scripts, tests, notes, data splits, view-classifier outputs, `pretrained/feature_norm_seed0.pt`.
Not tracked: images and reports, weights, `runs/`, `logs/`, `.venv/`, `third_party/`.
