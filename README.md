# mmgia — gradient inversion in multimodal medical image-text FL (ICS-500)

Design: `paper/main.tex` in the parent folder (the v7 outline PDF is superseded).

    data/iuxray/raw/        NLMCXR_png.tgz, NLMCXR_reports.tgz (Open-i originals, untouched)
    data/iuxray/extracted/  unpacked PNGs + report XMLs
    data/splits/            filtered pair list, labels, client partitions (per seed)
    pretrained/hf           HF_HOME (GPT-2 small)
    pretrained/torch        TORCH_HOME (ResNet-18, LPIPS AlexNet)
    src/                    simulator, fusion model, attacks, metrics
    configs/                YAML configs per run
    scripts/                setup / download / launch scripts
    third_party/            DAGER and other external code
    runs/                   per-run outputs (checkpoints, intercepted updates, reconstructions)
    results/                aggregated tables and figures
    logs/                   stdout logs

Environment: conda activate mmgia; export HF_HOME=$PWD/pretrained/hf TORCH_HOME=$PWD/pretrained/torch
Seeds: 0 = calibration only; test victims = 4 clients from each of seeds 1-5.

## Repository and sync

The main repository is the Windows folder `C:/Users/Jacob/Assignments/Couse Project ICS-500/mmgia`.
WSL2 (Ubuntu-22.04) keeps a clone in `~/mmgia` whose remote `windows` points at that folder.

    # in WSL2: get changes made on Windows
    cd ~/mmgia && git pull
    # in WSL2: send your commits to Windows (the Windows folder must have no uncommitted changes)
    git push

Tracked: code, scripts, data splits, view-classifier outputs, `pretrained/feature_norm_seed0.pt`.
Not tracked: images and reports, weights, `runs/`, `logs/`, `.venv/`, `third_party/`
(run `scripts/get_third_party.sh` to fetch DAGER at the pinned commit).
