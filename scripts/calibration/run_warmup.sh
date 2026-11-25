#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
python src/calibration/warmup.py --seed 0 --rounds 9 --lr-img 0.01 --lr-txt 1e-4 --out runs/pilot > logs/pilot_warmup_seed0.log 2>&1
echo WARMUP_EXIT=$? >> logs/pilot_warmup_seed0.log
