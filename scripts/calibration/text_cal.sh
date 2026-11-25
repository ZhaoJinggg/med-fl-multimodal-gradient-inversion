#!/bin/bash
# One text-attack run (DAGER, batch 8, u = 1, float32 update, 128 px model) with the search throttled.
# Start scripts/main/gpu_watchdog.sh first. usage: bash scripts/calibration/text_cal.sh <tag> <idle> <timeout_s> <victims...> [-- extra args]
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
tag=$1; idle=$2; limit=$3; shift 3
timeout "$limit" python -u src/attack_text.py --global runs/pilot_r128/seed0_fedavg/global_round9.pt --seed 0 \
    --lr-img 0.01 --lr-txt 1e-2 --batch 8 --u 1 --res 128 --noise-factor 2 --l1-thresh 5e-2 --l2-thresh 2e-2 \
    --parallel 50 --idle "$idle" --tag "$tag" --out runs/pilot/text128 --victims "$@" 2>&1 \
    | grep --line-buffered -E "^victim|^SUMMARY|rank|Error|Traceback" | cut -c1-300
echo "exit ${PIPESTATUS[0]}"
echo TEXT_CAL_DONE
