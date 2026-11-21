#!/bin/bash
# Text attack (DAGER), batch 8, u = 1, float32 update, calibration clients 0-2: grid over the rank rule and the two
# span thresholds. ONE run at a time (a run takes seconds; a search that explodes is cut off after 2 minutes).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
for nf in 1.5 2 3; do for l1 in 5e-2 1e-1; do for l2 in 1e-2 2e-2 3e-2 5e-2; do
  tag=_g_nf${nf}_l1_${l1}_l2_${l2}
  timeout 120 python -u src/attack_text.py --global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 1 2 --batch 8 --u 1 --res 128 \
      --noise-factor $nf --l1-thresh $l1 --l2-thresh $l2 --tag $tag --out runs/pilot/text128 2>&1 \
      | grep --line-buffered -E "^victim|^SUMMARY|Error|Traceback" | cut -c1-260 | sed "s/^/$tag | /"
  echo "$tag | exit ${PIPESTATUS[0]}"
done; done; done
echo TEXT_GRID_DONE
