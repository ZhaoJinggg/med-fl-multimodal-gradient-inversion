#!/bin/bash
# Text attack (DAGER), batch 8, u = 1, float32 update, calibration client 0: grid over the span thresholds and the rank rule.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
for nf in 5 2; do
  for l1 in 1e-2 2e-2 5e-2; do
    for l2 in 1e-2 2e-2; do
      tag=_nf${nf}_l1_${l1}_l2_${l2}
      ( timeout 900 python -u src/attack_text.py --global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 --batch 8 --u 1 --res 128 \
          --noise-factor $nf --l1-thresh $l1 --l2-thresh $l2 --tag $tag --out runs/pilot/text128 2>&1 \
          | grep --line-buffered -E "^victim|^  rank|Error|Traceback" | sed "s/^/$tag | /"; echo "$tag | exit ${PIPESTATUS[0]}" ) >> logs/text_grid.log 2>&1 &
    done
  done
done
wait
echo TEXT_GRID_DONE >> logs/text_grid.log
