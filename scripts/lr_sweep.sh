#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
for li in 0.01 0.03; do for lt in 1e-5 1e-4 1e-3 1e-2; do
  echo "== lr_img $li lr_txt $lt"
  python src/warmup.py --seed 0 --rounds 9 --lr-img $li --lr-txt $lt --out runs/pilot/lr_sweep --save 2>&1 | grep "^{" | tail -3
done; done
echo SWEEP_DONE
