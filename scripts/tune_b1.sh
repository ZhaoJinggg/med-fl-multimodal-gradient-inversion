#!/bin/bash
# Tune the image attack at batch 1, u = 1, calibration victim 0 (3k iterations per setting).
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
G=runs/pilot/seed0_fedavg/global_round9.pt
for init in random mean; do for tv in 1e-4 1e-3 1e-2; do for bn in 1e-2 1e-1; do
  python src/attack_image.py --global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 --batch 1 --u 1 \
    --iters 3000 --checkpoints 1000 3000 --tv $tv --bn $bn --init $init --tag _tv${tv}_bn${bn} \
    --out runs/pilot/tune_b1 2>&1 | grep "it=3000" | sed "s/^/init=$init tv=$tv bn=$bn | /"
done; done; done
echo TUNE_DONE
