#!/bin/bash
# Second tuning pass at batch 1: Adam step size and a stronger BN prior (mean-image start, TV 1e-4).
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
while pgrep -f "scripts/calibration/tune_b1\.sh" > /dev/null; do sleep 10; done
G=runs/pilot/seed0_fedavg/global_round9.pt
for lr in 0.003 0.01 0.03; do for bn in 1e-1 1; do
  python src/attack_image.py --global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 --batch 1 --u 1 \
    --iters 3000 --checkpoints 1000 3000 --tv 1e-4 --bn $bn --init mean --opt-lr $lr --tag _lr${lr}_bn${bn} \
    --out runs/pilot/tune_b1 2>&1 | grep "it=3000" | sed "s/^/optlr=$lr bn=$bn | /"
done; done
echo TUNE2_DONE
