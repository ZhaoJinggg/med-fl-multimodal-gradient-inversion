#!/bin/bash
# Best batch-1 setting, 10k iterations, victims 0-2, at 224 and 128 (first fallback of the gate).
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
while pgrep -f "scripts/tune_b1_lr\.sh" > /dev/null; do sleep 10; done
for res in 224 128; do
  G=runs/pilot/seed0_fedavg/global_round9.pt
  if [ $res = 128 ]; then
    G=runs/pilot_r128/seed0_fedavg/global_round9.pt
    [ -f $G ] || python src/warmup.py --seed 0 --rounds 9 --lr-img 0.01 --lr-txt 1e-2 --res 128 --out runs/pilot_r128 --save 9 2>&1 | grep "round': 9" | sed "s/^/warm-up 128: /"
  fi
  python src/attack_image.py --global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 1 2 --batch 1 --u 1 \
    --iters 10000 --checkpoints 2000 5000 10000 --tv 1e-4 --bn 1e-1 --init mean --opt-lr 0.01 --res $res \
    --tag _best_r$res --out runs/pilot/b1_long 2>&1 | grep -E "it=10000|SUMMARY|Error" | sed "s/^/res=$res | /"
done
echo LONG_DONE
