#!/bin/bash
# Image gate and batch-size comparison on the held-out calibration clients 3-9 (seed 0, round 9, 128 px, u = 1).
# Setting frozen on 2026-11-22 from clients 0-2: both loss terms, BN weight 1.0, mean start (data outside the
# federation), plain Adam lr 0.01, TV 1e-2, 12k iterations. Batch sizes 8 (main setting), 4, 2, 1; at batch 1
# also the BN-statistics-only variant, which identified the patient there on the tuning clients.
# Refuses to start unless the GPU power limit is 250 W; runs under the watchdog (stop at 78 C).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
lim=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
[ "$lim" = 250 ] || { echo "power limit is '$lim' W, not 250: not starting"; exit 1; }
pgrep -f "[g]pu_watchdog.sh" >/dev/null || (setsid nohup bash scripts/main/gpu_watchdog.sh 78 9999 >/dev/null 2>&1 &)
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
COMMON="--global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 3 4 5 6 7 8 9 --u 1 --res 128 --opt-lr 0.01 --tv 1e-2 --iters 12000 --checkpoints 4000 8000 12000 --out runs/gate/image"
run() {  # tag, then extra args
  tag=$1; shift
  python -u src/attack_image.py $COMMON --tag $tag "$@" 2>&1 | grep --line-buffered -E "check at|it=|FINAL|SUMMARY|Error|Traceback"
}
echo "GATE_IMAGE_START $(date +%T)"
for b in 8 4 2 1; do
  ( run _gate --batch $b --init mean --bn 1.0 ) > logs/gate_img_b$b.log 2>&1 &
done
( run _gate_bnonly --batch 1 --init mean --cos-w 0 --bn 1e-1 ) > logs/gate_img_b1bn.log 2>&1 &
wait
echo "GATE_IMAGE_DONE $(date +%T)"
