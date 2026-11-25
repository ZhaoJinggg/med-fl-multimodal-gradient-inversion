#!/bin/bash
# Batch 2 and 4 with the settings that did best at batch 8. 128 px, round 9, u = 1, calibration clients 0-2.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
COMMON="--global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 1 2 --u 1 --res 128 --opt-lr 0.01 --tv 1e-2 --iters 12000 --checkpoints 4000 8000 12000 --out runs/pilot/push_b24"
run() {  # tag, then extra args
  tag=$1; shift
  python -u src/attack_image.py $COMMON --tag $tag "$@" 2>&1 | grep --line-buffered -E "check at|it=|FINAL|SUMMARY|Error|Traceback"
}
( run _bnonly12k --batch 2 --init random --cos-w 0 --bn 1e-1 ) > logs/push24_A.log 2>&1 &
( run _both_bn1_12k --batch 2 --init mean --bn 1.0 ) > logs/push24_B.log 2>&1 &
( run _bnonly12k --batch 4 --init random --cos-w 0 --bn 1e-1 ) > logs/push24_C.log 2>&1 &
( run _both_bn1_12k --batch 4 --init mean --bn 1.0 ) > logs/push24_D.log 2>&1 &
wait
echo PUSH24_DONE
