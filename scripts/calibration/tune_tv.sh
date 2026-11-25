#!/bin/bash
# Batch 1, u = 1, 128 px, calibration clients 0-2 of seed 0. Four streams run side by side on one GPU.
#   V2 control: start from the true image + noise; the attack must return to the truth.
#   TV sweep:   TV weight 1e-3, 1e-2, 1e-1 with plain Adam (lr 0.01) and with signed Adam (lr 0.1).
#   Random start with signed Adam, TV 1e-2.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
BASE="--global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 1 2 --batch 1 --u 1 --res 128 --bn 1e-1 --out runs/pilot/tune128"
run() {  # tag, then extra args
  tag=$1; shift
  python -u src/attack_image.py $BASE --tag $tag "$@" 2>&1 | grep --line-buffered -E "check at|it=|FINAL|SUMMARY|Error|Traceback"
}
PLAIN="--opt-lr 0.01 --iters 5000 --checkpoints 1000 2500 5000"
SIGN="--signed --opt-lr 0.1 --iters 5000 --checkpoints 1000 2500 5000"
NEAR="--init near --opt-lr 0.01 --iters 3000 --checkpoints 1000 3000"

( run _v2_tv1e-4 $NEAR --tv 1e-4; run _v2_tv1e-2 $NEAR --tv 1e-2; run _sign_tv1e-2 --init mean $SIGN --tv 1e-2 ) > logs/tune_tv_A.log 2>&1 &
( run _plain_tv1e-3 --init mean $PLAIN --tv 1e-3; run _plain_tv1e-2 --init mean $PLAIN --tv 1e-2 ) > logs/tune_tv_B.log 2>&1 &
( run _plain_tv1e-1 --init mean $PLAIN --tv 1e-1; run _sign_tv1e-1 --init mean $SIGN --tv 1e-1 ) > logs/tune_tv_C.log 2>&1 &
( run _sign_tv1e-3 --init mean $SIGN --tv 1e-3; run _sign_tv1e-2 --init random $SIGN --tv 1e-2 ) > logs/tune_tv_D.log 2>&1 &
wait
echo TUNE_DONE
