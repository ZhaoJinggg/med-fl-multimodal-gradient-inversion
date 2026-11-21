#!/bin/bash
# How far does the basic attack get at batch 8? 128 px, round 9, u = 1, calibration clients 0-2.
#   A  BN statistics only, 12k iterations
#   B  both terms with a 10x BN weight, 12k iterations
#   C  BN statistics only, 3 restarts of 4k iterations (best by matching loss)
#   D  BN statistics only from a random start, 12k iterations
#   E  control V1 at batch 1: our attack (signed, no BN term) and the reference code of Geiping et al.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
COMMON="--global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 1 2 --u 1 --res 128"
run() {  # tag, then extra args
  tag=$1; shift
  python -u src/attack_image.py $COMMON --tag $tag "$@" 2>&1 | grep --line-buffered -E "check at|it=|FINAL|SUMMARY|Error|Traceback"
}
B8="--batch 8 --opt-lr 0.01 --tv 1e-2 --out runs/pilot/push_b8"
LONG="--iters 12000 --checkpoints 4000 8000 12000"

( run _bnonly12k $B8 --init mean --cos-w 0 --bn 1e-1 $LONG ) > logs/push_A.log 2>&1 &
( run _both_bn1_12k $B8 --init mean --bn 1.0 $LONG ) > logs/push_B.log 2>&1 &
( run _bnonly_r3 $B8 --init mean --cos-w 0 --bn 1e-1 --iters 4000 --checkpoints 4000 --restarts 3 ) > logs/push_C.log 2>&1 &
( run _bnonly12k $B8 --init random --cos-w 0 --bn 1e-1 $LONG ) > logs/push_D.log 2>&1 &
( run _ours_v1 --batch 1 --init mean --signed --opt-lr 0.1 --tv 1e-2 --bn 0 --iters 5000 --checkpoints 2500 5000 --out runs/pilot/v1
  python -u src/control_reference.py --global $G --seed 0 --victims 0 1 2 --res 128 --tv 1e-2 --iters 5000 --out runs/pilot/v1 2>&1 | grep --line-buffered -E "FINAL|SUMMARY|Error|Traceback|Rec. loss" ) > logs/push_E.log 2>&1 &
wait
echo PUSH_DONE
