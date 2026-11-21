#!/bin/bash
# 128 px, calibration clients 0-2 of seed 0, u = 1. Four streams side by side on one GPU.
#   A  controls: BN in eval mode at batch 1 on a random, the pretrained (round 0) and the round-9 ResNet-18
#   B  attribution at batch 1 (round 9, BN in training mode): BN statistics only, matching loss only, both
#   C  batch 8: both terms, then BN statistics only
#   D  batch 2 and batch 4: both terms
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
COMMON="--seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 1 2 --u 1 --res 128 --iters 5000 --checkpoints 1000 2500 5000 --out runs/pilot/attrib128"
run() {  # tag, then extra args
  tag=$1; shift
  python -u src/attack_image.py $COMMON --tag $tag "$@" 2>&1 | grep --line-buffered -E "check at|it=|FINAL|SUMMARY|Error|Traceback"
}
GEIPING="--bn-mode eval --batch 1 --init random --signed --opt-lr 0.1 --tv 1e-2"
PLAIN="--init mean --opt-lr 0.01 --tv 1e-2"

( run _ctl_random --global random $GEIPING; run _ctl_round0 --global pretrained $GEIPING; run _ctl_round9 --global $G $GEIPING ) > logs/attrib_A.log 2>&1 &
( run _bnonly --global $G --batch 1 $PLAIN --cos-w 0 --bn 1e-1; run _gradonly --global $G --batch 1 $PLAIN --bn 0; run _both --global $G --batch 1 $PLAIN --bn 1e-1 ) > logs/attrib_B.log 2>&1 &
( run _both --global $G --batch 8 $PLAIN --bn 1e-1; run _bnonly --global $G --batch 8 $PLAIN --cos-w 0 --bn 1e-1 ) > logs/attrib_C.log 2>&1 &
( run _both --global $G --batch 2 $PLAIN --bn 1e-1; run _both --global $G --batch 4 $PLAIN --bn 1e-1 ) > logs/attrib_D.log 2>&1 &
wait
echo ATTRIB_DONE
