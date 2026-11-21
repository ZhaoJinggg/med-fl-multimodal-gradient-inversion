#!/bin/bash
# Pilot on calibration seed 0, one GPU, steps run one after another.
#   usage: scripts/run_pilot.sh LR_IMG LR_TXT
# 1. FedAvg warm-up (9 rounds) with the chosen learning rates
# 2. Image attack at u = 1: victims 0-2 x batch 1, 4, 8 (SSIM logged at 0.5k, 1k, 2k, 5k, 10k iterations)
# 3. Image attack at u = 1, batch 8, initialized from the mean image (victim 0)
# 4. Text attack (DAGER) at u = 1 and 4, batch 8, victims 0-2
# 5. Timing: image attack at u = 4 and 16, batch 8, victim 0 (2k iterations)
set -u
LI=$1; LT=$2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
G=runs/pilot/seed0_fedavg/global_round9.pt
COMMON="--seed 0 --lr-img $LI --lr-txt $LT"
log() { echo "[$(date '+%F %T')] $*"; }

log "1. warm-up"
python src/warmup.py --seed 0 --rounds 9 --lr-img $LI --lr-txt $LT --out runs/pilot --save 5 9 2>&1 | grep "^{"

log "2. image attack, u=1, batch 1/4/8"
for B in 1 4 8; do
  python src/attack_image.py --global $G $COMMON --victims 0 1 2 --batch $B --u 1 --iters 10000 --out runs/pilot/attacks 2>&1 | grep -E "check|it=|SUMMARY|Error"
done

log "3. image attack, u=1, batch 8, mean-image init"
python src/attack_image.py --global $G $COMMON --victims 0 --batch 8 --u 1 --iters 10000 --init mean --out runs/pilot/attacks 2>&1 | grep -E "it=|SUMMARY|Error"

log "4. text attack, u=1 and 4"
for U in 1 4; do
  python src/attack_text.py --global $G $COMMON --victims 0 1 2 --batch 8 --u $U --out runs/pilot/text 2>&1 | grep -E "^victim|SUMMARY|Error"
done

log "5. timing, image attack u=4 and 16"
for U in 4 16; do
  python src/attack_image.py --global $G $COMMON --victims 0 --batch 8 --u $U --iters 2000 --checkpoints 500 1000 2000 --out runs/pilot/attacks 2>&1 | grep -E "check|it=|SUMMARY|Error"
done
log "PILOT_DONE"
