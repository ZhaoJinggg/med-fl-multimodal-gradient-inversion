#!/bin/bash
# Calibration on seed 0, gate clients 3-9, FedAvg, u = 1, batch 8, frozen image setting: the image attack under
# the three text conditions. The gate of 2026-11-22 used the oracle text only; the dry run of 2026-11-23 showed
# a matching distance of 0.14-0.55 at the TRUE images when the recovered text is used, so the recovered-text
# cell, which opens stage 2 of the main experiment, may fall well below the oracle result. Oracle is repeated
# with the current code so that the three conditions are paired. Text results: runs/gate/text (u = 1).
# Three processes on one card (steady load, about 150 W). Output: runs/gate/textcond/<condition>.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
pgrep -f "[g]pu_watchdog.sh" >/dev/null || (setsid nohup bash scripts/main/gpu_watchdog.sh 120 9999 >/dev/null 2>&1 &)
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
COMMON="--global $G --proto fedavg --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 3 4 5 6 7 8 9 --u 1 --batch 8 --res 128 --init mean --opt-lr 0.01 --tv 1e-2 --bn 1.0 --iters 12000 --checkpoints 4000 8000 12000"
O=runs/gate/textcond
echo "CAL_TEXTCOND_START $(date +%T)"
python -u src/attack_image.py $COMMON --text recovered --text-from runs/gate/text --out $O/recovered > logs/cal_img_recovered.log 2>&1 &
python -u src/attack_image.py $COMMON --text none --out $O/none > logs/cal_img_none.log 2>&1 &
python -u src/attack_image.py $COMMON --text oracle --out $O/oracle > logs/cal_img_oracle.log 2>&1 &
wait
sync
for c in recovered none oracle; do echo "$c: $(ls $O/$c/*.json 2>/dev/null | wc -l) results; $(grep -cE 'Error|Traceback' logs/cal_img_$c.log) error lines"; done
echo "CAL_TEXTCOND_DONE $(date +%T)"
