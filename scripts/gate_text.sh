#!/bin/bash
# Text attack on the held-out calibration clients 3-9 (seed 0, round 9, 128 px, batch 8) with the setting frozen on
# clients 0-2: noise factor 1.5, l1 5e-2, l2 1.5e-2, float32 update; u = 1 and u = 4; exact update at u = 1 as a
# reference (DAGER's default thresholds). Decides nothing (the paper: reported next to the tuning clients).
# One process at a time. Refuses to start unless the GPU power limit is 250 W; runs under the watchdog (stop at 78 C).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
lim=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
[ "$lim" = 250 ] || { echo "power limit is '$lim' W, not 250: not starting"; exit 1; }
pgrep -f "[g]pu_watchdog.sh" >/dev/null || (setsid nohup bash scripts/gpu_watchdog.sh 78 9999 >/dev/null 2>&1 &)
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
COMMON="--global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 3 4 5 6 7 8 9 --batch 8 --res 128 --parallel 1000 --out runs/gate/text"
FROZEN="--noise-factor 1.5 --l1-thresh 5e-2 --l2-thresh 1.5e-2"
run() {
  timeout 900 python -u src/attack_text.py $COMMON "$@" 2>&1 | grep --line-buffered -E "^victim|^SUMMARY|Error|Traceback" | cut -c1-220
  echo "exit ${PIPESTATUS[0]} $(date +%T)"
}
echo "GATE_TEXT_START $(date +%T)"
run --u 1 $FROZEN
run --u 4 $FROZEN
run --u 1 --exact
echo "GATE_TEXT_DONE $(date +%T)"
