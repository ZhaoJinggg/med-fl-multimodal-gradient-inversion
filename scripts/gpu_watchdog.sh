#!/bin/bash
# GPU watchdog: samples temperature and power once per second into logs/gpu_watch.csv and stops our
# attack processes when either limit is reached (the card fell off the bus twice at 81-87 C, 350-377 W).
# usage: bash scripts/gpu_watchdog.sh [max_temp_C=72] [max_power_W=280]   (run in the background; exits after a trip)
cd "$(dirname "$0")/.." || exit 1
TMAX=${1:-72}; PMAX=${2:-280}
LOG=logs/gpu_watch.csv
echo "time,temp_C,power_W,util_pct,mem_MiB,fan_pct  (limits ${TMAX} C, ${PMAX} W)" >> "$LOG"
while true; do
  line=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,fan.speed --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$line" ]; then echo "$(date +%T),GPU NOT FOUND" >> "$LOG"; exit 2; fi
  echo "$(date +%T),${line// /}" >> "$LOG"
  t=$(echo "$line" | cut -d, -f1 | tr -d ' '); p=$(echo "$line" | cut -d, -f2 | tr -d ' ' | cut -d. -f1)
  if [ "$t" -ge "$TMAX" ] || [ "$p" -ge "$PMAX" ]; then
    for pid in $(pgrep -f "python.* src/(attack_|control_|text_diag|identify)"; pgrep -f "scripts/(run_pilot|tune_tv|attrib|pilot_b1_long|push_b8|push_b24|text_grid|text_grid2|text_cal)\.sh"); do kill "$pid" 2>/dev/null; done
    echo "$(date +%T),TRIPPED at ${t} C ${p} W: attack processes stopped" >> "$LOG"
    exit 1
  fi
  sleep 1
done
