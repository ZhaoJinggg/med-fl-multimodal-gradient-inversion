#!/bin/bash
# GPU watchdog: samples every visible card about once per second into logs/gpu_watch.csv and stops our attack
# processes when any card reaches a limit, or when a card disappears (card 41:00.0 fell off the bus at 76-87 C,
# 350-414 W on 2026-11-22). Log line: time,gpu index,temp_C,power_W,util_pct,mem_MiB,fan_pct (one line per card).
# usage: bash scripts/gpu_watchdog.sh [max_temp_C=72] [max_power_W=280]   (run in the background; exits after a trip)
cd "$(dirname "$0")/.." || exit 1
TMAX=${1:-72}; PMAX=${2:-280}
LOG=logs/gpu_watch.csv
Q="nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used,fan.speed --format=csv,noheader,nounits"
N0=$($Q 2>/dev/null | grep -c .)
echo "time,gpu,temp_C,power_W,util_pct,mem_MiB,fan_pct  (limits ${TMAX} C, ${PMAX} W, ${N0} cards)" >> "$LOG"
stop_jobs() {
  for pid in $(pgrep -f "python.* src/(attack_|control_|text_diag|identify|train_fl)"; pgrep -f "scripts/(run_pilot|tune_tv|attrib|pilot_b1_long|push_b8|push_b24|text_grid|text_grid2|text_cal|gate_image|gate_text|dryrun_seed0|main_text|main_image|main_warmup)\.sh"); do kill "$pid" 2>/dev/null; done
}
while true; do
  out=$($Q 2>/dev/null)
  n=$(echo "$out" | grep -c .)
  if [ -z "$out" ] || [ "$n" -lt "$N0" ]; then
    echo "$(date +%T),GPU LOST: $n of $N0 cards visible, attack processes stopped" >> "$LOG"; stop_jobs; sync; exit 2
  fi
  now=$(date +%T); trip=""
  while IFS= read -r line; do
    echo "$now,${line// /}" >> "$LOG"
    t=$(echo "$line" | cut -d, -f2 | tr -d ' '); p=$(echo "$line" | cut -d, -f3 | tr -d ' ' | cut -d. -f1)
    if [ "$t" -ge "$TMAX" ] || [ "$p" -ge "$PMAX" ]; then trip="card $(echo "$line" | cut -d, -f1) at ${t} C ${p} W"; fi
  done <<< "$out"
  if [ -n "$trip" ]; then
    stop_jobs
    echo "$(date +%T),TRIPPED $trip: attack processes stopped" >> "$LOG"; sync
    exit 1
  fi
  sleep 1
done
