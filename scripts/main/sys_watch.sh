#!/bin/bash
# System watch: one line per second with the CPU temperature (k10temp Tctl), the CPU frequency (lowest and
# highest core, MHz), the load, and the GPU (temperature, power, utilization, memory, fan). The file is flushed
# to disk after every line, so that a power cut keeps the last second (the machine switched itself off on
# 2026-11-23 15:25:26 with nothing in the logs and no CPU temperature on record).
# Optional guard: with CPU_MAX > 0 the text attack and scripts/main/main_text.sh are stopped when Tctl reaches it.
# usage: bash scripts/main/sys_watch.sh [CPU_MAX=0]        log: logs/sys_watch.csv
cd "$(dirname "$0")/../.." || exit 1
CPU_MAX=${1:-0}; LOG=logs/sys_watch.csv
H=""; for d in /sys/class/hwmon/hwmon*; do [ "$(cat $d/name 2>/dev/null)" = k10temp ] && H=$d; done
echo "time,cpu_Tctl_C,cpu_MHz_min,cpu_MHz_max,load1,gpu_temp_C,gpu_power_W,gpu_util_pct,gpu_mem_MiB,gpu_fan_pct  (guard: CPU ${CPU_MAX} C)" >> $LOG
while true; do
  t=$(( $(cat $H/temp1_input 2>/dev/null || echo 0) / 1000 ))
  f=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | sort -n | sed -n '1p;$p' | tr '\n' ' ')
  set -- $f; fmin=$(( ${1:-0} / 1000 )); fmax=$(( ${2:-0} / 1000 ))
  g=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,fan.speed --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "$(date +%T),$t,$fmin,$fmax,$(cut -d' ' -f1 /proc/loadavg),$g" >> $LOG; sync $LOG
  if [ "$CPU_MAX" -gt 0 ] && [ "$t" -ge "$CPU_MAX" ]; then
    for p in $(pgrep -f "scripts/main/main_text\.sh"; pgrep -f "^(timeout [0-9]+ )?python.* src/attack_text\.py"); do kill $p 2>/dev/null; done
    echo "$(date +%T),CPU GUARD at $t C: text attack stopped" >> $LOG; sync $LOG; exit 1
  fi
  sleep 1
done
