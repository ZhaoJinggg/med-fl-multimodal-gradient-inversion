#!/bin/bash
cd ~/jacob/mmgia
for p in $(pgrep -f "scripts/calibration/lr_sweep\.sh"; pgrep -f "python src/warmup\.py"); do kill $p 2>/dev/null; done
sleep 3
echo "still running: $(pgrep -f 'python src/warmup\.py' | wc -l)"
rm -rf runs/pilot/lr_sweep
setsid nohup scripts/calibration/lr_sweep.sh > logs/pilot_lr_sweep.log 2>&1 < /dev/null &
sleep 2
echo "sweep processes: $(pgrep -f 'scripts/calibration/lr_sweep\.sh' | wc -l)"
