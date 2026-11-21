#!/bin/bash
for p in $(pgrep -f "scripts/run_pilot\.sh"; pgrep -f "python src/attack_"); do kill $p 2>/dev/null; done
sleep 2; echo "attack processes left: $(pgrep -f 'python src/attack_' | wc -l)"
