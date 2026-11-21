#!/bin/bash
# Stop all pilot job scripts and attack processes (patterns live here so that the ssh command line is not matched).
for p in $(pgrep -f "scripts/(run_pilot|tune_tv|attrib|pilot_b1_long|push_b8|push_b24|text_grid)\.sh"; pgrep -f "python.* src/(attack_|control_)"); do kill $p 2>/dev/null; done
sleep 2; echo "attack processes left: $(pgrep -f 'python.* src/(attack_|control_)' | wc -l)"
