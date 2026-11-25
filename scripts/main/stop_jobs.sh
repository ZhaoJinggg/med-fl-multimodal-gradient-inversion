#!/bin/bash
# Stop all pilot job scripts and attack processes (patterns live here so that the ssh command line is not matched).
for p in $(pgrep -f "scripts/(main|calibration)/(run_pilot|tune_tv|attrib|pilot_b1_long|push_b8|push_b24|text_grid|text_grid2|text_cal|gate_image|gate_text|dryrun_seed0|main_text|main_image|main_warmup)\.sh"; pgrep -f "python.* src/(calibration/)?(attack_|control_|train_fl)"); do kill $p 2>/dev/null; done
sleep 2; echo "attack processes left: $(pgrep -f 'python.* src/(calibration/)?(attack_|control_|train_fl)' | wc -l)"
