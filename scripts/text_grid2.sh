#!/bin/bash
# Text attack (DAGER), batch 8, float32 update, tuning clients 0-2, 128 px model. One run at a time.
#  part 1: guessing baseline (CPU)
#  part 2: tight grid around the calibrated setting (noise factor x l2 threshold, then the l1 threshold)
#  part 3: local steps u = 2, 8, 16 at the calibrated setting (u = 1 and 4 exist), and the exact update at u = 4
# Temperature and power are logged every second to logs/gpu_watch.csv (log only, no limits).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
mkdir -p logs/done
bash scripts/gpu_watchdog.sh 120 9999 >/dev/null 2>&1 &
WD=$!
G=runs/pilot_r128/seed0_fedavg/global_round9.pt
run() {  # tag, then extra arguments
  local tag=$1; shift
  local mark=logs/done/$(echo "$tag $*" | tr -c 'A-Za-z0-9._\n-' '_')
  if [ -e "$mark" ]; then echo "$tag | already done"; return; fi
  timeout 240 python -u src/attack_text.py --global $G --seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims 0 1 2 --batch 8 --res 128 \
      --parallel 1000 --tag "$tag" --out runs/pilot/text128 "$@" 2>&1 \
      | grep --line-buffered -E "^victim|Error|Traceback" | cut -c1-200 | sed "s/^/$tag | /"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && touch "$mark"
  echo "$tag | exit $rc | $(date +%T)"
}
echo "PART 1 baseline $(date +%T)"
[ -e runs/pilot/text128/baseline_seed0_B8.json ] || python -u src/text_baseline.py --seed 0 --victims 0 1 2 --batch 8 2>&1 | grep -E "^victim|^MEAN|Error|Traceback" | cut -c1-200
echo "PART 2 grid $(date +%T)"
for nf in 1.5 2 3; do for l2 in 1e-2 1.5e-2 2e-2; do
  run _g2_nf${nf}_l1_5e-2_l2_${l2} --u 1 --noise-factor $nf --l1-thresh 5e-2 --l2-thresh $l2
done; done
for l1 in 2e-2 1e-1; do
  run _g2_nf2_l1_${l1}_l2_2e-2 --u 1 --noise-factor 2 --l1-thresh $l1 --l2-thresh 2e-2
done
echo "PART 3 steps $(date +%T)"
for u in 2 8 16; do
  run _cal --u $u --noise-factor 2 --l1-thresh 5e-2 --l2-thresh 2e-2
done
run _exact --u 4 --exact
kill $WD 2>/dev/null
echo "TEXT_GRID2_DONE $(date +%T)"
