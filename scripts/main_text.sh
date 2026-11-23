#!/bin/bash
# Text attack of the main experiment on the 20 test victims (seeds 1-5, four victims each), frozen setting
# (noise factor 1.5, l1 5e-2, l2 1.5e-2, float32 update, at most 500 partial sequences), throttled (--idle 2).
# Order: first the cells whose result the image attack of stage 1 needs (FedAvg u = 1 at batch 8, 4, 2, 1 and
# FedBN u = 1 at batch 1, 2, 8), then every other cell of the paper's Table of cells, then the exact-update
# reference (DAGER's default thresholds), then calibration step (vi) on seed 0 (SCAFFOLD and FedAvg, u = 4,
# clients 3-7). No stopping rule. A victim whose result file exists is skipped, so the script can be restarted.
# Output: runs/main/text/seed<s>/v<v>_B<B>_u<u><suffix>.json (the file name carries no seed: one directory per seed).
# Memory: one text process may take up to half of the card (--gpu-mem 0.5). On 2026-11-23 a trial with this
# script, a warm-up and two image processes at the same time ran out of GPU memory; do not run image attacks
# next to it (scripts/main_image_stage1.sh waits for MAIN_TEXT_DONE).
# usage: [GPU=0] [IDLE=2] [SEEDS="1 2 3 4 5"] [ROOT=runs/main/text] [FL=runs/fl] [WARMLOG=logs/main_warmup.log] bash scripts/main_text.sh
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
IDLE=${IDLE:-2}; SEEDS=${SEEDS:-"1 2 3 4 5"}; ROOT=${ROOT:-runs/main/text}; FL=${FL:-runs/fl}
pgrep -f "[g]pu_watchdog.sh" >/dev/null || (setsid nohup bash scripts/gpu_watchdog.sh 120 9999 >/dev/null 2>&1 &)
FROZEN="--noise-factor 1.5 --l1-thresh 5e-2 --l2-thresh 1.5e-2"
gdir() { case $2 in fedavg) echo $FL/seed${1}_fedavg;; fedprox) echo $FL/seed${1}_fedprox_mu1;; scaffold) echo $FL/seed${1}_scaffold_optII;; fedbn) echo $FL/seed${1}_fedbn;; esac; }
sfx() { case $1 in fedavg) echo "";; fedprox) echo _fedprox_mu1;; scaffold) echo _scaffold_optII;; fedbn) echo _fedbn;; esac; }
victims() {  # the test victims of a seed; VICTIMS overrides them (smoke tests on seed 0 only)
  if [ -n "${VICTIMS:-}" ]; then echo $VICTIMS; else python -c "import json; print(*sorted(map(int, json.load(open('data/splits/seed$1.json'))['victim_batches'])))"; fi
}
cell() {  # seed, protocol, u, batch, out dir, victims (quoted), then extra flags ("--exact" or the frozen thresholds)
  local s=$1 p=$2 u=$3 b=$4 out=$5 vs=$6; shift 6
  local ex=""; [[ " $* " == *" --exact "* ]] && ex=_exact
  local todo=""
  for v in $vs; do [ -s $out/v${v}_B${b}_u${u}$(sfx $p)$ex.json ] || todo="$todo $v"; done
  [ -z "$todo" ] && return
  local g=$(gdir $s $p)/global_round9.pt
  [ -s $g ] || { echo "MISSING $g: cell seed $s $p u=$u B=$b not run"; return; }
  local log=logs/main_text_s${s}_${p}_u${u}_B${b}$ex.log t0=$(date +%s)
  timeout 3600 python -u src/attack_text.py --global $g --proto $p --u $u --seed $s --lr-img 0.01 --lr-txt 1e-2 \
      --victims $todo --batch $b --res 128 --parallel 1000 --idle $IDLE --out $out "$@" > $log 2>&1
  local rc=$?
  sync
  echo "TEXT seed $s $p u=$u B=$b$ex victims$todo exit $rc $(( $(date +%s) - t0 ))s | $(grep -h "^victim" $log | sed -E 's/.*ROUGE-L ([0-9.]+).*recovered ([0-9]+\/[0-9]+).*/\1 (\2)/' | tr '\n' ' ')"
  grep -E "Error|Traceback|Killed" -A3 $log | head -6
}
wait_seed() {  # WARMLOG set: wait until the warm-ups of this seed are finished (FedBN is the last of a seed)
  [ -z "${WARMLOG:-}" ] && return
  until grep -q "^WARM seed $1 fedbn exit 0" $WARMLOG 2>/dev/null; do sleep 20; done
}
echo "MAIN_TEXT_START $(date +%T) seeds $SEEDS"
# 1. what the image attack of stage 1 needs
for s in $SEEDS; do
  wait_seed $s
  V=$(victims $s); O=$ROOT/seed$s
  for b in 8 4 2 1; do cell $s fedavg 1 $b $O "$V" $FROZEN; done
  for b in 1 2 8; do cell $s fedbn 1 $b $O "$V" $FROZEN; done
done
echo "TEXT_FOR_IMAGES_DONE $(date +%T)"
# 2. the other cells of the main experiment
for s in $SEEDS; do
  V=$(victims $s); O=$ROOT/seed$s
  for u in 2 4 8 16; do cell $s fedavg $u 8 $O "$V" $FROZEN; done
  for p in fedprox scaffold fedbn; do for u in 4 8; do cell $s $p $u 8 $O "$V" $FROZEN; done; done
done
echo "TEXT_CELLS_DONE $(date +%T)"
# 3. reference: the update without rounding, DAGER's default thresholds
for s in $SEEDS; do cell $s fedavg 1 8 $ROOT/seed$s "$(victims $s)" --exact; done
echo "TEXT_EXACT_DONE $(date +%T)"
# 4. calibration step (vi), text part: paired differences SCAFFOLD - FedAvg at u = 4 on five clients of seed 0
if [ "$SEEDS" = "1 2 3 4 5" ]; then
  [ -n "${WARMLOG:-}" ] && until grep -q "WARMUP_DONE" $WARMLOG 2>/dev/null; do sleep 20; done
  for p in fedavg scaffold; do cell 0 $p 4 8 runs/cal6/text "3 4 5 6 7" $FROZEN; done
fi
echo "results: $(find $ROOT -name '*.json' | wc -l) files; empty: $(find $ROOT -type f -size 0 | wc -l)"
echo "MAIN_TEXT_DONE $(date +%T)"
