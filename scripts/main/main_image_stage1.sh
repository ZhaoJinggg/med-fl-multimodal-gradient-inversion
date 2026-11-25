#!/bin/bash
# Image attack, STAGE 1 of the main experiment (paper, "Run order of the image cells"): 160 attacks on the 20
# test victims (seeds 1-5), all at u = 1, frozen setting (both terms, BN weight 1, TV 1e-2, mean start, Adam
# 0.01, 12,000 iterations):
#   fedavg_u1_B8_recovered            the cell whose floor test opens stage 2            (20)
#   fedavg_u1_B{4,2,1}_recovered      batch-size comparison                              (60)
#   fedavg_u1_B2_{oracle,none}        exploratory: RQ2 at batch 2                        (40)
#   fedbn_u1_B{1,2}_recovered         exploratory: FedBN at small batches                (40)
# The user decided on 2026-11-23 that the image part STOPS after stage 1 for now; later stages are not started
# by this script. One job = one cell of one seed (its four victims in one process); JOBS processes side by side
# (default 4, steady load). A victim whose .json exists is skipped, so the script can be restarted after a crash.
# Needs the text results of the same cells and the whole card: it waits for MAIN_TEXT_DONE in logs/main_text.log
# and until no warm-up or text process is left (a trial on 2026-11-23 ran out of GPU memory with a text process,
# a warm-up and two image processes side by side; one image process at batch 8 takes about 4.3 GB).
# Output: runs/main/image/<cell>/s<seed>_v<v>_B<B>_u1_mean.{json,pt} (one directory per cell).
# usage: [GPU=0] [JOBS=4] [SEEDS="1 2 3 4 5"] [ROOT=runs/main] [FL=runs/fl] [ITERS=12000] bash scripts/main/main_image_stage1.sh
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
JOBS=${JOBS:-4}; SEEDS=${SEEDS:-"1 2 3 4 5"}; ROOT=${ROOT:-runs/main}; FL=${FL:-runs/fl}; ITERS=${ITERS:-12000}
WAITLOG=${WAITLOG:-logs/main_text.log}; WAITFOR=${WAITFOR:-MAIN_TEXT_DONE}
pgrep -f "[g]pu_watchdog.sh" >/dev/null || (setsid nohup bash scripts/main/gpu_watchdog.sh 120 9999 >/dev/null 2>&1 &)
if [ "$ITERS" = 12000 ]; then CK="4000 8000 12000"; else CK=$ITERS; fi
gdir() { case $2 in fedavg) echo $FL/seed${1}_fedavg;; fedbn) echo $FL/seed${1}_fedbn;; esac; }
job() {  # protocol, batch, condition, seed
  local p=$1 b=$2 c=$3 s=$4
  local name=${p}_u1_B${b}_$c
  local out=$ROOT/image/$name
  local vs=${VICTIMS:-$(python -c "import json; print(*sorted(map(int, json.load(open('data/splits/seed$s.json'))['victim_batches'])))")}  # VICTIMS: smoke tests on seed 0 only
  local todo=""
  for v in $vs; do [ -s $out/s${s}_v${v}_B${b}_u1_mean.json ] || todo="$todo $v"; done
  [ -z "$todo" ] && { echo "IMG $name seed $s already there"; return; }
  local g=$(gdir $s $p)/global_round9.pt
  [ -s $g ] || { echo "MISSING $g: $name seed $s not run"; return; }
  local tf=""; [ $c = recovered ] && tf="--text-from $ROOT/text/seed$s"
  local log=logs/main_img_${name}_s$s.log t0=$(date +%s)
  python -u src/attack_image.py --global $g --proto $p --u 1 --seed $s --lr-img 0.01 --lr-txt 1e-2 --victims $todo \
      --batch $b --res 128 --init mean --opt-lr 0.01 --tv 1e-2 --bn 1.0 --iters $ITERS --checkpoints $CK \
      --text $c $tf --out $out > $log 2>&1
  local rc=$?
  sync
  echo "IMG $name seed $s victims$todo exit $rc $(( ($(date +%s) - t0) / 60 ))min $(date +%T) | cos $(grep -h FINAL $log | sed -E 's/.*cos=([0-9.]+).*/\1/' | tr '\n' ' ')"
  grep -E "Error|Traceback|Killed" -A3 $log | head -6
}
export -f job gdir; export ROOT FL ITERS CK; [ -n "${VICTIMS:-}" ] && export VICTIMS
echo "IMAGE_STAGE1_WAIT $(date +%T) for $WAITFOR in $WAITLOG and for the end of every warm-up and text process"
until grep -q "$WAITFOR" $WAITLOG 2>/dev/null; do sleep 20; done
while pgrep -f "^python.* src/(train_fl|attack_text)\.py" >/dev/null; do sleep 20; done
echo "IMAGE_STAGE1_START $(date +%T) jobs $JOBS seeds $SEEDS iterations $ITERS"
joblist() {
  for s in $SEEDS; do echo "fedavg 8 recovered $s"; done
  for b in 4 2 1; do for s in $SEEDS; do echo "fedavg $b recovered $s"; done; done
  for c in oracle none; do for s in $SEEDS; do echo "fedavg 2 $c $s"; done; done
  for b in 1 2; do for s in $SEEDS; do echo "fedbn $b recovered $s"; done; done
}
joblist | xargs -P $JOBS -L 1 bash -c 'job "$@"' _
sync
# second pass: a job that ended abnormally is repeated once for its missing victims (same seed, same start;
# the paper's rule for attacks that end abnormally); finished victims are skipped
echo "IMAGE_STAGE1_SECOND_PASS $(date +%T) $(find $ROOT/image -name '*.json' | wc -l) results so far"
joblist | xargs -P $JOBS -L 1 bash -c 'job "$@"' _ | grep -v "already there"
sync
echo "results: $(find $ROOT/image -name '*.json' | wc -l) of 160; empty files: $(find $ROOT/image -type f -size 0 | wc -l)"
echo "IMAGE_STAGE1_DONE $(date +%T)"
