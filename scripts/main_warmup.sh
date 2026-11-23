#!/bin/bash
# Warm-ups for the main experiment: seeds 1-5 under the four protocols (9 rounds, round 9 saved, with the
# auxiliary state of the four test victims of the seed), then seed 0 under FedAvg and SCAFFOLD with the state
# of all ten clients (calibration step vi). Throttled (--idle 2 by default). A finished run is skipped; files
# of 0 bytes, which a crash leaves behind, are removed first so that a lost run is repeated.
# usage: [GPU=0] [IDLE=2] bash scripts/main_warmup.sh      (flags of train_fl.py as in scripts/dryrun_seed0.sh)
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
IDLE=${IDLE:-2}; OUT=runs/fl
pgrep -f "[g]pu_watchdog.sh" >/dev/null || (setsid nohup bash scripts/gpu_watchdog.sh 120 9999 >/dev/null 2>&1 &)
dir() { case $2 in fedavg) echo $OUT/seed${1}_fedavg;; fedprox) echo $OUT/seed${1}_fedprox_mu1;; scaffold) echo $OUT/seed${1}_scaffold_optII;; fedbn) echo $OUT/seed${1}_fedbn;; esac; }
warm() {  # seed, protocol, then the clients whose auxiliary state is saved
  local s=$1 p=$2; shift 2
  local d=$(dir $s $p)
  [ -d $d ] && find $d -maxdepth 1 -type f -size 0 -delete
  [ -s $d/global_round9.pt ] && [ -s $d/train_log.json ] && { echo "seed $s $p already there"; return; }
  local t0=$(date +%s)
  python -u src/train_fl.py --proto $p --seed $s --rounds 9 --lr-img 0.01 --lr-txt 1e-2 --res 128 --out $OUT --save 9 --aux-clients "$@" --idle $IDLE > logs/warm_s${s}_$p.log 2>&1
  local rc=$?
  sync
  echo "WARM seed $s $p exit $rc $(( $(date +%s) - t0 ))s $(grep -h "'round': 9" logs/warm_s${s}_$p.log | tail -1 | grep -oE "'test_acc': [0-9.]+") $(date +%T)"
  grep -E "Error|Traceback" -A3 logs/warm_s${s}_$p.log | head -6
}
echo "WARMUP_START $(date +%T)"
for s in 1 2 3 4 5; do
  V=$(python -c "import json; print(*sorted(map(int, json.load(open('data/splits/seed$s.json'))['victim_batches'])))")
  echo "seed $s victims $V"
  for p in fedavg fedprox scaffold fedbn; do warm $s $p $V; done
done
warm 0 fedavg 0 1 2 3 4 5 6 7 8 9
warm 0 scaffold 0 1 2 3 4 5 6 7 8 9
echo "empty files: $(find $OUT -type f -size 0 | wc -l)"; du -sh $OUT
echo "WARMUP_DONE $(date +%T)"
