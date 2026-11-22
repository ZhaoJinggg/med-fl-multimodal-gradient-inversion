#!/bin/bash
# GPU dry run of the main-experiment code on the calibration seed (seed 0, clients 0 and 1): every protocol, the
# three text conditions, and u = 16 for time and memory. Short image budgets (300 iterations): this checks that the
# code runs and that its self-checks hold, it produces no result. One process at a time.
# Two cards since 2026-11-23: training and the image attack run on MAIN_GPU (default 0, PCI 41:00.0), the text attack
# on TEXT_GPU (default 1, PCI 42:00.0), because card 0 hung the machine during the text search five times.
# Refuses to start unless every card's power limit is at most MAXLIMIT (default 350 W, the user's setting);
# the watchdog logs every card. Finished warm-ups are not repeated. The text lane and the image lane run at the same time.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
export CUDA_DEVICE_ORDER=PCI_BUS_ID
MAIN_GPU=${MAIN_GPU:-0}; TEXT_GPU=${TEXT_GPU:-1}; MAXLIMIT=${MAXLIMIT:-350}
IDLE=${IDLE:-0}; WD_TEMP=${WD_TEMP:-120}   # user, 2026-11-23: no throttle and no temperature stop; the watchdog only logs (and stops the jobs if a card disappears)
for lim in $(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1); do
  [ "$lim" -le "$MAXLIMIT" ] || { echo "a power limit is '$lim' W, above $MAXLIMIT: not starting"; exit 1; }
done
pgrep -f "[g]pu_watchdog.sh" >/dev/null || (setsid nohup bash scripts/gpu_watchdog.sh $WD_TEMP 9999 >/dev/null 2>&1 &)
FL=runs/dry/fl; TX=runs/dry/text; IM=runs/dry/image
V="0 1"
step() {  # card, name, then command
  local gpu=$1 name=$2; shift 2
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu "$@" > logs/dry_$name.log 2>&1
  local rc=$?
  echo "STEP $name card $gpu exit $rc $(( $(date +%s) - t0 ))s mem,temp $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | tr -d ' ' | tr '\n' ' ')"; sync
  grep -E "Error|Traceback" -A3 logs/dry_$name.log | head -8
}
G() { case $1 in fedavg) echo $FL/seed0_fedavg;; fedprox) echo $FL/seed0_fedprox_mu1;; scaffold) echo $FL/seed0_scaffold_optII;; fedbn) echo $FL/seed0_fedbn;; esac; }
echo "DRY_START $(date +%T)"
# 1. warm-ups (9 rounds, only round 9 saved)
for p in fedavg fedprox scaffold fedbn; do
  [ -e $(G $p)/global_round9.pt ] && { echo "warm_$p already there"; continue; }
  step $MAIN_GPU warm_$p python -u src/train_fl.py --proto $p --seed 0 --rounds 9 --lr-img 0.01 --lr-txt 1e-2 --res 128 --out $FL --save 9 --aux-clients $V --idle $IDLE
  grep -hE "round +9|round 9|test" logs/dry_warm_$p.log | tail -2 | cut -c1-160
done
TXT="--seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims $V --batch 8 --res 128 --parallel 1000 --noise-factor 1.5 --l1-thresh 5e-2 --l2-thresh 1.5e-2 --idle $IDLE --out $TX"
IMG="--seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims $V --batch 8 --res 128 --init mean --opt-lr 0.01 --tv 1e-2 --bn 1.0 --iters 300 --checkpoints 300"
# 2. text on TEXT_GPU, in the background: every protocol at u = 4; FedAvg also at u = 1 and 16
text_lane() {
  for p in fedavg fedprox scaffold fedbn; do
    step $TEXT_GPU text_${p}_u4 timeout 1800 python -u src/attack_text.py --global $(G $p)/global_round9.pt --proto $p --u 4 $TXT
    grep -h "^victim" logs/dry_text_${p}_u4.log | cut -c1-150
  done
  for u in 1 16; do
    step $TEXT_GPU text_fedavg_u$u timeout 1800 python -u src/attack_text.py --global $(G fedavg)/global_round9.pt --proto fedavg --u $u $TXT
    grep -h "^victim" logs/dry_text_fedavg_u$u.log | cut -c1-150
  done
  echo "TEXT_LANE_DONE $(date +%T)"
}
text_lane & TPID=$!
# 3. image on MAIN_GPU, at the same time: first the cells that need no recovered text
for c in oracle none; do
  step $MAIN_GPU img_fedavg_u1_$c timeout 1200 python -u src/attack_image.py --global $(G fedavg)/global_round9.pt --proto fedavg --u 1 --text $c $IMG --out $IM/fedavg_u1_$c
done
wait $TPID
# then the cells with the recovered text: FedAvg u = 1, every protocol at u = 4, FedAvg u = 16
step $MAIN_GPU img_fedavg_u1_recovered timeout 1200 python -u src/attack_image.py --global $(G fedavg)/global_round9.pt --proto fedavg --u 1 --text recovered --text-from $TX $IMG --out $IM/fedavg_u1_recovered
for p in fedavg fedprox scaffold fedbn; do
  step $MAIN_GPU img_${p}_u4 timeout 1200 python -u src/attack_image.py --global $(G $p)/global_round9.pt --proto $p --u 4 --text recovered --text-from $TX $IMG --out $IM/${p}_u4_recovered
done
step $MAIN_GPU img_fedavg_u16 timeout 2400 python -u src/attack_image.py --global $(G fedavg)/global_round9.pt --proto fedavg --u 16 --text recovered --text-from $TX $IMG --out $IM/fedavg_u16_recovered
# 4. what the runs recorded about themselves
python - <<'EOF'
import glob, json, os
for f in sorted(glob.glob("runs/dry/image/*/*.json")):
    d = json.load(open(f))
    c = d.get("checks", {})
    print(os.path.basename(os.path.dirname(f)), "v%s" % d.get("victim"), "min", d.get("minutes"), "cond", d.get("condition"), "proto", d.get("proto"),
          "| checks:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items() if not isinstance(v, (list, dict))},
          "| text_feat_zerr", c.get("text_feat_zerr"))
EOF
for g in 0 1; do echo "card $g: peak temp $(awk -F, -v g=$g '$2==g && $3+0>t{t=$3+0} END{print t}' logs/gpu_watch.csv) C, peak power $(awk -F, -v g=$g '$2==g && $4+0>w{w=$4+0} END{print w}' logs/gpu_watch.csv) W, peak mem $(awk -F, -v g=$g '$2==g && $6+0>m{m=$6+0} END{print m}' logs/gpu_watch.csv) MiB"; done
du -sh runs/dry 2>/dev/null
echo "DRY_DONE $(date +%T)"
