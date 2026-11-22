#!/bin/bash
# GPU dry run of the main-experiment code on the calibration seed (seed 0, clients 0 and 1): every protocol, the
# three text conditions, and u = 16 for time and memory. Short image budgets (300 iterations): this checks that the
# code runs and that its self-checks hold, it produces no result. One process at a time.
# Refuses to start unless the GPU power limit is 250 W; runs under the watchdog (stop at 78 C).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
cd ~/jacob/mmgia
lim=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
[ "$lim" = 250 ] || { echo "power limit is '$lim' W, not 250: not starting"; exit 1; }
pgrep -f "[g]pu_watchdog.sh" >/dev/null || (setsid nohup bash scripts/gpu_watchdog.sh 78 9999 >/dev/null 2>&1 &)
FL=runs/dry/fl; TX=runs/dry/text; IM=runs/dry/image
V="0 1"
step() {  # name, then command
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > logs/dry_$name.log 2>&1
  local rc=$?
  echo "STEP $name exit $rc $(( $(date +%s) - t0 ))s mem $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | tr -d ' ')"
  grep -E "Error|Traceback" -A3 logs/dry_$name.log | head -8
}
echo "DRY_START $(date +%T)"
# 1. warm-ups (9 rounds, only round 9 saved)
for p in fedavg fedprox scaffold fedbn; do
  step warm_$p python -u src/train_fl.py --proto $p --seed 0 --rounds 9 --lr-img 0.01 --lr-txt 1e-2 --res 128 --out $FL --save 9 --aux-clients $V
  grep -hE "round +9|round 9|test" logs/dry_warm_$p.log | tail -2 | cut -c1-160
done
G() { case $1 in fedavg) echo $FL/seed0_fedavg;; fedprox) echo $FL/seed0_fedprox_mu1;; scaffold) echo $FL/seed0_scaffold_optII;; fedbn) echo $FL/seed0_fedbn;; esac; }
TXT="--seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims $V --batch 8 --res 128 --parallel 1000 --noise-factor 1.5 --l1-thresh 5e-2 --l2-thresh 1.5e-2 --idle 2 --out $TX"
IMG="--seed 0 --lr-img 0.01 --lr-txt 1e-2 --victims $V --batch 8 --res 128 --init mean --opt-lr 0.01 --tv 1e-2 --bn 1.0 --iters 300 --checkpoints 300"
# 2. text: every protocol at u = 4; FedAvg also at u = 1 and 16
for p in fedavg fedprox scaffold fedbn; do
  step text_${p}_u4 timeout 600 python -u src/attack_text.py --global $(G $p)/global_round9.pt --proto $p --u 4 $TXT
  grep -h "^victim" logs/dry_text_${p}_u4.log | cut -c1-150
done
for u in 1 16; do
  step text_fedavg_u$u timeout 600 python -u src/attack_text.py --global $(G fedavg)/global_round9.pt --proto fedavg --u $u $TXT
  grep -h "^victim" logs/dry_text_fedavg_u$u.log | cut -c1-150
done
# 3. image: every protocol at u = 4 with the recovered text; FedAvg u = 1 under the three conditions; FedAvg u = 16
for p in fedavg fedprox scaffold fedbn; do
  step img_${p}_u4 timeout 1200 python -u src/attack_image.py --global $(G $p)/global_round9.pt --proto $p --u 4 --text recovered --text-from $TX $IMG --out $IM/${p}_u4_recovered
done
for c in oracle recovered none; do
  extra=""; [ $c = recovered ] && extra="--text-from $TX"
  step img_fedavg_u1_$c timeout 1200 python -u src/attack_image.py --global $(G fedavg)/global_round9.pt --proto fedavg --u 1 --text $c $extra $IMG --out $IM/fedavg_u1_$c
done
step img_fedavg_u16 timeout 2400 python -u src/attack_image.py --global $(G fedavg)/global_round9.pt --proto fedavg --u 16 --text recovered --text-from $TX $IMG --out $IM/fedavg_u16_recovered
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
echo "peak temp $(awk -F, '$2+0>t{t=$2+0} END{print t}' logs/gpu_watch.csv), peak mem $(awk -F, '$5+0>m{m=$5+0} END{print m}' logs/gpu_watch.csv) MiB"
du -sh runs/dry 2>/dev/null
echo "DRY_DONE $(date +%T)"
