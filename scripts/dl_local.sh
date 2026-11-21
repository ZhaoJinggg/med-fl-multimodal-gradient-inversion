#!/bin/bash
# 1) HF mirror (R2Gen layout, 2-view studies only) as fallback
P="/c/Users/Jacob/Assignments/Couse Project ICS-500/mmgia"
cd "$P/data/iuxray/mirror_r2gen"
curl -fL --retry 10 -C - -o image.zip https://huggingface.co/datasets/dz-osamu/IU-Xray/resolve/main/image.zip && echo "MIRROR_DONE $(date +%T)"
# 2) Original Open-i PNG archive: retry every 10 min until it serves real bytes
cd "$P/data/iuxray/raw"
URL=https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz; TOTAL=1360814128
for i in $(seq 1 144); do
  SIZE=$(stat -c %s NLMCXR_png.tgz 2>/dev/null || echo 0)
  [ "$SIZE" -ge "$TOTAL" ] && { gzip -t NLMCXR_png.tgz && echo "OPENI_DONE_GZIP_OK $(date +%T)"; exit 0; }
  CT=$(curl -s -m 30 -r $SIZE-$((SIZE+1023)) -D - -o /dev/null $URL | awk -F': ' 'tolower($1)=="content-type"{print $2}' | tr -d '\r')
  if [ "$CT" = "text/html" ] || [ -z "$CT" ]; then echo "$(date +%T) Open-i not serving (ct=${CT:-none}), size=$SIZE"; sleep 600; continue; fi
  echo "$(date +%T) Open-i resuming at $SIZE"
  curl -sS -f --speed-limit 1000 --speed-time 120 -C - -o NLMCXR_png.tgz $URL || sleep 60
done
