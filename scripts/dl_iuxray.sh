#!/bin/bash
# Resume NLMCXR_png.tgz only when Open-i serves real bytes (it sometimes returns a CloudFront HTML error page).
URL=https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
F=~/jacob/mmgia/data/iuxray/raw/NLMCXR_png.tgz
TOTAL=1360814128
while true; do
  SIZE=$(stat -c %s $F 2>/dev/null || echo 0)
  if [ "$SIZE" -ge "$TOTAL" ]; then
    gzip -t $F && echo "DOWNLOAD_DONE_GZIP_OK $(date +%F\ %T)" && exit 0
    echo "GZIP_CHECK_FAILED $(date +%T)"; exit 1
  fi
  CT=$(curl -s -m 30 -r $SIZE-$((SIZE+1023)) -D - -o /dev/null $URL | awk -F": " "tolower(\$1)==\"content-type\"{print \$2}" | tr -d "\r")
  if [ "$CT" = "text/html" ] || [ -z "$CT" ]; then
    echo "$(date +%T) source not serving archive (content-type=${CT:-none}); size=$SIZE; retry in 10 min"
    sleep 600; continue
  fi
  echo "$(date +%T) resuming at $SIZE (content-type=$CT)"
  curl -sS -f --speed-limit 1000 --speed-time 120 -C - -o $F $URL || { echo "$(date +%T) curl exit $?"; sleep 60; }
done
