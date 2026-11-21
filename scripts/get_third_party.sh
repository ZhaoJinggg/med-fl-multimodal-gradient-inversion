#!/bin/bash
# Fetch external code into third_party/ (not tracked in this repo).
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/dager ]; then
  git clone -q https://github.com/insait-institute/dager-gradient-inversion.git third_party/dager
fi
git -C third_party/dager checkout -q 5f8e306   # commit used in this study (2025-08-13)
echo "DAGER at $(git -C third_party/dager log -1 --format='%h %cd')"
