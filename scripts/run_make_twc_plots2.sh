#!/bin/bash
set -euo pipefail
cd /home/rsadve1/PROJECT/Universal_Receiver

required=(
  "outputs/twc2_mild_main/metrics/curves.csv"
  "outputs/twc2_mild_clean/metrics/curves.csv"
  "outputs/twc2_mild_dmrsrich/metrics/curves.csv"
  "outputs/twc2_mild_main/artifacts/channel_example.npz"
)

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path"
    exit 1
  fi
done

python scripts/make_twc_plots2.py
