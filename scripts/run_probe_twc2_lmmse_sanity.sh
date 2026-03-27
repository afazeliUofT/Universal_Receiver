#!/bin/bash
set -euo pipefail

cd /home/rsadve1/PROJECT/Universal_Receiver
source /home/rsadve1/PROJECT/Universal_Receiver/venv_universal_receiver/bin/activate

python scripts/probe_twc2_lmmse_sanity.py \
  --config configs/twc_mild_main.yaml \
  --probe_batches 8 \
  --snrs 0 6 10
