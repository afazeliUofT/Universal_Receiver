#!/bin/bash
set -euo pipefail
cd /home/rsadve1/PROJECT/Universal_Receiver
python scripts/run_eval_twc2.py \
  --config configs/twc_mild_clean.yaml \
  --checkpoint outputs/twc_mild_clean/checkpoints/best.weights.h5
