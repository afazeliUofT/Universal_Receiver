#!/bin/bash
set -euo pipefail
PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"
source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1
python tests/regression_baseline_builds.py --config configs/twc_mild_main.yaml
python scripts/run_eval_add_2dlmmse.py \
  --config configs/twc_mild_main.yaml \
  --checkpoint outputs/twc_mild_main/checkpoints/best.weights.h5
