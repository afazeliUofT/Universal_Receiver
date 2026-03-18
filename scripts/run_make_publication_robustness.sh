#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"

source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1

python scripts/make_publication_robustness_figures.py
