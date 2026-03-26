#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"
cd "${PROJECT_ROOT}"

rm -f   logs/upair-twc-main-lmmse-*.out   logs/upair-twc-clean-lmmse-*.out   logs/upair-twc-dmrs-lmmse-*.out

echo "Removed old add-2dlmmse eval logs."
echo "Kept existing outputs/, checkpoints/, and TWC_plots/."
