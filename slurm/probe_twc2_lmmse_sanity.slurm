#!/bin/bash
#SBATCH --job-name=upair-twc2-lmmse-probe
#SBATCH --account=def-rsadve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/upair-twc2-lmmse-probe-%j.out

set -euo pipefail
bash scripts/run_probe_twc2_lmmse_sanity.sh
