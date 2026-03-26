#!/bin/bash
#SBATCH --job-name=upair-twc2-clean
#SBATCH --account=def-rsadve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/upair-twc2-clean-%j.out

set -euo pipefail
cd /home/rsadve1/PROJECT/Universal_Receiver
source /home/rsadve1/PROJECT/Universal_Receiver/venv_universal_receiver/bin/activate
bash scripts/run_twc2_mild_clean_eval.sh
