#!/bin/bash
#SBATCH --job-name=upair-twc2-plots
#SBATCH --account=def-rsadve
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/upair-twc2-plots-%j.out

set -euo pipefail
cd /home/rsadve1/PROJECT/Universal_Receiver
source /home/rsadve1/PROJECT/Universal_Receiver/venv_universal_receiver/bin/activate
bash scripts/run_make_twc_plots2.sh
