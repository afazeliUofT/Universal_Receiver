#!/bin/bash
#SBATCH --job-name=upair-twc-dmrsrich
#SBATCH --account=def-rsadve
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/rsadve1/PROJECT/Universal_Receiver/logs/%x-%j.out

set -euo pipefail
module purge || true
module load StdEnv/2023 || true
module load python/3.10 || true
module load cuda/12.2 || module load cuda/12.1 || true

cd /home/rsadve1/PROJECT/Universal_Receiver
mkdir -p logs
bash scripts/run_twc_mild_dmrsrich_full.sh
