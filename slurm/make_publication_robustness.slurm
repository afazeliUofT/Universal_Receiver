#!/bin/bash
#SBATCH --job-name=upair-publication-robustness
#SBATCH --account=def-rsadve
#SBATCH --time=08:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/rsadve1/PROJECT/Universal_Receiver/logs/%x-%j.out

set -euo pipefail
module purge || true
module load StdEnv/2023 || true
module load python/3.10 || true

cd /home/rsadve1/PROJECT/Universal_Receiver
mkdir -p logs
bash scripts/run_make_publication_robustness.sh
