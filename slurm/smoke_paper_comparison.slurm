#!/bin/bash
#SBATCH --job-name=upair-smoke-paper-phasefair
#SBATCH --account=def-rsadve
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/rsadve1/PROJECT/Universal_Receiver/logs/%x-%j.out

set -euo pipefail

module purge || true
module load StdEnv/2023 || true
module load python/3.10 || true
module load cuda/12.2 || module load cuda/12.1 || true

cd /home/rsadve1/PROJECT/Universal_Receiver
mkdir -p logs

echo "[ALIAS] slurm/smoke_paper_comparison.slurm now routes to the latest phase-aware paper comparator benchmark."
echo "[ALIAS] Outputs will be written under outputs/smoke_upair_cdlc_paper_phasefair_comparison/."

bash scripts/run_smoke_paper_phasefair_comparison.sh
