#!/bin/bash
#SBATCH --job-name=upair-twc-dmrs-lmmse
#SBATCH --account=def-rsadve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/upair-twc-dmrs-lmmse-%j.out

bash scripts/run_twc_mild_dmrsrich_add_2dlmmse_eval.sh
