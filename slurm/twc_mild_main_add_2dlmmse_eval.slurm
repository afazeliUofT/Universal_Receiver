#!/bin/bash
#SBATCH --job-name=upair-twc-main-lmmse
#SBATCH --account=def-rsadve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/upair-twc-main-lmmse-%j.out

bash scripts/run_twc_mild_main_add_2dlmmse_eval.sh
