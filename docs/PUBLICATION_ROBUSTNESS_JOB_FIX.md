# Publication robustness job fix

Problem:
- `make_publication_robustness.slurm` can start before the control-evaluation jobs finish.
- If that happens, the plotting script cannot find the required `curves.csv` files yet.

Fix:
- `scripts/run_make_publication_robustness.sh` now waits for the required curves files.
- `slurm/make_publication_robustness.slurm` now has enough wall time to wait safely.

Recommended rerun after applying this patch:
```bash
cd /home/rsadve1/PROJECT/Universal_Receiver
git pull
source /home/rsadve1/PROJECT/Universal_Receiver/venv_universal_receiver/bin/activate
python -m pip install -e .
deactivate
bash scripts/cleanup_publication_robustness_rerun.sh
sbatch slurm/make_publication_robustness.slurm
```

This rerun does not retrain anything and does not reevaluate the receivers.
It only regenerates the consolidated cross-scenario publication-robustness outputs.
