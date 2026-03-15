Overwrite these existing repo files:
- src/upair5g/baselines.py
- src/upair5g/evaluation.py
- src/upair5g/plotting.py

Add these new repo files:
- src/upair5g/phase_aware.py
- configs/smoke_phaseaware_baselines.yaml
- configs/target_cdlc_highmobility_phaseaware_baselines.yaml
- scripts/run_smoke_phaseaware_baselines.sh
- scripts/run_compare_phaseaware_baselines.sh
- scripts/cleanup_phaseaware_rerun.sh
- slurm/smoke_phaseaware_baselines.slurm
- slurm/compare_phaseaware_baselines.slurm
- tests/regression_phaseaware_identity.py
- docs/PHASE_AWARE_BASELINES_UPDATE.md
- FILE_ACTIONS.md
- REMOVE_ONLY_IF_RERUN.txt

Do not remove any of the old smoke/full/richer-baseline files or outputs.
