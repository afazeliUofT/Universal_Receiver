Overwrite these existing files:
- `src/upair5g/baselines.py`
- `src/upair5g/evaluation.py`
- `src/upair5g/plotting.py`

Add these new files:
- `src/upair5g/paper_configured_reservoir.py`
- `configs/smoke_paper_comparison.yaml`
- `configs/target_cdlc_highmobility_paper_comparison.yaml`
- `scripts/run_smoke_paper_comparison.sh`
- `scripts/run_compare_paper_comparison.sh`
- `scripts/cleanup_paper_comparison_rerun.sh`
- `slurm/smoke_paper_comparison.slurm`
- `slurm/compare_paper_comparison.slurm`
- `tests/regression_paper_cfgres_build.py`
- `docs/PAPER_COMPARATOR_UPDATE.md`
- `FILE_ACTIONS.md`
- `REMOVE_ONLY_IF_RERUN_PAPER.txt`

Remove nothing from the old smoke/full/richer-baseline/phase-aware code and outputs.
