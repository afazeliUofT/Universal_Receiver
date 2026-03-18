Overwrite these existing files:
- `src/upair5g/baselines.py`
- `src/upair5g/evaluation.py`
- `src/upair5g/paper_configured_reservoir.py`
- `src/upair5g/plotting.py`

Add these new files:
- `configs/smoke_paper_phasefair_comparison.yaml`
- `configs/target_cdlc_highmobility_paper_phasefair_comparison.yaml`
- `scripts/run_smoke_paper_phasefair_comparison.sh`
- `scripts/run_compare_paper_phasefair_comparison.sh`
- `scripts/cleanup_paper_phasefair_rerun.sh`
- `slurm/smoke_paper_phasefair_comparison.slurm`
- `slurm/compare_paper_phasefair_comparison.slurm`
- `tests/regression_paper_phasefair_build.py`
- `docs/PAPER_PHASEFAIR_COMPARATOR_UPDATE.md`
- `FILE_ACTIONS.md`
- `REMOVE_ONLY_IF_RERUN_PAPER_PHASEFAIR.txt`

Remove nothing from the old smoke/full/richer-baseline/phase-aware/paper-comparison code and outputs.
