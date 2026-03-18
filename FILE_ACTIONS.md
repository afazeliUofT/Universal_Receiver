Overwrite these existing files:
- `src/upair5g/paper_configured_reservoir.py`

Add these new files:
- `configs/target_cdlc_highmobility_paper_phasefair_cleancontrol.yaml`
- `configs/target_cdlc_highmobility_paper_phasefair_dmrsrich.yaml`
- `scripts/run_compare_paper_phasefair_cleancontrol.sh`
- `scripts/run_compare_paper_phasefair_dmrsrich.sh`
- `scripts/cleanup_control_rerun.sh`
- `scripts/make_publication_robustness_figures.py`
- `scripts/run_make_publication_robustness.sh`
- `slurm/compare_paper_phasefair_cleancontrol.slurm`
- `slurm/compare_paper_phasefair_dmrsrich.slurm`
- `slurm/make_publication_robustness.slurm`
- `tests/regression_paper_phasefair_warningfree.py`
- `docs/ROBUSTNESS_CONTROLS_UPDATE.md`
- `FILE_ACTIONS.md`
- `REMOVE_ONLY_IF_RERUN_CONTROLS.txt`

Remove nothing from the existing repo. Keep all old logs, outputs, checkpoints, and plots.
