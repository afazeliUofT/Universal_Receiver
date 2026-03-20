Overwrite these existing files:
- src/upair5g/evaluation.py
- scripts/make_twc_plots.py
- configs/twc_mild_main.yaml
- configs/twc_mild_clean.yaml
- configs/twc_mild_dmrsrich.yaml
- slurm/twc_mild_main_full.slurm
- slurm/twc_mild_clean_full.slurm
- slurm/twc_mild_dmrsrich_full.slurm

Add these new files:
- scripts/cleanup_twc_natural_rerun.sh
- docs/TWC_NATURAL_CURVES_REFINEMENT.md
- FILE_ACTIONS.md

Remove nothing from code.

Only if you are rerunning the refined mild campaign, remove:
- TWC_plots/
- outputs/twc_mild_main/
- outputs/twc_mild_clean/
- outputs/twc_mild_dmrsrich/
- old logs/upair-twc-*.out
