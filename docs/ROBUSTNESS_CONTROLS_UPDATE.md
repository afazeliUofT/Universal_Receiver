# Robustness controls + paper-phasefair warning fix

This update now has a final workflow fix for the consolidated publication-robustness plot job.

It already:
1. fixed the repeated Keras warning emitted by the phase-aware paper configured-reservoir estimator
2. added the two evaluation-only controls
   - clean control
   - DMRS-rich control

This patch adds:
- a wait-for-dependencies guard in `run_make_publication_robustness.sh`
- longer wall time for `make_publication_robustness.slurm`
- a cleanup helper for rerunning only the publication-robustness job

The publication-robustness job now waits until these files exist before plotting:
- `outputs/upair_cdlc_highmobility_paper_phasefair_comparison/metrics/curves.csv`
- `outputs/upair_cdlc_highmobility_paper_phasefair_cleancontrol/metrics/curves.csv`
- `outputs/upair_cdlc_highmobility_paper_phasefair_dmrsrich/metrics/curves.csv`

Old smoke/full/richer/phase-aware/paper/paper-phasefair/control outputs remain useful and should be kept.
