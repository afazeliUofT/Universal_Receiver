# Robustness controls + paper-phasefair warning fix

This update does two things.

1. It fixes the repeated Keras warning emitted by the phase-aware paper configured-reservoir estimator:
   - explicit `build()` added
   - more robust estimator input parsing added
   - new regression checks that those warning strings are gone

2. It adds the next publication-strength step:
   - **clean control**: same full trained UPAIR checkpoint, but evaluate with extra symbol-wise phase impairments disabled
   - **DMRS-rich control**: same full trained UPAIR checkpoint, but evaluate with one extra DMRS position (`additional_position: 1`)
   - consolidated robustness publication plots/tables across:
     - targeted hard case
     - clean control
     - DMRS-rich control

Notes:
- These new control jobs are **evaluation-only**; they reuse the existing full UPAIR checkpoint at
  `outputs/upair_cdlc_highmobility/checkpoints/best.weights.h5`.
- Old smoke/full/richer/phase-aware/paper-phasefair outputs remain useful and should be kept.
- Remove only the new control outputs/logs if you rerun those new control jobs.
