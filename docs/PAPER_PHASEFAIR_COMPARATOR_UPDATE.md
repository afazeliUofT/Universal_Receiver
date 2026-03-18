# Phase-aware paper comparator update

This update keeps the paper-style configured-reservoir comparator, but adds a fairer variant for the current one-DMRS, phase-distorted 5G PUSCH setting:

- `paper_cfgres_ls_lmmse`
- `paper_cfgres_phaseaware_ls_lmmse`

## Why this update is needed

The current paper-style comparator adapts from DMRS only, but in the present evaluation setup the strongest classical baseline is already phase-aware. That makes the paper comparator unfairly weak in a scenario dominated by symbol-wise phase distortion.

This update therefore adds a **phase-aware paper comparator** that:
- applies the same slot-local DDCPE-style phase front-end first
- solves the configured-reservoir readout on the phase-preconditioned LS estimate
- re-applies the phase profile after the configured solve
- keeps the same downstream Sionna 5G NR PUSCH LMMSE detector

## Added receiver

`paper_cfgres_phaseaware_ls_lmmse`

Interpretation:
- paper-style configured TF basis from covariance
- slot-local readout solve from current DMRS
- slot-local phase preconditioning before the configured solve
- same detector/decoder chain as all other receivers

## New outputs

The new compare jobs write to:
- `outputs/smoke_upair_cdlc_paper_phasefair_comparison/`
- `outputs/upair_cdlc_highmobility_paper_phasefair_comparison/`

Useful new artifacts:
- `artifacts/paper_cfgres_phasefair_covariances.npz`
- `artifacts/paper_cfgres_phasefair_basis.npz`

## Publication-level figures now added/updated

The plotting pipeline now also creates:
- `publication_paper_phasefair_focus.(png|pdf)`
- `publication_error_floor_zoom.(png|pdf)`

The update also removes the previous `tight_layout()` warning path during figure save.
