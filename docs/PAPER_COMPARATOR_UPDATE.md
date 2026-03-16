# Paper-style configured-reservoir comparator update

This update adds a fairer comparator inspired by the original Universal Neural Receiver philosophy:

- no offline training for the new comparator
- slot-local adaptation from current DMRS only
- domain-knowledge configuration from empirical time/frequency covariance
- same downstream Sionna 5G NR PUSCH LMMSE detector as all other receivers

## Added receiver

`paper_cfgres_ls_lmmse`

Interpretation:
- paper-style configured reservoir / WESN-inspired TF basis
- DMRS-driven online readout solve each slot
- same PUSCH chain and same decoder as other baselines and UPAIR-5G

## New outputs

The new compare jobs write to:
- `outputs/smoke_upair_cdlc_paper_comparison/`
- `outputs/upair_cdlc_highmobility_paper_comparison/`

Useful new artifacts:
- `artifacts/paper_cfgres_covariances.npz`
- `artifacts/paper_cfgres_basis.npz`
- publication-grade figures in `plots/`

## Publication-level figures now generated automatically

Besides the standard BER/BLER/NMSE curves, the plotting pipeline now also creates:
- `publication_main_curves.(png|pdf)`
- `publication_upair_gain_over_best_nonupair.(png|pdf)`
- `publication_paper_vs_upair.(png|pdf)`
- `publication_channel_error_maps.(png|pdf)`
- `publication_selected_points.csv`
- `publication_selected_points.tex`
- `publication_receiver_ranking.csv`
