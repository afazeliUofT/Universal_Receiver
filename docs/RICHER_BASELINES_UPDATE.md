# Richer baselines update

This update adds a fairer 5G NR PUSCH comparison stage without retraining the proposed UPAIR-5G weights.

Added classical baselines:
- `baseline_ls_lmmse`
- `baseline_ls_timeavg_lmmse`
- `baseline_ls_2dlmmse_lmmse`
- existing `upair5g_lmmse`
- existing `perfect_csi_lmmse`

How it works:
1. Reuse the already trained smoke/full checkpoints.
2. Re-evaluate the same checkpoints with richer baseline receivers.
3. Save new curves/plots under new output folders:
   - `outputs/smoke_upair_cdlc_richer_baselines/`
   - `outputs/upair_cdlc_highmobility_richer_baselines/`

No existing file needs to be deleted.
