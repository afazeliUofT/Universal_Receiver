# TWC final mild-scenario campaign

This update adds a new final campaign that is separate from the earlier hard-case experiments.

The new campaign is designed for publication-quality curves with milder settings:
- moderate mobility instead of high mobility,
- milder symbol-wise phase distortion in the main scenario,
- a clean control without the extra phase distortion,
- and a DMRS-rich control with one extra DMRS position.

The new final plots are collected in a single top-level folder named `TWC_plots/`.

## New scenarios

### `configs/twc_mild_main.yaml`
Main publication scenario.
- CDL-C uplink, 3.5 GHz, 30 kHz SCS
- moderate speed: 8.33 to 16.67 m/s
- one DMRS symbol (`additional_position: 0`)
- mild phase distortion enabled
- MCS 15 on 64-QAM
- large Monte-Carlo budget for smooth curves

### `configs/twc_mild_clean.yaml`
Same as the main scenario, but with the extra symbol-wise phase distortion disabled.

### `configs/twc_mild_dmrsrich.yaml`
Same as the main scenario, but with one extra DMRS position (`additional_position: 1`).

## Final figures written to `TWC_plots/`

- `Fig01_mild_main_curves.pdf/png`
- `Fig02_mild_main_focus.pdf/png`
- `Fig03_mild_main_upair_gain.pdf/png`
- `Fig04_mild_main_channel_error_maps.pdf/png`
- `Fig05_mild_crossscenario_metrics_grid.pdf/png`
- `Fig06_mild_crossscenario_upair_gain.pdf/png`

The folder also contains:
- `TWC_plot_manifest.txt`
- `tables/mild_main_upair_gain.csv`
- `tables/mild_crossscenario_gain.csv`

## Run order

Run these three full jobs first:
1. `sbatch slurm/twc_mild_main_full.slurm`
2. `sbatch slurm/twc_mild_clean_full.slurm`
3. `sbatch slurm/twc_mild_dmrsrich_full.slurm`

Then run:
4. `sbatch slurm/twc_make_plots.slurm`

## Clean rerun

If you want to rerun only this new final campaign, use:
```bash
bash scripts/cleanup_twc_final_rerun.sh
```
