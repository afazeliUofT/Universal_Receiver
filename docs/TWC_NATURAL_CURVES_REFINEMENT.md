TWC natural-curves refinement

What this update changes
- Replaces fixed-batch averaging in `src/upair5g/evaluation.py` with count-based evaluation.
- Saves exact Monte-Carlo counts in `curves.csv`:
  - `bit_errors`, `num_bits`
  - `block_errors`, `num_blocks`
  - `num_batches_run`
  - `reliable_ber`, `reliable_bler`
- Adds adaptive stopping controls through the mild configs.
- Updates `scripts/make_twc_plots.py` so BER/BLER points are shown only when enough observed errors exist.
- Removes artificial BER/BLER visual floors from the final TWC plots.
- Slightly rebalances the mild scenarios by using `num_rx_ant: 2` and a denser 1 dB Eb/N0 grid.

Why this update is needed
- The current mild plots show flat BER/BLER tails for UPAIR and perfect CSI because zero-observed-error points are plotted at a positive floor.
- Some classical high-SNR points fluctuate because only a few block errors were observed there.
- For publication-quality BER/BLER curves, the right fix is stronger Monte-Carlo accounting and reliability-aware plotting.

Expected outcome after rerun
- BER/BLER curves end naturally when the error statistics become unreliable.
- The remaining BER/BLER points become more defensible and less jagged.
- NMSE curves remain fully visible.
