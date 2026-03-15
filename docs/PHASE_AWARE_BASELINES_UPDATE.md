This update adds one new strong classical comparator for the current hard 5G PUSCH setting:

- `baseline_ddcpe_ls_lmmse`

What it does:
1. Starts from the same Sionna LS + linear-interpolation channel estimate used by the strongest current classical baseline.
2. Performs single-stream MMSE/MRC equalization on the received resource grid.
3. Uses hard QAM decisions over data REs to estimate one residual common phase per OFDM symbol.
4. Smooths that phase profile over time and anchors it at the DMRS symbol.
5. Rotates the classical channel estimate by the recovered phase profile.
6. Keeps the same downstream Sionna PUSCH receiver and LMMSE detection.

Why this is the right next baseline:
- The current stress test injects a symbol-wise common phase process.
- Standard LS + interpolation is not phase-aware under only one DMRS symbol.
- This added baseline is still fully classical, slot-local, and feed-forward.
- It is a much fairer comparator before moving to an implementation of the paper-style configured reservoir receiver.
