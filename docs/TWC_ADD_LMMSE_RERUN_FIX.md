# TWC add-2D-LMMSE rerun fix

Problem:
- The three add-2D-LMMSE eval jobs timed out before evaluation finished and before refreshed curves/plots were written.

Fix:
- Increase the three add-2D-LMMSE eval Slurm wall-times to 24 hours.
- Increase CPU reservation to 12 and memory to 48G for better throughput/headroom.
- Add a small cleanup script that removes only the failed add-2D-LMMSE log files.
