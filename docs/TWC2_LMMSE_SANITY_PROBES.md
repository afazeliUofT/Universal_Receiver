This package adds a diagnostic probe for the remaining LS+2D-LMMSE anomaly.

It does NOT change the current TWC or TWC2 figures.

What it probes:
- current twc2 2D-LMMSE settings
- same settings with normalize_trace disabled
- same settings with order changed from f-t to t-f
- a static clean version of the current twc2 settings

For each case it prints:
- covariance eigenvalue statistics
- direct channel-estimation NMSE
- channel-estimate power ratio
- channel-estimate correlation with the true channel

Run only:
sbatch slurm/probe_twc2_lmmse_sanity.slurm

Then paste back:
- logs/upair-twc2-lmmse-probe-<jobid>.out
- outputs/twc2_lmmse_probe/summary.json
