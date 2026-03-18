# Paper-comparison entrypoint alias update

This patch prevents accidental reruns of the older non-phase-aware paper comparator.

## Why this is needed
The repo already contains the newer `paper_phasefair` benchmark, but the older Slurm entrypoints
(`smoke_paper_comparison.slurm` and `compare_paper_comparison.slurm`) are easy to run by habit.
If those old entrypoints are used, they evaluate the older `paper_cfgres_ls_lmmse` setup instead
of the fairer phase-aware paper comparator.

## What this patch changes
- `slurm/smoke_paper_comparison.slurm` now routes to the newer phase-aware paper benchmark.
- `slurm/compare_paper_comparison.slurm` now routes to the newer phase-aware paper benchmark.
- the matching `scripts/run_*_paper_comparison.sh` files now delegate to the newer phase-aware scripts.
- the generated log filenames now clearly contain `paper-phasefair` in the Slurm job name.

## What this does **not** change
- it does **not** delete or overwrite old `*_paper_comparison` outputs.
- it does **not** remove the older non-phase-aware paper comparator from the repo.
- it does **not** change the dedicated `*_paper_phasefair_comparison` configs or scripts.

## Recommendation
From now on, either of these is acceptable and will run the fairer benchmark:
- `sbatch slurm/smoke_paper_phasefair_comparison.slurm`
- `sbatch slurm/smoke_paper_comparison.slurm`

and similarly:
- `sbatch slurm/compare_paper_phasefair_comparison.slurm`
- `sbatch slurm/compare_paper_comparison.slurm`

Both old names become safe aliases to the latest fair comparator.
