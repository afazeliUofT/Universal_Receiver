from __future__ import annotations

import argparse
from pathlib import Path

from upair5g.config import load_config
from upair5g.evaluation import evaluate_model

FINAL_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]

STOPPING_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
]


def _retarget_experiment_name(name: str) -> str:
    name = str(name)
    if name.startswith("twc_mild_"):
        return name.replace("twc_mild_", "twc2_mild_", 1)
    if name.startswith("twc_"):
        return name.replace("twc_", "twc2_", 1)
    return f"{name}_twc2"


def _prepare_cfg(cfg: dict) -> dict:
    cfg = dict(cfg)
    experiment = cfg.setdefault("experiment", {})
    experiment["name"] = _retarget_experiment_name(experiment.get("name", "twc2_run"))

    baselines = cfg.setdefault("baselines", {})
    baselines["enabled_receivers"] = list(FINAL_RECEIVERS)

    # Correct the currently misleading "2D-LMMSE" setup into a true
    # frequency-time empirical LMMSE interpolator with its own cache.
    cov = baselines.setdefault("covariance_estimation", {})
    cov["cache_name"] = "empirical_covariances_ft2d.npz"
    cov["reuse_cache"] = True
    cov["num_batches"] = max(int(cov.get("num_batches", 24)), 96)
    cov["batch_size"] = max(int(cov.get("batch_size", 48)), 96)
    cov["order"] = "f-t"
    cov["use_spatial_smoothing"] = False
    cov["normalize_trace"] = True
    cov["diagonal_loading"] = max(float(cov.get("diagonal_loading", 1.0e-4)), 1.0e-3)
    cov["use_training_impairments"] = False

    evaluation = cfg.setdefault("evaluation", {})
    evaluation["stopping_receivers"] = list(STOPPING_RECEIVERS)
    evaluation["min_num_batches_per_point"] = max(int(evaluation.get("min_num_batches_per_point", 256)), 512)
    evaluation["max_num_batches_per_point"] = max(int(evaluation.get("max_num_batches_per_point", 1024)), 1536)
    evaluation["num_batches_per_point"] = max(int(evaluation.get("num_batches_per_point", 1024)), 1536)
    evaluation["target_block_errors_per_receiver"] = max(int(evaluation.get("target_block_errors_per_receiver", 100)), 150)
    evaluation["reliable_min_block_errors"] = max(int(evaluation.get("reliable_min_block_errors", 20)), 30)
    evaluation["reliable_min_bit_errors"] = max(int(evaluation.get("reliable_min_bit_errors", 100)), 200)
    evaluation["save_example_batch"] = True

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a final TWC mild checkpoint into a new twc2_* output folder "
            "with a corrected pure 2D frequency-time empirical LMMSE baseline, "
            "no configured-reservoir curves, and stricter BER/BLER reliability bookkeeping."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    cfg = load_config(args.config)
    cfg = _prepare_cfg(cfg)
    result = evaluate_model(cfg, checkpoint_path=str(ckpt))
    print(result)


if __name__ == "__main__":
    main()
