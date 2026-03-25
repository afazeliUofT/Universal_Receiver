from __future__ import annotations

import argparse
from pathlib import Path

from upair5g.config import load_config
from upair5g.evaluation import evaluate_model
from upair5g.plotting import make_all_plots

BASELINE = "baseline_ls_2dlmmse_lmmse"


def _ensure_once(lst: list[str], item: str, before: str | None = None) -> list[str]:
    out = [str(x) for x in lst if str(x) != item]
    if before is not None and before in out:
        idx = out.index(before)
        out.insert(idx, item)
    else:
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Add empirical 2D-LMMSE baseline to an existing TWC mild config and rerun eval only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    baselines = cfg.setdefault("baselines", {})
    enabled = [str(x) for x in baselines.get("enabled_receivers", [])]
    enabled = _ensure_once(enabled, BASELINE, before="baseline_ddcpe_ls_lmmse")
    baselines["enabled_receivers"] = enabled

    evaluation = cfg.setdefault("evaluation", {})
    stopping = [str(x) for x in evaluation.get("stopping_receivers", [])]
    stopping = _ensure_once(stopping, BASELINE, before="baseline_ddcpe_ls_lmmse")
    evaluation["stopping_receivers"] = stopping

    # Give the empirical covariance baseline a slightly stronger covariance estimate
    # without changing the trained UPAIR checkpoint.
    cov = baselines.setdefault("covariance_estimation", {})
    cov.setdefault("reuse_cache", True)
    cov["num_batches"] = max(int(cov.get("num_batches", 24)), 48)
    cov["batch_size"] = max(int(cov.get("batch_size", 48)), 64)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    eval_result = evaluate_model(cfg, checkpoint_path=str(ckpt))
    plot_result = make_all_plots(cfg)
    print({"eval": eval_result, "plots": plot_result})


if __name__ == "__main__":
    main()
