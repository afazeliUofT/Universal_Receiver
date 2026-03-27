from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from upair5g.baselines import estimate_empirical_covariances, build_empirical_lmmse_interpolator
from upair5g.builders import build_channel, build_ls_estimator, build_pusch_transmitter
from upair5g.config import ensure_output_tree, load_config
from upair5g.evaluation import _call_channel_estimator, _make_eval_batch
from upair5g.utils import save_json, set_global_seed


def _nmse(h_true: tf.Tensor, h_hat: tf.Tensor) -> float:
    h_true = tf.cast(tf.convert_to_tensor(h_true), tf.complex64)
    h_hat = tf.cast(tf.convert_to_tensor(h_hat), tf.complex64)
    num = tf.reduce_mean(tf.abs(h_hat - h_true) ** 2)
    den = tf.reduce_mean(tf.abs(h_true) ** 2) + tf.constant(1e-12, tf.float32)
    return float((tf.cast(num, tf.float32) / den).numpy())


def _power_ratio(h_true: tf.Tensor, h_hat: tf.Tensor) -> float:
    h_true = tf.cast(tf.convert_to_tensor(h_true), tf.complex64)
    h_hat = tf.cast(tf.convert_to_tensor(h_hat), tf.complex64)
    p_true = tf.reduce_mean(tf.abs(h_true) ** 2) + tf.constant(1e-12, tf.float32)
    p_hat = tf.reduce_mean(tf.abs(h_hat) ** 2)
    return float((tf.cast(p_hat, tf.float32) / p_true).numpy())


def _corr_abs(h_true: tf.Tensor, h_hat: tf.Tensor) -> float:
    h_true = tf.cast(tf.reshape(tf.convert_to_tensor(h_true), [-1]), tf.complex64)
    h_hat = tf.cast(tf.reshape(tf.convert_to_tensor(h_hat), [-1]), tf.complex64)
    num = tf.abs(tf.reduce_mean(h_hat * tf.math.conj(h_true)))
    den = tf.sqrt(
        (tf.reduce_mean(tf.abs(h_true) ** 2) + tf.constant(1e-12, tf.float32))
        * (tf.reduce_mean(tf.abs(h_hat) ** 2) + tf.constant(1e-12, tf.float32))
    )
    return float((tf.cast(num, tf.float32) / den).numpy())


def _eig_stats(mat: tf.Tensor) -> dict[str, float]:
    mat = tf.cast(tf.convert_to_tensor(mat), tf.complex64)
    eig = tf.linalg.eigvalsh(mat)
    eig_r = tf.math.real(eig).numpy()
    eig_r = np.asarray(eig_r, dtype=np.float64)
    return {
        "dim": int(eig_r.size),
        "min_eig": float(np.min(eig_r)),
        "max_eig": float(np.max(eig_r)),
        "mean_eig": float(np.mean(eig_r)),
        "trace": float(np.sum(eig_r)),
        "cond_like": float(np.max(eig_r) / max(np.min(eig_r), 1e-12)),
    }


def _prepare_cfg(base_cfg: dict[str, Any], variant: str, static_clean: bool) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"probe_{exp.get('name', 'run')}_{variant}" + ("_staticclean" if static_clean else "")
    cov = cfg.setdefault("baselines", {}).setdefault("covariance_estimation", {})
    cov["cache_name"] = f"probe_cov_{variant}.npz"
    cov["reuse_cache"] = False
    cov["num_batches"] = max(int(cov.get("num_batches", 24)), 24)
    cov["batch_size"] = max(int(cov.get("batch_size", 48)), 48)
    cov["order"] = "f-t"
    cov["use_spatial_smoothing"] = False
    cov["use_training_impairments"] = False

    if variant == "current_twc2":
        cov["normalize_trace"] = True
        cov["diagonal_loading"] = max(float(cov.get("diagonal_loading", 1e-4)), 1e-3)
    elif variant == "no_trace_norm":
        cov["normalize_trace"] = False
        cov["diagonal_loading"] = float(cov.get("diagonal_loading", 1e-4))
    elif variant == "tf_order":
        cov["normalize_trace"] = True
        cov["diagonal_loading"] = max(float(cov.get("diagonal_loading", 1e-4)), 1e-3)
        cov["order"] = "t-f"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if static_clean:
        cfg.setdefault("impairments", {})["enabled"] = False
        ch = cfg.setdefault("channel", {})
        ch["min_speed_mps"] = 0.0
        ch["max_speed_mps"] = 0.0

    return cfg


def _probe_variant(base_cfg: dict[str, Any], variant: str, static_clean: bool, snrs: list[float], num_batches: int) -> dict[str, Any]:
    cfg = _prepare_cfg(base_cfg, variant=variant, static_clean=static_clean)
    set_global_seed(int(cfg.get("seed", 1234)))
    paths = ensure_output_tree(cfg)
    tx, _ = build_pusch_transmitter(cfg)
    channel = build_channel(cfg, tx)

    covs = estimate_empirical_covariances(tx=tx, channel=channel, cfg=cfg, paths=paths)
    interpolator = build_empirical_lmmse_interpolator(tx=tx, channel=channel, cfg=cfg, paths=paths)
    est_lin = build_ls_estimator(tx, cfg, interpolation_type="lin")
    est_lmmse = build_ls_estimator(tx, cfg, interpolator=interpolator)

    result: dict[str, Any] = {
        "variant": variant,
        "static_clean": static_clean,
        "experiment_name": cfg["experiment"]["name"],
        "cache_name": cfg["baselines"]["covariance_estimation"]["cache_name"],
        "covariance": {
            "time": _eig_stats(covs["cov_mat_time"]),
            "freq": _eig_stats(covs["cov_mat_freq"]),
        },
        "snr_points": {},
    }

    batch_size = int(cfg["system"]["batch_size_eval"])
    for snr in snrs:
        acc = {
            "ls_lin_nmse": [],
            "ls_lin_power_ratio": [],
            "ls_lin_corr_abs": [],
            "ls_2dlmmse_nmse": [],
            "ls_2dlmmse_power_ratio": [],
            "ls_2dlmmse_corr_abs": [],
        }
        for _ in range(num_batches):
            batch = _make_eval_batch(tx=tx, channel=channel, cfg=cfg, batch_size=batch_size, ebno_db=float(snr))
            h_true = batch["h"]
            h_lin, _ = _call_channel_estimator(est_lin, batch["y"], batch["no"])
            h_2d, _ = _call_channel_estimator(est_lmmse, batch["y"], batch["no"])
            acc["ls_lin_nmse"].append(_nmse(h_true, h_lin))
            acc["ls_lin_power_ratio"].append(_power_ratio(h_true, h_lin))
            acc["ls_lin_corr_abs"].append(_corr_abs(h_true, h_lin))
            acc["ls_2dlmmse_nmse"].append(_nmse(h_true, h_2d))
            acc["ls_2dlmmse_power_ratio"].append(_power_ratio(h_true, h_2d))
            acc["ls_2dlmmse_corr_abs"].append(_corr_abs(h_true, h_2d))
        result["snr_points"][str(snr)] = {k: float(np.mean(v)) for k, v in acc.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the twc2 LS+2D-LMMSE baseline to isolate whether the failure comes from covariance construction or estimator wiring.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe_batches", type=int, default=8)
    parser.add_argument("--snrs", type=float, nargs="+", default=[0.0, 6.0, 10.0])
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    out_root = Path("outputs") / "twc2_lmmse_probe"
    out_root.mkdir(parents=True, exist_ok=True)

    variants = [
        ("current_twc2", False),
        ("no_trace_norm", False),
        ("tf_order", False),
        ("current_twc2", True),
    ]
    all_results = []
    for variant, static_clean in variants:
        res = _probe_variant(base_cfg, variant=variant, static_clean=static_clean, snrs=list(args.snrs), num_batches=int(args.probe_batches))
        all_results.append(res)

    save_json({"results": all_results}, out_root / "summary.json")

    print("=" * 100)
    print("TWC2 LMMSE sanity probe")
    print("=" * 100)
    for res in all_results:
        print(f"VARIANT={res['variant']} static_clean={res['static_clean']} cache={res['cache_name']}")
        print("  COV-T:", res["covariance"]["time"])
        print("  COV-F:", res["covariance"]["freq"])
        for snr, vals in res["snr_points"].items():
            print(
                f"  SNR={snr:>4} dB | "
                f"LS-lin NMSE={vals['ls_lin_nmse']:.4e} power={vals['ls_lin_power_ratio']:.4f} corr={vals['ls_lin_corr_abs']:.4f} | "
                f"LS-2D NMSE={vals['ls_2dlmmse_nmse']:.4e} power={vals['ls_2dlmmse_power_ratio']:.4f} corr={vals['ls_2dlmmse_corr_abs']:.4f}"
            )
        print("-" * 100)


if __name__ == "__main__":
    main()
