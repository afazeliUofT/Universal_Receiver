from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from .baselines import (
    PERFECT_RECEIVER,
    PROPOSED_RECEIVER,
    BASELINE_PAPER_CFGRES_LS_LMMSE,
    build_classical_baseline_suite,
    classical_receivers_from_cfg,
    enabled_receivers_from_cfg,
    wants_receiver,
)
from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, build_receiver, get_resource_grid
from .compat import safe_call_variants
from .config import ensure_output_tree, get_cfg
from .estimator import UPAIRChannelEstimator
from .impairments import apply_symbol_phase_impairment
from .utils import (
    call_channel,
    call_receiver,
    call_transmitter,
    compute_ber,
    compute_bler_from_crc,
    compute_nmse,
    ebno_db_to_no,
    save_json,
    set_global_seed,
)


def _call_channel_estimator(estimator: Any, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    out = safe_call_variants(estimator, y, no)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError("Channel estimator must return (h_hat, err_var).")
    return tf.convert_to_tensor(out[0]), tf.convert_to_tensor(out[1])


def _make_eval_batch(
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
    batch_size: int,
    ebno_db: float,
) -> dict[str, tf.Tensor]:
    x, bits = call_transmitter(tx, batch_size)
    no = ebno_db_to_no(tf.constant(float(ebno_db), tf.float32), tx=tx, resource_grid=get_resource_grid(tx))
    y, h = call_channel(channel, x, no)
    y, h = apply_symbol_phase_impairment(y, h, cfg, training=False)
    return {"x": x, "b": bits, "y": y, "h": h, "no": no, "ebno_db": tf.constant(float(ebno_db), tf.float32)}


def _metric_min(df: pd.DataFrame, receiver: str, metric: str) -> float | None:
    sub = df[df["receiver"] == receiver][metric].dropna()
    if sub.empty:
        return None
    return float(sub.min())


def _best_classical_row(df: pd.DataFrame, metric: str) -> dict[str, float | str] | None:
    sub = df[["receiver", metric]].dropna()
    if sub.empty:
        return None
    idx = sub[metric].idxmin()
    row = sub.loc[idx]
    return {"receiver": str(row["receiver"]), "value": float(row[metric])}


def _build_summary(
    df: pd.DataFrame,
    checkpoint_path: str | None,
    enabled_receivers: list[str],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    curves_rows = len(df)
    classical_receivers = classical_receivers_from_cfg({"baselines": {"enabled_receivers": enabled_receivers}})
    classical_df = df[df["receiver"].isin(classical_receivers)].copy()

    summary: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "enabled_receivers": enabled_receivers,
        "classical_receivers": classical_receivers,
        "num_curve_rows": int(curves_rows),
    }
    summary.update(artifacts)

    if PROPOSED_RECEIVER in enabled_receivers:
        summary["best_ber_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "ber")
        summary["best_bler_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "bler")
        summary["best_nmse_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "nmse")

    if classical_receivers:
        summary["best_ber_classical"] = _best_classical_row(classical_df, "ber")
        summary["best_bler_classical"] = _best_classical_row(classical_df, "bler")
        summary["best_nmse_classical"] = _best_classical_row(classical_df, "nmse")

    per_ebno_best_classical: list[dict[str, Any]] = []
    if classical_receivers and PROPOSED_RECEIVER in enabled_receivers:
        for ebno_db in sorted(df["ebno_db"].unique().tolist()):
            row_summary: dict[str, Any] = {"ebno_db": float(ebno_db)}
            classical_slice = classical_df[classical_df["ebno_db"] == ebno_db]
            proposed_slice = df[(df["receiver"] == PROPOSED_RECEIVER) & (df["ebno_db"] == ebno_db)]
            if proposed_slice.empty:
                continue
            proposed_row = proposed_slice.iloc[0]
            for metric in ["ber", "bler", "nmse"]:
                best_classical = _best_classical_row(classical_slice, metric)
                if best_classical is None:
                    continue
                row_summary[f"best_{metric}_classical_receiver"] = best_classical["receiver"]
                row_summary[f"best_{metric}_classical"] = best_classical["value"]
                if pd.notna(proposed_row[metric]):
                    upair_value = float(proposed_row[metric])
                    row_summary[f"{metric}_upair5g"] = upair_value
                    row_summary[f"{metric}_gap_upair_minus_best_classical"] = upair_value - float(best_classical["value"])
            per_ebno_best_classical.append(row_summary)
    summary["per_ebno_best_classical"] = per_ebno_best_classical
    return summary


def evaluate_model(cfg: dict[str, Any], checkpoint_path: str | None = None) -> dict[str, Any]:
    set_global_seed(int(cfg["system"]["seed"]))
    paths = ensure_output_tree(cfg)

    tx, _ = build_pusch_transmitter(cfg)
    channel = build_channel(cfg, tx)

    ls_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
    estimator = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=get_resource_grid(tx), cfg=cfg)

    warmup_batch = _make_eval_batch(
        tx=tx,
        channel=channel,
        cfg=cfg,
        batch_size=int(cfg["system"]["batch_size_eval"]),
        ebno_db=float(get_cfg(cfg, "system.ebno_db_eval", [10])[0]),
    )
    estimator.estimate_with_ls(warmup_batch["y"], warmup_batch["no"], training=False)

    if checkpoint_path is not None:
        estimator.load_weights(str(checkpoint_path))

    enabled_receivers = enabled_receivers_from_cfg(cfg)
    classical_receivers, classical_estimators, baseline_artifacts = build_classical_baseline_suite(
        cfg=cfg,
        tx=tx,
        channel=channel,
        paths=paths,
    )

    proposed_rx = None
    if wants_receiver(cfg, PROPOSED_RECEIVER):
        proposed_rx = build_receiver(tx, cfg, channel_estimator=estimator, perfect_csi=False)

    perfect_rx = None
    if wants_receiver(cfg, PERFECT_RECEIVER):
        perfect_rx = build_receiver(tx, cfg, channel_estimator=None, perfect_csi=True)

    rows: list[dict[str, float | str]] = []
    example_saved = False
    ebno_grid = [float(x) for x in get_cfg(cfg, "system.ebno_db_eval", [0, 4, 8, 12])]
    num_batches = int(cfg["evaluation"]["num_batches_per_point"])

    for ebno_db in ebno_grid:
        agg: dict[str, dict[str, list[float]]] = {
            receiver_name: {"ber": [], "bler": [], "nmse": []}
            for receiver_name in enabled_receivers
        }

        for _ in range(num_batches):
            batch = _make_eval_batch(
                tx=tx,
                channel=channel,
                cfg=cfg,
                batch_size=int(cfg["system"]["batch_size_eval"]),
                ebno_db=ebno_db,
            )

            h_hat_prop = None
            h_ls = None
            if proposed_rx is not None:
                h_hat_prop, _, h_ls, _ = estimator.estimate_with_ls(batch["y"], batch["no"], training=False)

            classical_h_hats: dict[str, tf.Tensor] = {}
            for receiver_name, estimator_block in classical_estimators.items():
                h_hat_base, _ = _call_channel_estimator(estimator_block, batch["y"], batch["no"])
                classical_h_hats[receiver_name] = h_hat_base

            for receiver_name, receiver_block in classical_receivers.items():
                b_hat, crc = call_receiver(receiver_block, batch["y"], batch["no"])
                h_hat_base = classical_h_hats[receiver_name]
                if batch["b"] is not None:
                    agg[receiver_name]["ber"].append(float(compute_ber(batch["b"], b_hat).numpy()))
                if crc is not None:
                    agg[receiver_name]["bler"].append(float(compute_bler_from_crc(crc).numpy()))
                agg[receiver_name]["nmse"].append(float(compute_nmse(batch["h"], h_hat_base).numpy()))

            if proposed_rx is not None and h_hat_prop is not None:
                b_hat_prop, crc_prop = call_receiver(proposed_rx, batch["y"], batch["no"])
                if batch["b"] is not None:
                    agg[PROPOSED_RECEIVER]["ber"].append(float(compute_ber(batch["b"], b_hat_prop).numpy()))
                if crc_prop is not None:
                    agg[PROPOSED_RECEIVER]["bler"].append(float(compute_bler_from_crc(crc_prop).numpy()))
                agg[PROPOSED_RECEIVER]["nmse"].append(float(compute_nmse(batch["h"], h_hat_prop).numpy()))

            if perfect_rx is not None:
                b_hat_perf, crc_perf = call_receiver(perfect_rx, batch["y"], batch["no"], h=batch["h"])
                if batch["b"] is not None:
                    agg[PERFECT_RECEIVER]["ber"].append(float(compute_ber(batch["b"], b_hat_perf).numpy()))
                if crc_perf is not None:
                    agg[PERFECT_RECEIVER]["bler"].append(float(compute_bler_from_crc(crc_perf).numpy()))
                agg[PERFECT_RECEIVER]["nmse"].append(0.0)

            if not example_saved and bool(get_cfg(cfg, "evaluation.save_example_batch", True)):
                example_payload: dict[str, Any] = {
                    "h_true": np.asarray(batch["h"].numpy()),
                    "y": np.asarray(batch["y"].numpy()),
                    "ebno_db": np.asarray([ebno_db]),
                }
                if "baseline_ls_lmmse" in classical_h_hats:
                    example_payload["h_ls_linear"] = np.asarray(classical_h_hats["baseline_ls_lmmse"].numpy())
                elif h_ls is not None:
                    example_payload["h_ls_linear"] = np.asarray(h_ls.numpy())
                if "baseline_ls_timeavg_lmmse" in classical_h_hats:
                    example_payload["h_ls_timeavg"] = np.asarray(classical_h_hats["baseline_ls_timeavg_lmmse"].numpy())
                if "baseline_ls_2dlmmse_lmmse" in classical_h_hats:
                    example_payload["h_ls_2dlmmse"] = np.asarray(classical_h_hats["baseline_ls_2dlmmse_lmmse"].numpy())
                if "baseline_ddcpe_ls_lmmse" in classical_h_hats:
                    example_payload["h_ddcpe_ls"] = np.asarray(classical_h_hats["baseline_ddcpe_ls_lmmse"].numpy())
                if BASELINE_PAPER_CFGRES_LS_LMMSE in classical_h_hats:
                    example_payload["h_paper_cfgres"] = np.asarray(classical_h_hats[BASELINE_PAPER_CFGRES_LS_LMMSE].numpy())
                if h_hat_prop is not None:
                    example_payload["h_prop"] = np.asarray(h_hat_prop.numpy())
                np.savez_compressed(paths["artifacts"] / "channel_example.npz", **example_payload)
                example_saved = True

        for receiver_name in enabled_receivers:
            values = agg[receiver_name]
            row = {
                "receiver": receiver_name,
                "ebno_db": ebno_db,
                "ber": float(np.mean(values["ber"])) if values["ber"] else np.nan,
                "bler": float(np.mean(values["bler"])) if values["bler"] else np.nan,
                "nmse": float(np.mean(values["nmse"])) if values["nmse"] else np.nan,
            }
            rows.append(row)
            print(
                f"[EVAL] receiver={receiver_name:>24s} "
                f"Eb/N0={ebno_db:>4.1f} dB "
                f"BER={row['ber']:.5e} "
                f"BLER={row['bler']:.5e} "
                f"NMSE={row['nmse']:.5e}"
            )

    df = pd.DataFrame(rows)
    curves_path = paths["metrics"] / "curves.csv"
    df.to_csv(curves_path, index=False)

    summary = _build_summary(
        df=df,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        enabled_receivers=enabled_receivers,
        artifacts=baseline_artifacts,
    )
    summary["curves_csv"] = str(curves_path)
    save_json(summary, paths["metrics"] / "evaluation_summary.json")

    return {
        "output_dir": str(paths["root"]),
        "curves_path": str(curves_path),
        "summary_path": str(paths["metrics"] / "evaluation_summary.json"),
    }
