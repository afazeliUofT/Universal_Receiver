from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from .builders import (
    build_channel,
    build_ls_estimator,
    build_pusch_transmitter,
    build_receiver,
    get_resource_grid,
)
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


def evaluate_model(cfg: dict[str, Any], checkpoint_path: str | None = None) -> dict[str, Any]:
    set_global_seed(int(cfg["system"]["seed"]))
    paths = ensure_output_tree(cfg)

    tx, _ = build_pusch_transmitter(cfg)
    channel = build_channel(cfg, tx)

    ls_estimator = build_ls_estimator(tx, cfg)
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

    baseline_rx = build_receiver(tx, cfg, channel_estimator=None, perfect_csi=False)
    proposed_rx = build_receiver(tx, cfg, channel_estimator=estimator, perfect_csi=False)
    perfect_rx = build_receiver(tx, cfg, channel_estimator=None, perfect_csi=True)

    rows: list[dict[str, float | str]] = []
    example_saved = False
    ebno_grid = [float(x) for x in get_cfg(cfg, "system.ebno_db_eval", [0, 4, 8, 12])]
    num_batches = int(cfg["evaluation"]["num_batches_per_point"])

    for ebno_db in ebno_grid:
        agg: dict[str, dict[str, list[float]]] = {
            "baseline_ls_lmmse": {"ber": [], "bler": [], "nmse": []},
            "upair5g_lmmse": {"ber": [], "bler": [], "nmse": []},
            "perfect_csi_lmmse": {"ber": [], "bler": [], "nmse": []},
        }

        for batch_idx in range(num_batches):
            batch = _make_eval_batch(
                tx=tx,
                channel=channel,
                cfg=cfg,
                batch_size=int(cfg["system"]["batch_size_eval"]),
                ebno_db=ebno_db,
            )

            h_hat_prop, _, h_ls, _ = estimator.estimate_with_ls(batch["y"], batch["no"], training=False)

            b_hat_base, crc_base = call_receiver(baseline_rx, batch["y"], batch["no"])
            b_hat_prop, crc_prop = call_receiver(proposed_rx, batch["y"], batch["no"])
            b_hat_perf, crc_perf = call_receiver(perfect_rx, batch["y"], batch["no"], h=batch["h"])

            if batch["b"] is not None:
                agg["baseline_ls_lmmse"]["ber"].append(float(compute_ber(batch["b"], b_hat_base).numpy()))
                agg["upair5g_lmmse"]["ber"].append(float(compute_ber(batch["b"], b_hat_prop).numpy()))
                agg["perfect_csi_lmmse"]["ber"].append(float(compute_ber(batch["b"], b_hat_perf).numpy()))

            if crc_base is not None:
                agg["baseline_ls_lmmse"]["bler"].append(float(compute_bler_from_crc(crc_base).numpy()))
            if crc_prop is not None:
                agg["upair5g_lmmse"]["bler"].append(float(compute_bler_from_crc(crc_prop).numpy()))
            if crc_perf is not None:
                agg["perfect_csi_lmmse"]["bler"].append(float(compute_bler_from_crc(crc_perf).numpy()))

            agg["baseline_ls_lmmse"]["nmse"].append(float(compute_nmse(batch["h"], h_ls).numpy()))
            agg["upair5g_lmmse"]["nmse"].append(float(compute_nmse(batch["h"], h_hat_prop).numpy()))
            agg["perfect_csi_lmmse"]["nmse"].append(0.0)

            if not example_saved and bool(get_cfg(cfg, "evaluation.save_example_batch", True)):
                np.savez_compressed(
                    paths["artifacts"] / "channel_example.npz",
                    h_true=np.asarray(batch["h"].numpy()),
                    h_ls=np.asarray(h_ls.numpy()),
                    h_prop=np.asarray(h_hat_prop.numpy()),
                    y=np.asarray(batch["y"].numpy()),
                    ebno_db=np.asarray([ebno_db]),
                )
                example_saved = True

        for receiver_name, values in agg.items():
            row = {
                "receiver": receiver_name,
                "ebno_db": ebno_db,
                "ber": float(np.mean(values["ber"])) if values["ber"] else np.nan,
                "bler": float(np.mean(values["bler"])) if values["bler"] else np.nan,
                "nmse": float(np.mean(values["nmse"])) if values["nmse"] else np.nan,
            }
            rows.append(row)
            print(
                f"[EVAL] receiver={receiver_name:>18s} "
                f"Eb/N0={ebno_db:>4.1f} dB "
                f"BER={row['ber']:.5e} "
                f"BLER={row['bler']:.5e} "
                f"NMSE={row['nmse']:.5e}"
            )

    df = pd.DataFrame(rows)
    curves_path = paths["metrics"] / "curves.csv"
    df.to_csv(curves_path, index=False)

    summary = {
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "curves_csv": str(curves_path),
        "best_bler_baseline": float(df[df["receiver"] == "baseline_ls_lmmse"]["bler"].min()),
        "best_bler_upair5g": float(df[df["receiver"] == "upair5g_lmmse"]["bler"].min()),
        "best_nmse_baseline": float(df[df["receiver"] == "baseline_ls_lmmse"]["nmse"].min()),
        "best_nmse_upair5g": float(df[df["receiver"] == "upair5g_lmmse"]["nmse"].min()),
    }
    save_json(summary, paths["metrics"] / "evaluation_summary.json")

    return {
        "output_dir": str(paths["root"]),
        "curves_path": str(curves_path),
        "summary_path": str(paths["metrics"] / "evaluation_summary.json"),
    }
