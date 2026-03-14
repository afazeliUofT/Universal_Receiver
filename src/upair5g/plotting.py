from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ensure_output_tree
from .utils import tensor7_to_btfnc

PLOT_LABELS = {
    "baseline_ls_lmmse": "LS+linear+LMMSE",
    "baseline_ls_timeavg_lmmse": "LS+lin_time_avg+LMMSE",
    "baseline_ls_2dlmmse_lmmse": "LS+2D-LMMSE+LMMSE",
    "upair5g_lmmse": "UPAIR-5G+LMMSE",
    "perfect_csi_lmmse": "Perfect CSI+LMMSE",
}

PLOT_ORDER = [
    "baseline_ls_lmmse",
    "baseline_ls_timeavg_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]


def _label(receiver: str) -> str:
    return PLOT_LABELS.get(receiver, receiver)


def plot_training_history(history_path: str | Path, out_dir: str | Path) -> None:
    history_path = Path(history_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        return

    with open(history_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("history", [])
    if not rows:
        return

    df = pd.DataFrame(rows)

    plt.figure()
    plt.plot(df["step"], df["loss"], label="train loss")
    if "val_nmse_prop" in df.columns:
        valid = df["val_nmse_prop"].notna()
        if valid.any():
            plt.plot(df.loc[valid, "step"], df.loc[valid, "val_nmse_prop"], label="val NMSE (proposed)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["step"], df["nmse_ls"], label="LS NMSE")
    plt.plot(df["step"], df["nmse_prop"], label="UPAIR-5G NMSE")
    plt.xlabel("Step")
    plt.ylabel("NMSE")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_nmse.png", dpi=200)
    plt.close()


def _receiver_sort_key(receiver: str) -> tuple[int, str]:
    try:
        return (PLOT_ORDER.index(receiver), receiver)
    except ValueError:
        return (len(PLOT_ORDER), receiver)


def _plot_curve(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plt.figure()
    for receiver in sorted(df["receiver"].unique(), key=_receiver_sort_key):
        sub = df[df["receiver"] == receiver].sort_values("ebno_db")
        y = np.maximum(sub[metric].to_numpy(dtype=float), 1e-12)
        plt.semilogy(sub["ebno_db"], y, marker="o", label=_label(receiver))
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel(metric.upper())
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_curves(curves_path: str | Path, out_dir: str | Path) -> None:
    curves_path = Path(curves_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not curves_path.exists():
        return

    df = pd.read_csv(curves_path)
    for metric in ["ber", "bler", "nmse"]:
        if metric in df.columns and df[metric].notna().any():
            _plot_curve(df, metric, out_dir / f"{metric}_vs_ebno.png")


def _to_magnitude_panel(tensor_like: np.ndarray) -> np.ndarray:
    h_btfnc = tensor7_to_btfnc(tensor_like).numpy()
    return np.abs(h_btfnc[0, :, :, 0])


def plot_channel_example(example_npz: str | Path, out_dir: str | Path) -> None:
    example_npz = Path(example_npz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not example_npz.exists():
        return

    data = np.load(example_npz)
    panels: list[tuple[str, np.ndarray]] = [("True |H|", _to_magnitude_panel(data["h_true"]))]

    optional_keys = [
        ("h_ls_linear", "LS+linear |H|"),
        ("h_ls_timeavg", "LS+time-avg |H|"),
        ("h_ls_2dlmmse", "LS+2D-LMMSE |H|"),
        ("h_prop", "UPAIR-5G |H|"),
    ]
    for key, title in optional_keys:
        if key in data:
            panels.append((title, _to_magnitude_panel(data[key])))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, mag) in zip(axes, panels):
        ax.imshow(mag, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Subcarrier")
        ax.set_ylabel("OFDM symbol")

    plt.tight_layout()
    plt.savefig(out_dir / "channel_refinement_example.png", dpi=200)
    plt.close(fig)


def make_all_plots(cfg: dict) -> dict[str, str]:
    paths = ensure_output_tree(cfg)
    plot_training_history(paths["metrics"] / "history.json", paths["plots"])
    plot_curves(paths["metrics"] / "curves.csv", paths["plots"])
    plot_channel_example(paths["artifacts"] / "channel_example.npz", paths["plots"])
    return {"plots_dir": str(paths["plots"])}
