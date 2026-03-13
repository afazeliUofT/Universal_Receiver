from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ensure_output_tree
from .utils import tensor7_to_btfnc


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


def _plot_curve(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plt.figure()
    for receiver in sorted(df["receiver"].unique()):
        sub = df[df["receiver"] == receiver].sort_values("ebno_db")
        y = np.maximum(sub[metric].to_numpy(dtype=float), 1e-12)
        plt.semilogy(sub["ebno_db"], y, marker="o", label=receiver)
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


def plot_channel_example(example_npz: str | Path, out_dir: str | Path) -> None:
    example_npz = Path(example_npz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not example_npz.exists():
        return

    data = np.load(example_npz)
    h_true = tensor7_to_btfnc(data["h_true"]).numpy()
    h_ls = tensor7_to_btfnc(data["h_ls"]).numpy()
    h_prop = tensor7_to_btfnc(data["h_prop"]).numpy()

    # Plot magnitude for the first batch sample and first RX antenna
    idx_ant = 0
    true_mag = np.abs(h_true[0, :, :, idx_ant])
    ls_mag = np.abs(h_ls[0, :, :, idx_ant])
    prop_mag = np.abs(h_prop[0, :, :, idx_ant])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(true_mag, aspect="auto")
    axes[0].set_title("True |H|")
    axes[1].imshow(ls_mag, aspect="auto")
    axes[1].set_title("LS |H|")
    axes[2].imshow(prop_mag, aspect="auto")
    axes[2].set_title("UPAIR-5G |H|")
    for ax in axes:
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
