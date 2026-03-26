from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "TWC_plots2"

SCENARIOS = {
    "mild_main": ROOT / "outputs" / "twc2_mild_main" / "metrics" / "curves.csv",
    "mild_clean": ROOT / "outputs" / "twc2_mild_clean" / "metrics" / "curves.csv",
    "mild_dmrsrich": ROOT / "outputs" / "twc2_mild_dmrsrich" / "metrics" / "curves.csv",
}

CHANNEL_EXAMPLE = ROOT / "outputs" / "twc2_mild_main" / "artifacts" / "channel_example.npz"

RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]

CLASSICAL = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
]

LABELS = {
    "baseline_ls_lmmse": "LS+linear+LMMSE",
    "baseline_ls_2dlmmse_lmmse": "LS+2D-LMMSE+LMMSE",
    "baseline_ddcpe_ls_lmmse": "DD-CPE+LS+LMMSE",
    "upair5g_lmmse": "UPAIR-5G+LMMSE",
    "perfect_csi_lmmse": "Perfect CSI+LMMSE",
}

ORDER = {name: idx for idx, name in enumerate(RECEIVERS)}


def _style(receiver: str) -> dict:
    if receiver == "upair5g_lmmse":
        return {"linewidth": 2.8, "marker": "o"}
    if receiver == "perfect_csi_lmmse":
        return {"linewidth": 2.1, "marker": "^"}
    if receiver == "baseline_ls_2dlmmse_lmmse":
        return {"linewidth": 1.9, "marker": "s"}
    if receiver == "baseline_ddcpe_ls_lmmse":
        return {"linewidth": 2.0, "marker": "D"}
    return {"linewidth": 1.7, "marker": "o"}


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "reliable_ber" not in df.columns:
        df["reliable_ber"] = True
    if "reliable_bler" not in df.columns:
        df["reliable_bler"] = True
    if "bit_errors" not in df.columns:
        df["bit_errors"] = np.nan
    if "block_errors" not in df.columns:
        df["block_errors"] = np.nan
    return df


def _metric_series(df: pd.DataFrame, receiver: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["receiver"] == receiver].sort_values("ebno_db").copy()
    if sub.empty:
        return np.asarray([]), np.asarray([])
    x = sub["ebno_db"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)
    if metric == "ber":
        keep = sub["reliable_ber"].astype(bool).to_numpy() & (sub["bit_errors"].to_numpy(dtype=float) > 0)
        y = np.where(keep, y, np.nan)
    elif metric == "bler":
        keep = sub["reliable_bler"].astype(bool).to_numpy() & (sub["block_errors"].to_numpy(dtype=float) > 0)
        y = np.where(keep, y, np.nan)
    elif metric == "nmse":
        y = np.where(y > 0, y, np.nan)
    return x, y


def _plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, receivers: Iterable[str]) -> None:
    for receiver in sorted(receivers, key=lambda x: ORDER.get(x, 999)):
        x, y = _metric_series(df, receiver, metric)
        if x.size == 0 or np.all(np.isnan(y)):
            continue
        ax.plot(x, y, label=LABELS.get(receiver, receiver), **_style(receiver))
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel(metric.upper() if metric != "nmse" else "NMSE")
    ax.grid(True, which="both", alpha=0.30)
    ax.set_yscale("log")


def _best_classical(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for ebno in sorted(df["ebno_db"].unique().tolist()):
        sub = df[(df["ebno_db"] == ebno) & (df["receiver"].isin(CLASSICAL))].copy()
        if metric == "ber":
            sub = sub[sub["reliable_ber"].astype(bool) & (sub["bit_errors"] > 0)]
        elif metric == "bler":
            sub = sub[sub["reliable_bler"].astype(bool) & (sub["block_errors"] > 0)]
        else:
            sub = sub[sub[metric] > 0]
        if sub.empty:
            rows.append({"ebno_db": float(ebno), metric: np.nan, "receiver": None})
        else:
            row = sub.sort_values(metric, ascending=True).iloc[0]
            rows.append({"ebno_db": float(ebno), metric: float(row[metric]), "receiver": row["receiver"]})
    return pd.DataFrame(rows)


def _proposed(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df["receiver"] == "upair5g_lmmse"].sort_values("ebno_db").copy()
    if metric == "ber":
        sub.loc[~sub["reliable_ber"].astype(bool) | (sub["bit_errors"] <= 0), metric] = np.nan
    elif metric == "bler":
        sub.loc[~sub["reliable_bler"].astype(bool) | (sub["block_errors"] <= 0), metric] = np.nan
    else:
        sub.loc[sub[metric] <= 0, metric] = np.nan
    return sub[["ebno_db", metric]]


def _gain_df(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    merged = pd.DataFrame({"ebno_db": sorted(df["ebno_db"].unique().tolist())})
    for metric in ["ber", "bler", "nmse"]:
        best = _best_classical(df, metric)
        prop = _proposed(df, metric)
        tmp = merged.merge(best, on="ebno_db", how="left", suffixes=("", "_best"))
        tmp = tmp.merge(prop, on="ebno_db", how="left", suffixes=("_best", "_upair"))
        num = tmp[f"{metric}_best"].to_numpy(dtype=float)
        den = tmp[metric].to_numpy(dtype=float)
        ratio = np.where((num > 0) & (den > 0), num / den, np.nan)
        merged[f"{metric}_gain_over_best_classical"] = ratio
    merged["scenario"] = scenario
    return merged


def _load_channel_example(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {k: payload[k] for k in payload.files}


def _tensor7_to_tfr(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 7:
        arr = np.squeeze(arr, axis=(1, 3, 4))
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected channel-example rank: {arr.ndim}")
    return arr[0, :, :, 0]


def _plot_channel_maps(example: dict[str, np.ndarray]) -> None:
    h_true = _tensor7_to_tfr(example["h_true"])
    candidates = {
        "LS+linear": example.get("h_ls_linear"),
        "LS+2D-LMMSE": example.get("h_ls_2dlmmse"),
        "DD-CPE+LS": example.get("h_ddcpe_ls"),
        "UPAIR-5G": example.get("h_prop"),
    }
    maps: list[tuple[str, np.ndarray]] = []
    for name, arr in candidates.items():
        if arr is None:
            continue
        h_hat = _tensor7_to_tfr(arr)
        err = np.abs(h_hat - h_true) ** 2 / (np.abs(h_true) ** 2 + 1.0e-9)
        maps.append((name, err))
    if not maps:
        return
    vmax = float(np.nanpercentile(np.concatenate([m.ravel() for _, m in maps]), 95))
    vmax = max(vmax, 1.0e-3)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), constrained_layout=True)
    for ax, (name, err) in zip(axes.flat, maps):
        im = ax.imshow(err, aspect="auto", origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Subcarrier index")
        ax.set_ylabel("OFDM symbol index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.flat[len(maps):]:
        ax.axis("off")
    _save(fig, "Fig04_twc2_main_channel_error_maps")


def _main_curves(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.5), constrained_layout=True)
    _plot_metric(axes[0], df, "ber", RECEIVERS)
    _plot_metric(axes[1], df, "bler", RECEIVERS)
    _plot_metric(axes[2], df, "nmse", RECEIVERS)
    axes[0].legend(loc="best", fontsize=8)
    _save(fig, "Fig01_twc2_main_curves")


def _main_focus(df: pd.DataFrame) -> None:
    focus = [
        "baseline_ls_lmmse",
        "baseline_ls_2dlmmse_lmmse",
        "baseline_ddcpe_ls_lmmse",
        "upair5g_lmmse",
        "perfect_csi_lmmse",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.5), constrained_layout=True)
    _plot_metric(axes[0], df, "ber", focus)
    _plot_metric(axes[1], df, "bler", focus)
    _plot_metric(axes[2], df, "nmse", focus)
    axes[0].legend(loc="best", fontsize=8)
    _save(fig, "Fig02_twc2_main_focus")


def _main_gain(df: pd.DataFrame) -> None:
    gain = _gain_df(df, "mild_main")
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.2), constrained_layout=True)
    for ax, metric in zip(axes, ["ber", "bler", "nmse"]):
        y = gain[f"{metric}_gain_over_best_classical"].to_numpy(dtype=float)
        x = gain["ebno_db"].to_numpy(dtype=float)
        y = np.where(y > 0, y, np.nan)
        if not np.all(np.isnan(y)):
            ax.plot(x, y, linewidth=2.5, marker="o")
        ax.set_xlabel(r"$E_b/N_0$ (dB)")
        ax.set_ylabel(f"Best classical / UPAIR ({metric.upper() if metric != 'nmse' else 'NMSE'})")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.30)
    _save(fig, "Fig03_twc2_main_upair_gain")
    gain.to_csv(OUT_DIR / "twc2_main_gain_table.csv", index=False)


def _crossscenario_grid(dfs: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15.6, 12.0), constrained_layout=True)
    metrics = ["ber", "bler", "nmse"]
    titles = {
        "mild_main": "Mild main scenario",
        "mild_clean": "Mild clean control",
        "mild_dmrsrich": "Mild DMRS-rich control",
    }
    for col, (scenario, df) in enumerate(dfs.items()):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            _plot_metric(ax, df, metric, RECEIVERS)
            if row == 0:
                ax.set_title(titles[scenario])
            if col == 0:
                ax.set_ylabel(metric.upper() if metric != "nmse" else "NMSE")
            else:
                ax.set_ylabel("")
    axes[0, 0].legend(loc="best", fontsize=8)
    _save(fig, "Fig05_twc2_crossscenario_metrics_grid")


def _crossscenario_gain(dfs: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.2), constrained_layout=True)
    gains = []
    for scenario, df in dfs.items():
        gains.append(_gain_df(df, scenario))
    gain_df = pd.concat(gains, ignore_index=True)
    labels = {
        "mild_main": "Mild main",
        "mild_clean": "Mild clean",
        "mild_dmrsrich": "Mild DMRS-rich",
    }
    for ax, metric in zip(axes, ["ber", "bler", "nmse"]):
        for scenario in ["mild_main", "mild_clean", "mild_dmrsrich"]:
            sub = gain_df[gain_df["scenario"] == scenario].copy()
            y = sub[f"{metric}_gain_over_best_classical"].to_numpy(dtype=float)
            x = sub["ebno_db"].to_numpy(dtype=float)
            y = np.where(y > 0, y, np.nan)
            if np.all(np.isnan(y)):
                continue
            ax.plot(x, y, linewidth=2.2, marker="o", label=labels[scenario])
        ax.set_xlabel(r"$E_b/N_0$ (dB)")
        ax.set_ylabel(f"Best classical / UPAIR ({metric.upper() if metric != 'nmse' else 'NMSE'})")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.30)
    axes[0].legend(loc="best", fontsize=8)
    _save(fig, "Fig06_twc2_crossscenario_upair_gain")
    gain_df.to_csv(OUT_DIR / "twc2_crossscenario_gain_table.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = {name: _load_df(path) for name, path in SCENARIOS.items()}
    _main_curves(dfs["mild_main"])
    _main_focus(dfs["mild_main"])
    _main_gain(dfs["mild_main"])
    if CHANNEL_EXAMPLE.exists():
        _plot_channel_maps(_load_channel_example(CHANNEL_EXAMPLE))
    _crossscenario_grid(dfs)
    _crossscenario_gain(dfs)
    manifest = [
        "Fig01_twc2_main_curves.pdf/png",
        "Fig02_twc2_main_focus.pdf/png",
        "Fig03_twc2_main_upair_gain.pdf/png",
        "Fig04_twc2_main_channel_error_maps.pdf/png",
        "Fig05_twc2_crossscenario_metrics_grid.pdf/png",
        "Fig06_twc2_crossscenario_upair_gain.pdf/png",
        "twc2_main_gain_table.csv",
        "twc2_crossscenario_gain_table.csv",
    ]
    (OUT_DIR / "TWC_plots2_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
