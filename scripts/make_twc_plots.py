from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from upair5g.utils import tensor7_to_btfnc

ROOT = Path("/home/rsadve1/PROJECT/Universal_Receiver")
FIG_DIR = ROOT / "TWC_plots"
TABLE_DIR = FIG_DIR / "tables"

MAIN_CURVES = ROOT / "outputs" / "twc_mild_main" / "metrics" / "curves.csv"
MAIN_EXAMPLE = ROOT / "outputs" / "twc_mild_main" / "artifacts" / "channel_example.npz"
CLEAN_CURVES = ROOT / "outputs" / "twc_mild_clean" / "metrics" / "curves.csv"
DMRSRICH_CURVES = ROOT / "outputs" / "twc_mild_dmrsrich" / "metrics" / "curves.csv"

SCENARIOS = {
    "Mild main": MAIN_CURVES,
    "Mild clean": CLEAN_CURVES,
    "Mild DMRS-rich": DMRSRICH_CURVES,
}

LABELS = {
    "baseline_ls_lmmse": "LS+LMMSE",
    "baseline_ddcpe_ls_lmmse": "DD-CPE+LS+LMMSE",
    "paper_cfgres_phaseaware_ls_lmmse": "Phase-aware paper cfg reservoir+LMMSE",
    "upair5g_lmmse": "UPAIR-5G+LMMSE",
    "perfect_csi_lmmse": "Perfect CSI+LMMSE",
}

MAIN_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
    "upair5g_lmmse",
]

FOCUS_RECEIVERS = [
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]

CLASSICAL_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
]

METRICS = ["ber", "bler", "nmse"]
LINESTYLES = {
    "baseline_ls_lmmse": "--",
    "baseline_ddcpe_ls_lmmse": "-.",
    "paper_cfgres_phaseaware_ls_lmmse": ":",
    "upair5g_lmmse": "-",
    "perfect_csi_lmmse": "-",
}
MARKERS = {
    "baseline_ls_lmmse": "s",
    "baseline_ddcpe_ls_lmmse": "D",
    "paper_cfgres_phaseaware_ls_lmmse": "^",
    "upair5g_lmmse": "o",
    "perfect_csi_lmmse": "v",
}
LINEWIDTHS = {
    "baseline_ls_lmmse": 1.7,
    "baseline_ddcpe_ls_lmmse": 1.8,
    "paper_cfgres_phaseaware_ls_lmmse": 1.9,
    "upair5g_lmmse": 2.4,
    "perfect_csi_lmmse": 1.8,
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 13,
    }
)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing = []
    for name, path in SCENARIOS.items():
        if not path.exists():
            missing.append(str(path))
        else:
            frames[name] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError("Missing curves.csv files:\n" + "\n".join(missing))
    return frames


def _positive_floor(frames: dict[str, pd.DataFrame], metric: str, receivers: list[str]) -> float:
    positives: list[float] = []
    for df in frames.values():
        sub = df[df["receiver"].isin(receivers)][metric].to_numpy(dtype=float)
        positives.extend(sub[sub > 0.0].tolist())
    if not positives:
        return 1e-6 if metric in {"ber", "bler"} else 1e-5
    minimum = min(positives)
    lower_bound = 1e-7 if metric in {"ber", "bler"} else 1e-5
    return max(minimum / 5.0, lower_bound)


def _plot_receiver_curve(
    ax: plt.Axes,
    sub: pd.DataFrame,
    receiver: str,
    metric: str,
    floor: float,
) -> None:
    x = sub["ebno_db"].to_numpy(dtype=float)
    raw = sub[metric].to_numpy(dtype=float)
    y = np.where(raw > 0.0, raw, floor)
    line = ax.semilogy(
        x,
        y,
        linestyle=LINESTYLES[receiver],
        marker=MARKERS[receiver],
        linewidth=LINEWIDTHS[receiver],
        markersize=5.5 if receiver == "upair5g_lmmse" else 4.8,
        label=LABELS[receiver],
    )[0]
    zero_mask = raw <= 0.0
    if np.any(zero_mask):
        ax.plot(
            x[zero_mask],
            y[zero_mask],
            linestyle="None",
            marker="v",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            color=line.get_color(),
        )


def _best_classical(df: pd.DataFrame, ebno_db: float, metric: str) -> tuple[str, float]:
    classical = df[(df["receiver"].isin(CLASSICAL_RECEIVERS)) & (df["ebno_db"] == ebno_db)].copy()
    classical = classical.dropna(subset=[metric]).sort_values(metric)
    best = classical.iloc[0]
    return str(best["receiver"]), float(best[metric])


def _gain_ratio(best_value: float, upair_value: float) -> float:
    if best_value == 0.0 and upair_value == 0.0:
        return 1.0
    if upair_value == 0.0 and best_value > 0.0:
        return np.inf
    return best_value / max(upair_value, 1e-15)


def make_fig01_main_curves(frames: dict[str, pd.DataFrame]) -> None:
    df = frames["Mild main"]
    floor_map = {metric: _positive_floor(frames, metric, MAIN_RECEIVERS) for metric in METRICS}
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for receiver in MAIN_RECEIVERS:
            sub = df[df["receiver"] == receiver].sort_values("ebno_db")
            if sub.empty:
                continue
            _plot_receiver_curve(ax, sub, receiver, metric, floor_map[metric])
        ax.set_ylabel(metric.upper())
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("Mild main scenario")
    _save(fig, FIG_DIR / "Fig01_mild_main_curves")


def make_fig02_main_focus(frames: dict[str, pd.DataFrame]) -> None:
    df = frames["Mild main"]
    floor_map = {metric: _positive_floor(frames, metric, FOCUS_RECEIVERS) for metric in METRICS}
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for receiver in FOCUS_RECEIVERS:
            sub = df[df["receiver"] == receiver].sort_values("ebno_db")
            if sub.empty:
                continue
            _plot_receiver_curve(ax, sub, receiver, metric, floor_map[metric])
        ax.set_ylabel(metric.upper())
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    axes[0].legend(loc="best", frameon=False)
    axes[0].text(
        0.98,
        0.05,
        "Hollow downward marker = 0 observed error\nplotted at a positive visual floor.",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.suptitle("Mild main scenario, strongest comparators and upper bound")
    _save(fig, FIG_DIR / "Fig02_mild_main_focus")


def make_fig03_main_gain(frames: dict[str, pd.DataFrame]) -> None:
    df = frames["Mild main"]
    upair = df[df["receiver"] == "upair5g_lmmse"].sort_values("ebno_db")
    rows = []
    ratio_caps: dict[str, float] = {}
    for metric in METRICS:
        ratios = []
        for ebno_db in upair["ebno_db"].tolist():
            _, best_value = _best_classical(df, float(ebno_db), metric)
            upair_value = float(upair[upair["ebno_db"] == ebno_db][metric].iloc[0])
            ratios.append(_gain_ratio(best_value, upair_value))
        finite = np.asarray([r for r in ratios if np.isfinite(r)], dtype=float)
        ratio_caps[metric] = max(8.0, float(np.max(finite)) * 1.5) if finite.size else 8.0

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        xs = []
        ys = []
        inf_x = []
        cap = ratio_caps[metric]
        for ebno_db in upair["ebno_db"].tolist():
            best_name, best_value = _best_classical(df, float(ebno_db), metric)
            upair_value = float(upair[upair["ebno_db"] == ebno_db][metric].iloc[0])
            ratio = _gain_ratio(best_value, upair_value)
            rows.append(
                {
                    "ebno_db": float(ebno_db),
                    "metric": metric,
                    "best_classical_receiver": best_name,
                    "best_classical_value": best_value,
                    "upair_value": upair_value,
                    "best_classical_over_upair": ratio,
                }
            )
            xs.append(float(ebno_db))
            ys.append(cap if np.isinf(ratio) else float(ratio))
            if np.isinf(ratio):
                inf_x.append(float(ebno_db))
        ax.semilogy(xs, ys, marker="o", linewidth=2.2)
        if inf_x:
            ax.plot(inf_x, [cap] * len(inf_x), linestyle="None", marker="^", markersize=7)
        ax.axhline(1.0, linestyle=":", linewidth=1.0)
        ax.set_ylabel("Best classical / UPAIR")
        ax.set_title(metric.upper())
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Eb/N0 [dB]")
    _save(fig, FIG_DIR / "Fig03_mild_main_upair_gain")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "mild_main_upair_gain.csv", index=False)


def _to_magnitude_panel(x: np.ndarray) -> np.ndarray:
    return np.abs(tensor7_to_btfnc(tf.convert_to_tensor(x)).numpy()[0, :, :, 0])


def _to_error_panel(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    ref = tensor7_to_btfnc(tf.convert_to_tensor(reference)).numpy()[0, :, :, 0]
    est = tensor7_to_btfnc(tf.convert_to_tensor(estimate)).numpy()[0, :, :, 0]
    return np.abs(ref - est)


def make_fig04_channel_maps() -> None:
    if not MAIN_EXAMPLE.exists():
        raise FileNotFoundError(str(MAIN_EXAMPLE))
    data = np.load(MAIN_EXAMPLE)
    keys = [
        ("h_ls_linear", "LS+LMMSE"),
        ("h_ddcpe_ls", "DD-CPE+LS+LMMSE"),
        ("h_paper_cfgres_phaseaware", "Phase-aware paper cfg reservoir+LMMSE"),
        ("h_prop", "UPAIR-5G+LMMSE"),
    ]
    keys = [(k, t) for k, t in keys if k in data]
    if not keys:
        return

    mags = [_to_magnitude_panel(data["h_true"]) ]
    errs = [np.zeros_like(mags[0])]
    titles_top = ["True |H|"]
    titles_bottom = ["Absolute error"]
    for key, title in keys:
        mags.append(_to_magnitude_panel(data[key]))
        errs.append(_to_error_panel(data["h_true"], data[key]))
        titles_top.append(title)
        titles_bottom.append(title)

    vmax_mag = max(float(np.max(panel)) for panel in mags)
    vmax_err = max(float(np.max(panel)) for panel in errs[1:]) if len(errs) > 1 else 1.0
    fig, axes = plt.subplots(2, len(mags), figsize=(3.4 * len(mags), 6.0))
    for idx in range(len(mags)):
        im0 = axes[0, idx].imshow(mags[idx], aspect="auto", vmin=0.0, vmax=max(vmax_mag, 1e-6))
        axes[0, idx].set_title(titles_top[idx])
        axes[0, idx].set_xlabel("Subcarrier")
        axes[0, idx].set_ylabel("OFDM symbol")
        im1 = axes[1, idx].imshow(errs[idx], aspect="auto", vmin=0.0, vmax=max(vmax_err, 1e-6))
        axes[1, idx].set_title(titles_bottom[idx])
        axes[1, idx].set_xlabel("Subcarrier")
        axes[1, idx].set_ylabel("OFDM symbol")
    fig.colorbar(im0, ax=axes[0, :].ravel().tolist(), shrink=0.82)
    fig.colorbar(im1, ax=axes[1, :].ravel().tolist(), shrink=0.82)
    _save(fig, FIG_DIR / "Fig04_mild_main_channel_error_maps")


def make_fig05_crossscenario_grid(frames: dict[str, pd.DataFrame]) -> None:
    floor_map = {metric: _positive_floor(frames, metric, MAIN_RECEIVERS) for metric in METRICS}
    scenarios = list(SCENARIOS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15.4, 11.0), sharex="col")
    for col, scenario in enumerate(scenarios):
        df = frames[scenario]
        axes[0, col].set_title(scenario)
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for receiver in MAIN_RECEIVERS:
                sub = df[df["receiver"] == receiver].sort_values("ebno_db")
                if sub.empty:
                    continue
                _plot_receiver_curve(ax, sub, receiver, metric, floor_map[metric])
            if col == 0:
                ax.set_ylabel(metric.upper())
            if row == 2:
                ax.set_xlabel("Eb/N0 [dB]")
            ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    _save(fig, FIG_DIR / "Fig05_mild_crossscenario_metrics_grid")


def make_fig06_crossscenario_gain(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    scenarios = list(SCENARIOS.keys())
    for scenario, df in frames.items():
        upair = df[df["receiver"] == "upair5g_lmmse"].sort_values("ebno_db")
        for ebno_db in upair["ebno_db"].tolist():
            row = {"scenario": scenario, "ebno_db": float(ebno_db)}
            for metric in METRICS:
                best_name, best_value = _best_classical(df, float(ebno_db), metric)
                upair_value = float(upair[upair["ebno_db"] == ebno_db][metric].iloc[0])
                row[f"best_{metric}_classical_receiver"] = best_name
                row[f"best_{metric}_classical"] = best_value
                row[f"{metric}_upair"] = upair_value
                row[f"{metric}_ratio_best_classical_over_upair"] = _gain_ratio(best_value, upair_value)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "mild_crossscenario_gain.csv", index=False)

    ratio_caps = {}
    for metric in METRICS:
        finite = summary[f"{metric}_ratio_best_classical_over_upair"].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        ratio_caps[metric] = max(8.0, float(np.max(finite)) * 1.5) if finite.size else 8.0

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharex=True)
    for ax, metric in zip(axes, METRICS):
        cap = ratio_caps[metric]
        for scenario in scenarios:
            sub = summary[summary["scenario"] == scenario].copy().sort_values("ebno_db")
            x = sub["ebno_db"].to_numpy(dtype=float)
            raw = sub[f"{metric}_ratio_best_classical_over_upair"].to_numpy(dtype=float)
            y = np.where(np.isfinite(raw), raw, cap)
            ax.semilogy(x, y, marker="o", linewidth=2.0, label=scenario)
            if np.isinf(raw).any():
                inf_x = x[np.isinf(raw)]
                ax.plot(inf_x, np.full_like(inf_x, cap, dtype=float), linestyle="None", marker="^", markersize=7)
        ax.axhline(1.0, linestyle=":", linewidth=1.0)
        ax.set_title(metric.upper())
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel("Best classical / UPAIR")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(frameon=False)
    _save(fig, FIG_DIR / "Fig06_mild_crossscenario_upair_gain")


def write_manifest() -> None:
    lines = [
        "Fig01_mild_main_curves.pdf/png",
        "Fig02_mild_main_focus.pdf/png",
        "Fig03_mild_main_upair_gain.pdf/png",
        "Fig04_mild_main_channel_error_maps.pdf/png",
        "Fig05_mild_crossscenario_metrics_grid.pdf/png",
        "Fig06_mild_crossscenario_upair_gain.pdf/png",
        "tables/mild_main_upair_gain.csv",
        "tables/mild_crossscenario_gain.csv",
    ]
    (FIG_DIR / "TWC_plot_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frames = _load_frames()
    make_fig01_main_curves(frames)
    make_fig02_main_focus(frames)
    make_fig03_main_gain(frames)
    make_fig04_channel_maps()
    make_fig05_crossscenario_grid(frames)
    make_fig06_crossscenario_gain(frames)
    write_manifest()
    print({"twc_plots_dir": str(FIG_DIR)})


if __name__ == "__main__":
    main()
