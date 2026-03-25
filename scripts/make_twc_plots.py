from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
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
    "baseline_ls_2dlmmse_lmmse": "LS+empirical 2D-LMMSE+LMMSE",
    "baseline_ddcpe_ls_lmmse": "DD-CPE+LS+LMMSE",
    "paper_cfgres_phaseaware_ls_lmmse": "Phase-aware paper cfg reservoir+LMMSE",
    "upair5g_lmmse": "UPAIR-5G+LMMSE",
    "perfect_csi_lmmse": "Perfect CSI+LMMSE",
}

MAIN_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
    "upair5g_lmmse",
]

FOCUS_RECEIVERS = [
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]

CLASSICAL_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
]

METRICS = ["ber", "bler", "nmse"]
LINESTYLES = {
    "baseline_ls_lmmse": "--",
    "baseline_ls_2dlmmse_lmmse": (0, (5, 1.5)),
    "baseline_ddcpe_ls_lmmse": "-.",
    "paper_cfgres_phaseaware_ls_lmmse": ":",
    "upair5g_lmmse": "-",
    "perfect_csi_lmmse": "-",
}
MARKERS = {
    "baseline_ls_lmmse": "s",
    "baseline_ls_2dlmmse_lmmse": "P",
    "baseline_ddcpe_ls_lmmse": "D",
    "paper_cfgres_phaseaware_ls_lmmse": "^",
    "upair5g_lmmse": "o",
    "perfect_csi_lmmse": "v",
}
LINEWIDTHS = {
    "baseline_ls_lmmse": 1.6,
    "baseline_ls_2dlmmse_lmmse": 1.7,
    "baseline_ddcpe_ls_lmmse": 1.8,
    "paper_cfgres_phaseaware_ls_lmmse": 1.9,
    "upair5g_lmmse": 2.4,
    "perfect_csi_lmmse": 2.0,
}


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    missing = []
    for name, path in SCENARIOS.items():
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        if "reliable_ber" not in df.columns:
            df["reliable_ber"] = df["ber"] > 0
        if "reliable_bler" not in df.columns:
            df["reliable_bler"] = df["bler"] > 0
        frames[name] = df
    if missing:
        raise FileNotFoundError("Missing curves.csv files:\n" + "\n".join(missing))
    return frames


def _positive_floor(frames: dict[str, pd.DataFrame], metric: str, receivers: list[str]) -> float:
    positives = []
    rel_col = {"ber": "reliable_ber", "bler": "reliable_bler"}.get(metric)
    for df in frames.values():
        for rx in receivers:
            s = df[df["receiver"] == rx]
            if s.empty:
                continue
            vals = s[metric].astype(float).to_numpy()
            if rel_col is not None:
                rel = s[rel_col].astype(bool).to_numpy()
                vals = vals[(vals > 0) & rel]
            else:
                vals = vals[vals > 0]
            positives.extend(vals.tolist())
    if not positives:
        return 1e-8 if metric in {"ber", "bler"} else 1e-4
    return max(min(positives) * 0.5, 1e-10 if metric in {"ber", "bler"} else 1e-6)


def _plot_metric(ax, df: pd.DataFrame, metric: str, receivers: list[str], floor: float) -> None:
    rel_col = {"ber": "reliable_ber", "bler": "reliable_bler"}.get(metric)
    for rx in receivers:
        s = df[df["receiver"] == rx].sort_values("ebno_db")
        if s.empty:
            continue
        x = s["ebno_db"].astype(float).to_numpy()
        y = s[metric].astype(float).to_numpy()
        if rel_col is not None:
            rel = s[rel_col].astype(bool).to_numpy()
            plot_mask = rel & np.isfinite(y) & (y > 0)
        else:
            plot_mask = np.isfinite(y)
        if np.any(plot_mask):
            ax.plot(
                x[plot_mask], y[plot_mask],
                label=LABELS[rx],
                linestyle=LINESTYLES[rx],
                marker=MARKERS[rx],
                linewidth=LINEWIDTHS[rx],
                markersize=4.2,
            )
        # mark zero-observed / unreliable points softly at the floor so the reader sees truncation
        if rel_col is not None:
            zero_mask = np.isfinite(y) & ~plot_mask
            if np.any(zero_mask):
                ax.plot(x[zero_mask], np.full(np.sum(zero_mask), floor), linestyle="None", marker=MARKERS[rx], markersize=3.5)
    if metric in {"ber", "bler"}:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel(metric.upper())


def _best_classical_row(df: pd.DataFrame, ebno: float, metric: str):
    rel_col = {"ber": "reliable_ber", "bler": "reliable_bler"}.get(metric)
    s = df[(df["ebno_db"] == ebno) & (df["receiver"].isin(CLASSICAL_RECEIVERS))].copy()
    if s.empty:
        return None
    if rel_col is not None:
        s = s[s[rel_col].astype(bool)]
    if s.empty:
        return None
    return s.sort_values(metric).iloc[0]


def _make_main_curves(frames):
    df = frames["Mild main"]
    floors = {m: _positive_floor(frames, m, MAIN_RECEIVERS + ["perfect_csi_lmmse"]) for m in METRICS}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, metric in zip(axes, METRICS):
        recs = MAIN_RECEIVERS + (["perfect_csi_lmmse"] if metric in {"ber", "bler", "nmse"} else [])
        _plot_metric(ax, df, metric, recs, floors[metric])
    axes[0].legend(loc="best", fontsize=8)
    _save(fig, "Fig01_mild_main_curves")


def _make_focus(frames):
    df = frames["Mild main"]
    floors = {m: _positive_floor(frames, m, FOCUS_RECEIVERS) for m in METRICS}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, metric in zip(axes, METRICS):
        _plot_metric(ax, df, metric, FOCUS_RECEIVERS, floors[metric])
    axes[0].legend(loc="best", fontsize=8)
    _save(fig, "Fig02_mild_main_focus")


def _make_gain_tables(frames):
    main = frames["Mild main"]
    rows = []
    for ebno in sorted(main["ebno_db"].unique().tolist()):
        up = main[(main["receiver"] == "upair5g_lmmse") & (main["ebno_db"] == ebno)]
        if up.empty:
            continue
        up = up.iloc[0]
        row = {"ebno_db": float(ebno)}
        for metric in ["ber", "bler", "nmse"]:
            best = _best_classical_row(main, ebno, metric)
            if best is None:
                continue
            row[f"best_{metric}_classical_receiver"] = best["receiver"]
            row[f"best_{metric}_classical"] = float(best[metric])
            row[f"{metric}_upair"] = float(up[metric])
            if metric in {"ber", "bler"}:
                b = float(best[metric]); u = float(up[metric])
                row[f"{metric}_ratio_best_over_upair"] = (b / u) if (u > 0 and b > 0) else np.nan
            else:
                row[f"{metric}_relative_reduction"] = (float(best[metric]) - float(up[metric])) / float(best[metric]) if float(best[metric]) > 0 else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "mild_main_gain.csv", index=False)
    return df


def _make_gain_plot(main_gain: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    axes[0].plot(main_gain["ebno_db"], main_gain["ber_ratio_best_over_upair"], marker="o")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BER ratio (best classical / UPAIR)")
    axes[1].plot(main_gain["ebno_db"], main_gain["bler_ratio_best_over_upair"], marker="o")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("BLER ratio (best classical / UPAIR)")
    axes[2].plot(main_gain["ebno_db"], 100.0*main_gain["nmse_relative_reduction"], marker="o")
    axes[2].set_ylabel("NMSE reduction (%)")
    for ax in axes:
        ax.set_xlabel("Eb/N0 (dB)")
        ax.grid(True, which="both", alpha=0.25)
    _save(fig, "Fig03_mild_main_upair_gain")


def _make_channel_error_maps():
    import numpy as np
    path = MAIN_EXAMPLE
    if not path.exists():
        return
    data = np.load(path)
    h_true = data["h_true"]
    h_ls = data.get("h_ls_linear")
    h_up = data.get("h_upair") if "h_upair" in data else data.get("h_proposed")
    if h_up is None:
        return
    # take first batch/user/layer slices conservatively
    ht = np.asarray(h_true).reshape(-1)
    hl = np.asarray(h_ls).reshape(-1) if h_ls is not None else None
    hu = np.asarray(h_up).reshape(-1)
    n = int(np.sqrt(min(len(ht), 1024)))
    ht = ht[: n*n].reshape(n, n)
    hu = hu[: n*n].reshape(n, n)
    err_up = np.abs(ht - hu)
    fig, axes = plt.subplots(1, 2 if hl is None else 3, figsize=(10.5, 3.2))
    idx = 0
    if hl is not None:
        hl = hl[: n*n].reshape(n, n)
        err_ls = np.abs(ht - hl)
        axes[idx].imshow(err_ls)
        axes[idx].set_title("|H-Hhat| : LS")
        idx += 1
    axes[idx].imshow(err_up)
    axes[idx].set_title("|H-Hhat| : UPAIR")
    idx += 1
    axes[idx].imshow(np.abs(ht))
    axes[idx].set_title("|H| true")
    for ax in np.atleast_1d(axes):
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, "Fig04_mild_main_channel_error_maps")


def _make_crossscenario_tables(frames):
    rows = []
    for scenario, df in frames.items():
        for metric in ["ber", "bler", "nmse"]:
            ups = df[df["receiver"] == "upair5g_lmmse"]
            if ups.empty:
                continue
            vals = []
            for ebno in sorted(df["ebno_db"].unique().tolist()):
                up = ups[ups["ebno_db"] == ebno]
                if up.empty:
                    continue
                best = _best_classical_row(df, ebno, metric)
                if best is None:
                    continue
                u = float(up.iloc[0][metric])
                b = float(best[metric])
                if metric in {"ber", "bler"}:
                    ratio = (b / u) if (u > 0 and b > 0) else (1.0 if u == 0 and b == 0 else np.nan)
                    vals.append(ratio)
                else:
                    vals.append((b - u) / b if b > 0 else np.nan)
            rows.append({"scenario": scenario, "metric": metric, "mean_gain": float(np.nanmean(vals)) if len(vals) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "mild_crossscenario_gain.csv", index=False)
    return out


def _make_crossscenario_grid(frames):
    floors = {m: _positive_floor(frames, m, CLASSICAL_RECEIVERS + ["upair5g_lmmse"]) for m in METRICS}
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.2), sharex='col')
    scenario_names = list(SCENARIOS.keys())
    for i, scen in enumerate(scenario_names):
        df = frames[scen]
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            _plot_metric(ax, df, metric, CLASSICAL_RECEIVERS + ["upair5g_lmmse"], floors[metric])
            if i == 0:
                ax.set_title(metric.upper())
            if j == 0:
                ax.text(-0.25, 0.5, scen, transform=ax.transAxes, rotation=90, va='center', ha='center')
    axes[0,0].legend(loc='best', fontsize=7)
    _save(fig, "Fig05_mild_crossscenario_metrics_grid")


def _make_crossscenario_gain_plot(cross_gain: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    for ax, metric in zip(axes, ["ber", "bler", "nmse"]):
        s = cross_gain[cross_gain["metric"] == metric]
        ax.bar(s["scenario"], s["mean_gain"])
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_title(metric.upper())
        ax.tick_params(axis='x', rotation=20)
        if metric in {"ber", "bler"}:
            ax.set_ylabel("Mean ratio\n(best classical / UPAIR)")
        else:
            ax.set_ylabel("Mean relative NMSE reduction")
    _save(fig, "Fig06_mild_crossscenario_upair_gain")


def main() -> None:
    _ensure_dirs()
    frames = _load_frames()
    _make_main_curves(frames)
    _make_focus(frames)
    gain = _make_gain_tables(frames)
    _make_gain_plot(gain)
    _make_channel_error_maps()
    cross = _make_crossscenario_tables(frames)
    _make_crossscenario_grid(frames)
    _make_crossscenario_gain_plot(cross)
    manifest = FIG_DIR / "TWC_plot_manifest.txt"
    manifest.write_text(
        "Fig01_mild_main_curves.pdf/png\n"
        "Fig02_mild_main_focus.pdf/png\n"
        "Fig03_mild_main_upair_gain.pdf/png\n"
        "Fig04_mild_main_channel_error_maps.pdf/png\n"
        "Fig05_mild_crossscenario_metrics_grid.pdf/png\n"
        "Fig06_mild_crossscenario_upair_gain.pdf/png\n"
        "tables/mild_main_gain.csv\n"
        "tables/mild_crossscenario_gain.csv\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
