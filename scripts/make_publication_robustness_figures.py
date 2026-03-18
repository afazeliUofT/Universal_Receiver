from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/rsadve1/PROJECT/Universal_Receiver")
OUT_ROOT = ROOT / "outputs" / "publication_robustness_summary"
METRICS_DIR = OUT_ROOT / "metrics"
PLOTS_DIR = OUT_ROOT / "plots"

SCENARIOS = {
    "Targeted hard case": ROOT / "outputs" / "upair_cdlc_highmobility_paper_phasefair_comparison" / "metrics" / "curves.csv",
    "Clean control": ROOT / "outputs" / "upair_cdlc_highmobility_paper_phasefair_cleancontrol" / "metrics" / "curves.csv",
    "DMRS-rich control": ROOT / "outputs" / "upair_cdlc_highmobility_paper_phasefair_dmrsrich" / "metrics" / "curves.csv",
}

RECEIVERS = [
    "baseline_ddcpe_ls_lmmse",
    "paper_cfgres_phaseaware_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]

LABELS = {
    "baseline_ddcpe_ls_lmmse": "DD-CPE+LS+LMMSE",
    "paper_cfgres_phaseaware_ls_lmmse": "Phase-aware paper cfg reservoir+LMMSE",
    "upair5g_lmmse": "UPAIR-5G+LMMSE",
    "perfect_csi_lmmse": "Perfect CSI+LMMSE",
}

LINESTYLES = {
    "baseline_ddcpe_ls_lmmse": "--",
    "paper_cfgres_phaseaware_ls_lmmse": "-.",
    "upair5g_lmmse": "-",
    "perfect_csi_lmmse": ":",
}

MARKERS = {
    "baseline_ddcpe_ls_lmmse": "s",
    "paper_cfgres_phaseaware_ls_lmmse": "D",
    "upair5g_lmmse": "o",
    "perfect_csi_lmmse": "^",
}

METRICS = ["ber", "bler", "nmse"]


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _require_curves() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for scenario, path in SCENARIOS.items():
        if not path.exists():
            missing.append(str(path))
            continue
        frames[scenario] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError(
            "Missing required curves.csv files for robustness publication summary:\n" + "\n".join(missing)
        )
    return frames


def _best_classical(df: pd.DataFrame, ebno_db: float, metric: str) -> tuple[str, float]:
    classical = df[
        (df["receiver"] != "upair5g_lmmse") &
        (df["receiver"] != "perfect_csi_lmmse")
    ].copy()
    classical = classical[classical["ebno_db"] == ebno_db]
    classical = classical.dropna(subset=[metric])
    best = classical.loc[classical[metric].idxmin()]
    return str(best["receiver"]), float(best[metric])


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    frames = _require_curves()

    # 3x3 grid: metrics x scenarios
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex="col")
    scenario_names = list(SCENARIOS.keys())
    for col, scenario in enumerate(scenario_names):
        df = frames[scenario]
        axes[0, col].set_title(scenario)
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for receiver in RECEIVERS:
                sub = df[df["receiver"] == receiver].copy()
                if sub.empty:
                    continue
                ax.semilogy(
                    sub["ebno_db"].to_numpy(),
                    sub[metric].to_numpy(),
                    linestyle=LINESTYLES[receiver],
                    marker=MARKERS[receiver],
                    linewidth=2.2 if receiver == "upair5g_lmmse" else 1.8,
                    markersize=5.0,
                    label=LABELS[receiver],
                )
            ax.grid(True, which="both", alpha=0.3)
            if col == 0:
                ax.set_ylabel(metric.upper())
            if row == 2:
                ax.set_xlabel("Eb/N0 [dB]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    _save(fig, PLOTS_DIR / "publication_crossscenario_metrics_grid")

    # UPAIR gain vs best classical across scenarios
    rows: list[dict[str, float | str]] = []
    for scenario, df in frames.items():
        ebnos = sorted(df["ebno_db"].unique().tolist())
        for ebno_db in ebnos:
            upair = df[(df["receiver"] == "upair5g_lmmse") & (df["ebno_db"] == ebno_db)]
            if upair.empty:
                continue
            upair_row = upair.iloc[0]
            row: dict[str, float | str] = {
                "scenario": scenario,
                "ebno_db": float(ebno_db),
            }
            for metric in METRICS:
                best_name, best_value = _best_classical(df, ebno_db=float(ebno_db), metric=metric)
                upair_value = float(upair_row[metric])
                row[f"best_{metric}_classical_receiver"] = best_name
                row[f"best_{metric}_classical"] = best_value
                row[f"{metric}_upair5g"] = upair_value
                row[f"{metric}_gap_upair_minus_best_classical"] = upair_value - best_value
                if metric in {"ber", "bler"}:
                    row[f"{metric}_ratio_best_classical_over_upair"] = np.inf if upair_value == 0.0 else best_value / upair_value
                else:
                    row[f"{metric}_reduction_pct"] = 100.0 * (best_value - upair_value) / max(best_value, 1e-15)
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(METRICS_DIR / "publication_crossscenario_upair_gain.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for scenario in scenario_names:
            sub = summary_df[summary_df["scenario"] == scenario].copy()
            if metric in {"ber", "bler"}:
                y = np.clip(sub[f"{metric}_ratio_best_classical_over_upair"].to_numpy(dtype=float), 1e0, None)
                ylabel = "Best classical / UPAIR"
            else:
                y = sub[f"{metric}_reduction_pct"].to_numpy(dtype=float)
                ylabel = "NMSE reduction [%]"
            ax.plot(sub["ebno_db"].to_numpy(), y, marker="o", linewidth=2.0, label=scenario)
        if metric in {"ber", "bler"}:
            ax.set_yscale("log")
        ax.set_title(metric.upper())
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(frameon=False)
    _save(fig, PLOTS_DIR / "publication_crossscenario_upair_gain")

    selected = summary_df[summary_df["ebno_db"].isin([0.0, 8.0, 16.0])].copy()
    selected.to_csv(METRICS_DIR / "publication_crossscenario_selected_points.csv", index=False)

    print({
        "summary_csv": str(METRICS_DIR / "publication_crossscenario_upair_gain.csv"),
        "selected_points_csv": str(METRICS_DIR / "publication_crossscenario_selected_points.csv"),
        "plots_dir": str(PLOTS_DIR),
    })


if __name__ == "__main__":
    main()
