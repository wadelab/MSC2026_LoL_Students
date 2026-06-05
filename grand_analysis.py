"""Across-server summary analysis for the League of Legends rhythm workflow.

This module consumes the per-server outputs written by
`Parquet_longerAnalyses_May_26.py` and builds a final meta-analysis in
`results/GRAND/`. It avoids re-querying DuckDB and keeps each server as one
analysis unit before averaging across servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import vonmises

from riot_analysis import COLORS, configure_plot_style, fit_vonmises_1comp, fit_vonmises_2comp, save_table, style_axes


GRAND_DIR_NAME = "GRAND"

# Canonical server set for the grand analysis: the eight regions used by
# published source-dataset reports (Aung et al. 2018; Vardal et al. 2022).
# The remaining platforms (ID1, TR1, PBE1) are excluded as small / unreliable.
# JP1 is retained; its spurious DeltaMMR period is still screened by
# drop_period_outliers().
ANALYSIS_PLATFORMS = ["BR1", "EUN1", "EUW1", "JP1", "LA1", "LA2", "NA1", "OC1"]

COMPONENTS_TO_PLOT = ["PC1", "PC2", "PC3"]
METRIC_ORDER = {"PC1": 0, "PC2": 1, "DeltaMMR": 2}
PC_DENSITY_METRICS = ["PC1", "PC2"]
CIRCULAR_DENSITY_KAPPA = 4.0
CIRCULAR_DENSITY_GRID_POINTS = 240
CIRCULAR_DENSITY_MIN_PHASES = 20
SUMMARY_COLUMNS = [
    "platform",
    "target_best_period",
    "win_rate_best_period",
    "performance_pc1_explained",
    "performance_pc2_explained",
    "performance_pc3_explained",
    "success_pc1_win_rate_loading",
    "success_pc2_win_rate_loading",
    "success_pc3_win_rate_loading",
    "pc1_phase_fdr_significant",
    "pc2_phase_fdr_significant",
    "deltammr_phase_fdr_significant",
]


def weighted_stats(values: Any, weights: Any) -> dict[str, float]:
    """Return weighted mean, SD, and SEM for numeric values."""

    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    value_array = value_array[valid]
    weight_array = weight_array[valid]

    if len(value_array) == 0:
        return {
            "n": 0,
            "weight_sum": 0.0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    weight_sum = float(weight_array.sum())
    mean = float(np.average(value_array, weights=weight_array))
    variance = float(np.average((value_array - mean) ** 2, weights=weight_array))
    sd = float(np.sqrt(variance))
    n_effective = (weight_sum**2) / float(np.sum(weight_array**2))
    sem = float(sd / np.sqrt(n_effective)) if n_effective > 0 else np.nan
    return {
        "n": int(len(value_array)),
        "weight_sum": weight_sum,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "min": float(value_array.min()),
        "max": float(value_array.max()),
    }


def discover_server_dirs(output_root: Path, platforms: list[str] | None = None) -> list[Path]:
    """Find server result folders with per-server analysis summaries."""

    requested = {platform.upper() for platform in platforms} if platforms else None
    server_dirs = []
    for path in sorted(output_root.iterdir()):
        if not path.is_dir() or path.name == GRAND_DIR_NAME:
            continue
        if requested is not None and path.name.upper() not in requested:
            continue
        if (path / "analysis_summary.csv").exists():
            server_dirs.append(path)
    return server_dirs


def clean_grand_outputs(grand_dir: Path) -> None:
    """Remove generated grand-analysis artifacts before writing fresh outputs."""

    if not grand_dir.exists():
        return
    for pattern in ("*.png", "*.csv", "*.md", "*.html"):
        for path in grand_dir.glob(pattern):
            path.unlink()


def read_one_row_csv(path: Path) -> pd.DataFrame:
    """Read a CSV expected to contain one logical table."""

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_server_summaries(server_dirs: list[Path]) -> pd.DataFrame:
    """Combine each server's one-row analysis summary."""

    frames = []
    for server_dir in server_dirs:
        summary = read_one_row_csv(server_dir / "analysis_summary.csv")
        summary.insert(0, "server_dir", server_dir.name)
        frames.append(summary)
    combined = pd.concat(frames, ignore_index=True)
    if "platform" not in combined.columns:
        combined["platform"] = combined["server_dir"]
    return combined


def load_server_weights(server_dirs: list[Path]) -> pd.DataFrame:
    """Load available server-size weights from existing output tables."""

    rows = []
    for server_dir in server_dirs:
        row = {"platform": server_dir.name}

        win_path = server_dir / "win_rate_by_local_hour.csv"
        if win_path.exists():
            win_rate = pd.read_csv(win_path)
            row["server_n_win_games"] = float(win_rate["n_win_games"].sum())

        top_players_path = server_dir / f"top_players_{server_dir.name}.csv"
        if top_players_path.exists():
            top_players = pd.read_csv(top_players_path)
            row["server_top_player_games"] = float(top_players["game_count"].sum())

        rows.append(row)

    return pd.DataFrame(rows)


def add_server_weights(server_summary: pd.DataFrame, server_dirs: list[Path]) -> pd.DataFrame:
    """Attach server-size columns used by weighted grand summaries."""

    weights = load_server_weights(server_dirs)
    if weights.empty:
        return server_summary
    return server_summary.merge(weights, on="platform", how="left")


def weight_column_for_metric(metric: str, data: pd.DataFrame) -> str:
    """Choose the most relevant N column for one server-level metric."""

    metric_lower = metric.lower()
    if metric_lower.startswith("pc1_player_periodogram") and "pc1_player_periodogram_valid_players" in data:
        return "pc1_player_periodogram_valid_players"
    if metric_lower.startswith("pc2_player_periodogram") and "pc2_player_periodogram_valid_players" in data:
        return "pc2_player_periodogram_valid_players"
    if metric_lower.startswith("deltammr_player_periodogram") and "deltammr_player_periodogram_valid_players" in data:
        return "deltammr_player_periodogram_valid_players"
    if metric_lower.startswith("pc1_phase") and "pc1_phase_players" in data:
        return "pc1_phase_players"
    if metric_lower.startswith("pc2_phase") and "pc2_phase_players" in data:
        return "pc2_phase_players"
    if metric_lower.startswith("deltammr_phase") and "deltammr_phase_players" in data:
        return "deltammr_phase_players"
    if metric_lower.startswith("pc1_circular") and "pc1_circular_n" in data:
        return "pc1_circular_n"
    if metric_lower.startswith("pc2_circular") and "pc2_circular_n" in data:
        return "pc2_circular_n"
    if metric_lower.startswith("deltammr_circular") and "deltammr_circular_n" in data:
        return "deltammr_circular_n"
    if "server_n_win_games" in data:
        return "server_n_win_games"
    if "server_top_player_games" in data:
        return "server_top_player_games"
    return ""


def summarize_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric server-level columns with N-aware weights."""

    rows = []
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith("server_"):
            continue
        values = data[col].dropna()
        if values.empty:
            continue
        weight_col = weight_column_for_metric(col, data)
        if weight_col:
            weights = data.loc[values.index, weight_col]
        else:
            weights = pd.Series(np.ones(len(values)), index=values.index)
            weight_col = "equal_weight_fallback"
        stats = weighted_stats(values, weights)
        rows.append(
            {
                "metric": col,
                "n_servers": stats["n"],
                "weight_col": weight_col,
                "weight_sum": stats["weight_sum"],
                "weighted_mean": stats["mean"],
                "weighted_sd": stats["sd"],
                "weighted_sem": stats["sem"],
                "min": stats["min"],
                "max": stats["max"],
            }
        )
    return pd.DataFrame(rows)


def load_pca_loadings(server_dirs: list[Path], filename: str) -> pd.DataFrame:
    """Load and stack PCA loading tables from every server."""

    frames = []
    for server_dir in server_dirs:
        path = server_dir / filename
        if not path.exists():
            continue
        loadings = pd.read_csv(path)
        loadings.insert(0, "platform", server_dir.name)
        frames.append(loadings)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_loadings(loadings: pd.DataFrame, server_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute N-weighted across-server mean and spread for PCA loadings."""

    if loadings.empty:
        return pd.DataFrame()

    weight_cols = ["platform", "server_n_win_games"]
    weights = server_summary[[col for col in weight_cols if col in server_summary.columns]].drop_duplicates("platform")
    loadings = loadings.merge(weights, on="platform", how="left")
    loadings["server_n_win_games"] = loadings["server_n_win_games"].fillna(1.0)

    value_cols = [col for col in loadings.columns if col not in {"platform", "component"}]
    value_cols = [col for col in value_cols if not col.startswith("server_")]
    long = loadings.melt(
        id_vars=["platform", "component", "server_n_win_games"],
        value_vars=value_cols,
        var_name="feature",
        value_name="loading",
    )

    rows = []
    for (component, feature), group in long.groupby(["component", "feature"]):
        stats = weighted_stats(group["loading"], group["server_n_win_games"])
        rows.append(
            {
                "component": component,
                "feature": feature,
                "n_servers": stats["n"],
                "weight_col": "server_n_win_games",
                "weight_sum": stats["weight_sum"],
                "weighted_mean_loading": stats["mean"],
                "weighted_sd_loading": stats["sd"],
                "min_loading": stats["min"],
                "max_loading": stats["max"],
            }
        )
    return pd.DataFrame(rows).sort_values(["component", "feature"]).reset_index(drop=True)


def load_local_win_rates(server_dirs: list[Path]) -> pd.DataFrame:
    """Load server-local win-rate curves and add within-server z scores."""

    frames = []
    for server_dir in server_dirs:
        path = server_dir / "win_rate_by_local_hour.csv"
        if not path.exists():
            continue
        win_rate = pd.read_csv(path)
        win_rate.insert(0, "platform", server_dir.name)
        mean = win_rate["win_rate"].mean()
        sd = win_rate["win_rate"].std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            win_rate["win_rate_z"] = 0.0
        else:
            win_rate["win_rate_z"] = (win_rate["win_rate"] - mean) / sd
        frames.append(win_rate)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_local_win_rates(local_win: pd.DataFrame) -> pd.DataFrame:
    """Average local-hour win-rate curves with game-count weights."""

    if local_win.empty:
        return pd.DataFrame()

    rows = []
    for local_hour, group in local_win.groupby("local_hour"):
        weights = group["n_win_games"].to_numpy(dtype=float)
        z_stats = weighted_stats(group["win_rate_z"], weights)
        raw_stats = weighted_stats(group["win_rate"], weights)
        rows.append(
            {
                "local_hour": int(local_hour),
                "n_servers": z_stats["n"],
                "weight_col": "n_win_games",
                "weight_sum": z_stats["weight_sum"],
                "weighted_win_rate_z": z_stats["mean"],
                "weighted_sem_win_rate_z": z_stats["sem"],
                "weighted_win_rate": raw_stats["mean"],
                "weighted_sem_win_rate": raw_stats["sem"],
                "n_win_games": int(group["n_win_games"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("local_hour").reset_index(drop=True)


def load_metric_table(server_dirs: list[Path], filename: str) -> pd.DataFrame:
    """Load a repeated metric table from each server folder."""

    frames = []
    for server_dir in server_dirs:
        path = server_dir / filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "platform", server_dir.name)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def sort_metric_table(table: pd.DataFrame) -> pd.DataFrame:
    """Sort rows in a human-friendly metric order."""

    if table.empty or "metric" not in table.columns:
        return table
    sorted_table = table.copy()
    sorted_table["_metric_order"] = sorted_table["metric"].map(METRIC_ORDER).fillna(99)
    sorted_table = sorted_table.sort_values(["_metric_order", "metric"]).drop(columns="_metric_order")
    return sorted_table.reset_index(drop=True)


def drop_period_outliers(
    periodograms: pd.DataFrame,
    mad_z_threshold: float = 3.5,
    abs_floor_h: float = 3.0,
) -> pd.DataFrame:
    """Drop per-server periodogram peaks that are robust outliers within a metric.

    A peak is dropped only if it is *both* a robust outlier and at least
    `abs_floor_h` hours from the metric median. The absolute-deviation gate keeps
    legitimate near-circadian values (e.g. peaks one or two grid bins either side
    of 24 h) while still removing harmonics (e.g. an 8 h or 12 h peak) and
    long-period artifacts (e.g. a 44 h peak). When the grid anchors most servers
    on the same bin the MAD collapses to zero; in that case the absolute-deviation
    gate alone decides, so a lone aberrant server is still caught.
    """

    if periodograms.empty or "best_period" not in periodograms.columns:
        return periodograms

    keep = pd.Series(True, index=periodograms.index)
    for metric, group in periodograms.groupby("metric"):
        values = pd.to_numeric(group["best_period"], errors="coerce")
        median = values.median()
        abs_dev = (values - median).abs()
        mad = abs_dev.median()
        far = abs_dev > abs_floor_h
        if np.isfinite(mad) and mad > 0:
            robust = (0.6745 * (values - median) / mad).abs() > mad_z_threshold
            outliers = robust & far
        else:
            # Degenerate scale (most servers share a bin): the gate alone decides.
            outliers = far
        if outliers.any():
            servers = group.loc[outliers, "platform"].tolist() if "platform" in group else outliers.index.tolist()
            print(f"  dropping {metric} periodogram outliers: {', '.join(map(str, servers))}")
            keep.loc[group.index[outliers.to_numpy()]] = False

    return periodograms[keep].reset_index(drop=True)


def summarize_periodograms(periodograms: pd.DataFrame) -> pd.DataFrame:
    """Average within-subject periodogram summaries with valid-player weights."""

    if periodograms.empty:
        return pd.DataFrame()

    rows = []
    for metric, group in periodograms.groupby("metric"):
        weights = group["valid_players"].to_numpy(dtype=float)
        period_stats = weighted_stats(group["best_period"], weights)
        power_stats = weighted_stats(group["best_power"], weights)
        rows.append(
            {
                "metric": metric,
                "n_servers": period_stats["n"],
                "weight_col": "valid_players",
                "weight_sum": period_stats["weight_sum"],
                "total_valid_players": int(group["valid_players"].sum()),
                "weighted_best_period": period_stats["mean"],
                "weighted_sd_best_period": period_stats["sd"],
                "weighted_sem_best_period": period_stats["sem"],
                "weighted_best_power": power_stats["mean"],
            }
        )
    return sort_metric_table(pd.DataFrame(rows))


def summarize_phase_counts(phases: pd.DataFrame) -> pd.DataFrame:
    """Summarize FDR-significant phase counts with analyzed-player weights."""

    if phases.empty:
        return pd.DataFrame()

    rows = []
    phases = phases.copy()
    phases["fdr_fraction"] = phases["fdr_significant"] / phases["players_analyzed"].where(
        phases["players_analyzed"] > 0
    )
    for metric, group in phases.groupby("metric"):
        fraction_stats = weighted_stats(group["fdr_fraction"], group["players_analyzed"])
        total_players = int(group["players_analyzed"].sum())
        total_fdr = int(group["fdr_significant"].sum())
        rows.append(
            {
                "metric": metric,
                "n_servers": fraction_stats["n"],
                "weight_col": "players_analyzed",
                "weight_sum": fraction_stats["weight_sum"],
                "total_players_analyzed": total_players,
                "total_fdr_significant": total_fdr,
                "weighted_fdr_fraction": float(total_fdr / total_players) if total_players > 0 else np.nan,
                "weighted_sd_fdr_fraction": fraction_stats["sd"],
                "weighted_sem_fdr_fraction": fraction_stats["sem"],
            }
        )
    return sort_metric_table(pd.DataFrame(rows))


def summarize_circular_models(circular: pd.DataFrame) -> pd.DataFrame:
    """Count preferred circular model outcomes, weighted by phase counts."""

    if circular.empty:
        return pd.DataFrame()

    rows = []
    for metric, group in circular.groupby("metric"):
        phases = pd.to_numeric(group["n_fdr_significant"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "metric": metric,
                "n_servers": int(group["platform"].nunique()),
                "weight_col": "n_fdr_significant",
                "total_fdr_significant": int(phases.sum()),
                "fit_servers": int((group["status"] == "fit").sum()),
                "skipped_servers": int((group["status"] == "skipped").sum()),
                "preferred_1_component": int((group["preferred"] == "1-component").sum()),
                "preferred_2_component": int((group["preferred"] == "2-component").sum()),
                "preferred_1_component_phases": int(phases[group["preferred"] == "1-component"].sum()),
                "preferred_2_component_phases": int(phases[group["preferred"] == "2-component"].sum()),
                "skipped_phases": int(phases[group["status"] == "skipped"].sum()),
            }
        )
    return sort_metric_table(pd.DataFrame(rows))


def circular_density_hours(phases_hours: np.ndarray, grid_hours: np.ndarray, kappa: float) -> np.ndarray:
    """Estimate a smooth circular density over local-hour phase values."""

    theta_grid = (grid_hours / 24.0) * 2.0 * np.pi
    theta_values = ((phases_hours % 24.0) / 24.0) * 2.0 * np.pi
    density_radians = vonmises.pdf(theta_grid[:, None], kappa, loc=theta_values[None, :]).mean(axis=1)
    return density_radians * (2.0 * np.pi / 24.0)


def load_pc_peak_densities(
    server_dirs: list[Path],
    metrics: list[str] | None = None,
    min_phases: int = CIRCULAR_DENSITY_MIN_PHASES,
    kappa: float = CIRCULAR_DENSITY_KAPPA,
    n_grid: int = CIRCULAR_DENSITY_GRID_POINTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PC peak phases and estimate one circular density per server."""

    metrics = metrics or PC_DENSITY_METRICS
    grid_hours = np.linspace(0.0, 24.0, n_grid, endpoint=False)
    server_rows = []

    for server_dir in server_dirs:
        for metric in metrics:
            phase_path = server_dir / f"significant_phases_{metric.lower()}_{server_dir.name}.csv"
            if not phase_path.exists():
                continue
            phases = pd.read_csv(phase_path)
            phase_values = pd.to_numeric(phases.get("phase_local_peak"), errors="coerce").dropna().to_numpy()
            if len(phase_values) < min_phases:
                continue

            density = circular_density_hours(phase_values, grid_hours, kappa=kappa)
            for local_hour, density_value in zip(grid_hours, density):
                server_rows.append(
                    {
                        "platform": server_dir.name,
                        "metric": metric,
                        "local_hour": float(local_hour),
                        "density": float(density_value),
                        "n_phases": int(len(phase_values)),
                        "kappa": float(kappa),
                        "min_phases": int(min_phases),
                    }
                )

    server_curves = pd.DataFrame(server_rows)
    if server_curves.empty:
        return pd.DataFrame(), server_curves

    summary_rows = []
    for (metric, local_hour), group in server_curves.groupby(["metric", "local_hour"]):
        weights = group["n_phases"].to_numpy(dtype=float)
        density_stats = weighted_stats(group["density"], weights)
        summary_rows.append(
            {
                "metric": metric,
                "local_hour": float(local_hour),
                "n_servers": density_stats["n"],
                "weight_col": "n_phases",
                "weight_sum": density_stats["weight_sum"],
                "total_phases": int(group.drop_duplicates("platform")["n_phases"].sum()),
                "weighted_density": density_stats["mean"],
                "weighted_sd_density": density_stats["sd"],
                "weighted_sem_density": density_stats["sem"],
                "kappa": float(kappa),
                "min_phases": int(min_phases),
            }
        )
    density_summary = pd.DataFrame(summary_rows)
    density_summary = sort_metric_table(density_summary).sort_values(["metric", "local_hour"]).reset_index(drop=True)
    return density_summary, server_curves


def summarize_peak_density_modes(density_summary: pd.DataFrame) -> pd.DataFrame:
    """Find the peak hour of each phase-count weighted circular density."""

    if density_summary.empty:
        return pd.DataFrame()

    rows = []
    for metric, group in density_summary.groupby("metric"):
        peak_row = group.loc[group["weighted_density"].idxmax()]
        rows.append(
            {
                "metric": metric,
                "n_servers": int(peak_row["n_servers"]),
                "total_phases": int(peak_row["total_phases"]),
                "weighted_peak_hour": float(peak_row["local_hour"]),
                "weighted_density_peak": float(peak_row["weighted_density"]),
                "kappa": float(peak_row["kappa"]),
                "min_phases": int(peak_row["min_phases"]),
            }
        )
    return sort_metric_table(pd.DataFrame(rows))


def load_pooled_significant_phases(
    server_dirs: list[Path],
    metrics: list[str] | None = None,
    min_phases: int = CIRCULAR_DENSITY_MIN_PHASES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pooled phase values from server-metric files that pass the N threshold."""

    metrics = metrics or (PC_DENSITY_METRICS + ["DeltaMMR"])
    phase_rows = []
    inclusion_rows = []

    for metric in metrics:
        for server_dir in server_dirs:
            phase_path = server_dir / f"significant_phases_{metric.lower()}_{server_dir.name}.csv"
            if not phase_path.exists():
                continue
            phases = pd.read_csv(phase_path)
            values = pd.to_numeric(phases.get("phase_local_peak"), errors="coerce").dropna().to_numpy(dtype=float)
            included = len(values) >= min_phases
            inclusion_rows.append(
                {
                    "metric": metric,
                    "platform": server_dir.name,
                    "n_phases": int(len(values)),
                    "included": bool(included),
                    "min_phases": int(min_phases),
                }
            )
            if not included:
                continue
            for value in values % 24.0:
                phase_rows.append(
                    {
                        "metric": metric,
                        "platform": server_dir.name,
                        "phase_local_peak": float(value),
                    }
                )

    return pd.DataFrame(phase_rows), pd.DataFrame(inclusion_rows)


def pooled_circular_bimodality_tests(
    server_dirs: list[Path],
    metrics: list[str] | None = None,
    min_phases: int = CIRCULAR_DENSITY_MIN_PHASES,
) -> pd.DataFrame:
    """Compare one- vs two-component circular models on pooled significant phases."""

    metrics = metrics or PC_DENSITY_METRICS
    pooled_phases, inclusion = load_pooled_significant_phases(
        server_dirs,
        metrics=metrics + ["DeltaMMR"],
        min_phases=min_phases,
    )
    rows = []

    for metric in metrics + ["DeltaMMR"]:
        metric_inclusion = inclusion[inclusion["metric"] == metric]
        phase_values = pooled_phases.loc[pooled_phases["metric"] == metric, "phase_local_peak"].to_numpy(dtype=float)
        contributing_servers = int(metric_inclusion["included"].sum()) if not metric_inclusion.empty else 0
        skipped_servers = int((~metric_inclusion["included"]).sum()) if not metric_inclusion.empty else 0
        if len(phase_values) < min_phases:
            rows.append(
                {
                    "metric": metric,
                    "n_servers": contributing_servers,
                    "skipped_servers": skipped_servers,
                    "n_phases": int(len(phase_values)),
                    "status": "skipped",
                    "reason": f"fewer than {min_phases} pooled phases",
                    "min_phases": min_phases,
                }
            )
            continue

        theta = (phase_values / 24.0) * 2.0 * np.pi
        vm1 = fit_vonmises_1comp(theta)
        vm2 = fit_vonmises_2comp(theta)
        delta_bic = vm1["bic"] - vm2["bic"]

        rows.append(
            {
                "metric": metric,
                "n_servers": contributing_servers,
                "skipped_servers": skipped_servers,
                "n_phases": int(len(phase_values)),
                "status": "fit",
                "preferred": "2-component" if delta_bic > 0 else "1-component",
                "vm1_bic": vm1["bic"],
                "vm2_bic": vm2["bic"],
                "delta_bic_1_minus_2": delta_bic,
                "vm1_peak_h": float((vm1["mu"] / (2.0 * np.pi)) * 24.0),
                "vm1_kappa": vm1["kappa"],
                "component_1_h": float((vm2["mu1"] / (2.0 * np.pi)) * 24.0),
                "component_2_h": float((vm2["mu2"] / (2.0 * np.pi)) * 24.0),
                "component_1_weight": vm2["pi1"],
                "component_2_weight": vm2["pi2"],
                "component_1_kappa": vm2["kappa1"],
                "component_2_kappa": vm2["kappa2"],
                "min_phases": min_phases,
            }
        )

    return sort_metric_table(pd.DataFrame(rows))


def plot_grand_loadings(loadings_summary: pd.DataFrame, title: str, output_path: Path) -> None:
    """Save grand weighted PCA loading bars with across-server SD."""

    if loadings_summary.empty:
        return

    components = [component for component in COMPONENTS_TO_PLOT if component in set(loadings_summary["component"])]
    fig, axes = plt.subplots(1, len(components), figsize=(6.2 * len(components), 5.3), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, component in zip(axes, components):
        subset = loadings_summary[loadings_summary["component"] == component].copy()
        subset = subset.sort_values("weighted_mean_loading")
        colors = [COLORS["accent"] if value >= 0 else COLORS["primary"] for value in subset["weighted_mean_loading"]]
        ax.barh(
            subset["feature"],
            subset["weighted_mean_loading"],
            xerr=subset["weighted_sd_loading"].fillna(0.0),
            color=colors,
            alpha=0.88,
            edgecolor="white",
        )
        ax.axvline(0.0, color=COLORS["ink"], linewidth=1.0)
        ax.set_title(f"{component} N-Weighted Loadings")
        ax.set_xlabel("Weighted loading across servers")
        style_axes(ax, grid_axis="x")

    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grand_win_rate(local_summary: pd.DataFrame, output_path: Path) -> None:
    """Save the N-weighted grand local-hour win-rate curve."""

    if local_summary.empty:
        return

    x = local_summary["local_hour"].to_numpy(dtype=float)
    y = local_summary["weighted_win_rate_z"].to_numpy(dtype=float)
    ci = 1.96 * local_summary["weighted_sem_win_rate_z"].fillna(0.0).to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax = axes[0]
    ax.plot(x, y, color=COLORS["primary"], marker="o", linewidth=2.2)
    ax.fill_between(x, y - ci, y + ci, color=COLORS["primary"], alpha=0.16, linewidth=0)
    ax.axhline(0.0, color=COLORS["muted"], linestyle="--", linewidth=1.1)
    ax.set_title("Mean Local-Hour Win-Rate Shape")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Game-count weighted z-scored win rate")
    ax.set_xticks(np.arange(0, 24, 2))
    style_axes(ax, grid_axis="y")

    ax = axes[1]
    ax.plot(
        x,
        local_summary["weighted_win_rate"],
        color=COLORS["accent"],
        marker="o",
        linewidth=2.2,
        label="Game-count weighted",
    )
    ax.set_title("Raw Win Rate by Local Hour")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Win rate")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(frameon=False)
    style_axes(ax, grid_axis="y")

    fig.suptitle("Grand N-Weighted Local-Hour Win-Rate Analysis", fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pc_peak_densities(density_summary: pd.DataFrame, output_path: Path) -> None:
    """Save phase-count weighted circular PC peak density plots."""

    if density_summary.empty:
        return

    metrics = [metric for metric in PC_DENSITY_METRICS if metric in set(density_summary["metric"])]
    fig = plt.figure(figsize=(14.5, 5.7 * len(metrics)))

    for row_idx, metric in enumerate(metrics):
        subset = density_summary[density_summary["metric"] == metric].sort_values("local_hour").copy()
        if subset.empty:
            continue

        x = subset["local_hour"].to_numpy(dtype=float)
        y = subset["weighted_density"].to_numpy(dtype=float)
        ci = 1.96 * subset["weighted_sem_density"].fillna(0.0).to_numpy(dtype=float)
        x_wrap = np.r_[x, 24.0]
        y_wrap = np.r_[y, y[0]]
        ci_wrap = np.r_[ci, ci[0]]

        n_servers = int(subset["n_servers"].iloc[0])
        total_phases = int(subset["total_phases"].iloc[0])

        ax1 = fig.add_subplot(len(metrics), 2, 2 * row_idx + 1)
        ax1.plot(x_wrap, y_wrap, color=COLORS["primary"], linewidth=2.4, label="Phase-count weighted")
        ax1.fill_between(
            x_wrap,
            y_wrap - ci_wrap,
            y_wrap + ci_wrap,
            color=COLORS["primary"],
            alpha=0.16,
            linewidth=0,
            label="95% CI across servers",
        )
        ax1.set_title(f"{metric} Peak Density by Local Hour")
        ax1.set_xlabel("Local peak hour")
        ax1.set_ylabel("Density")
        ax1.set_xlim(0, 24)
        ax1.set_xticks(np.arange(0, 25, 2))
        ax1.legend(frameon=False, loc="upper right")
        style_axes(ax1, grid_axis="y")

        ax2 = fig.add_subplot(len(metrics), 2, 2 * row_idx + 2, projection="polar")
        theta = (x_wrap / 24.0) * 2.0 * np.pi
        ax2.plot(theta, y_wrap, color=COLORS["primary"], linewidth=2.4)
        ax2.fill_between(theta, np.maximum(y_wrap - ci_wrap, 0.0), y_wrap + ci_wrap, color=COLORS["primary"], alpha=0.16)
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax2.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
        ax2.set_title(f"{metric} Circular Density\n{n_servers} servers, {total_phases:,} phases", va="bottom")
        ax2.grid(alpha=0.24)

    fig.suptitle(
        "Grand Phase-Count Weighted PC Peak Density",
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_circular_bimodality_fits(
    pooled_phases: pd.DataFrame,
    pooled_bimodality: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save pooled phase histograms with one- and two-component fits overlaid."""

    if pooled_phases.empty or pooled_bimodality.empty:
        return

    metrics = [metric for metric in PC_DENSITY_METRICS if metric in set(pooled_bimodality["metric"])]
    fig = plt.figure(figsize=(15.0, 5.8 * len(metrics)))
    x_hours = np.linspace(0.0, 24.0, 1000)
    x_theta = (x_hours / 24.0) * 2.0 * np.pi
    hour_scale = 2.0 * np.pi / 24.0

    for row_idx, metric in enumerate(metrics):
        row = pooled_bimodality[pooled_bimodality["metric"] == metric].iloc[0]
        if row.get("status") != "fit":
            continue

        phases = pooled_phases.loc[pooled_phases["metric"] == metric, "phase_local_peak"].to_numpy(dtype=float)
        theta_data = (phases / 24.0) * 2.0 * np.pi

        vm1_theta = vonmises.pdf(
            x_theta,
            float(row["vm1_kappa"]),
            loc=(float(row["vm1_peak_h"]) / 24.0) * 2.0 * np.pi,
        )
        component_1_theta = float(row["component_1_weight"]) * vonmises.pdf(
            x_theta,
            float(row["component_1_kappa"]),
            loc=(float(row["component_1_h"]) / 24.0) * 2.0 * np.pi,
        )
        component_2_theta = float(row["component_2_weight"]) * vonmises.pdf(
            x_theta,
            float(row["component_2_kappa"]),
            loc=(float(row["component_2_h"]) / 24.0) * 2.0 * np.pi,
        )
        vm2_theta = component_1_theta + component_2_theta

        vm1_hours = vm1_theta * hour_scale
        vm2_hours = vm2_theta * hour_scale
        component_1_hours = component_1_theta * hour_scale
        component_2_hours = component_2_theta * hour_scale

        ax1 = fig.add_subplot(len(metrics), 2, 2 * row_idx + 1)
        ax1.hist(
            phases,
            bins=np.arange(0, 25, 1),
            density=True,
            color=COLORS["secondary"],
            edgecolor="white",
            alpha=0.46,
            label="Pooled FDR phases",
        )
        ax1.plot(x_hours, vm1_hours, color=COLORS["ink"], linewidth=2.0, linestyle="--", label="1-component fit")
        ax1.plot(x_hours, vm2_hours, color=COLORS["accent"], linewidth=2.5, label="2-component fit")
        ax1.plot(x_hours, component_1_hours, color=COLORS["accent"], linewidth=1.3, linestyle=":", alpha=0.8)
        ax1.plot(x_hours, component_2_hours, color=COLORS["accent"], linewidth=1.3, linestyle=":", alpha=0.8)
        ax1.set_title(f"{metric} Pooled Phase Fit")
        ax1.set_xlabel("Local peak hour")
        ax1.set_ylabel("Density")
        ax1.set_xlim(0, 24)
        ax1.set_xticks(np.arange(0, 25, 2))
        ax1.legend(frameon=False)
        bic_text = (
            f"N phases = {int(row['n_phases']):,}\n"
            f"BIC 1-comp = {float(row['vm1_bic']):.1f}\n"
            f"BIC 2-comp = {float(row['vm2_bic']):.1f}\n"
            f"Delta BIC = {float(row['delta_bic_1_minus_2']):.1f}"
        )
        ax1.text(
            0.02,
            0.96,
            bic_text,
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color=COLORS["ink"],
            bbox={"facecolor": "#fcfbf8", "edgecolor": "#d9d1c8", "alpha": 0.9, "pad": 6},
        )
        style_axes(ax1, grid_axis="y")

        ax2 = fig.add_subplot(len(metrics), 2, 2 * row_idx + 2, projection="polar")
        bins = np.linspace(0, 2.0 * np.pi, 25)
        counts, edges = np.histogram(theta_data, bins=bins, density=True)
        ax2.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=COLORS["secondary"],
            alpha=0.44,
            edgecolor="white",
            linewidth=0.8,
        )
        ax2.plot(x_theta, vm1_theta, color=COLORS["ink"], linewidth=1.9, linestyle="--")
        ax2.plot(x_theta, vm2_theta, color=COLORS["accent"], linewidth=2.4)
        ax2.plot(x_theta, component_1_theta, color=COLORS["accent"], linewidth=1.2, linestyle=":", alpha=0.8)
        ax2.plot(x_theta, component_2_theta, color=COLORS["accent"], linewidth=1.2, linestyle=":", alpha=0.8)
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax2.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
        ax2.set_title(
            f"{metric}: 2-component peaks "
            f"{float(row['component_1_h']):.1f} h / {float(row['component_2_h']):.1f} h",
            va="bottom",
        )
        ax2.grid(alpha=0.24)

    fig.suptitle(
        "Pooled Circular Bimodality Fits for Final FDR Phase Data",
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_periods_and_phases(
    period_summary: pd.DataFrame,
    periodograms: pd.DataFrame,
    phase_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save grand periodogram and FDR phase-count summary panels."""

    if period_summary.empty or phase_summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax = axes[0]
    period_order = sort_metric_table(period_summary)
    period_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]][: len(period_order)]
    period_labels = period_order["metric"].tolist()
    period_positions = np.arange(1, len(period_labels) + 1)

    box_data = []
    for _, row in period_order.iterrows():
        if periodograms.empty:
            values = np.array([], dtype=float)
        else:
            values = pd.to_numeric(
                periodograms.loc[periodograms["metric"] == row["metric"], "best_period"],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
        if len(values) == 0 and pd.notna(row["weighted_best_period"]):
            values = np.array([float(row["weighted_best_period"])])
        box_data.append(values)

    box = ax.boxplot(
        box_data,
        positions=period_positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.8},
        whiskerprops={"color": COLORS["muted"], "linewidth": 1.2},
        capprops={"color": COLORS["muted"], "linewidth": 1.2},
        boxprops={"edgecolor": "white", "linewidth": 1.1},
    )
    for patch, color in zip(box["boxes"], period_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    for pos, metric, color in zip(period_positions, period_labels, period_colors):
        metric_rows = periodograms[periodograms["metric"] == metric] if not periodograms.empty else pd.DataFrame()
        if metric_rows.empty:
            continue
        values = pd.to_numeric(metric_rows["best_period"], errors="coerce")
        valid = values.notna()
        values = values[valid].to_numpy(dtype=float)
        weights = pd.to_numeric(metric_rows.loc[valid, "valid_players"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
        max_weight = np.nanmax(weights) if np.isfinite(weights).any() else 1.0
        max_weight = max(max_weight, 1.0)
        size = 28 + 58 * np.sqrt(weights / max_weight)
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=size,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label="Server peak" if pos == period_positions[0] else None,
        )

    means = period_order["weighted_best_period"].to_numpy(dtype=float)
    sems = period_order["weighted_sem_best_period"].to_numpy(dtype=float)
    ax.errorbar(
        period_positions,
        means,
        yerr=sems,
        fmt="D",
        markersize=5.8,
        color=COLORS["ink"],
        ecolor=COLORS["ink"],
        elinewidth=1.4,
        capsize=4,
        label="N-weighted mean +/- SEM",
        zorder=4,
    )
    ax.axhline(24.0, color=COLORS["ink"], linestyle="--", linewidth=1.2, label="24 h")

    # Tight, data-driven y-limits so the (outlier-free) peaks fill the panel.
    all_periods = pd.to_numeric(periodograms["best_period"], errors="coerce").dropna() if not periodograms.empty else pd.Series(dtype=float)
    if not all_periods.empty:
        lo = min(all_periods.min(), 24.0)
        hi = max(all_periods.max(), 24.0)
        pad = max((hi - lo) * 0.08, 0.05)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_title("Player Periodogram Peak by Server")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Best period (hours)")
    ax.set_xticks(period_positions)
    ax.set_xticklabels(period_labels)
    ax.legend(frameon=False)
    style_axes(ax, grid_axis="y")

    ax = axes[1]
    phase_order = sort_metric_table(phase_summary)
    ax.bar(
        phase_order["metric"],
        phase_order["weighted_fdr_fraction"],
        color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"]][: len(phase_order)],
        alpha=0.86,
        edgecolor="white",
    )
    ax.set_title("Weighted FDR-Significant Player Fraction")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Fraction of analyzed players")
    style_axes(ax, grid_axis="y")

    fig.suptitle("Grand Within-Subject Rhythm Summary", fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _format_markdown_value(value: Any) -> str:
    """Format values for Markdown tables."""

    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(data: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return a compact Markdown table."""

    if data.empty:
        return ["No rows."]
    columns = [col for col in columns if col in data.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data[columns].iterrows():
        values = [_format_markdown_value(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_grand_markdown(
    grand_dir: Path,
    server_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    circular_summary: pd.DataFrame,
    density_modes: pd.DataFrame,
    pooled_bimodality: pd.DataFrame,
) -> None:
    """Write a tracked Markdown report for the grand analysis."""

    key_metrics = metric_summary[metric_summary["metric"].isin(SUMMARY_COLUMNS[1:])].copy()
    lines = [
        "# Grand Analysis",
        "",
        f"Servers included: {len(server_summary)}",
        "",
        "This is an across-server meta-analysis. Server-level outputs are summarized first, then combined with metric-specific N weights.",
        "",
        "## Key N-Weighted Server Metrics",
        "",
    ]
    lines.extend(
        _markdown_table(
            key_metrics,
            ["metric", "n_servers", "weight_col", "weight_sum", "weighted_mean", "weighted_sd", "min", "max"],
        )
    )
    lines.extend(["", "## Within-Subject Period Peaks", ""])
    lines.extend(
        _markdown_table(
            period_summary,
            [
                "metric",
                "n_servers",
                "weight_col",
                "total_valid_players",
                "weighted_best_period",
                "weighted_sd_best_period",
                "weighted_sem_best_period",
            ],
        )
    )
    lines.extend(["", "## FDR Phase Counts", ""])
    lines.extend(
        _markdown_table(
            phase_summary,
            [
                "metric",
                "n_servers",
                "weight_col",
                "total_players_analyzed",
                "total_fdr_significant",
                "weighted_fdr_fraction",
            ],
        )
    )
    lines.extend(["", "## Circular Model Preference", ""])
    lines.extend(
        _markdown_table(
            circular_summary,
            [
                "metric",
                "n_servers",
                "weight_col",
                "total_fdr_significant",
                "fit_servers",
                "skipped_servers",
                "preferred_1_component_phases",
                "preferred_2_component_phases",
            ],
        )
    )
    lines.extend(["", "## Phase-Count Weighted PC Peak Density", ""])
    lines.extend(
        _markdown_table(
            density_modes,
            [
                "metric",
                "n_servers",
                "total_phases",
                "weighted_peak_hour",
                "kappa",
                "min_phases",
            ],
        )
    )
    lines.extend(["", "## Pooled Circular Bimodality Test", ""])
    lines.extend(
        _markdown_table(
            pooled_bimodality,
            [
                "metric",
                "n_servers",
                "n_phases",
                "preferred",
                "delta_bic_1_minus_2",
                "component_1_h",
                "component_2_h",
                "component_1_weight",
                "component_2_weight",
            ],
        )
    )
    (grand_dir / "grand_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _html_table(data: pd.DataFrame, columns: list[str]) -> str:
    """Return a small HTML table with rounded numeric columns."""

    if data.empty:
        return "<p>No rows.</p>"
    columns = [col for col in columns if col in data.columns]
    table = data[columns].copy()
    return table.to_html(
        index=False,
        border=0,
        classes="data-table",
        justify="left",
        na_rep="",
        float_format=lambda value: f"{value:.3f}",
    )


def write_final_html_report(
    grand_dir: Path,
    metric_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    circular_summary: pd.DataFrame,
    density_modes: pd.DataFrame,
    pooled_bimodality: pd.DataFrame,
) -> None:
    """Write a browser-friendly final report for the grand analysis."""

    bic_rows = pooled_bimodality[pooled_bimodality["status"] == "fit"].copy()
    key_metrics = metric_summary[metric_summary["metric"].isin(SUMMARY_COLUMNS[1:])].copy()

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>League of Legends Grand Rhythm Analysis</title>
  <style>
    :root {{
      --ink: #264653;
      --primary: #1d3557;
      --accent: #e76f51;
      --page-bg: #fcfbf8;
      --panel: #f8f5ef;
      --rule: #d9d1c8;
    }}
    body {{
      margin: 0;
      background: var(--page-bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 28px 56px;
    }}
    h1, h2, h3 {{
      color: var(--primary);
      line-height: 1.15;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 2.1rem;
    }}
    h2 {{
      margin-top: 34px;
      padding-top: 18px;
      border-top: 1px solid var(--rule);
    }}
    p {{
      max-width: 900px;
    }}
    .lead {{
      font-size: 1.05rem;
    }}
    .callout {{
      background: var(--panel);
      border-left: 4px solid var(--accent);
      padding: 14px 16px;
      margin: 20px 0;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--rule);
      background: white;
      margin: 14px 0 22px;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 24px;
      font-size: 0.92rem;
      background: white;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--rule);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: var(--panel);
      color: var(--primary);
      font-weight: 700;
    }}
    code {{
      background: var(--panel);
      padding: 1px 4px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>
<main>
  <h1>League of Legends Grand Rhythm Analysis</h1>
  <p class="lead">Final across-server analysis using N-aware weighting. The headline circular test pools FDR-significant player peak phases from server-metric cells with at least {CIRCULAR_DENSITY_MIN_PHASES} significant phases.</p>

  <div class="callout">
    <strong>Final bimodality result:</strong> PC1 and PC2 both prefer a two-component circular von Mises model by BIC in the pooled final data.
  </div>

  <h2>Pooled Circular Bimodality Fits</h2>
  <p>The histogram bars are the final pooled FDR-significant peak phases. The dashed curve is the one-component von Mises fit; the solid curve is the two-component mixture fit, with dotted component curves.</p>
  <img src="grand_pooled_circular_bimodality_fits.png" alt="Pooled circular bimodality fits with one- and two-component von Mises curves">
  {_html_table(
      bic_rows,
      [
          "metric",
          "n_servers",
          "n_phases",
          "preferred",
          "vm1_bic",
          "vm2_bic",
          "delta_bic_1_minus_2",
          "component_1_h",
          "component_2_h",
          "component_1_weight",
          "component_2_weight",
      ],
  )}

  <h2>Phase-Count Weighted PC Peak Density</h2>
  <p>This companion plot summarizes the same local-hour phase landscape with phase-count weighted smoothed circular densities.</p>
  <img src="grand_pc_peak_density.png" alt="Phase-count weighted PC peak density">
  {_html_table(density_modes, ["metric", "n_servers", "total_phases", "weighted_peak_hour", "weighted_density_peak", "kappa", "min_phases"])}

  <h2>Weighted Rhythm and Phase Summaries</h2>
  <img src="grand_within_subject_summary.png" alt="Weighted within-subject period and phase summary">
  {_html_table(period_summary, ["metric", "n_servers", "weight_col", "total_valid_players", "weighted_best_period", "weighted_sd_best_period", "weighted_sem_best_period"])}
  {_html_table(phase_summary, ["metric", "n_servers", "weight_col", "total_players_analyzed", "total_fdr_significant", "weighted_fdr_fraction"])}

  <h2>Circular Model Preference Across Servers</h2>
  {_html_table(
      circular_summary,
      [
          "metric",
          "n_servers",
          "weight_col",
          "total_fdr_significant",
          "fit_servers",
          "skipped_servers",
          "preferred_1_component_phases",
          "preferred_2_component_phases",
      ],
  )}

  <h2>Key N-Weighted Server Metrics</h2>
  {_html_table(key_metrics, ["metric", "n_servers", "weight_col", "weight_sum", "weighted_mean", "weighted_sd", "min", "max"])}

  <h2>PCA Loading Summaries</h2>
  <img src="grand_performance_pca_loadings.png" alt="N-weighted performance PCA loadings">
  <img src="grand_success_aware_pca_loadings.png" alt="N-weighted success-aware PCA loadings">

  <h2>Local-Hour Win Rate</h2>
  <img src="grand_win_rate_by_local_hour.png" alt="N-weighted local-hour win rate">
</main>
</body>
</html>
"""
    (grand_dir / "grand_final_report.html").write_text(html, encoding="utf-8")


def run_grand_analysis(
    output_root: str | Path = "results",
    platforms: list[str] | None = ANALYSIS_PLATFORMS,
    clean: bool = True,
) -> dict[str, Any]:
    """Run the across-server grand analysis and write `results/GRAND/` outputs."""

    configure_plot_style()
    output_root = Path(output_root)
    grand_dir = output_root / GRAND_DIR_NAME
    if clean:
        clean_grand_outputs(grand_dir)
    grand_dir.mkdir(parents=True, exist_ok=True)

    server_dirs = discover_server_dirs(output_root, platforms=platforms)
    if not server_dirs:
        raise RuntimeError("No per-server analysis_summary.csv files found for grand analysis.")

    server_summary = load_server_summaries(server_dirs)
    server_summary = add_server_weights(server_summary, server_dirs)
    metric_summary = summarize_numeric_columns(server_summary)
    save_table(server_summary, grand_dir / "grand_server_metrics.csv")
    save_table(metric_summary, grand_dir / "grand_metric_summary.csv")

    performance_loadings = load_pca_loadings(server_dirs, "performance_pca_loadings.csv")
    performance_summary = summarize_loadings(performance_loadings, server_summary)
    save_table(performance_summary, grand_dir / "grand_performance_pca_loadings.csv")
    plot_grand_loadings(
        performance_summary,
        "Grand Performance PCA Loadings",
        grand_dir / "grand_performance_pca_loadings.png",
    )

    success_loadings = load_pca_loadings(server_dirs, "success_aware_pca_loadings.csv")
    success_summary = summarize_loadings(success_loadings, server_summary)
    save_table(success_summary, grand_dir / "grand_success_aware_pca_loadings.csv")
    plot_grand_loadings(
        success_summary,
        "Grand Success-Aware PCA Loadings",
        grand_dir / "grand_success_aware_pca_loadings.png",
    )

    local_win = load_local_win_rates(server_dirs)
    local_summary = summarize_local_win_rates(local_win)
    save_table(local_summary, grand_dir / "grand_win_rate_by_local_hour.csv")
    plot_grand_win_rate(local_summary, grand_dir / "grand_win_rate_by_local_hour.png")

    periodograms = load_metric_table(server_dirs, "within_subject_periodogram_summary.csv")
    periodograms = drop_period_outliers(periodograms)
    period_summary = summarize_periodograms(periodograms)
    save_table(period_summary, grand_dir / "grand_within_subject_periodograms.csv")

    phases = load_metric_table(server_dirs, "phase_summary.csv")
    phase_summary = summarize_phase_counts(phases)
    save_table(phase_summary, grand_dir / "grand_phase_summary.csv")
    plot_periods_and_phases(
        period_summary,
        periodograms,
        phase_summary,
        grand_dir / "grand_within_subject_summary.png",
    )

    circular = load_metric_table(server_dirs, "circular_modality_summary.csv")
    circular_summary = summarize_circular_models(circular)
    save_table(circular_summary, grand_dir / "grand_circular_modality_summary.csv")

    density_summary, density_server_curves = load_pc_peak_densities(server_dirs)
    density_modes = summarize_peak_density_modes(density_summary)
    save_table(density_summary, grand_dir / "grand_pc_peak_density.csv")
    save_table(density_server_curves, grand_dir / "grand_pc_peak_density_server_curves.csv")
    save_table(density_modes, grand_dir / "grand_pc_peak_density_modes.csv")
    plot_pc_peak_densities(density_summary, grand_dir / "grand_pc_peak_density.png")

    pooled_phases, pooled_phase_inclusion = load_pooled_significant_phases(server_dirs)
    save_table(pooled_phases, grand_dir / "grand_pooled_phase_values.csv")
    save_table(pooled_phase_inclusion, grand_dir / "grand_pooled_phase_inclusion.csv")

    pooled_bimodality = pooled_circular_bimodality_tests(server_dirs)
    save_table(pooled_bimodality, grand_dir / "grand_pooled_circular_bimodality.csv")
    plot_pooled_circular_bimodality_fits(
        pooled_phases,
        pooled_bimodality,
        grand_dir / "grand_pooled_circular_bimodality_fits.png",
    )

    write_grand_markdown(
        grand_dir,
        server_summary,
        metric_summary,
        period_summary,
        phase_summary,
        circular_summary,
        density_modes,
        pooled_bimodality,
    )
    write_final_html_report(
        grand_dir,
        metric_summary,
        period_summary,
        phase_summary,
        circular_summary,
        density_modes,
        pooled_bimodality,
    )

    return {
        "servers": len(server_dirs),
        "grand_dir": str(grand_dir),
        "figures": len(list(grand_dir.glob("*.png"))),
    }


if __name__ == "__main__":
    result = run_grand_analysis()
    print(f"Grand analysis complete: {result['servers']} servers, {result['figures']} figures")
