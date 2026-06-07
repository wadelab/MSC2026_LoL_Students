"""Generate focused Colab-ready teaching notebooks.

The notebooks are intentionally generated from this script so shared setup
cells stay consistent across the teaching sequence.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent / "teaching"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


DEPENDENCY_CELL = r"""
# Local users should normally use the uv environment from README.md.
# This cell only installs missing packages when the notebook is opened in Colab.
import importlib.util
import subprocess
import sys

MODULE_TO_PACKAGE = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "duckdb": "duckdb",
    "astropy": "astropy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "joblib": "joblib",
}

missing = [
    package
    for module, package in MODULE_TO_PACKAGE.items()
    if importlib.util.find_spec(module) is None
]

if missing:
    print("Installing missing packages:", ", ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("Notebook packages are available.")
"""


PATH_CELL = r"""
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


IN_COLAB = running_in_colab()
if IN_COLAB:
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount("/content/drive")


# Override this if your repository folder has a different Colab/Drive location.
ROOT_OVERRIDE = None
REPO_URL = "https://github.com/wadelab/MSC2026_LoL_Students.git"


def find_repo_root() -> Path | None:
    if ROOT_OVERRIDE is not None:
        candidate = Path(ROOT_OVERRIDE).expanduser()
        if (candidate / "riot_analysis.py").exists():
            return candidate.resolve()
        raise FileNotFoundError(f"ROOT_OVERRIDE does not contain riot_analysis.py: {candidate}")

    candidates = list(Path.cwd().resolve().parents)
    candidates.insert(0, Path.cwd().resolve())
    candidates.extend(
        [
            Path("/content/MSC2026_LoL_Students"),
            Path("/content/drive/MyDrive/MSC2026_LoL_Students"),
            Path("/content/drive/Shareddrives/MSc_2026_Riot/MSC2026_LoL_Students"),
        ]
    )
    for candidate in candidates:
        if (candidate / "riot_analysis.py").exists():
            return candidate.resolve()
    return None


ROOT = find_repo_root()
if ROOT is None and IN_COLAB:
    clone_target = Path("/content/MSC2026_LoL_Students")
    if not clone_target.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(clone_target)], check=True)
    ROOT = find_repo_root()

if ROOT is None:
    raise FileNotFoundError(
        "Could not find riot_analysis.py. Set ROOT_OVERRIDE to the repository folder."
    )

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from riot_analysis import resolve_analysis_db_file

ANALYSIS_DB_FILE = resolve_analysis_db_file() if IN_COLAB else ROOT / "riot_local.duckdb"

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
plt.rcParams["figure.dpi"] = 120

print(f"Repository root: {ROOT}")
print(f"Running in Colab: {IN_COLAB}")
print(f"Persistent DuckDB cache: {ANALYSIS_DB_FILE}")
"""


def common_setup_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            ## Setup

            In Colab, the notebook mounts Google Drive and looks for the raw Riot
            Parquet at the same shared-drive path used by the other release
            notebooks: `/content/drive/Shareddrives/MSc_2026_Riot/db/riotData.parquet`.
            The materialized `hourly_agg` cache is stored beside it as
            `riot_local.duckdb`. The first run creates the cache; later Colab
            sessions reuse it without rebuilding the aggregate table.

            The notebook also needs the repository Python files. If they are not
            already present in the runtime or Drive, the setup cell tries to clone
            the student repository into `/content/MSC2026_LoL_Students`.
            """
        ),
        code(DEPENDENCY_CELL),
        code(PATH_CELL),
    ]


def write_notebook(filename: str, cells: list[nbf.NotebookNode]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    cell_id_prefix = Path(filename).stem.replace("_", "-")
    for index, cell in enumerate(cells):
        cell["id"] = f"{cell_id_prefix}-{index:02d}"

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    path = NOTEBOOK_DIR / filename
    nbf.write(notebook, path)
    print(f"Wrote {path.relative_to(Path.cwd())}")


def hourly_activity_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # EUW Hourly Activity and Mean Game Duration

            This notebook starts with the simplest time series: how many game
            records appear in each hour, and the mean game duration in those same
            hourly bins. It is a useful first check before fitting rhythms or PCA.
            """
        ),
        *common_setup_cells(),
        code(
            r"""
            from riot_analysis import (
                AnalysisConfig,
                available_platforms,
                configure_plot_style,
                connect_analysis_database,
                filter_hourly_window,
                load_hourly_target,
                platform_overview,
                style_axes,
            )
            from server_timezones import hour_idx_to_local_hour_of_day

            configure_plot_style()

            PLATFORM = "EUW1"
            DB_FILE = ANALYSIS_DB_FILE
            PARQUET_FILE = None  # Auto-detects RIOT_DB_PATH, this server, or the Colab shared Drive path.
            REBUILD_HOURLY_AGG = False

            config = AnalysisConfig(
                platform=PLATFORM,
                target_col="TIMEPLAYED",
                max_hour_limit=10000,
                output_root=ROOT / "results",
            )

            conn = connect_analysis_database(
                DB_FILE,
                parquet_file=PARQUET_FILE,
                rebuild_hourly_agg=REBUILD_HOURLY_AGG,
            )

            print("Available platforms:", available_platforms(conn))
            display(platform_overview(conn).head(12))
            """
        ),
        md(
            """
            ## Load the hourly series

            `n` is the number of game records in an hourly bin. `target_mean` is
            the hourly mean of `TIMEPLAYED`, which we convert to minutes.
            """
        ),
        code(
            r"""
            hourly = load_hourly_target(conn, config)
            hourly = filter_hourly_window(hourly, config.max_hour_limit)
            hourly = hourly.sort_values("hour_idx").reset_index(drop=True)

            hourly["hours_since_start"] = hourly["hour_idx"] - hourly["hour_idx"].min()
            hourly["date_utc"] = pd.to_datetime(hourly["hour_idx"], unit="h", origin="unix", utc=True)
            hourly["local_hour"] = hour_idx_to_local_hour_of_day(hourly["hour_idx"], PLATFORM)
            hourly["game_records"] = hourly["n"]
            hourly["mean_duration_min"] = hourly["target_mean"] / 60.0
            hourly["game_records_7d_mean"] = hourly["game_records"].rolling(
                24 * 7,
                center=True,
                min_periods=24,
            ).mean()

            print(f"{PLATFORM}: {len(hourly):,} hourly bins after filtering")
            display(hourly.head())
            """
        ),
        md(
            """
            ## Plot activity across calendar time

            The weekly rolling line is not a model. It is only a visual guide for
            slow changes in game volume.
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)

            ax = axes[0]
            ax.plot(hourly["date_utc"], hourly["game_records"], color="#8d99ae", alpha=0.35, linewidth=0.7, label="Hourly game records")
            ax.plot(hourly["date_utc"], hourly["game_records_7d_mean"], color="#1d3557", linewidth=2.0, label="7-day rolling mean")
            ax.set_title(f"Game records per hour over time ({PLATFORM})")
            ax.set_ylabel("Game records")
            ax.legend(frameon=False)
            style_axes(ax)

            ax = axes[1]
            ax.plot(hourly["date_utc"], hourly["mean_duration_min"], color="#2a9d8f", alpha=0.85, linewidth=1.0)
            ax.set_title(f"Mean game duration per hour over time ({PLATFORM})")
            ax.set_xlabel("UTC date")
            ax.set_ylabel("Mean TIMEPLAYED (minutes)")
            style_axes(ax)

            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Fold into the 24-hour local cycle

            Folding by local hour makes the daily pattern easier to see. The game
            duration curve uses `n` as weights so sparse hourly bins contribute
            less.
            """
        ),
        code(
            r"""
            volume_local = (
                hourly.groupby("local_hour", as_index=False)["game_records"]
                .mean()
                .set_index("local_hour")
                .reindex(range(24))
                .reset_index()
            )

            duration_rows = []
            for local_hour, group in hourly.groupby("local_hour"):
                duration_rows.append(
                    {
                        "local_hour": local_hour,
                        "mean_duration_min": np.average(group["mean_duration_min"], weights=group["game_records"]),
                    }
                )
            duration_local = (
                pd.DataFrame(duration_rows)
                .set_index("local_hour")
                .reindex(range(24))
                .reset_index()
            )

            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

            ax = axes[0]
            ax.plot(volume_local["local_hour"], volume_local["game_records"], marker="o", color="#1d3557", linewidth=2.0)
            ax.set_title(f"Mean hourly game records by local hour ({PLATFORM})")
            ax.set_xlabel("Local hour")
            ax.set_ylabel("Mean game records")
            ax.set_xticks(range(0, 24, 2))
            style_axes(ax, grid_axis="y")

            ax = axes[1]
            ax.plot(duration_local["local_hour"], duration_local["mean_duration_min"], marker="o", color="#e76f51", linewidth=2.0)
            ax.set_title(f"Mean game duration by local hour ({PLATFORM})")
            ax.set_xlabel("Local hour")
            ax.set_ylabel("Mean TIMEPLAYED (minutes)")
            ax.set_xticks(range(0, 24, 2))
            style_axes(ax, grid_axis="y")

            fig.tight_layout()
            plt.show()

            display(volume_local.merge(duration_local, on="local_hour"))
            conn.close()
            """
        ),
    ]


def pca_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # EUW Performance PCA

            This notebook builds the hourly performance PCA used later for PC
            periodograms and player-level projections. The focus is on what goes
            into the PCA, how much variance each component explains, and how the
            loading vectors should be read.
            """
        ),
        *common_setup_cells(),
        code(
            r"""
            from riot_analysis import (
                AnalysisConfig,
                GOOD_PCA_COLS,
                add_time_normalized_features,
                compute_pca,
                configure_plot_style,
                connect_analysis_database,
                filter_hourly_window,
                filter_metric_outliers,
                load_hourly_metrics,
                style_axes,
            )
            from server_timezones import hour_idx_to_local_hour_of_day

            configure_plot_style()

            PLATFORM = "EUW1"
            DB_FILE = ANALYSIS_DB_FILE
            PARQUET_FILE = None

            config = AnalysisConfig(platform=PLATFORM, max_hour_limit=10000, output_root=ROOT / "results")
            conn = connect_analysis_database(DB_FILE, parquet_file=PARQUET_FILE)
            """
        ),
        md(
            """
            ## Build time-normalized hourly features

            The raw counts are divided by mean `TIMEPLAYED` so PCA reflects
            performance rate rather than simply longer games.
            """
        ),
        code(
            r"""
            hourly_metrics = load_hourly_metrics(conn, PLATFORM)
            hourly_metrics = filter_hourly_window(hourly_metrics, config.max_hour_limit)
            hourly_metrics, numeric_cols = add_time_normalized_features(hourly_metrics)

            rows_before_outlier_filter = len(hourly_metrics)
            hourly_metrics = filter_metric_outliers(hourly_metrics, numeric_cols)

            print("PCA input columns:", numeric_cols)
            print(f"Rows before outlier filter: {rows_before_outlier_filter:,}")
            print(f"Rows after outlier filter:  {len(hourly_metrics):,}")
            display(hourly_metrics[["hour_idx", "n", *numeric_cols]].head())
            """
        ),
        md(
            """
            ## Run PCA and inspect component loadings

            The helper uses SVD on standardized features, then orients component
            signs so positive performance-rate variables tend to load positively.
            """
        ),
        code(
            r"""
            pca = compute_pca(hourly_metrics, numeric_cols, GOOD_PCA_COLS)
            loadings = pca["loadings"].iloc[:3].copy()
            explained = pd.DataFrame(
                {
                    "component": [f"PC{i + 1}" for i in range(6)],
                    "explained_variance": pca["explained"][:6],
                    "cumulative": np.cumsum(pca["explained"][:6]),
                }
            )

            pc_scores = hourly_metrics.loc[pca["features"].index, ["hour_idx"]].reset_index(drop=True)
            for i in range(3):
                pc_scores[f"PC{i + 1}"] = pca["scores"][:, i]
            pc_scores["hours_since_start"] = pc_scores["hour_idx"] - pc_scores["hour_idx"].min()
            pc_scores["local_hour"] = hour_idx_to_local_hour_of_day(pc_scores["hour_idx"], PLATFORM)

            display(explained)
            display(loadings)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

            ax = axes[0]
            ax.bar(explained["component"], explained["explained_variance"], color="#2a9d8f", alpha=0.9)
            ax.plot(explained["component"], explained["cumulative"], color="#1d3557", marker="o", linewidth=2.0, label="Cumulative")
            ax.set_title(f"PCA explained variance ({PLATFORM})")
            ax.set_ylabel("Fraction of variance")
            ax.legend(frameon=False)
            style_axes(ax, grid_axis="y")

            ax = axes[1]
            image = ax.imshow(loadings, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_title("First three PCA loading vectors")
            ax.set_yticks(range(len(loadings.index)))
            ax.set_yticklabels(loadings.index)
            ax.set_xticks(range(len(loadings.columns)))
            ax.set_xticklabels(loadings.columns, rotation=45, ha="right")
            fig.colorbar(image, ax=ax, label="Loading")

            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Plot loadings and PC scores

            A loading is the weight assigned to a standardized input variable.
            A score is the value of that component for one hourly bin.
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=False)
            for ax, component in zip(axes, loadings.index):
                ordered = loadings.loc[component].sort_values()
                colors = ["#e76f51" if value >= 0 else "#1d3557" for value in ordered]
                ax.barh(ordered.index, ordered.values, color=colors, alpha=0.9)
                ax.axvline(0, color="#264653", linewidth=1.0)
                ax.set_title(f"{component} loadings")
                ax.set_xlabel("Loading")
                style_axes(ax, grid_axis="x")
            fig.tight_layout()
            plt.show()

            fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True)
            for ax, component in zip(axes, ["PC1", "PC2", "PC3"]):
                ax.plot(pc_scores["hours_since_start"], pc_scores[component], color="#2a9d8f", linewidth=0.9, alpha=0.9)
                ax.axhline(0, color="#8d99ae", linestyle="--", linewidth=1.0)
                ax.set_title(f"{component} score over hourly bins")
                ax.set_ylabel("Score")
                style_axes(ax)
            axes[-1].set_xlabel("Hours since first retained bin")
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Local-hour PC profiles

            This is a descriptive fold of the hourly scores. It does not replace
            the periodogram, but it helps interpret a 24-hour signal if one is
            present.
            """
        ),
        code(
            r"""
            local_pc = pc_scores.groupby("local_hour", as_index=False)[["PC1", "PC2", "PC3"]].mean()
            local_pc = local_pc.set_index("local_hour").reindex(range(24)).reset_index()

            fig, ax = plt.subplots(figsize=(12, 4.8))
            for component, color in zip(["PC1", "PC2", "PC3"], ["#1d3557", "#2a9d8f", "#e76f51"]):
                ax.plot(local_pc["local_hour"], local_pc[component], marker="o", linewidth=2.0, label=component, color=color)
            ax.axhline(0, color="#8d99ae", linestyle="--", linewidth=1.0)
            ax.set_title(f"Mean PC score by local hour ({PLATFORM})")
            ax.set_xlabel("Local hour")
            ax.set_ylabel("Mean score")
            ax.set_xticks(range(0, 24, 2))
            ax.legend(frameon=False)
            style_axes(ax, grid_axis="y")
            plt.show()

            display(local_pc)
            conn.close()
            """
        ),
    ]


def pc_periodogram_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # EUW Periodograms of Hourly PCA Scores

            This notebook asks whether the hourly PCA scores have regular cycles.
            It computes Lomb-Scargle periodograms for PC1, PC2, and PC3 over a
            6-48 hour period range.
            """
        ),
        *common_setup_cells(),
        code(
            r"""
            from riot_analysis import (
                AnalysisConfig,
                GOOD_PCA_COLS,
                add_time_normalized_features,
                compute_pca,
                configure_plot_style,
                connect_analysis_database,
                filter_hourly_window,
                filter_metric_outliers,
                fixed_period_lag_table,
                load_hourly_metrics,
                lomb_scargle_summary,
                period_grid,
                style_axes,
            )

            configure_plot_style()

            PLATFORM = "EUW1"
            DB_FILE = ANALYSIS_DB_FILE
            PARQUET_FILE = None

            config = AnalysisConfig(
                platform=PLATFORM,
                max_hour_limit=10000,
                min_period_h=6,
                max_period_h=48,
                period_step_h=0.25,
                output_root=ROOT / "results",
            )
            conn = connect_analysis_database(DB_FILE, parquet_file=PARQUET_FILE)
            """
        ),
        code(
            r"""
            def hourly_pca_scores(conn, platform: str, config: AnalysisConfig) -> tuple[pd.DataFrame, dict]:
                hourly_metrics = load_hourly_metrics(conn, platform)
                hourly_metrics = filter_hourly_window(hourly_metrics, config.max_hour_limit)
                hourly_metrics, numeric_cols = add_time_normalized_features(hourly_metrics)
                hourly_metrics = filter_metric_outliers(hourly_metrics, numeric_cols)
                pca = compute_pca(hourly_metrics, numeric_cols, GOOD_PCA_COLS)

                scores = hourly_metrics.loc[pca["features"].index, ["hour_idx"]].reset_index(drop=True)
                scores["hours_since_start"] = scores["hour_idx"] - scores["hour_idx"].min()
                for i in range(3):
                    scores[f"PC{i + 1}"] = pca["scores"][:, i]
                return scores, pca


            pc_scores, pca = hourly_pca_scores(conn, PLATFORM, config)
            display(pc_scores.head())
            """
        ),
        md(
            """
            ## Compute periodograms

            Lomb-Scargle is useful here because it handles gaps in the hourly
            sequence without requiring interpolation.
            """
        ),
        code(
            r"""
            frequency, period = period_grid(config)
            periodogram_curves = {}
            summary_rows = []
            fixed_period_tables = []

            for component in ["PC1", "PC2", "PC3"]:
                result = lomb_scargle_summary(
                    pc_scores["hour_idx"].to_numpy(dtype=float),
                    pc_scores[component].to_numpy(dtype=float),
                    frequency,
                    period,
                )
                periodogram_curves[component] = result
                summary_rows.append(
                    {
                        "component": component,
                        "best_period_h": result["best_period"],
                        "power_at_24h": result["power_24"],
                    }
                )

                fit_table = fixed_period_lag_table(
                    pc_scores["hours_since_start"].to_numpy(dtype=float),
                    pc_scores[component].to_numpy(dtype=float),
                    periods_h=(12.0, 24.0, 36.0, 48.0),
                    robust=True,
                )
                fit_table.insert(0, "component", component)
                fixed_period_tables.append(fit_table)

            periodogram_summary = pd.DataFrame(summary_rows)
            fixed_period_summary = pd.concat(fixed_period_tables, ignore_index=True)

            display(periodogram_summary)
            display(fixed_period_summary)
            """
        ),
        md(
            """
            ## Plot a time-window and its periodogram

            The left panels show the first three weeks of retained hourly bins.
            The right panels show the periodogram over the full filtered window.
            """
        ),
        code(
            r"""
            plot_window_hours = 24 * 21
            window = pc_scores[pc_scores["hours_since_start"] <= plot_window_hours]

            fig, axes = plt.subplots(3, 2, figsize=(15, 10.5))
            for row, component in enumerate(["PC1", "PC2", "PC3"]):
                ax = axes[row, 0]
                ax.plot(window["hours_since_start"], window[component], color="#2a9d8f", linewidth=1.1)
                ax.axhline(0, color="#8d99ae", linestyle="--", linewidth=1.0)
                ax.set_title(f"{component}: first {plot_window_hours // 24} days")
                ax.set_ylabel("Score")
                style_axes(ax)

                ax = axes[row, 1]
                result = periodogram_curves[component]
                ax.plot(period, result["power"], color="#1d3557", linewidth=2.0)
                ax.axvline(24.0, color="#e76f51", linestyle="--", linewidth=1.3, label="24 h")
                ax.axvline(result["best_period"], color="#264653", linestyle=":", linewidth=1.3, label=f"Peak {result['best_period']:.2f} h")
                ax.set_title(f"{component}: Lomb-Scargle periodogram")
                ax.set_xlabel("Period (hours)")
                ax.set_ylabel("Power")
                ax.legend(frameon=False)
                style_axes(ax)

            axes[-1, 0].set_xlabel("Hours since first retained bin")
            fig.tight_layout()
            plt.show()
            conn.close()
            """
        ),
    ]


def mean_vs_player_periodogram_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # EUW Mean-Data Periodograms vs Mean Player Periodograms

            This notebook compares two different summaries:

            - periodogram of the hourly mean score, after averaging games within each hour
            - mean of the individual player periodograms, after computing one periodogram per player

            These are not equivalent operations. The comparison shows whether a
            pooled rhythm survives averaging across players, and whether player
            rhythms are coherent in phase.
            """
        ),
        *common_setup_cells(),
        code(
            r"""
            from riot_analysis import (
                AnalysisConfig,
                GOOD_PCA_COLS,
                add_time_normalized_features,
                average_player_periodograms,
                compute_pca,
                configure_plot_style,
                connect_analysis_database,
                filter_hourly_window,
                filter_metric_outliers,
                load_hourly_metrics,
                load_top_players,
                lomb_scargle_summary,
                period_grid,
                project_player_pca,
                style_axes,
            )

            configure_plot_style()

            PLATFORM = "EUW1"
            TOP_N_PLAYERS = 250
            DB_FILE = ANALYSIS_DB_FILE
            PARQUET_FILE = None

            config = AnalysisConfig(
                platform=PLATFORM,
                top_n_players=TOP_N_PLAYERS,
                max_hour_limit=10000,
                min_period_h=6,
                max_period_h=48,
                player_period_step_h=0.5,
                n_jobs=2 if IN_COLAB else 8,
                output_root=ROOT / "results",
            )
            conn = connect_analysis_database(DB_FILE, parquet_file=PARQUET_FILE)
            """
        ),
        md(
            """
            ## Build the hourly PCA basis, then project player games

            We fit PCA on hourly aggregate rates, then project individual player
            games into that same PCA space. This keeps PC1 and PC2 comparable to
            the rest of the workflow.
            """
        ),
        code(
            r"""
            hourly_metrics = load_hourly_metrics(conn, PLATFORM)
            hourly_metrics = filter_hourly_window(hourly_metrics, config.max_hour_limit)
            hourly_metrics, numeric_cols = add_time_normalized_features(hourly_metrics)
            hourly_metrics = filter_metric_outliers(hourly_metrics, numeric_cols)
            pca = compute_pca(hourly_metrics, numeric_cols, GOOD_PCA_COLS)

            top_players, player_data = load_top_players(
                conn,
                PLATFORM,
                TOP_N_PLAYERS,
                min_hour_idx=int(hourly_metrics["hour_idx"].min()),
                max_hour_idx=int(hourly_metrics["hour_idx"].max()),
            )
            player_data = project_player_pca(player_data, numeric_cols, pca)
            player_data["TIMESTAMP"] = pd.to_numeric(player_data["TIMESTAMP"], errors="coerce")
            player_data["hour_idx"] = np.floor(player_data["TIMESTAMP"] / 3600000.0)

            metric_map = {
                "PC1": "perf_factor_pc1",
                "PC2": "perf_factor_pc2",
                "DeltaMMR": "delta_mmr",
            }

            print(f"Loaded {len(top_players):,} players and {len(player_data):,} game rows")
            display(top_players.head(10))
            """
        ),
        md(
            """
            ## Compute the two periodogram summaries

            The mean-data periodogram is computed after grouping the same selected
            players into hourly means. The individual summary first computes one
            periodogram per player and then averages the powers.
            """
        ),
        code(
            r"""
            frequency, period = period_grid(config, player=True)


            def aggregate_periodogram(data: pd.DataFrame, score_col: str) -> dict | None:
                grouped = (
                    data.dropna(subset=["hour_idx", score_col])
                    .groupby("hour_idx", as_index=False)[score_col]
                    .mean()
                    .sort_values("hour_idx")
                )
                if len(grouped) <= 10 or grouped[score_col].std() == 0:
                    return None
                result = lomb_scargle_summary(
                    grouped["hour_idx"].to_numpy(dtype=float),
                    grouped[score_col].to_numpy(dtype=float),
                    frequency,
                    period,
                )
                result["n_hour_bins"] = len(grouped)
                return result


            mean_data_periodograms = {
                label: aggregate_periodogram(player_data, score_col)
                for label, score_col in metric_map.items()
            }

            individual_periodograms = average_player_periodograms(player_data, metric_map, config)

            rows = []
            for label in metric_map:
                mean_result = mean_data_periodograms[label]
                individual_result = individual_periodograms[label]
                rows.append(
                    {
                        "metric": label,
                        "mean_data_hour_bins": None if mean_result is None else mean_result["n_hour_bins"],
                        "mean_data_best_period_h": None if mean_result is None else mean_result["best_period"],
                        "individual_valid_players": individual_result["valid_players"],
                        "individual_mean_best_period_h": individual_result["best_period"],
                    }
                )

            comparison_summary = pd.DataFrame(rows)
            display(comparison_summary)
            """
        ),
        md(
            """
            ## Plot normalized curves and their difference

            Lomb-Scargle power scales with the variance of the series being
            analyzed, so the comparison below normalizes each curve by its own
            maximum. The difference plot is therefore about shape and peak
            location, not absolute power.
            """
        ),
        code(
            r"""
            def normalize_power(power):
                power = np.asarray(power, dtype=float)
                max_power = np.nanmax(power)
                if not np.isfinite(max_power) or max_power <= 0:
                    return np.full_like(power, np.nan)
                return power / max_power


            fig, axes = plt.subplots(len(metric_map), 2, figsize=(15, 4.5 * len(metric_map)), sharex=True)
            axes = np.atleast_2d(axes)
            difference_rows = []

            for row, label in enumerate(metric_map):
                mean_result = mean_data_periodograms[label]
                individual_result = individual_periodograms[label]
                ax_curve = axes[row, 0]
                ax_diff = axes[row, 1]

                if mean_result is None or individual_result["mean_power"] is None:
                    ax_curve.set_title(f"{label}: missing periodogram")
                    ax_diff.set_title(f"{label}: missing difference")
                    continue

                mean_scaled = normalize_power(mean_result["power"])
                individual_scaled = normalize_power(individual_result["mean_power"])
                difference = mean_scaled - individual_scaled
                largest_gap_idx = int(np.nanargmax(np.abs(difference)))

                ax_curve.plot(period, mean_scaled, color="#1d3557", linewidth=2.2, label="Periodogram of hourly mean")
                ax_curve.plot(period, individual_scaled, color="#e76f51", linewidth=2.0, label="Mean of player periodograms")
                ax_curve.axvline(24.0, color="#8d99ae", linestyle="--", linewidth=1.1)
                ax_curve.set_title(f"{label}: normalized periodogram summaries")
                ax_curve.set_ylabel("Power / max power")
                ax_curve.legend(frameon=False)
                style_axes(ax_curve)

                ax_diff.plot(period, difference, color="#2a9d8f", linewidth=2.0)
                ax_diff.axhline(0.0, color="#8d99ae", linestyle="--", linewidth=1.0)
                ax_diff.axvline(period[largest_gap_idx], color="#264653", linestyle=":", linewidth=1.2)
                ax_diff.set_title(f"{label}: hourly-mean minus mean-player power")
                ax_diff.set_ylabel("Normalized difference")
                style_axes(ax_diff)

                difference_rows.append(
                    {
                        "metric": label,
                        "largest_absolute_difference_period_h": period[largest_gap_idx],
                        "largest_absolute_difference": difference[largest_gap_idx],
                        "difference_at_24h": difference[int(np.argmin(np.abs(period - 24.0)))],
                    }
                )

            for ax in axes[-1, :]:
                ax.set_xlabel("Period (hours)")

            fig.tight_layout()
            plt.show()

            display(pd.DataFrame(difference_rows))
            conn.close()
            """
        ),
    ]


def main() -> None:
    notebooks = {
        "01_euw_hourly_activity.ipynb": hourly_activity_notebook(),
        "02_euw_pca_analysis.ipynb": pca_notebook(),
        "03_euw_pc_periodograms.ipynb": pc_periodogram_notebook(),
        "04_euw_mean_vs_player_periodograms.ipynb": mean_vs_player_periodogram_notebook(),
    }
    for filename, cells in notebooks.items():
        write_notebook(filename, cells)


if __name__ == "__main__":
    main()
