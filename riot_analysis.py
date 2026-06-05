"""Reusable League of Legends rhythm analysis helpers.

This module is a script-friendly version of the teaching notebook workflow.
It keeps the main analysis steps explicit:

- hourly target and win-rate rhythm checks
- performance PCA and success-aware PCA
- selected-player within-subject periodograms
- per-player 24 h phase extraction
- circular one-vs-two component von Mises comparisons
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from astropy.timeseries import LombScargle
from joblib import Parallel, delayed
from scipy.stats import vonmises

from server_timezones import utc_offset_hours


COLORS = {
    "ink": "#264653",
    "primary": "#1d3557",
    "secondary": "#2a9d8f",
    "accent": "#e76f51",
    "highlight": "#f4a261",
    "muted": "#8d99ae",
    "panel": "#f8f5ef",
}

AGG_COL_MAP = {
    "NEUTRALCREEP": "neutralcreep_mean",
    "ENEMYCREEP": "enemycreep_mean",
    "GOLD": "gold_mean",
    "DAMDEALT": "damdealt_mean",
    "TIMEDEAD": "timedead_mean",
    "TIMEPLAYED": "timeplayed_mean",
    "KILLS": "kills_mean",
    "DEATHS": "deaths_mean",
    "ASSISTS": "assists_mean",
}

PCA_RATE_MAP = {
    "neutralcreep_per_min": "neutralcreep_mean",
    "enemycreep_per_min": "enemycreep_mean",
    "gold_per_min": "gold_mean",
    "damdealt_per_min": "damdealt_mean",
    "kills_per_min": "kills_mean",
    "deaths_per_min": "deaths_mean",
    "assists_per_min": "assists_mean",
}

PLAYER_PCA_FEATURE_MAP = {
    "neutralcreep_per_min": "NEUTRALCREEP",
    "enemycreep_per_min": "ENEMYCREEP",
    "gold_per_min": "GOLD",
    "damdealt_per_min": "DAMDEALT",
    "kills_per_min": "KILLS",
    "deaths_per_min": "DEATHS",
    "assists_per_min": "ASSISTS",
}

GOOD_PCA_COLS = ["gold_per_min", "damdealt_per_min", "kills_per_min", "assists_per_min"]

SERVER_RIOT_PARQUET = Path("/raid/data/riot/riotData.parquet")
COLAB_RIOT_PARQUET = Path("/content/drive/Shareddrives/MSc_2026_Riot/db/riotData.parquet")
RIOT_PARQUET_ENV_VARS = ("RIOT_DB_PATH", "RIOT_PARQUET_PATH")


@dataclass
class AnalysisConfig:
    """Configuration for one server/platform analysis run."""

    platform: str
    target_col: str = "TIMEPLAYED"
    top_n_players: int = 1000
    max_hour_limit: int = 5000
    min_period_h: float = 6.0
    max_period_h: float = 48.0
    period_step_h: float = 1.0
    player_period_step_h: float = 1.0
    n_freq: int = 500
    player_n_freq: int = 2000
    phase_alpha: float = 0.05
    n_jobs: int = 8
    output_root: Path = Path("results")


def configure_plot_style() -> None:
    """Apply the notebook plot style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfbf8",
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": "#d9d1c8",
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": "#cfc7bc",
            "grid.linestyle": "--",
            "grid.alpha": 0.28,
            "savefig.facecolor": "#fcfbf8",
        }
    )


def style_axes(ax: Any, grid_axis: str = "both") -> Any:
    """Apply the shared notebook axis style."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d9d1c8")
    ax.spines["bottom"].set_color("#d9d1c8")
    ax.grid(True, axis=grid_axis)
    return ax


def styled_hist(ax: Any, data: Any, bins: Any = 50, variant: str = "secondary", **kwargs: Any) -> Any:
    """Draw a histogram using the shared notebook style."""

    hist_kwargs = {
        "color": COLORS.get(variant, COLORS["secondary"]),
        "alpha": 0.85,
        "edgecolor": "white",
    }
    hist_kwargs.update(kwargs)
    ax.hist(data, bins=bins, **hist_kwargs)
    style_axes(ax, grid_axis="y")
    return ax


def running_in_colab() -> bool:
    """Return True when the code is running in a Google Colab runtime."""

    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def mount_colab_drive() -> None:
    """Mount Google Drive in Colab so the shared Riot Parquet can be found."""

    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount("/content/drive")


def unique_paths(paths: list[Path]) -> list[Path]:
    """Return paths in first-seen order without duplicates."""

    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_riot_parquet(parquet_file: str | Path | None = None, *, mount_drive: bool = True) -> Path:
    """Find the raw Riot Parquet on this server or in the Colab shared drive."""

    if parquet_file is not None:
        path = Path(parquet_file).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Explicit Riot Parquet path does not exist: {path}")

    candidates = []
    for env_var in RIOT_PARQUET_ENV_VARS:
        env_value = os.environ.get(env_var)
        if env_value:
            candidates.append(Path(env_value).expanduser())

    candidates.extend([SERVER_RIOT_PARQUET, COLAB_RIOT_PARQUET])
    candidates = unique_paths(candidates)

    for path in candidates:
        if path.exists():
            return path

    mount_error = None
    if mount_drive and running_in_colab():
        try:
            mount_colab_drive()
        except Exception as exc:  # pragma: no cover - depends on Colab runtime UI.
            mount_error = exc
        if COLAB_RIOT_PARQUET.exists():
            return COLAB_RIOT_PARQUET

    checked = "\n".join(f"  - {path}" for path in candidates)
    message = (
        "Could not locate riotData.parquet. Set RIOT_DB_PATH or RIOT_PARQUET_PATH, "
        "or place the file at one of:\n"
        f"{checked}"
    )
    if mount_error is not None:
        message += f"\nColab Drive mount failed with: {mount_error}"
    raise FileNotFoundError(message)


def duckdb_string_literal(value: str | Path) -> str:
    """Return a single-quoted SQL literal for DuckDB path strings."""

    return "'" + str(value).replace("'", "''") + "'"


def duckdb_relation_exists(conn: duckdb.DuckDBPyConnection, relation_name: str) -> bool:
    """Return True if a table or view exists in the active DuckDB file."""

    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = ?;
        """,
        [relation_name],
    ).fetchone()
    return bool(row and row[0])


def duckdb_relation_columns(conn: duckdb.DuckDBPyConnection, relation_name: str) -> set[str]:
    """Return lower-case column names for a DuckDB table or view."""

    safe_name = relation_name.replace("'", "''")
    rows = conn.execute(f"PRAGMA table_info('{safe_name}')").fetchall()
    return {row[1].lower() for row in rows}


def create_riot_view(conn: duckdb.DuckDBPyConnection, parquet_path: Path) -> None:
    """Point the DuckDB `riotData` view at the resolved raw Parquet file."""

    conn.execute(
        f"CREATE OR REPLACE VIEW riotData AS "
        f"SELECT * FROM read_parquet({duckdb_string_literal(parquet_path)})"
    )


def build_hourly_agg_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Materialize hourly platform aggregates used by the analysis workflow."""

    conn.execute("DROP TABLE IF EXISTS hourly_agg")
    conn.execute(
        """
        CREATE TABLE hourly_agg AS
        SELECT
          PLATFORMID AS platformid,
          CAST(FLOOR(CAST(TIMESTAMP AS DOUBLE) / 3600000.0) AS BIGINT) AS hour_idx,
          AVG(CAST(NULLIF(NEUTRALCREEP, '') AS DOUBLE)) AS neutralcreep_mean,
          AVG(CAST(NULLIF(ENEMYCREEP, '') AS DOUBLE)) AS enemycreep_mean,
          AVG(CAST(NULLIF(GOLD, '') AS DOUBLE)) AS gold_mean,
          AVG(CAST(NULLIF(DAMDEALT, '') AS DOUBLE)) AS damdealt_mean,
          AVG(CAST(NULLIF(TIMEDEAD, '') AS DOUBLE)) AS timedead_mean,
          AVG(CAST(NULLIF(TIMEPLAYED, '') AS DOUBLE)) AS timeplayed_mean,
          AVG(CAST(NULLIF(KILLS, '') AS DOUBLE)) AS kills_mean,
          AVG(CAST(NULLIF(DEATHS, '') AS DOUBLE)) AS deaths_mean,
          AVG(CAST(NULLIF(ASSISTS, '') AS DOUBLE)) AS assists_mean,
          COUNT(*) AS n
        FROM riotData
        WHERE PLATFORMID IS NOT NULL
          AND TIMESTAMP IS NOT NULL
        GROUP BY PLATFORMID, hour_idx;
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hourly_agg_platform_hour ON hourly_agg(platformid, hour_idx)")


def ensure_hourly_agg_table(
    conn: duckdb.DuckDBPyConnection,
    *,
    rebuild: bool = False,
    verbose: bool = True,
) -> bool:
    """Create hourly_agg when missing, or rebuild it when requested."""

    required_columns = {"platformid", "hour_idx", "n"} | {
        col.lower() for col in AGG_COL_MAP.values()
    }
    exists = duckdb_relation_exists(conn, "hourly_agg")

    if exists and not rebuild:
        existing_columns = duckdb_relation_columns(conn, "hourly_agg")
        if required_columns.issubset(existing_columns):
            if verbose:
                print("hourly_agg table found; using existing aggregate table.", flush=True)
            return False
        if verbose:
            print("hourly_agg exists but is missing expected columns; rebuilding it.", flush=True)

    if verbose:
        if rebuild and exists:
            print("Rebuilding hourly_agg table from riotData. This may take a few minutes.", flush=True)
        elif not exists:
            print("Building hourly_agg table from riotData. This may take a few minutes.", flush=True)

    build_hourly_agg_table(conn)
    if verbose:
        print("hourly_agg rebuilt and indexed.", flush=True)
    return True


def existing_database_is_usable(db_file: str | Path) -> bool:
    """Return True when an existing DuckDB has readable raw and aggregate relations."""

    try:
        conn = duckdb.connect(str(db_file), read_only=True)
        try:
            conn.execute("SELECT 1 FROM riotData LIMIT 1").fetchone()
            conn.execute("SELECT 1 FROM hourly_agg LIMIT 1").fetchone()
            return True
        finally:
            conn.close()
    except Exception:
        return False


def connect_analysis_database(
    db_file: str | Path = "riot_local.duckdb",
    *,
    parquet_file: str | Path | None = None,
    rebuild_hourly_agg: bool = False,
    read_only: bool = True,
    verbose: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Prepare and open the local DuckDB cache for the full analysis workflow."""

    db_path = Path(db_file).expanduser()
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        parquet_path = resolve_riot_parquet(parquet_file)
    except FileNotFoundError:
        if db_path.exists() and existing_database_is_usable(db_path):
            if verbose:
                print(f"Using existing DuckDB cache without rebuilding: {db_path}", flush=True)
            return duckdb.connect(str(db_path), read_only=read_only)
        raise

    if verbose:
        print(f"Using Riot Parquet: {parquet_path}", flush=True)
        print(f"Using DuckDB cache: {db_path}", flush=True)

    conn = duckdb.connect(str(db_path), read_only=False)
    try:
        create_riot_view(conn, parquet_path)
        ensure_hourly_agg_table(conn, rebuild=rebuild_hourly_agg, verbose=verbose)
        conn.execute("CHECKPOINT")
    finally:
        conn.close()

    return duckdb.connect(str(db_path), read_only=read_only)


def connect_read_only(db_file: str | Path = "riot_local.duckdb") -> duckdb.DuckDBPyConnection:
    """Open the local DuckDB analysis file in read-only mode."""

    return duckdb.connect(str(db_file), read_only=True)


def available_platforms(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return platform IDs available in hourly_agg."""

    rows = conn.execute(
        """
        SELECT platformid
        FROM hourly_agg
        WHERE platformid IS NOT NULL
        GROUP BY platformid
        ORDER BY platformid;
        """
    ).fetchall()
    return [row[0] for row in rows]


def platform_overview(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Summarize hourly_agg coverage by platform."""

    return conn.execute(
        """
        SELECT
            platformid AS PLATFORMID,
            COUNT(*) AS n_hour_bins,
            SUM(n) AS n_rows
        FROM hourly_agg
        WHERE platformid IS NOT NULL
        GROUP BY platformid
        ORDER BY n_rows DESC;
        """
    ).df()


def load_hourly_target(conn: duckdb.DuckDBPyConnection, config: AnalysisConfig) -> pd.DataFrame:
    """Load hourly target means for the configured platform."""

    agg_col = AGG_COL_MAP[config.target_col]
    return conn.execute(
        f"""
        SELECT
            hour_idx,
            {agg_col} AS target_mean,
            n
        FROM hourly_agg
        WHERE platformid = ?
        ORDER BY hour_idx;
        """,
        [config.platform],
    ).df()


def filter_hourly_window(hourly: pd.DataFrame, max_hour_limit: int) -> pd.DataFrame:
    """Keep the first max_hour_limit hours and remove extreme hour-index outliers."""

    if hourly.empty:
        raise RuntimeError("No hourly rows to filter.")

    filtered = hourly.copy()
    min_hour = filtered["hour_idx"].min()
    max_hour = filtered["hour_idx"].max()
    if max_hour - min_hour > max_hour_limit:
        filtered = filtered[filtered["hour_idx"] <= min_hour + max_hour_limit].copy()

    q1 = filtered["hour_idx"].quantile(0.25)
    q3 = filtered["hour_idx"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 3.0 * iqr
    filtered = filtered[(filtered["hour_idx"] >= lower) & (filtered["hour_idx"] <= upper)].copy()
    return filtered


def period_grid(config: AnalysisConfig, *, player: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency and period arrays.

    The grid is evenly spaced in period hours (default 1 h bins over ~6-48 h).
    This keeps reporting interpretable and avoids overly fine period bins.
    """

    period_step_h = config.player_period_step_h if player else config.period_step_h
    if period_step_h <= 0:
        raise ValueError("period_step_h must be > 0")

    period = np.arange(
        config.min_period_h,
        config.max_period_h + (0.5 * period_step_h),
        period_step_h,
        dtype=float,
    )
    period = period[period > 0]
    if len(period) == 0:
        raise ValueError("Period grid is empty; check min/max period bounds.")

    frequency = 1.0 / period
    return frequency, 1.0 / frequency


def lomb_scargle_summary(
    t_hours: np.ndarray,
    values: np.ndarray,
    frequency: np.ndarray,
    period: np.ndarray,
) -> dict[str, Any]:
    """Run Lomb-Scargle and return key periodogram values."""

    y = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(y)
    t = np.asarray(t_hours, dtype=float)[valid].copy()
    y = y[valid]
    if len(y) == 0:
        raise RuntimeError("No valid values for periodogram.")

    t -= t.min()
    y -= np.mean(y)
    power = LombScargle(t, y).power(frequency)
    best_idx = int(np.argmax(power))
    return {
        "power": power,
        "best_period": float(period[best_idx]),
        "power_24": float(power[int(np.argmin(np.abs(period - 24.0)))]),
    }


def fit_sinusoid_ols(
    t_hours: Any,
    y: Any,
    period_h: float,
    weights: Any | None = None,
    fit_intercept: bool = True,
    robust: bool = False,
    cov_type: str = "HC1",
) -> dict[str, float]:
    """Fit y ~ cos + sin at a fixed period and return amplitude, lag, and tests."""

    t_hours = np.asarray(t_hours, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_mask = np.isfinite(y) & np.isfinite(t_hours)
    t_hours = t_hours[valid_mask]
    y = y[valid_mask]

    if weights is not None:
        weights = np.asarray(weights, dtype=float)[valid_mask]

    if period_h <= 0:
        raise ValueError("period_h must be > 0")
    if len(y) == 0:
        raise ValueError("No valid data points left after dropping NaNs.")

    omega = 2.0 * np.pi / period_h
    cos_col = np.cos(omega * t_hours)
    sin_col = np.sin(omega * t_hours)

    if fit_intercept:
        x = pd.DataFrame({"const": 1.0, "cos": cos_col, "sin": sin_col})
    else:
        x = pd.DataFrame({"cos": cos_col, "sin": sin_col})
    y_s = pd.Series(y, name="y")

    if weights is None:
        model = sm.OLS(y_s, x)
    else:
        weights = weights / np.nanmean(weights)
        model = sm.WLS(y_s, x, weights=weights)

    result = model.fit(cov_type=cov_type) if robust else model.fit()

    a = float(result.params["cos"])
    b = float(result.params["sin"])
    c = float(result.params["const"]) if fit_intercept else 0.0

    try:
        if fit_intercept:
            ftest = result.f_test(np.array([[0, 1, 0], [0, 0, 1]]))
        else:
            ftest = result.f_test(np.array([[1, 0], [0, 1]]))
        p_joint = float(np.asarray(ftest.pvalue).reshape(-1)[0])
        f_joint = float(np.asarray(ftest.fvalue).reshape(-1)[0])
    except Exception:
        p_joint = np.nan
        f_joint = np.nan

    amp = float(np.hypot(a, b))
    tau = float(np.arctan2(b, a) / omega)
    tau_mod = float(tau % period_h)

    return {
        "period_h": float(period_h),
        "freq_per_h": float(1.0 / period_h),
        "n": int(result.nobs),
        "amp": amp,
        "lag_h": tau,
        "lag_h_mod_period": tau_mod,
        "p_joint": p_joint,
        "f_joint": f_joint,
        "r2": float(result.rsquared) if hasattr(result, "rsquared") else np.nan,
        "a_cos": a,
        "b_sin": b,
        "c_const": c,
    }


def fixed_period_lag_table(
    t_hours: Any,
    y: Any,
    periods_h: tuple[float, ...] = (12.0, 24.0, 36.0, 48.0, 168.0),
    weights: Any | None = None,
    alpha: float = 0.05,
    robust: bool = False,
    cov_type: str = "HC1",
) -> pd.DataFrame:
    """Fit fixed-period sinusoids for several periods."""

    rows = []
    for period_h in periods_h:
        rows.append(
            fit_sinusoid_ols(
                t_hours=t_hours,
                y=y,
                period_h=period_h,
                weights=weights,
                fit_intercept=True,
                robust=robust,
                cov_type=cov_type,
            )
        )
    table = pd.DataFrame(rows).sort_values("period_h").reset_index(drop=True)
    table["significant"] = table["p_joint"] < alpha
    return table


def load_hourly_win_rate(
    conn: duckdb.DuckDBPyConnection,
    platform: str,
    analysis_hours: pd.DataFrame,
) -> pd.DataFrame:
    """Load hourly win rate and align it to the filtered hourly analysis window."""

    win_rate = conn.execute(
        """
        WITH game_wins AS (
            SELECT
                CAST(FLOOR(CAST(TIMESTAMP AS DOUBLE) / 3600000.0) AS BIGINT) AS hour_idx,
                CASE
                    WHEN LOWER(WINLOSE) = 'true' THEN 1.0
                    WHEN LOWER(WINLOSE) = 'false' THEN 0.0
                    ELSE NULL
                END AS win_flag
            FROM riotData
            WHERE PLATFORMID = ?
              AND TIMESTAMP IS NOT NULL
              AND WINLOSE IS NOT NULL
              AND WINLOSE <> ''
        )
        SELECT
            hour_idx,
            AVG(win_flag) AS win_rate,
            COUNT(win_flag) AS n_win_games
        FROM game_wins
        WHERE win_flag IS NOT NULL
        GROUP BY hour_idx
        ORDER BY hour_idx;
        """,
        [platform],
    ).df()
    return analysis_hours[["hour_idx"]].drop_duplicates().merge(win_rate, on="hour_idx", how="inner")


def load_hourly_metrics(conn: duckdb.DuckDBPyConnection, platform: str) -> pd.DataFrame:
    """Load hourly aggregate metrics needed for PCA."""

    metric_cols = sorted(set(PCA_RATE_MAP.values()) | {"timedead_mean", "timeplayed_mean"})
    select_metrics = ",\n            ".join(metric_cols)
    return conn.execute(
        f"""
        SELECT
            hour_idx,
            {select_metrics},
            n
        FROM hourly_agg
        WHERE platformid = ?
        ORDER BY hour_idx;
        """,
        [platform],
    ).df()


def add_time_normalized_features(hourly_metrics: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add per-minute and fraction columns used as PCA inputs."""

    data = hourly_metrics.copy()
    timeplayed_minutes = (data["timeplayed_mean"] / 60.0).where(data["timeplayed_mean"] > 0)
    for rate_col, source_col in PCA_RATE_MAP.items():
        data[rate_col] = data[source_col] / timeplayed_minutes
    data["timedead_fraction"] = data["timedead_mean"] / data["timeplayed_mean"].where(data["timeplayed_mean"] > 0)
    numeric_cols = list(PCA_RATE_MAP.keys()) + ["timedead_fraction"]
    return data, numeric_cols


def filter_metric_outliers(data: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Sequentially remove IQR outliers from PCA input columns."""

    filtered = data.copy()
    for col in numeric_cols:
        q1 = filtered[col].quantile(0.25)
        q3 = filtered[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = filtered[(filtered[col] >= lower) & (filtered[col] <= upper)].copy()
    if filtered.empty:
        raise RuntimeError("All rows were filtered out during PCA outlier rejection.")
    return filtered


def orient_pca(
    vt: np.ndarray,
    numeric_cols: list[str],
    positive_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Flip PCA component signs so selected columns point positive on average."""

    oriented = vt.copy()
    signs = np.ones(vt.shape[0])
    for idx in range(vt.shape[0]):
        loadings = pd.Series(oriented[idx].copy(), index=numeric_cols)
        if loadings[positive_cols].mean() < 0:
            oriented[idx, :] *= -1.0
            signs[idx] = -1.0
    return oriented, signs


def compute_pca(
    data: pd.DataFrame,
    numeric_cols: list[str],
    positive_cols: list[str],
) -> dict[str, Any]:
    """Standardize features and compute sign-oriented PCA via SVD."""

    features = data[numeric_cols].dropna().copy()
    x = features.to_numpy(dtype=float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0
    x_z = (x - x_mean) / x_std
    u, s, vt = np.linalg.svd(x_z, full_matrices=False)
    vt, signs = orient_pca(vt, numeric_cols, positive_cols)
    explained = (s**2) / np.sum(s**2)
    scores = u * s * signs
    loadings = pd.DataFrame(vt, columns=numeric_cols)
    loadings.index = [f"PC{i + 1}" for i in range(len(loadings))]
    return {
        "features": features,
        "x_mean": x_mean,
        "x_std": x_std,
        "scores": scores,
        "vt": vt,
        "explained": explained,
        "loadings": loadings,
    }


def load_top_players(
    conn: duckdb.DuckDBPyConnection,
    platform: str,
    top_n_players: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load top players by game count and all their game rows."""

    top_players = conn.execute(
        """
        SELECT ACCOUNTID, COUNT(*) AS game_count
        FROM riotData
        WHERE PLATFORMID = ? AND ACCOUNTID IS NOT NULL
        GROUP BY ACCOUNTID
        ORDER BY game_count DESC
        LIMIT ?;
        """,
        [platform, int(top_n_players)],
    ).df()

    if top_players.empty:
        return top_players, pd.DataFrame()

    account_ids = top_players["ACCOUNTID"].tolist()
    placeholders = ",".join(["?"] * len(account_ids))
    player_data = conn.execute(
        f"""
        SELECT *
        FROM riotData
        WHERE PLATFORMID = ? AND ACCOUNTID IN ({placeholders})
        ORDER BY ACCOUNTID, TIMESTAMP;
        """,
        [platform] + account_ids,
    ).df()
    return top_players, add_delta_mmr(player_data)


def add_delta_mmr(player_data: pd.DataFrame) -> pd.DataFrame:
    """Add next-rating and delta_mmr columns within each player."""

    data = player_data.copy()
    data["TIMESTAMP"] = pd.to_numeric(data["TIMESTAMP"], errors="coerce")
    data["RATING"] = pd.to_numeric(data["RATING"], errors="coerce")
    data = data.sort_values(["ACCOUNTID", "TIMESTAMP"]).copy()
    data["next_rating"] = data.groupby("ACCOUNTID")["RATING"].shift(-1)
    data["delta_mmr"] = data["next_rating"] - data["RATING"]
    return data


def project_player_pca(
    player_data: pd.DataFrame,
    numeric_cols: list[str],
    pca_result: dict[str, Any],
) -> pd.DataFrame:
    """Project individual player-game rows onto the hourly PCA basis."""

    source_cols = sorted(set(PLAYER_PCA_FEATURE_MAP.values()) | {"TIMEDEAD", "TIMEPLAYED"})
    work = player_data[source_cols].apply(pd.to_numeric, errors="coerce")
    timeplayed_minutes = (work["TIMEPLAYED"] / 60.0).where(work["TIMEPLAYED"] > 0)

    player_features = pd.DataFrame(index=work.index)
    for feature_col in numeric_cols:
        if feature_col == "timedead_fraction":
            player_features[feature_col] = work["TIMEDEAD"] / work["TIMEPLAYED"].where(work["TIMEPLAYED"] > 0)
        else:
            source_col = PLAYER_PCA_FEATURE_MAP[feature_col]
            player_features[feature_col] = work[source_col] / timeplayed_minutes

    valid = player_features.notna().all(axis=1)
    projected = player_data.copy()
    if valid.sum() == 0:
        projected["perf_factor_pc1"] = np.nan
        projected["perf_factor_pc2"] = np.nan
        return projected

    basis_mean = pd.Series(pca_result["x_mean"], index=numeric_cols)
    basis_std = pd.Series(pca_result["x_std"], index=numeric_cols).replace(0, 1.0)
    x_player_z = (player_features.loc[valid, numeric_cols] - basis_mean) / basis_std
    x_player_z = x_player_z.to_numpy(dtype=float, copy=True)
    vt = pca_result["vt"]

    projected["perf_factor_pc1"] = np.nan
    projected["perf_factor_pc2"] = np.nan
    projected.loc[valid, "perf_factor_pc1"] = x_player_z @ vt[0]
    projected.loc[valid, "perf_factor_pc2"] = x_player_z @ vt[1]
    return projected


def player_periodogram(group: pd.DataFrame, score_col: str, frequency: np.ndarray) -> np.ndarray | None:
    """Compute one player's periodogram for one metric."""

    group = group.dropna(subset=["TIMESTAMP", score_col]).sort_values("TIMESTAMP")
    if len(group) <= 10:
        return None
    t = group["TIMESTAMP"].to_numpy(dtype=float) / 3600000.0
    t = t - t.min()
    y = group[score_col].to_numpy(dtype=float)
    if np.nanstd(y) == 0:
        return None
    y = y - np.nanmean(y)
    try:
        return LombScargle(t, y).power(frequency)
    except Exception:
        return None


def average_player_periodograms(
    player_data: pd.DataFrame,
    metric_map: dict[str, str],
    config: AnalysisConfig,
) -> dict[str, dict[str, Any]]:
    """Average per-player periodograms for each metric."""

    frequency, period = period_grid(config, player=True)
    groups = [group for _, group in player_data.groupby("ACCOUNTID", sort=False)]
    n_jobs = max(1, min(config.n_jobs, 8))
    results: dict[str, dict[str, Any]] = {}

    for label, score_col in metric_map.items():
        power_list = Parallel(n_jobs=n_jobs)(
            delayed(player_periodogram)(group, score_col, frequency)
            for group in groups
        )
        valid_powers = [power for power in power_list if power is not None]
        if not valid_powers:
            results[label] = {
                "score_col": score_col,
                "valid_players": 0,
                "valid_powers": [],
                "mean_power": None,
                "best_period": np.nan,
                "best_power": np.nan,
            }
            continue

        mean_power = np.mean(valid_powers, axis=0)
        best_idx = int(np.argmax(mean_power))
        results[label] = {
            "score_col": score_col,
            "valid_players": len(valid_powers),
            "valid_powers": valid_powers,
            "mean_power": mean_power,
            "best_period": float(period[best_idx]),
            "best_power": float(mean_power[best_idx]),
            "period": period,
        }
    return results


def benjamini_hochberg_mask(p_values: Any, alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask for Benjamini-Hochberg FDR significance."""

    p_values = np.asarray(p_values, dtype=float)
    keep = np.zeros(len(p_values), dtype=bool)
    finite = np.isfinite(p_values)
    finite_p = p_values[finite]
    if len(finite_p) == 0:
        return keep

    order = np.argsort(finite_p)
    sorted_p = finite_p[order]
    ranks = np.arange(1, len(sorted_p) + 1)
    passed = sorted_p <= alpha * ranks / len(sorted_p)
    if not np.any(passed):
        return keep

    threshold = sorted_p[np.max(np.where(passed))]
    keep[finite] = finite_p <= threshold
    return keep


def player_phase(group: pd.DataFrame, score_col: str, offset_hours: int) -> tuple[float, float, float] | None:
    """Fit one player's 24 h sinusoid and return local peak phase."""

    group = group.dropna(subset=["TIMESTAMP", score_col]).sort_values("TIMESTAMP")
    if len(group) <= 10:
        return None
    t_utc_hours = group["TIMESTAMP"].to_numpy(dtype=float) / 3600000.0
    y = group[score_col].to_numpy(dtype=float)
    if np.nanstd(y) == 0:
        return None
    try:
        result = fit_sinusoid_ols(t_utc_hours, y, period_h=24.0)
    except Exception:
        return None
    peak_local = (result["lag_h_mod_period"] + offset_hours) % 24.0
    return peak_local, result["amp"], result["p_joint"]


def extract_player_phases(
    player_data: pd.DataFrame,
    metric_map: dict[str, str],
    config: AnalysisConfig,
) -> dict[str, dict[str, Any]]:
    """Extract per-player 24 h peak phases for each metric."""

    offset_hours = int(utc_offset_hours(config.platform))
    groups = [group for _, group in player_data.groupby("ACCOUNTID", sort=False)]
    n_jobs = max(1, min(config.n_jobs, 32))
    phase_results: dict[str, dict[str, Any]] = {}

    for label, score_col in metric_map.items():
        raw = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(player_phase)(group, score_col, offset_hours)
            for group in groups
        )
        phases = []
        amps = []
        p_values = []
        for item in raw:
            if item is None:
                continue
            phase, amp, p_value = item
            phases.append(phase)
            amps.append(amp)
            p_values.append(p_value)

        phases_array = np.asarray(phases, dtype=float)
        amps_array = np.asarray(amps, dtype=float)
        p_array = np.asarray(p_values, dtype=float)
        nominal = p_array < config.phase_alpha
        fdr = benjamini_hochberg_mask(p_array, alpha=config.phase_alpha)
        phase_results[label] = {
            "score_col": score_col,
            "phases_local": phases_array,
            "amplitudes": amps_array,
            "p_values": p_array,
            "nominal_sig_mask": nominal,
            "fdr_sig_mask": fdr,
            "phases_sig": phases_array[fdr],
            "alpha": config.phase_alpha,
            "correction": "Benjamini-Hochberg FDR",
        }
    return phase_results


def _wrap_mu(mu: float) -> float:
    return float(np.mod(mu, 2.0 * np.pi))


def _kappa_from_rbar(rbar: float) -> float:
    rbar = float(np.clip(rbar, 1e-8, 0.999999))
    if rbar < 0.53:
        return 2 * rbar + rbar**3 + 5 * (rbar**5) / 6
    if rbar < 0.85:
        return -0.4 + 1.39 * rbar + 0.43 / (1 - rbar)
    return 1 / (rbar**3 - 4 * rbar**2 + 3 * rbar)


def fit_vonmises_1comp(theta_vals: np.ndarray) -> dict[str, float]:
    """Fit a one-component circular von Mises model."""

    z = np.exp(1j * theta_vals)
    z_mean = np.mean(z)
    mu = _wrap_mu(np.angle(z_mean))
    rbar = np.abs(z_mean)
    kappa = max(1e-4, _kappa_from_rbar(rbar))
    loglik = float(np.sum(vonmises.logpdf(theta_vals, kappa, loc=mu)))
    bic = float(2 * np.log(len(theta_vals)) - 2 * loglik)
    return {"mu": mu, "kappa": kappa, "loglik": loglik, "bic": bic}


def fit_vonmises_2comp(theta_vals: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> dict[str, float]:
    """Fit a two-component circular von Mises mixture with a simple EM loop."""

    n = len(theta_vals)
    z = np.exp(1j * theta_vals)
    mu_global = _wrap_mu(np.angle(np.mean(z)))
    rbar_global = np.abs(np.mean(z))
    kappa_global = max(1e-4, _kappa_from_rbar(rbar_global))

    pi1 = 0.5
    mu1 = _wrap_mu(mu_global)
    mu2 = _wrap_mu(mu_global + np.pi)
    kappa1 = kappa_global
    kappa2 = kappa_global
    loglik = -np.inf

    for _ in range(max_iter):
        f1 = vonmises.pdf(theta_vals, kappa1, loc=mu1)
        f2 = vonmises.pdf(theta_vals, kappa2, loc=mu2)
        mix = np.clip(pi1 * f1 + (1 - pi1) * f2, 1e-300, None)
        next_loglik = float(np.sum(np.log(mix)))

        gamma1 = (pi1 * f1) / mix
        gamma2 = 1.0 - gamma1
        pi1 = float(np.clip(np.mean(gamma1), 1e-4, 1 - 1e-4))

        z1 = np.sum(gamma1 * np.exp(1j * theta_vals))
        z2 = np.sum(gamma2 * np.exp(1j * theta_vals))
        mu1 = _wrap_mu(np.angle(z1))
        mu2 = _wrap_mu(np.angle(z2))

        rbar1 = np.abs(z1) / np.sum(gamma1)
        rbar2 = np.abs(z2) / np.sum(gamma2)
        kappa1 = max(1e-4, _kappa_from_rbar(rbar1))
        kappa2 = max(1e-4, _kappa_from_rbar(rbar2))

        if np.abs(next_loglik - loglik) < tol:
            loglik = next_loglik
            break
        loglik = next_loglik

    bic = float(5 * np.log(n) - 2 * loglik)
    return {
        "pi1": pi1,
        "pi2": 1 - pi1,
        "mu1": mu1,
        "mu2": mu2,
        "kappa1": kappa1,
        "kappa2": kappa2,
        "loglik": loglik,
        "bic": bic,
    }


def circular_modality_tests(phase_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare one- and two-component circular models for FDR-significant phases."""

    circular: dict[str, dict[str, Any]] = {}
    for label, result in phase_results.items():
        data_hours = np.asarray(result["phases_sig"], dtype=float)
        if len(data_hours) < 20:
            circular[label] = {
                "data_hours": data_hours,
                "n": len(data_hours),
                "status": "skipped",
                "reason": "fewer than 20 FDR-significant players",
            }
            continue
        theta = (data_hours / 24.0) * 2.0 * np.pi
        vm1 = fit_vonmises_1comp(theta)
        vm2 = fit_vonmises_2comp(theta)
        circular[label] = {
            "data_hours": data_hours,
            "theta": theta,
            "n": len(data_hours),
            "status": "fit",
            "vm1": vm1,
            "vm2": vm2,
            "preferred": "2-component" if vm2["bic"] < vm1["bic"] else "1-component",
            "mu1_h": float((vm2["mu1"] / (2 * np.pi)) * 24.0),
            "mu2_h": float((vm2["mu2"] / (2 * np.pi)) * 24.0),
        }
    return circular


def output_dir_for(config: AnalysisConfig) -> Path:
    """Return and create the output directory for one platform."""

    output_dir = config.output_root / config.platform
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_table(table: pd.DataFrame, path: Path) -> None:
    """Save a table as CSV with a parent directory check."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def plot_win_rate(
    hourly_win: pd.DataFrame,
    win_summary: dict[str, Any],
    win_fit: pd.DataFrame,
    config: AnalysisConfig,
    output_dir: Path,
) -> None:
    """Save the standalone win-rate rhythm figure."""

    if hourly_win.empty:
        return
    period = win_summary["period"]
    power = win_summary["power"]
    t_hours = hourly_win["hour_idx"].to_numpy(dtype=float)
    t_hours -= t_hours.min()
    y = hourly_win["win_rate"].to_numpy(dtype=float)
    y_demeaned = y - np.mean(y)

    local = hourly_win.copy()
    local["local_hour"] = np.floor((local["hour_idx"] + utc_offset_hours(config.platform)) % 24.0).astype(int)
    rows = []
    for local_hour, group in local.groupby("local_hour"):
        rows.append(
            {
                "local_hour": local_hour,
                "win_rate": np.average(group["win_rate"], weights=group["n_win_games"]),
                "n_win_games": group["n_win_games"].sum(),
            }
        )
    local_win = pd.DataFrame(rows).set_index("local_hour").reindex(range(24)).reset_index()
    save_table(local_win, output_dir / "win_rate_by_local_hour.csv")
    save_table(win_fit, output_dir / "win_rate_fixed_period_ols.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    ax = axes[0]
    ax.plot(t_hours, y_demeaned, color=COLORS["primary"], linewidth=1.0, alpha=0.9)
    ax.axhline(0.0, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    ax.set_title(f"Hourly Win Rate Over Time ({config.platform})")
    ax.set_xlabel("Hours Since Start")
    ax.set_ylabel("Win rate (demeaned)")
    style_axes(ax)

    ax = axes[1]
    ax.plot(period, power, color=COLORS["secondary"], linewidth=2.0)
    ax.axvline(24.0, color=COLORS["accent"], linestyle="--", linewidth=1.3, label="24 h")
    ax.axvline(win_summary["best_period"], color=COLORS["ink"], linestyle=":", linewidth=1.3, label=f"Peak {win_summary['best_period']:.1f} h")
    ax.set_xlim(0, config.max_period_h)
    ax.set_title(f"Win-Rate Periodogram ({config.platform})")
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("Power")
    ax.legend(frameon=False)
    style_axes(ax)

    ax = axes[2]
    ax.plot(local_win["local_hour"], local_win["win_rate"], color=COLORS["accent"], marker="o", linewidth=2.0)
    ax.axhline(hourly_win["win_rate"].mean(), color=COLORS["muted"], linestyle="--", linewidth=1.1, label="Mean")
    ax.set_title(f"Win Rate by Local Hour ({config.platform})")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Win rate")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(frameon=False)
    style_axes(ax, grid_axis="y")

    fig.suptitle(f"Standalone Win-Rate Rhythm: PLATFORMID={config.platform}", fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(output_dir / "win_rate_rhythm.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loadings(loadings: pd.DataFrame, title: str, output_path: Path, components: int = 3) -> None:
    """Save horizontal bar charts for the first PCA loading vectors."""

    n_components = min(components, len(loadings))
    fig, axes = plt.subplots(1, n_components, figsize=(5.8 * n_components, 5.2), sharex=False)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes[:n_components]):
        label = loadings.index[idx]
        ordered = loadings.iloc[idx].sort_values()
        colors = [COLORS["accent"] if value >= 0 else COLORS["primary"] for value in ordered]
        ax.barh(ordered.index, ordered.values, color=colors, alpha=0.88)
        ax.axvline(0.0, color=COLORS["ink"], linewidth=1.0)
        ax.set_title(f"{label} Loadings")
        ax.set_xlabel("Loading")
        style_axes(ax, grid_axis="x")
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_player_periodograms(
    periodogram_results: dict[str, dict[str, Any]],
    config: AnalysisConfig,
    output_dir: Path,
) -> None:
    """Save mean per-player periodograms for PC1, PC2, and DeltaMMR."""

    labels = list(periodogram_results)
    fig, axes = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 5), sharey=False)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, label in zip(axes, labels):
        result = periodogram_results[label]
        rows.append(
            {
                "metric": label,
                "valid_players": result["valid_players"],
                "best_period": result["best_period"],
                "best_power": result["best_power"],
            }
        )
        if result["mean_power"] is None:
            ax.set_title(f"{label}: no valid players")
            ax.set_xlabel("Period (hours)")
            continue
        ax.plot(result["period"], result["mean_power"], color=COLORS["primary"], linewidth=2.5, label="Mean power")
        ax.axvline(result["best_period"], color=COLORS["accent"], linestyle="--", linewidth=1.5, label=f"Peak = {result['best_period']:.2f} h")
        ax.set_title(f"{label}: {result['valid_players']:,} players, {config.platform}")
        ax.set_xlabel("Period (hours)")
        ax.set_ylabel("Mean Lomb-Scargle power")
        ax.legend(frameon=False)
        style_axes(ax)
    fig.suptitle(f"Incoherent averaged periodograms: {config.platform}", y=1.03, fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(output_dir / "within_subject_periodograms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_table(pd.DataFrame(rows), output_dir / "within_subject_periodogram_summary.csv")


def plot_phase_results(
    phase_results: dict[str, dict[str, Any]],
    config: AnalysisConfig,
    output_dir: Path,
) -> None:
    """Save phase histograms and FDR significant phase tables."""

    labels = list(phase_results)
    fig = plt.figure(figsize=(14.5, 5.2 * len(labels)))
    summary_rows = []

    for row_idx, label in enumerate(labels):
        result = phase_results[label]
        phases = result["phases_local"]
        phases_sig = result["phases_sig"]
        summary_rows.append(
            {
                "metric": label,
                "players_analyzed": len(phases),
                "nominal_significant": int(np.sum(result["nominal_sig_mask"])),
                "fdr_significant": len(phases_sig),
                "alpha": result["alpha"],
                "correction": result["correction"],
            }
        )
        phase_table = pd.DataFrame({"phase_local_peak": phases_sig})
        phase_table.insert(0, "metric", label)
        phase_table["correction"] = result["correction"]
        phase_table["alpha"] = result["alpha"]
        save_table(phase_table, output_dir / f"significant_phases_{label.lower()}_{config.platform}.csv")

        ax1 = fig.add_subplot(len(labels), 2, 2 * row_idx + 1)
        ax2 = fig.add_subplot(len(labels), 2, 2 * row_idx + 2, projection="polar")

        bins = np.arange(0, 25, 1)
        ax1.hist(phases, bins=bins, color=COLORS["secondary"], edgecolor="white", alpha=0.55, label="All analyzed")
        ax1.hist(phases_sig, bins=bins, color=COLORS["accent"], edgecolor="white", alpha=0.88, label="FDR-significant")
        ax1.set_title(f"{label} Peak Hour Distribution ({config.platform})")
        ax1.set_xlabel("Local Hour of Day")
        ax1.set_ylabel("Players")
        ax1.set_xticks(np.arange(0, 25, 2))
        style_axes(ax1, grid_axis="y")
        ax1.legend(frameon=False)

        if len(phases_sig) > 0:
            theta_sig = (phases_sig / 24.0) * 2.0 * np.pi
            polar_bins = np.linspace(0, 2.0 * np.pi, 25)
            counts, edges = np.histogram(theta_sig, bins=polar_bins)
            ax2.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=COLORS["primary"], alpha=0.85, edgecolor="white")

        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax2.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
        ax2.set_title(f"{label} FDR-Significant Peaks ({config.platform})", va="bottom")
        ax2.grid(alpha=0.25)

    fig.suptitle(f"Circadian Phase Landscape: {config.platform}", fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(output_dir / "phase_landscape.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_table(pd.DataFrame(summary_rows), output_dir / "phase_summary.csv")


def plot_circular_results(
    circular_results: dict[str, dict[str, Any]],
    config: AnalysisConfig,
    output_dir: Path,
) -> None:
    """Save circular mixture figures and summary table."""

    summary_rows = []
    for label, result in circular_results.items():
        row = {
            "metric": label,
            "n_fdr_significant": result["n"],
            "status": result["status"],
            "preferred": result.get("preferred"),
            "vm1_bic": result.get("vm1", {}).get("bic"),
            "vm2_bic": result.get("vm2", {}).get("bic"),
            "component_peak_1_h": result.get("mu1_h"),
            "component_peak_2_h": result.get("mu2_h"),
        }
        summary_rows.append(row)
        if result["status"] != "fit":
            continue

        data_hours = result["data_hours"]
        vm1 = result["vm1"]
        vm2 = result["vm2"]
        x_hours = np.linspace(0, 24, 1000)
        x_theta = (x_hours / 24.0) * 2.0 * np.pi
        scale = 2.0 * np.pi / 24.0
        y_vm1 = vonmises.pdf(x_theta, vm1["kappa"], loc=vm1["mu"]) * scale
        y_vm2 = (
            vm2["pi1"] * vonmises.pdf(x_theta, vm2["kappa1"], loc=vm2["mu1"])
            + vm2["pi2"] * vonmises.pdf(x_theta, vm2["kappa2"], loc=vm2["mu2"])
        ) * scale

        fig = plt.figure(figsize=(14.2, 5.6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")

        ax1.hist(data_hours, bins=np.arange(0, 25, 1), density=True, color=COLORS["secondary"], edgecolor="white", alpha=0.5, label="FDR-significant phases")
        ax1.plot(x_hours, y_vm1, linestyle="--", linewidth=2.0, color=COLORS["ink"], label="1-component von Mises")
        ax1.plot(x_hours, y_vm2, linewidth=2.4, color=COLORS["accent"], label="2-component von Mises")
        ax1.set_title(f"{label} Hour-Domain Density Fit ({config.platform})")
        ax1.set_xlabel("Local Peak Performance Hour")
        ax1.set_ylabel("Density")
        ax1.set_xlim(0, 24)
        ax1.legend(frameon=False, loc="upper right")
        style_axes(ax1, grid_axis="y")

        theta_data = (data_hours / 24.0) * 2.0 * np.pi
        polar_bins = np.linspace(0, 2.0 * np.pi, 25)
        counts, edges = np.histogram(theta_data, bins=polar_bins, density=True)
        ax2.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=COLORS["primary"], alpha=0.75, edgecolor="white", linewidth=0.8)
        ax2.plot(x_theta, y_vm2, color=COLORS["accent"], linewidth=2.2)
        ax2.plot(x_theta, y_vm1, color=COLORS["ink"], linewidth=1.8, linestyle="--")
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax2.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
        ax2.set_title(f"{label} Circular Density View ({config.platform})", va="bottom")
        ax2.grid(alpha=0.22)

        fig.suptitle(f"Owls vs Larks Circular Modality: {config.platform}\nMetric: {label} Peak Local Hour", fontsize=13.5, fontweight="bold", color=COLORS["ink"])
        fig.tight_layout()
        fig_path = output_dir / f"owls_vs_larks_circular_vm_{label.lower()}_{config.platform}.png"
        fig.savefig(fig_path, dpi=320, bbox_inches="tight")
        if label == "PC1":
            fig.savefig(output_dir / f"owls_vs_larks_circular_vm_{config.platform}.png", dpi=320, bbox_inches="tight")
        plt.close(fig)

    save_table(pd.DataFrame(summary_rows), output_dir / "circular_modality_summary.csv")


def run_platform_analysis(conn: duckdb.DuckDBPyConnection, config: AnalysisConfig) -> dict[str, Any]:
    """Run the full script version of the notebook for one platform."""

    configure_plot_style()
    output_dir = output_dir_for(config)
    summary: dict[str, Any] = {"platform": config.platform}

    hourly = load_hourly_target(conn, config)
    hourly = filter_hourly_window(hourly, config.max_hour_limit)
    frequency, period = period_grid(config)
    target_periodogram = lomb_scargle_summary(
        hourly["hour_idx"].to_numpy(dtype=float),
        hourly["target_mean"].to_numpy(dtype=float),
        frequency,
        period,
    )
    target_fit = fixed_period_lag_table(
        hourly["hour_idx"].to_numpy(dtype=float) - hourly["hour_idx"].min(),
        hourly["target_mean"].to_numpy(dtype=float),
        robust=False,
    )
    save_table(target_fit, output_dir / "target_fixed_period_ols.csv")
    summary["target_best_period"] = target_periodogram["best_period"]
    summary["target_power_24"] = target_periodogram["power_24"]

    hourly_win = load_hourly_win_rate(conn, config.platform, hourly)
    win_frequency, win_period = period_grid(config)
    win_periodogram = lomb_scargle_summary(
        hourly_win["hour_idx"].to_numpy(dtype=float),
        hourly_win["win_rate"].to_numpy(dtype=float),
        win_frequency,
        win_period,
    )
    win_periodogram["period"] = win_period
    win_fit = fixed_period_lag_table(
        hourly_win["hour_idx"].to_numpy(dtype=float) - hourly_win["hour_idx"].min(),
        hourly_win["win_rate"].to_numpy(dtype=float),
        weights=hourly_win["n_win_games"].to_numpy(dtype=float),
        robust=True,
    )
    plot_win_rate(hourly_win, win_periodogram, win_fit, config, output_dir)
    summary["win_rate_mean"] = float(hourly_win["win_rate"].mean())
    summary["win_rate_best_period"] = win_periodogram["best_period"]
    summary["win_rate_power_24"] = win_periodogram["power_24"]

    hourly_metrics = load_hourly_metrics(conn, config.platform)
    hourly_metrics = filter_hourly_window(hourly_metrics, config.max_hour_limit)
    hourly_metrics, numeric_cols = add_time_normalized_features(hourly_metrics)
    hourly_metrics = filter_metric_outliers(hourly_metrics, numeric_cols)

    pca = compute_pca(hourly_metrics, numeric_cols, GOOD_PCA_COLS)
    loadings = pca["loadings"]
    save_table(loadings.reset_index(names="component"), output_dir / "performance_pca_loadings.csv")
    plot_loadings(loadings, f"Performance PCA Loadings: {config.platform}", output_dir / "performance_pca_loadings.png")
    summary["performance_pc1_explained"] = float(pca["explained"][0])
    summary["performance_pc2_explained"] = float(pca["explained"][1])
    summary["performance_pc3_explained"] = float(pca["explained"][2])

    success_input = hourly_metrics.merge(hourly_win[["hour_idx", "win_rate", "n_win_games"]], on="hour_idx", how="inner")
    success_cols = numeric_cols + ["win_rate"]
    success_pca = compute_pca(success_input, success_cols, GOOD_PCA_COLS + ["win_rate"])
    success_loadings = success_pca["loadings"]
    save_table(success_loadings.reset_index(names="component"), output_dir / "success_aware_pca_loadings.csv")
    plot_loadings(success_loadings, f"Success-Aware PCA Loadings: {config.platform}", output_dir / "success_aware_pca_loadings.png")
    summary["success_pc1_explained"] = float(success_pca["explained"][0])
    summary["success_pc2_explained"] = float(success_pca["explained"][1])
    summary["success_pc3_explained"] = float(success_pca["explained"][2])
    summary["success_pc1_win_rate_loading"] = float(success_loadings.loc["PC1", "win_rate"])
    summary["success_pc2_win_rate_loading"] = float(success_loadings.loc["PC2", "win_rate"])
    summary["success_pc3_win_rate_loading"] = float(success_loadings.loc["PC3", "win_rate"])

    top_players, player_data = load_top_players(conn, config.platform, config.top_n_players)
    save_table(top_players, output_dir / f"top_players_{config.platform}.csv")
    if player_data.empty:
        raise RuntimeError(f"No player rows found for {config.platform}.")

    player_data = project_player_pca(player_data, numeric_cols, pca)
    metric_map = {
        "PC1": "perf_factor_pc1",
        "PC2": "perf_factor_pc2",
        "DeltaMMR": "delta_mmr",
    }

    periodograms = average_player_periodograms(player_data, metric_map, config)
    plot_player_periodograms(periodograms, config, output_dir)
    for label, result in periodograms.items():
        summary[f"{label.lower()}_player_periodogram_best_period"] = result["best_period"]
        summary[f"{label.lower()}_player_periodogram_valid_players"] = result["valid_players"]

    phases = extract_player_phases(player_data, metric_map, config)
    plot_phase_results(phases, config, output_dir)
    for label, result in phases.items():
        summary[f"{label.lower()}_phase_players"] = len(result["phases_local"])
        summary[f"{label.lower()}_phase_fdr_significant"] = len(result["phases_sig"])

    circular = circular_modality_tests(phases)
    plot_circular_results(circular, config, output_dir)
    for label, result in circular.items():
        summary[f"{label.lower()}_circular_status"] = result["status"]
        summary[f"{label.lower()}_circular_preferred"] = result.get("preferred")
        summary[f"{label.lower()}_circular_n"] = result["n"]

    save_table(pd.DataFrame([summary]), output_dir / "analysis_summary.csv")
    return summary
