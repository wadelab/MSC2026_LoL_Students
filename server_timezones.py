"""Fixed UTC offset helpers for Riot platform/server local time conversion.

This uses one static UTC correction per server (no DST handling).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# Aliases to canonical platform IDs.
SERVER_ALIASES: dict[str, str] = {
    "EUW": "EUW1",
    "EUNE": "EUN1",
    "JP": "JP1",
    "ID": "ID1",
    "BR": "BR1",
    "TR": "TR1",
    "NA": "NA1",
    "LAN": "LA1",
    "LAS": "LA2",
    "OCE": "OC1",
    "PBE": "PBE1",
}


# One fixed UTC correction per server.
# These are practical approximations for analysis (DST ignored).
UTC_OFFSET_HOURS: dict[str, int] = {
    # Platforms in current dataset
    "EUW1": 1,    # Western Europe
    "EUN1": 2,    # Eastern/Nordic Europe
    "NA1": -5,    # North America (ET approximation)
    "BR1": -3,    # Brazil
    "TR1": 3,     # Turkey
    "JP1": 9,     # Japan
    "LA1": -6,    # Latin America North
    "LA2": -3,    # Latin America South
    "OC1": 10,    # Oceania (Sydney approximation)
    "ID1": 7,     # Indonesia
    "PBE1": -8,   # PBE (US West approximation)
    # Common extra Riot platforms
    "KR": 9,
    "RU": 3,
    "ME1": 3,
    "PH2": 8,
    "SG2": 8,
    "TH2": 7,
    "TW2": 8,
    "VN2": 7,
}


def canonical_server(server: str) -> str:
    """Normalize aliases (e.g., EUW -> EUW1, JP -> JP1)."""
    key = server.strip().upper()
    return SERVER_ALIASES.get(key, key)


def utc_offset_hours(server: str) -> int:
    """Return fixed UTC offset hours for a server."""
    canonical = canonical_server(server)
    if canonical not in UTC_OFFSET_HOURS:
        known = ", ".join(sorted(UTC_OFFSET_HOURS))
        raise KeyError(f"Unknown server '{server}' (canonical '{canonical}'). Known: {known}")
    return UTC_OFFSET_HOURS[canonical]


def hour_idx_to_local_hour_idx(hour_idx: Any, server: str) -> Any:
    """Shift UTC hour index by server fixed UTC offset.

    Works with scalar, numpy array, or pandas Series.
    """
    return hour_idx + utc_offset_hours(server)


def hour_idx_to_local_hour_of_day(hour_idx: Any, server: str) -> Any:
    """Map UTC hour index to local hour-of-day in [0, 23]."""
    local = hour_idx_to_local_hour_idx(hour_idx, server) % 24
    if hasattr(local, "astype"):
        return local.astype(int)
    return int(local)


def utc_ms_to_local_datetime(utc_ms: Any, server: str) -> pd.Series:
    """Convert UTC milliseconds since epoch to local datetime (fixed offset).

    Returns timezone-aware UTC datetimes shifted by offset hours.
    """
    dt_utc = pd.to_datetime(utc_ms, unit="ms", utc=True)
    return dt_utc + pd.to_timedelta(utc_offset_hours(server), unit="h")


def local_24h_cycle_mean(
    df: pd.DataFrame,
    value_col: str,
    server: str,
    *,
    hour_idx_col: str = "hour_idx",
) -> pd.DataFrame:
    """Fold data into local hour-of-day and return per-hour mean.

    Parameters
    ----------
    df
        DataFrame containing UTC hour index and a metric column.
    value_col
        Metric column to average (e.g., ``target_mean``).
    server
        Server/platform code (supports aliases like ``EUW``, ``JP``, ``ID``).
    hour_idx_col
        Column containing integer UTC hour index.
    """
    missing = [c for c in (hour_idx_col, value_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df[[hour_idx_col, value_col]].dropna().copy()
    work["local_hour"] = hour_idx_to_local_hour_of_day(work[hour_idx_col], server)
    cycle = (
        work.groupby("local_hour", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "mean_value"})
    )
    cycle = cycle.set_index("local_hour").reindex(range(24)).reset_index()
    return cycle


def plot_local_24h_cycle(
    df: pd.DataFrame,
    value_col: str,
    server: str,
    *,
    hour_idx_col: str = "hour_idx",
    ax: Any = None,
    title: str | None = None,
):
    """Plot mean metric over local 24-hour cycle.

    Returns
    -------
    tuple
        ``(ax, cycle_df)`` where ``cycle_df`` has columns:
        ``local_hour`` and ``mean_value``.
    """
    import matplotlib.pyplot as plt

    cycle = local_24h_cycle_mean(df, value_col, server, hour_idx_col=hour_idx_col)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(cycle["local_hour"], cycle["mean_value"], marker="o", linewidth=1.5)
    ax.set_xlim(0, 23)
    xticks = list(range(0, 24, 2))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{h:02d}:00" for h in xticks])
    ax.set_xlabel(f"Local time ({canonical_server(server)})")
    ax.set_ylabel(f"Mean {value_col}")
    ax.grid(alpha=0.3)

    if title is None:
        title = f"Mean {value_col} across 24h local cycle ({canonical_server(server)})"
    ax.set_title(title)

    return ax, cycle
