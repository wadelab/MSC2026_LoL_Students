"""Run the League of Legends rhythm analysis outside the notebook.

By default this script runs the original 8 Aung et al. platforms found in
`hourly_agg` and writes outputs to separate folders under `results/`, for
example `results/EUW1/`. The reusable functions live in `riot_analysis.py`
so the notebook and script workflow can share the same analysis logic.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from grand_analysis import ANALYSIS_PLATFORMS, run_grand_analysis
from riot_analysis import AnalysisConfig, available_platforms, connect_analysis_database, run_platform_analysis


SUMMARY_COLUMNS = [
    "platform",
    "status",
    "elapsed_seconds",
    "target_best_period",
    "win_rate_best_period",
    "performance_pc1_explained",
    "performance_pc2_explained",
    "performance_pc3_explained",
    "success_pc1_explained",
    "success_pc2_explained",
    "success_pc3_explained",
    "success_pc1_win_rate_loading",
    "success_pc2_win_rate_loading",
    "success_pc3_win_rate_loading",
    "pc1_phase_fdr_significant",
    "pc2_phase_fdr_significant",
    "deltammr_phase_fdr_significant",
    "error",
]


def default_n_jobs() -> int:
    """Choose a conservative default number of workers."""

    cpu_count = os.cpu_count() or 2
    return max(1, min(cpu_count - 1, 8))


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Run LoL rhythm analyses by server.")
    parser.add_argument(
        "--duckdb-file",
        default="riot_local.duckdb",
        help="Path to the local DuckDB cache file. It is created or refreshed from the raw Parquet when needed.",
    )
    parser.add_argument(
        "--parquet-file",
        default=None,
        help=(
            "Path to raw riotData.parquet. Defaults to RIOT_DB_PATH/RIOT_PARQUET_PATH, "
            "/raid/data/riot/riotData.parquet, or the Colab shared-drive path."
        ),
    )
    parser.add_argument("--output-root", default="results", help="Folder where per-server outputs are written.")
    parser.add_argument(
        "--platform",
        action="append",
        help=(
            "Platform/server to run. Repeat this option to run several. "
            "Defaults to the original 8 Aung et al. servers."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Platform/server to skip. Repeat this option to skip several.",
    )
    parser.add_argument("--target-col", default="TIMEPLAYED", help="Hourly target column for the main rhythm check.")
    parser.add_argument(
        "--top-n-players",
        type=int,
        default=1000,
        help="Top players per server for within-subject analyses.",
    )
    parser.add_argument("--max-hour-limit", type=int, default=5000, help="Maximum hourly span to analyze per server.")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs(), help="Parallel workers for player analyses.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove existing generated files from each server output folder before rerunning.",
    )
    parser.add_argument(
        "--skip-grand",
        action="store_true",
        help="Skip the across-server grand analysis after per-server runs.",
    )
    parser.add_argument(
        "--grand-only",
        action="store_true",
        help="Only regenerate results/GRAND from existing per-server outputs.",
    )
    parser.add_argument(
        "--rebuild-hourly-agg",
        action="store_true",
        help="Force rebuild of hourly_agg from raw riotData.parquet before running platforms.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Combined summary CSV path. Defaults to <output-root>/all_servers_summary.csv.",
    )
    return parser.parse_args()


def selected_platforms(conn, requested: list[str] | None, excluded: list[str]) -> list[str]:
    """Return the platform list requested by the user."""

    available = {platform.upper() for platform in available_platforms(conn)}
    if requested:
        platforms = [platform.upper() for platform in requested if platform.upper() in available]
    else:
        platforms = [platform.upper() for platform in ANALYSIS_PLATFORMS if platform.upper() in available]

    excluded_set = {platform.upper() for platform in excluded}
    return [platform for platform in platforms if platform.upper() not in excluded_set]


def clean_platform_outputs(output_root: Path, platform: str) -> None:
    """Remove generated files for one platform before a fresh run."""

    output_dir = output_root / platform
    if not output_dir.exists():
        return

    for pattern in ("*.png", "*.csv", "error.json"):
        for path in output_dir.glob(pattern):
            path.unlink()


def format_markdown_value(value: object) -> str:
    """Format a summary value for the Markdown table."""

    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown_summary(rows: list[dict], summary_path: Path) -> None:
    """Write a compact Markdown version of the combined run summary."""

    markdown_path = summary_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        markdown_path.write_text("# All-Server Analysis Summary\n\nNo rows yet.\n", encoding="utf-8")
        return

    summary = pd.DataFrame(rows)
    columns = [col for col in SUMMARY_COLUMNS if col in summary.columns]
    summary = summary[columns]

    lines = [
        "# All-Server Analysis Summary",
        "",
        "Generated by `Parquet_longerAnalyses_May_26.py`.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in summary.iterrows():
        values = [format_markdown_value(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: list[dict], summary_path: Path) -> None:
    """Write the combined run summaries."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    write_markdown_summary(rows, summary_path)


def main() -> int:
    """Run analyses for all selected platforms."""

    args = parse_args()
    output_root = Path(args.output_root)
    summary_path = Path(args.summary_csv) if args.summary_csv else output_root / "all_servers_summary.csv"

    if args.grand_only:
        print("Running grand analysis from existing per-server outputs", flush=True)
        result = run_grand_analysis(
            output_root=output_root,
            platforms=args.platform,
            clean=not args.keep_existing,
        )
        print(
            f"Grand analysis complete: {result['servers']} server(s), "
            f"{result['figures']} figure(s), output: {result['grand_dir']}",
            flush=True,
        )
        return 0

    conn = connect_analysis_database(
        args.duckdb_file,
        parquet_file=args.parquet_file,
        rebuild_hourly_agg=args.rebuild_hourly_agg,
    )
    try:
        platforms = selected_platforms(conn, args.platform, args.exclude)
        if not platforms:
            raise SystemExit("No platforms selected.")

        print(f"Running {len(platforms)} platform(s): {', '.join(platforms)}", flush=True)
        rows = []

        for index, platform in enumerate(platforms, start=1):
            print(f"[{index}/{len(platforms)}] Starting {platform}", flush=True)
            start_time = time.time()
            config = AnalysisConfig(
                platform=platform,
                target_col=args.target_col,
                top_n_players=args.top_n_players,
                max_hour_limit=args.max_hour_limit,
                n_jobs=args.n_jobs,
                output_root=output_root,
            )

            try:
                if not args.keep_existing:
                    clean_platform_outputs(output_root, platform)
                summary = run_platform_analysis(conn, config)
                summary["status"] = "ok"
                summary["elapsed_seconds"] = round(time.time() - start_time, 2)
                rows.append(summary)
                print(
                    f"[{index}/{len(platforms)}] Finished {platform} "
                    f"in {summary['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                elapsed_seconds = round(time.time() - start_time, 2)
                error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                error_payload = {
                    "platform": platform,
                    "status": "error",
                    "elapsed_seconds": elapsed_seconds,
                    "error": error_text,
                }
                rows.append(error_payload)
                error_path = output_root / platform / "error.json"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(json.dumps(error_payload, indent=2) + "\n")
                print(f"[{index}/{len(platforms)}] ERROR {platform}: {error_text}", flush=True)

            write_summary(rows, summary_path)

        ok_count = sum(row.get("status") == "ok" for row in rows)
        error_count = sum(row.get("status") == "error" for row in rows)
        if not args.skip_grand and ok_count > 0:
            ok_platforms = [row["platform"] for row in rows if row.get("status") == "ok"]
            print("Starting grand analysis", flush=True)
            try:
                result = run_grand_analysis(
                    output_root=output_root,
                    platforms=ok_platforms,
                    clean=not args.keep_existing,
                )
                print(
                    f"Grand analysis complete: {result['servers']} server(s), "
                    f"{result['figures']} figure(s), output: {result['grand_dir']}",
                    flush=True,
                )
            except Exception as exc:
                error_count += 1
                error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                print(f"Grand analysis ERROR: {error_text}", flush=True)

        print(f"Complete: {ok_count} ok, {error_count} error(s). Summary: {summary_path}", flush=True)
        return 0 if error_count == 0 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
