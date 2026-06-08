from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import vonmises

from grand_analysis import ANALYSIS_PLATFORMS, write_full_html_report
from riot_analysis import (
    AGG_COL_MAP,
    AnalysisConfig,
    COLAB_RIOT_DUCKDB,
    configure_plot_style,
    connect_analysis_database,
    existing_database_is_usable,
    fit_vonmises_2comp,
    load_hourly_win_rate,
    load_top_players,
    resolve_analysis_db_file,
    save_figure,
)
from server_timezones import utc_ms_to_local_datetime


class AnalysisTests(unittest.TestCase):
    def test_default_window_covers_current_dataset(self) -> None:
        self.assertEqual(AnalysisConfig("EUW1").max_hour_limit, 10000)

    def test_default_platforms_match_canonical_eight(self) -> None:
        self.assertEqual(len(ANALYSIS_PLATFORMS), 8)
        self.assertNotIn("TR1", ANALYSIS_PLATFORMS)
        self.assertNotIn("ID1", ANALYSIS_PLATFORMS)
        self.assertNotIn("PBE1", ANALYSIS_PLATFORMS)

    def test_duckdb_path_can_be_overridden_for_drive_cache(self) -> None:
        expected = Path("/content/drive/MyDrive/lol/riot_local.duckdb")
        with patch.dict("os.environ", {"RIOT_DUCKDB_PATH": str(expected)}):
            self.assertEqual(resolve_analysis_db_file(), expected)

    def test_colab_defaults_to_shared_drive_cache(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("riot_analysis.running_in_colab", return_value=True),
        ):
            self.assertEqual(resolve_analysis_db_file(), COLAB_RIOT_DUCKDB)

    def test_valid_existing_cache_is_reused_without_parquet_resolution(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "riot_local.duckdb"
            conn = duckdb.connect(str(db_path))
            conn.execute("CREATE TABLE riotData (value INTEGER)")
            aggregate_columns = ", ".join(
                f"{column} DOUBLE" for column in sorted(set(AGG_COL_MAP.values()))
            )
            conn.execute(
                f"CREATE TABLE hourly_agg (platformid VARCHAR, hour_idx BIGINT, "
                f"n BIGINT, {aggregate_columns})"
            )
            conn.execute("INSERT INTO riotData VALUES (1)")
            conn.execute(
                f"INSERT INTO hourly_agg VALUES ('EUW1', 1, 1, "
                + ", ".join(["1"] * len(set(AGG_COL_MAP.values())))
                + ")"
            )
            conn.close()

            with patch("riot_analysis.resolve_riot_parquet", side_effect=AssertionError):
                reused = connect_analysis_database(db_path, verbose=False)
            try:
                self.assertEqual(reused.execute("SELECT COUNT(*) FROM hourly_agg").fetchone()[0], 1)
            finally:
                reused.close()

    def test_cache_validation_does_not_scan_raw_parquet_view(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parquet_path = temp_path / "raw.parquet"
            db_path = temp_path / "riot_local.duckdb"
            conn = duckdb.connect(str(db_path))
            conn.execute(f"COPY (SELECT 1 AS value) TO '{parquet_path}' (FORMAT PARQUET)")
            conn.execute(f"CREATE VIEW riotData AS SELECT * FROM read_parquet('{parquet_path}')")
            aggregate_columns = ", ".join(
                f"{column} DOUBLE" for column in sorted(set(AGG_COL_MAP.values()))
            )
            conn.execute(
                f"CREATE TABLE hourly_agg (platformid VARCHAR, hour_idx BIGINT, "
                f"n BIGINT, {aggregate_columns})"
            )
            conn.close()
            parquet_path.unlink()

            self.assertTrue(existing_database_is_usable(db_path))

    def test_svg_first_full_report_links_editable_figures_and_all_servers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            grand_dir = output_root / "GRAND"
            server_dir = output_root / "EUW1"
            grand_dir.mkdir()
            server_dir.mkdir()
            pd.DataFrame([{"platform": "EUW1", "status": "ok"}]).to_csv(
                output_root / "all_servers_summary.csv",
                index=False,
            )
            pd.DataFrame([{"metric": "PC1", "value": 1.0}]).to_csv(
                server_dir / "phase_summary.csv",
                index=False,
            )

            configure_plot_style()
            fig, ax = plt.subplots()
            ax.set_title("Editable title")
            save_figure(fig, server_dir / "phase_landscape.png")
            plt.close(fig)

            report_path = write_full_html_report(output_root, [server_dir], grand_dir)
            report = report_path.read_text(encoding="utf-8")
            svg = (server_dir / "phase_landscape.svg").read_text(encoding="utf-8")

            self.assertTrue((server_dir / "phase_landscape.png").exists())
            self.assertIn("EUW1 Server Analysis", report)
            self.assertIn("EUW1/phase_landscape.svg", report)
            self.assertIn("<text", svg)

    def test_fixed_offset_datetime_has_correct_timezone(self) -> None:
        scalar = utc_ms_to_local_datetime(0, "EUW1")
        series = utc_ms_to_local_datetime(pd.Series([0]), "EUW1")

        self.assertEqual(scalar.isoformat(), "1970-01-01T01:00:00+01:00")
        self.assertEqual(series.iloc[0].isoformat(), "1970-01-01T01:00:00+01:00")

    def test_win_rate_count_is_labeled_as_player_match_records(self) -> None:
        conn = duckdb.connect()
        conn.execute("CREATE TABLE riotData (PLATFORMID VARCHAR, TIMESTAMP VARCHAR, WINLOSE VARCHAR)")
        conn.executemany(
            "INSERT INTO riotData VALUES (?, ?, ?)",
            [
                ("EUW1", "0", "true"),
                ("EUW1", "1", "false"),
                ("EUW1", "2", "true"),
            ],
        )

        result = load_hourly_win_rate(conn, "EUW1", pd.DataFrame({"hour_idx": [0]}))

        self.assertEqual(result.loc[0, "n_win_records"], 3)
        self.assertAlmostEqual(result.loc[0, "win_rate"], 2 / 3)
        self.assertNotIn("n_win_games", result.columns)

    def test_top_player_rows_respect_analysis_window(self) -> None:
        conn = duckdb.connect()
        conn.execute(
            """
            CREATE TABLE riotData (
                ACCOUNTID VARCHAR,
                PLATFORMID VARCHAR,
                TIMESTAMP VARCHAR,
                RATING VARCHAR
            )
            """
        )
        hour_ms = 3600000
        conn.executemany(
            "INSERT INTO riotData VALUES (?, ?, ?, ?)",
            [
                ("a", "EUW1", str(0 * hour_ms), "10"),
                ("a", "EUW1", str(1 * hour_ms), "11"),
                ("a", "EUW1", str(3 * hour_ms), "12"),
                ("b", "EUW1", str(1 * hour_ms), "20"),
            ],
        )

        top, rows = load_top_players(conn, "EUW1", 1, min_hour_idx=0, max_hour_idx=1)

        self.assertEqual(top.loc[0, "ACCOUNTID"], "a")
        self.assertEqual(top.loc[0, "game_count"], 2)
        self.assertEqual(len(rows), 2)
        self.assertLess(rows["TIMESTAMP"].max(), 2 * hour_ms)

    def test_two_component_loglik_matches_returned_parameters(self) -> None:
        theta = np.concatenate([np.linspace(-0.3, 0.3, 30), np.linspace(2.8, 3.4, 30)])
        result = fit_vonmises_2comp(theta)
        mixture = (
            result["pi1"] * vonmises.pdf(theta, result["kappa1"], loc=result["mu1"])
            + result["pi2"] * vonmises.pdf(theta, result["kappa2"], loc=result["mu2"])
        )

        self.assertAlmostEqual(result["loglik"], float(np.log(mixture).sum()), places=10)


if __name__ == "__main__":
    unittest.main()
