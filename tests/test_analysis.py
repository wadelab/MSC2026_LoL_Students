from __future__ import annotations

import unittest

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import vonmises

from grand_analysis import ANALYSIS_PLATFORMS
from riot_analysis import AnalysisConfig, fit_vonmises_2comp, load_hourly_win_rate, load_top_players
from server_timezones import utc_ms_to_local_datetime


class AnalysisTests(unittest.TestCase):
    def test_default_window_covers_current_dataset(self) -> None:
        self.assertEqual(AnalysisConfig("EUW1").max_hour_limit, 10000)

    def test_default_platforms_match_canonical_eight(self) -> None:
        self.assertEqual(len(ANALYSIS_PLATFORMS), 8)
        self.assertNotIn("TR1", ANALYSIS_PLATFORMS)
        self.assertNotIn("ID1", ANALYSIS_PLATFORMS)
        self.assertNotIn("PBE1", ANALYSIS_PLATFORMS)

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
