"""
test_module1.py
---------------
Unit + connectivity tests for Module 1 (Data Ingestion Layer).

Tests are grouped into:
  A. Pure-function tests (no I/O, no network) — always fast
  B. NHL API connectivity tests (live network) — skipped if offline
  C. GameContext construction tests (dataclass integrity)
  D. Parquet integration tests (skipped if files absent)

Run with:
    python test_module1.py
    python test_module1.py -v
"""

import sys
import math
import json
import time
import unittest
from pathlib import Path
from datetime import date, datetime
from unittest.mock import patch, MagicMock

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

# Import the real dataclasses
from game_context import GameContext, TeamContext, GoalieContext, OddsContext

# Import module1 functions directly (patching the Windows BASE path)
# We patch before importing so the path constants resolve correctly
SANDBOX_BASE = str(BASE)

import module1_ingest as m1

# Patch the BASE path to point to our sandbox location
m1.BASE      = BASE
m1.PBP_CACHE = BASE / "pbp_cache"
m1.ODDS_LOG  = BASE / "odds_history.csv"
m1.PBP_CACHE.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — Pure function tests (no I/O)
# ═══════════════════════════════════════════════════════════════════════════════

class TestXGFromShot(unittest.TestCase):
    """xg_from_shot() — distance-decay + shot-type xG model."""

    def test_tap_in_high_xg(self):
        """Shot right in front of net (x=87, y=0, tip-in) should be highest xG
        in our simplified model — well above the base rate."""
        xg       = m1.xg_from_shot(87, 0, shot_type="tip-in")
        far_shot = m1.xg_from_shot(30, 10, shot_type="wrist")
        self.assertGreater(xg, m1.XG_BASE_RATE, "Tap-in must exceed base rate")
        self.assertGreater(xg, far_shot * 2, "Tap-in should be 2x+ a far wrist shot")

    def test_long_shot_low_xg(self):
        """Shot from center ice (x=0, y=0) should be well below base rate."""
        xg = m1.xg_from_shot(0, 0, shot_type="slap")
        self.assertLess(xg, m1.XG_BASE_RATE, "Center-ice shot should be below base rate")

    def test_slot_wrist_reasonable(self):
        """Wrist shot from slot (~30ft) should be between center ice and tap-in."""
        slot = m1.xg_from_shot(60, 5, shot_type="wrist")
        far  = m1.xg_from_shot(20, 5, shot_type="wrist")   # well back
        near = m1.xg_from_shot(85, 2, shot_type="tip-in")  # right in front
        self.assertGreater(slot, far,  "Slot shot must beat long-range shot")
        self.assertLess(slot, near, "Slot shot must be less than close-range tip-in")

    def test_tip_in_beats_slap(self):
        """Tip-in from same distance should outperform slap shot."""
        tip  = m1.xg_from_shot(75, 5, shot_type="tip-in")
        slap = m1.xg_from_shot(75, 5, shot_type="slap")
        self.assertGreater(tip, slap)

    def test_backhand_lower_than_wrist(self):
        """Backhand should have lower xG than wrist from same position."""
        wrist    = m1.xg_from_shot(70, 8, shot_type="wrist")
        backhand = m1.xg_from_shot(70, 8, shot_type="backhand")
        self.assertGreater(wrist, backhand)

    def test_missing_coords_returns_base_rate(self):
        """None coordinates should return the base rate fallback."""
        xg = m1.xg_from_shot(None, None)
        self.assertAlmostEqual(xg, m1.XG_BASE_RATE, places=5)

    def test_xg_capped_at_95_pct(self):
        """xG should never exceed 0.95."""
        xg = m1.xg_from_shot(88, 0.1, shot_type="tip-in")
        self.assertLessEqual(xg, 0.95)

    def test_xg_positive(self):
        """xG should always be positive."""
        for x, y, st in [(88, 0, "wrist"), (50, 20, "slap"), (30, 30, "backhand")]:
            self.assertGreater(m1.xg_from_shot(x, y, st), 0)


class TestParsePBP(unittest.TestCase):
    """parse_pbp() — play-by-play event attribution at 5v5."""

    def _make_play(self, type_key, owner_id, x=70, y=5, shot_type="wrist",
                   situation="1515"):
        return {
            "typeDescKey":  type_key,
            "situationCode": situation,
            "details": {
                "eventOwnerTeamId": owner_id,
                "xCoord": x,
                "yCoord": y,
                "shotType": shot_type,
            }
        }

    def _make_pbp(self, plays):
        return {
            "homeTeam": {"id": 1, "abbrev": "BOS"},
            "awayTeam": {"id": 2, "abbrev": "TOR"},
            "plays":    plays,
        }

    def test_shot_on_goal_counted_for_home(self):
        pbp    = self._make_pbp([self._make_play("shot-on-goal", 1)])
        result = m1.parse_pbp(pbp, "BOS", "TOR")
        self.assertEqual(result["BOS"]["cf"], 1)
        self.assertEqual(result["BOS"]["ff"], 1)
        self.assertEqual(result["BOS"]["sog"], 1)
        self.assertGreater(result["BOS"]["xgf"], 0)

    def test_blocked_shot_counts_corsi_not_fenwick(self):
        pbp    = self._make_pbp([self._make_play("blocked-shot", 1)])
        result = m1.parse_pbp(pbp, "BOS", "TOR")
        self.assertEqual(result["BOS"]["cf"], 1)   # Corsi counts it
        self.assertEqual(result["BOS"]["ff"], 0)   # Fenwick does not
        self.assertEqual(result["BOS"]["sog"], 0)

    def test_missed_shot_counted_for_away(self):
        pbp    = self._make_pbp([self._make_play("missed-shot", 2)])
        result = m1.parse_pbp(pbp, "BOS", "TOR")
        self.assertEqual(result["TOR"]["cf"], 1)
        self.assertEqual(result["TOR"]["ff"], 1)

    def test_non_5v5_excluded(self):
        """Power-play shots (1614 = 5v4) should be excluded."""
        pbp    = self._make_pbp([self._make_play("shot-on-goal", 1, situation="1614")])
        result = m1.parse_pbp(pbp, "BOS", "TOR")
        self.assertEqual(result["BOS"]["cf"], 0)

    def test_xg_accumulates_across_multiple_shots(self):
        plays = [
            self._make_play("shot-on-goal", 1, x=75),
            self._make_play("shot-on-goal", 1, x=80),
            self._make_play("shot-on-goal", 2, x=72),
        ]
        result = m1.parse_pbp(self._make_pbp(plays), "BOS", "TOR")
        self.assertEqual(result["BOS"]["sog"], 2)
        self.assertEqual(result["TOR"]["sog"], 1)
        self.assertGreater(result["BOS"]["xgf"], result["TOR"]["xgf"])

    def test_xga_equals_opponent_xgf(self):
        """xGA for one team should equal the other team's xGF."""
        plays = [self._make_play("shot-on-goal", 1, x=78)]
        result = m1.parse_pbp(self._make_pbp(plays), "BOS", "TOR")
        self.assertAlmostEqual(result["TOR"]["xga"], result["BOS"]["xgf"], places=5)

    def test_empty_pbp_returns_zero_stats(self):
        result = m1.parse_pbp(self._make_pbp([]), "BOS", "TOR")
        self.assertEqual(result["BOS"]["cf"], 0)
        self.assertEqual(result["TOR"]["cf"], 0)

    def test_goal_event_counted_as_sog(self):
        pbp    = self._make_pbp([self._make_play("goal", 1)])
        result = m1.parse_pbp(pbp, "BOS", "TOR")
        self.assertEqual(result["BOS"]["sog"], 1)
        self.assertEqual(result["BOS"]["cf"], 1)


class TestAmericanToImplied(unittest.TestCase):
    """american_to_implied() — moneyline → implied probability."""

    def test_even_money_is_50_pct(self):
        # +100 should be 50%
        self.assertAlmostEqual(m1.american_to_implied(100), 0.50, places=4)

    def test_heavy_favorite(self):
        # -200 = 66.7% implied
        implied = m1.american_to_implied(-200)
        self.assertAlmostEqual(implied, 200 / 300, places=4)

    def test_underdog(self):
        # +150 = 40% implied
        implied = m1.american_to_implied(150)
        self.assertAlmostEqual(implied, 100 / 250, places=4)

    def test_none_returns_none(self):
        self.assertIsNone(m1.american_to_implied(None))

    def test_invalid_string_returns_none(self):
        self.assertIsNone(m1.american_to_implied("N/A"))

    def test_zero_returns_none(self):
        self.assertIsNone(m1.american_to_implied(0))

    def test_implied_between_0_and_1(self):
        for ml in [-300, -150, -110, 110, 150, 300]:
            imp = m1.american_to_implied(ml)
            self.assertGreater(imp, 0.0)
            self.assertLess(imp, 1.0)


class TestGSAA(unittest.TestCase):
    """compute_gsaa() — goals saved above average."""

    def _make_row(self, shots, saves):
        """Minimal dict simulating a goalie parquet row."""
        return {"shotsAgainst": shots, "saves": saves}

    def test_elite_goalie_positive_gsaa(self):
        # 1000 shots, 921 saves = .921 SV% vs .906 avg → +15 GSAA
        row = self._make_row(1000, 921)
        gsaa = m1.compute_gsaa(row)
        self.assertGreater(gsaa, 0)
        self.assertAlmostEqual(gsaa, 921 - 1000 * m1.LEAGUE_AVG_SAVE_PCT, places=1)

    def test_bad_goalie_negative_gsaa(self):
        row = self._make_row(500, 440)   # .880 SV%
        gsaa = m1.compute_gsaa(row)
        self.assertLess(gsaa, 0)

    def test_league_average_near_zero(self):
        saves = round(500 * m1.LEAGUE_AVG_SAVE_PCT)
        row   = self._make_row(500, saves)
        gsaa  = m1.compute_gsaa(row)
        self.assertAlmostEqual(gsaa, 0.0, delta=1.0)

    def test_none_shots_returns_none(self):
        self.assertIsNone(m1.compute_gsaa({"shotsAgainst": None, "saves": 300}))

    def test_missing_key_returns_none(self):
        self.assertIsNone(m1.compute_gsaa({}))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — NHL API connectivity (live, skipped if offline)
# ═══════════════════════════════════════════════════════════════════════════════

def _api_reachable() -> bool:
    try:
        import requests
        r = requests.get(
            "https://api-web.nhle.com/v1/schedule/2026-04-08",
            timeout=8
        )
        return r.status_code == 200
    except Exception:
        return False

API_AVAILABLE = _api_reachable()


@unittest.skipUnless(API_AVAILABLE, "NHL API not reachable from this environment")
class TestNHLApiConnectivity(unittest.TestCase):
    """Live NHL API calls — only run when network is available."""

    def test_schedule_returns_games(self):
        games = m1.get_schedule(date(2026, 4, 8))
        self.assertIsInstance(games, list)
        # April 8 2026 is a regular season date — should have games
        self.assertGreater(len(games), 0, "Expected at least 1 game on 2026-04-08")

    def test_schedule_game_has_required_fields(self):
        games = m1.get_schedule(date(2026, 4, 8))
        if games:
            g = games[0]
            self.assertIn("game_id",   g)
            self.assertIn("home_team", g)
            self.assertIn("away_team", g)
            self.assertIn("start_time", g)

    def test_team_abbrevs_are_valid(self):
        games = m1.get_schedule(date(2026, 4, 8))
        known = {
            "ANA","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL",
            "DET","EDM","FLA","LAK","MIN","MTL","NSH","NJD","NYI",
            "NYR","OTT","PHI","PIT","SJS","SEA","STL","TBL","TOR",
            "UTA","VAN","VGK","WSH","WPG"
        }
        for g in games:
            self.assertIn(g["home_team"], known, f"Unknown team: {g['home_team']}")
            self.assertIn(g["away_team"], known, f"Unknown team: {g['away_team']}")

    def test_rest_days_returns_dict(self):
        result = m1.get_team_rest(["BOS", "TOR"], date(2026, 4, 8))
        self.assertIn("BOS", result)
        self.assertIn("TOR", result)
        for team, info in result.items():
            self.assertIn("rest_days", info)
            self.assertIn("is_b2b",    info)

    def test_rest_days_values_are_reasonable(self):
        result = m1.get_team_rest(["BOS"], date(2026, 4, 8))
        rest = result["BOS"]["rest_days"]
        if rest is not None:
            self.assertGreater(rest, 0)
            self.assertLess(rest, 15)   # no NHL team rests >2 weeks mid-season

    def test_pbp_endpoint_returns_plays(self):
        """Grab a known recent game and verify play-by-play structure."""
        # Use the schedule to find a completed game in the last few days
        for offset in range(1, 5):
            check = date(2026, 4, 8).replace(day=8 - offset)
            import requests
            data = requests.get(
                f"https://api-web.nhle.com/v1/score/{check}",
                timeout=10
            ).json()
            final_games = [
                g for g in data.get("games", [])
                if g.get("gameState") in ("FINAL", "OFF")
            ]
            if final_games:
                gid  = final_games[0]["id"]
                home = final_games[0]["homeTeam"]["abbrev"]
                away = final_games[0]["awayTeam"]["abbrev"]
                pbp  = m1.fetch_pbp_cached(gid)
                self.assertIsNotNone(pbp)
                self.assertIn("plays", pbp)
                self.assertIsInstance(pbp["plays"], list)
                # Parse it
                stats = m1.parse_pbp(pbp, home, away)
                self.assertIn(home, stats)
                self.assertIn(away, stats)
                # Both teams should have at least 1 shot attempt
                total_cf = stats[home]["cf"] + stats[away]["cf"]
                self.assertGreater(total_cf, 0, "Expected shot attempts in a completed game")
                return
        self.skipTest("No completed games found in last 4 days")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — GameContext dataclass integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestGameContextDataclass(unittest.TestCase):
    """Verify GameContext dataclass builds correctly and fields work."""

    def _make_goalie(self, name="Test Goalie", status="confirmed"):
        return GoalieContext(
            player_id=1001, name=name, status=status,
            gaa_live=2.50, svpct_live=0.915,
            rolling_savePct_avg5=0.917, rolling_gaa_avg5=2.45,
            rolling_win_rate_5=0.60, gsaa_season=8.0,
        )

    def _make_team(self, team="BOS", is_home=True):
        return TeamContext(
            team=team, is_home=is_home,
            rest_days=2, is_b2b=False,
            sk_goals_avg5_mean=0.27, sk_shots_avg5_mean=3.1,
            cf_pct_last5=0.525, xg_for_last5=2.80, xg_against_last5=2.40,
            goalie=self._make_goalie(),
        )

    def test_default_construction(self):
        ctx = GameContext()
        self.assertEqual(ctx.game_id, 0)
        self.assertIsInstance(ctx.home, TeamContext)
        self.assertIsInstance(ctx.away, TeamContext)
        self.assertIsInstance(ctx.odds, OddsContext)

    def test_full_construction(self):
        ctx = GameContext(
            game_id=2026020500,
            game_date="2026-04-08",
            home=self._make_team("BOS", True),
            away=self._make_team("TOR", False),
            odds=OddsContext(home_ml=-140, away_ml=120),
            goalie_confirmed=True,
            has_advanced_stats=True,
            has_odds=True,
        )
        self.assertEqual(ctx.game_id, 2026020500)
        self.assertEqual(ctx.home.team, "BOS")
        self.assertEqual(ctx.away.team, "TOR")
        self.assertTrue(ctx.goalie_confirmed)
        self.assertTrue(ctx.has_advanced_stats)

    def test_goalie_is_trusted_property(self):
        confirmed = self._make_goalie(status="confirmed")
        likely    = self._make_goalie(status="likely")
        unknown   = self._make_goalie(status="unknown")
        self.assertTrue(confirmed.is_trusted)
        self.assertTrue(likely.is_trusted)
        self.assertFalse(unknown.is_trusted)

    def test_odds_has_movement_property(self):
        with_open = OddsContext(home_ml=-140, opening_home_ml=-130)
        no_open   = OddsContext(home_ml=-140)
        self.assertTrue(with_open.has_movement)
        self.assertFalse(no_open.has_movement)

    def test_game_context_str(self):
        ctx = GameContext(
            game_id=1, game_date="2026-04-08",
            home=self._make_team("BOS", True),
            away=self._make_team("TOR", False),
            goalie_confirmed=True,
            has_advanced_stats=True,
            has_odds=True,
        )
        s = str(ctx)
        self.assertIn("BOS", s)
        self.assertIn("TOR", s)
        self.assertIn("2026-04-08", s)

    def test_team_context_news_headlines_default_empty(self):
        t = TeamContext(team="BOS", is_home=True)
        self.assertEqual(t.news_headlines, [])

    def test_optional_fields_default_none(self):
        g = GoalieContext()
        self.assertIsNone(g.player_id)
        self.assertIsNone(g.gsaa_season)
        self.assertIsNone(g.rolling_savePct_avg5)

    def test_odds_context_defaults_none(self):
        o = OddsContext()
        self.assertIsNone(o.home_ml)
        self.assertIsNone(o.line_movement)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — Parquet integration (skipped if files absent)
# ═══════════════════════════════════════════════════════════════════════════════

SKATER_PARQUET = BASE / "skater_features.parquet"
GOALIE_PARQUET = BASE / "goalie_features.parquet"

@unittest.skipUnless(SKATER_PARQUET.exists(), "skater_features.parquet not found")
class TestSkaterParquet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.sk = pd.read_parquet(SKATER_PARQUET)

    def test_parquet_loads(self):
        self.assertGreater(len(self.sk), 0)

    def test_required_columns_present(self):
        required = [
            "player_id", "game_pk", "game_date", "team",
            "goals", "assists", "points", "shots", "toi_min",
        ]
        for col in required:
            self.assertIn(col, self.sk.columns, f"Missing column: {col}")

    def test_rolling_columns_exist(self):
        for window in [3, 5, 10, 20]:
            self.assertIn(f"goals_avg{window}", self.sk.columns)
            self.assertIn(f"points_avg{window}", self.sk.columns)

    def test_pts_per60_columns_exist(self):
        for window in [3, 5, 10, 20]:
            self.assertIn(f"pts_per60_avg{window}", self.sk.columns)

    def test_no_future_data_leakage(self):
        """Rolling features use shift(1) — so for active scorers (goals>0),
        goals_avg5 should NOT equal goals exactly (would mean the current game
        is included in its own rolling average)."""
        # Filter to scoring games only (goals > 0) — avoids 0==0 false matches
        scoring = self.sk[self.sk["goals"] > 0].dropna(subset=["goals_avg5"])
        if len(scoring) == 0:
            self.skipTest("No scoring rows to test")
        # For active scorers, goals_avg5 is an average over prior games,
        # so exact equality with goals is rare and suspicious
        exact_match_rate = (scoring["goals_avg5"] == scoring["goals"]).mean()
        self.assertLess(exact_match_rate, 0.25,
            f"Too many exact matches ({exact_match_rate:.1%}) in scoring rows — "
            "possible data leakage in rolling calculation")

    def test_skater_features_function(self):
        teams = self.sk["team"].unique()[:3]
        for team in teams:
            feats = m1.get_skater_features(team, self.sk)
            self.assertIsInstance(feats, dict)
            if feats:
                self.assertIn("sk_goals_avg5_mean", feats)


@unittest.skipUnless(GOALIE_PARQUET.exists(), "goalie_features.parquet not found")
class TestGoalieParquet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.gl = pd.read_parquet(GOALIE_PARQUET)

    def test_parquet_loads(self):
        self.assertGreater(len(self.gl), 0)

    def test_rolling_goalie_columns_exist(self):
        for window in [3, 5, 10]:
            self.assertIn(f"g_savePct_avg{window}", self.gl.columns)
            self.assertIn(f"g_goalsAgainst_avg{window}", self.gl.columns)

    def test_win_rate_columns_exist(self):
        for window in [3, 5, 10]:
            self.assertIn(f"g_win_rate_{window}", self.gl.columns)

    def test_gsaa_computable(self):
        """compute_gsaa should work on real parquet rows."""
        sample = self.gl.dropna(subset=["shotsAgainst", "saves"]).head(10)
        for _, row in sample.iterrows():
            gsaa = m1.compute_gsaa(row)
            self.assertIsNotNone(gsaa)
            self.assertIsInstance(gsaa, float)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Module 1 Tests")
    print(f"  Parquet files: {'✅' if SKATER_PARQUET.exists() else '❌'} skaters  "
          f"{'✅' if GOALIE_PARQUET.exists() else '❌'} goalies")
    print(f"  NHL API:       {'✅ reachable' if API_AVAILABLE else '❌ offline (API tests skipped)'}")
    print("=" * 65 + "\n")
    unittest.main(verbosity=2)
