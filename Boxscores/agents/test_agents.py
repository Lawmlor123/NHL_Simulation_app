"""
test_agents.py
--------------
Unit tests for all NHL MAS agents.

Uses mock GameContext objects — no parquet files, no API calls required.
Run with:
    python -m pytest agents/test_agents.py -v
    OR
    python agents/test_agents.py
"""

import sys
import unittest
from pathlib import Path
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

# ── Allow running from Boxscores/ directory ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_base        import AgentSignal, NHLAgent, logistic, clamp
from agents.team_form_agent   import TeamFormAgent
from agents.player_form_agent import PlayerFormAgent
from agents.goalie_form_agent import GoalieFormAgent
from agents.schedule_agent    import ScheduleAgent, travel_distance
from agents.sentiment_agent   import SentimentAgent, _score_headlines, _score_to_multiplier


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK OBJECTS — no file I/O, no network
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockGoalieContext:
    player_id:               Optional[int]   = 1001
    name:                    str             = "Marc-Andre Fleury"
    status:                  str             = "confirmed"
    gaa_live:                Optional[float] = 2.45
    svpct_live:              Optional[float] = 0.918
    wins_live:               Optional[int]   = 22
    losses_live:             Optional[int]   = 14
    rolling_savePct_avg3:    Optional[float] = 0.921
    rolling_savePct_avg5:    Optional[float] = 0.917
    rolling_savePct_avg10:   Optional[float] = 0.915
    rolling_gaa_avg3:        Optional[float] = 2.40
    rolling_gaa_avg5:        Optional[float] = 2.45
    rolling_win_rate_3:      Optional[float] = 0.667
    rolling_win_rate_5:      Optional[float] = 0.600
    rolling_win_rate_10:     Optional[float] = 0.580
    rolling_shotsAgainst_avg5: Optional[float] = 30.2
    gsaa_season:             Optional[float] = 8.4
    gsaa_last10:             Optional[float] = 2.1

    @property
    def is_trusted(self) -> bool:
        return self.status.lower() in ("confirmed", "likely", "expected", "manual")


@dataclass
class MockTeamContext:
    team:     str  = "BOS"
    is_home:  bool = True
    rest_days:    Optional[int]  = 2
    is_b2b:       bool           = False
    sk_goals_avg3_mean:    Optional[float] = 0.28
    sk_goals_avg5_mean:    Optional[float] = 0.27
    sk_goals_avg10_mean:   Optional[float] = 0.25
    sk_goals_avg20_mean:   Optional[float] = 0.24
    sk_shots_avg3_mean:    Optional[float] = 3.2
    sk_shots_avg5_mean:    Optional[float] = 3.1
    sk_shots_avg10_mean:   Optional[float] = 3.0
    sk_points_avg5_mean:   Optional[float] = 0.55
    sk_points_avg10_mean:  Optional[float] = 0.52
    sk_pp_goals_avg5_mean: Optional[float] = 0.08
    sk_pp_goals_avg10_mean: Optional[float] = 0.07
    sk_hits_avg5_mean:      Optional[float] = 2.1
    sk_takeaways_avg5_mean: Optional[float] = 0.9
    sk_giveaways_avg5_mean: Optional[float] = 0.8
    sk_plus_minus_avg5_mean: Optional[float] = 0.15
    sk_goals_avg5_sum:    Optional[float] = 3.5
    sk_shots_avg5_sum:    Optional[float] = 30.2
    sk_points_avg5_sum:   Optional[float] = 7.1
    sk_pp_goals_avg5_sum: Optional[float] = 2.3
    cf_pct_last5:     Optional[float] = 0.525
    ff_pct_last5:     Optional[float] = 0.530
    xg_for_last5:     Optional[float] = 2.85
    xg_against_last5: Optional[float] = 2.40
    xg_pct_last5:     Optional[float] = 0.543
    sh_attempts_last5: Optional[float] = 58.2
    goalie:            Optional[MockGoalieContext] = field(default_factory=MockGoalieContext)
    news_headlines:    list = field(default_factory=list)


@dataclass
class MockOddsContext:
    home_ml:       Optional[float] = -140
    away_ml:       Optional[float] = +120
    point_spread:  Optional[float] = -1.5
    home_implied:  Optional[float] = 0.583
    away_implied:  Optional[float] = 0.455
    opening_home_ml: Optional[float] = -130
    opening_away_ml: Optional[float] = +110
    line_movement:   Optional[float] = -10.0

    @property
    def has_movement(self) -> bool:
        return self.opening_home_ml is not None and self.home_ml is not None


@dataclass
class MockGameContext:
    game_id:    int = 2026020500
    game_date:  str = "2026-04-08"
    start_time: str = "2026-04-08T23:00:00Z"
    season:     str = "20252026"
    game_type:  int = 2
    home:       MockTeamContext = field(default_factory=lambda: MockTeamContext(team="BOS", is_home=True))
    away:       MockTeamContext = field(default_factory=lambda: MockTeamContext(
        team="TOR", is_home=False,
        rest_days=1, is_b2b=True,
        cf_pct_last5=0.490,
        xg_for_last5=2.50, xg_against_last5=2.70, xg_pct_last5=0.481,
        goalie=MockGoalieContext(
            name="Ilya Samsonov", status="confirmed",
            rolling_savePct_avg5=0.899, rolling_gaa_avg5=3.10,
            rolling_win_rate_5=0.400, gsaa_season=-2.1,
        )
    ))
    odds: MockOddsContext = field(default_factory=MockOddsContext)
    goalie_confirmed:    bool = True
    has_advanced_stats:  bool = True
    has_odds:            bool = True
    has_news:            bool = False


def make_game(
    home_team: str = "BOS",
    away_team: str = "TOR",
    home_rest: int = 2,
    away_rest: int = 1,
    home_b2b:  bool = False,
    away_b2b:  bool = True,
    home_cf:   float = 0.525,
    away_cf:   float = 0.490,
    home_xgf:  float = 2.85,
    away_xgf:  float = 2.50,
    home_news: list = None,
    away_news: list = None,
    home_goalie_status: str = "confirmed",
    away_goalie_status: str = "confirmed",
    home_goalie_spp5: float = 0.917,
    away_goalie_spp5: float = 0.899,
) -> MockGameContext:
    """Factory to create mock games with specific parameters."""
    home_goalie = MockGoalieContext(
        name="Home Goalie", status=home_goalie_status,
        rolling_savePct_avg5=home_goalie_spp5,
        rolling_gaa_avg5=2.5, rolling_win_rate_5=0.60, gsaa_season=5.0,
    )
    away_goalie = MockGoalieContext(
        name="Away Goalie", status=away_goalie_status,
        rolling_savePct_avg5=away_goalie_spp5,
        rolling_gaa_avg5=3.0, rolling_win_rate_5=0.40, gsaa_season=-2.0,
    )
    home = MockTeamContext(
        team=home_team, is_home=True,
        rest_days=home_rest, is_b2b=home_b2b,
        cf_pct_last5=home_cf,
        xg_for_last5=home_xgf, xg_against_last5=2.50, xg_pct_last5=home_cf,
        goalie=home_goalie,
        news_headlines=home_news or [],
    )
    away = MockTeamContext(
        team=away_team, is_home=False,
        rest_days=away_rest, is_b2b=away_b2b,
        cf_pct_last5=away_cf,
        xg_for_last5=away_xgf, xg_against_last5=2.70, xg_pct_last5=away_cf,
        goalie=away_goalie,
        news_headlines=away_news or [],
    )
    return MockGameContext(home=home, away=away)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentSignal(unittest.TestCase):
    """Tests for the AgentSignal contract."""

    def test_score_clamped(self):
        s = AgentSignal("test", "home", raw_score=1.5, confidence=0.8, reasoning="")
        self.assertEqual(s.raw_score, 1.0)
        s2 = AgentSignal("test", "away", raw_score=-0.5, confidence=0.8, reasoning="")
        self.assertEqual(s2.raw_score, 0.0)

    def test_confidence_clamped(self):
        s = AgentSignal("test", "neutral", raw_score=0.5, confidence=2.0, reasoning="")
        self.assertEqual(s.confidence, 1.0)

    def test_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            AgentSignal("test", "sideways", raw_score=0.5, confidence=0.5, reasoning="")

    def test_home_edge(self):
        s = AgentSignal("test", "home", raw_score=0.62, confidence=0.7, reasoning="")
        self.assertAlmostEqual(s.home_edge, 0.12, places=5)

    def test_weighted_score(self):
        s = AgentSignal("test", "home", raw_score=0.7, confidence=0.6, reasoning="")
        expected = 0.5 + (0.7 - 0.5) * 0.6   # = 0.62
        self.assertAlmostEqual(s.weighted_score, expected, places=5)

    def test_str_output(self):
        s = AgentSignal("team_form", "home", 0.63, 0.71, "test reasoning")
        self.assertIn("HOME", str(s))
        self.assertIn("team_form", str(s))


class TestAgentBase(unittest.TestCase):
    """Tests for NHLAgent base class utilities."""

    def _make_agent(self):
        # Concrete subclass for testing the base
        class DummyAgent(NHLAgent):
            def analyze(self, ctx):
                return self.neutral_signal("dummy")
        return DummyAgent("dummy", weight=1.0, confidence_floor=0.55)

    def test_neutral_signal(self):
        agent  = self._make_agent()
        signal = agent.neutral_signal("no data")
        self.assertEqual(signal.pick_direction, "neutral")
        self.assertEqual(signal.raw_score, 0.5)
        self.assertEqual(signal.confidence, 0.0)

    def test_is_signal_valid(self):
        agent  = self._make_agent()
        high   = AgentSignal("dummy", "home", 0.7, 0.8, "")
        low    = AgentSignal("dummy", "home", 0.7, 0.40, "")
        self.assertTrue(agent.is_signal_valid(high))
        self.assertFalse(agent.is_signal_valid(low))

    def test_logistic(self):
        self.assertAlmostEqual(logistic(0), 0.5, places=5)
        self.assertGreater(logistic(5), 0.99)
        self.assertLess(logistic(-5), 0.01)

    def test_clamp(self):
        self.assertEqual(clamp(1.5), 1.0)
        self.assertEqual(clamp(-0.1), 0.0)
        self.assertEqual(clamp(0.7), 0.7)


class TestTeamFormAgent(unittest.TestCase):

    def setUp(self):
        # Patch Elo loader to avoid parquet dependency
        self.agent = TeamFormAgent(target_date=date(2026, 4, 8))
        self.agent._elo_ratings = {"BOS": 1560.0, "TOR": 1490.0}

    def test_returns_agent_signal(self):
        ctx    = make_game("BOS", "TOR")
        signal = self.agent.analyze(ctx)
        self.assertIsInstance(signal, AgentSignal)

    def test_home_advantage_when_better_xg(self):
        ctx    = make_game("BOS", "TOR", home_xgf=3.2, away_xgf=2.0, home_cf=0.54, away_cf=0.46)
        signal = self.agent.analyze(ctx)
        self.assertGreater(signal.raw_score, 0.5)
        self.assertEqual(signal.pick_direction, "home")

    def test_away_lean_when_away_dominates(self):
        # Away team has much higher Elo + better stats
        self.agent._elo_ratings = {"BOS": 1450.0, "TOR": 1590.0}
        ctx = make_game("BOS", "TOR", home_xgf=2.0, away_xgf=3.2, home_cf=0.45, away_cf=0.55)
        signal = self.agent.analyze(ctx)
        self.assertLess(signal.raw_score, 0.5)
        self.assertEqual(signal.pick_direction, "away")

    def test_neutral_without_advanced_stats(self):
        ctx = make_game()
        ctx.home.cf_pct_last5 = None
        ctx.home.xg_for_last5 = None
        ctx.away.cf_pct_last5 = None
        ctx.away.xg_for_last5 = None
        signal = self.agent.analyze(ctx)
        # Should still produce a signal (Elo still works)
        self.assertIsInstance(signal, AgentSignal)

    def test_factors_populated(self):
        ctx    = make_game("BOS", "TOR")
        signal = self.agent.analyze(ctx)
        self.assertIn("elo_home", signal.factors)
        self.assertIn("elo_away", signal.factors)

    def test_confidence_in_range(self):
        ctx    = make_game()
        signal = self.agent.analyze(ctx)
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)


class TestPlayerFormAgent(unittest.TestCase):

    def setUp(self):
        # Inject empty parquet — agent will use GameContext aggregates as fallback
        import pandas as pd
        self.agent = PlayerFormAgent(sk=pd.DataFrame())

    def test_returns_agent_signal(self):
        ctx    = make_game()
        signal = self.agent.analyze(ctx)
        self.assertIsInstance(signal, AgentSignal)

    def test_injury_news_reduces_score(self):
        ctx = make_game(
            home_news=["Star forward ruled out with lower-body injury"],
            away_news=[],
        )
        signal = self.agent.analyze(ctx)
        # Home team has injury → score should lean away
        self.assertLess(signal.raw_score, 0.5)

    def test_return_news_boosts_score(self):
        ctx = make_game(
            home_news=["Key player returns to lineup after being cleared"],
            away_news=["Star center day-to-day with lower-body injury"],
        )
        signal = self.agent.analyze(ctx)
        # Home has good news, away has bad news → lean home
        self.assertGreater(signal.raw_score, 0.5)

    def test_pp_strength_factor_in_factors(self):
        ctx    = make_game()
        signal = self.agent.analyze(ctx)
        self.assertIn("pp_score", signal.factors)

    def test_confidence_range(self):
        ctx    = make_game()
        signal = self.agent.analyze(ctx)
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)


class TestGoalieFormAgent(unittest.TestCase):

    def setUp(self):
        import pandas as pd
        self.agent = GoalieFormAgent(gl=pd.DataFrame())

    def test_returns_signal(self):
        ctx    = make_game()
        signal = self.agent.analyze(ctx)
        self.assertIsInstance(signal, AgentSignal)

    def test_better_goalie_wins_matchup(self):
        ctx = make_game(home_goalie_spp5=0.935, away_goalie_spp5=0.880)
        signal = self.agent.analyze(ctx)
        self.assertGreater(signal.raw_score, 0.5)
        self.assertEqual(signal.pick_direction, "home")

    def test_unconfirmed_reduces_confidence(self):
        ctx_conf   = make_game(home_goalie_status="confirmed", away_goalie_status="confirmed")
        ctx_unconf = make_game(home_goalie_status="unconfirmed", away_goalie_status="unconfirmed")
        sig_conf   = self.agent.analyze(ctx_conf)
        sig_unconf = self.agent.analyze(ctx_unconf)
        self.assertGreater(sig_conf.confidence, sig_unconf.confidence)

    def test_away_elite_goalie_signals_away(self):
        ctx = make_game(home_goalie_spp5=0.885, away_goalie_spp5=0.940)
        signal = self.agent.analyze(ctx)
        self.assertLess(signal.raw_score, 0.5)

    def test_reasoning_contains_goalie_names(self):
        ctx    = make_game()
        signal = self.agent.analyze(ctx)
        self.assertIn("Home Goalie", signal.reasoning)

    def test_no_goalie_returns_neutral(self):
        ctx = make_game()
        ctx.home.goalie = None
        ctx.away.goalie = None
        signal = self.agent.analyze(ctx)
        self.assertEqual(signal.pick_direction, "neutral")
        self.assertEqual(signal.confidence, 0.0)


class TestScheduleAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ScheduleAgent()

    def test_b2b_away_favors_home(self):
        ctx = make_game(home_b2b=False, away_b2b=True, home_rest=2, away_rest=1)
        signal = self.agent.analyze(ctx)
        self.assertGreater(signal.raw_score, 0.5)

    def test_b2b_home_favors_away(self):
        ctx = make_game(home_b2b=True, away_b2b=False, home_rest=1, away_rest=2)
        signal = self.agent.analyze(ctx)
        self.assertLess(signal.raw_score, 0.5)

    def test_equal_rest_is_neutral(self):
        ctx = make_game(home_b2b=False, away_b2b=False, home_rest=2, away_rest=2)
        signal = self.agent.analyze(ctx)
        # Should be close to 0.5 (small home advantage from travel)
        self.assertAlmostEqual(signal.raw_score, 0.5, delta=0.10)

    def test_altitude_factor_for_col(self):
        ctx = make_game(home_team="COL", away_team="FLA",
                        home_b2b=False, away_b2b=False, home_rest=2, away_rest=2)
        signal = self.agent.analyze(ctx)
        # Colorado home altitude should give slight boost
        self.assertGreater(signal.raw_score, 0.50)
        self.assertIn("altitude", signal.reasoning.lower())

    def test_travel_distance_bos_to_lak(self):
        dist = travel_distance("BOS", "LAK")
        self.assertIsNotNone(dist)
        self.assertGreater(dist, 2500)   # Boston → LA is ~2600 miles

    def test_travel_distance_bos_to_phi(self):
        dist = travel_distance("BOS", "PHI")
        self.assertIsNotNone(dist)
        self.assertLess(dist, 500)       # Boston → Philly is ~300 miles

    def test_returns_signal(self):
        ctx    = make_game()
        signal = self.agent.analyze(ctx)
        self.assertIsInstance(signal, AgentSignal)

    def test_b2b_mismatch_raises_confidence(self):
        ctx_even   = make_game(home_b2b=False, away_b2b=False)
        ctx_mismatch = make_game(home_b2b=False, away_b2b=True)
        sig_even   = self.agent.analyze(ctx_even)
        sig_mismatch = self.agent.analyze(ctx_mismatch)
        self.assertGreater(sig_mismatch.confidence, sig_even.confidence)


class TestSentimentAgent(unittest.TestCase):

    def setUp(self):
        self.agent = SentimentAgent()

    def test_neutral_no_news(self):
        ctx = make_game(home_news=[], away_news=[])
        signal = self.agent.analyze(ctx)
        self.assertEqual(signal.pick_direction, "neutral")
        self.assertAlmostEqual(signal.raw_score, 0.5, delta=0.01)
        self.assertEqual(signal.confidence, 0.0)

    def test_injury_news_negative(self):
        score, matched = _score_headlines(["Star player ruled out with lower-body injury"])
        self.assertLess(score, 0.0)
        self.assertGreater(len(matched), 0)

    def test_return_news_positive(self):
        score, matched = _score_headlines(["Key player returns to lineup after being cleared"])
        self.assertGreater(score, 0.0)

    def test_multiplier_neutral_at_zero(self):
        self.assertAlmostEqual(_score_to_multiplier(0.0), 1.0, places=5)

    def test_multiplier_capped(self):
        mult_high = _score_to_multiplier(100.0)
        mult_low  = _score_to_multiplier(-100.0)
        self.assertEqual(mult_high, 1.30)
        self.assertEqual(mult_low,  0.70)

    def test_away_injury_boosts_home(self):
        ctx = make_game(
            away_news=["Key forward ruled out, day-to-day with concussion"],
            home_news=[],
        )
        signal = self.agent.analyze(ctx)
        self.assertGreater(signal.raw_score, 0.5)   # home benefits from away injury

    def test_home_injury_reduces_home_signal(self):
        ctx = make_game(
            home_news=["Top scorer ruled out for tonight, scratched from lineup"],
            away_news=[],
        )
        signal = self.agent.analyze(ctx)
        self.assertLess(signal.raw_score, 0.5)

    def test_multiplier_in_factors(self):
        ctx    = make_game(home_news=["Player returns"], away_news=[])
        signal = self.agent.analyze(ctx)
        self.assertIn("multiplier", signal.factors)

    def test_both_teams_injured_near_neutral(self):
        ctx = make_game(
            home_news=["Forward day-to-day with injury"],
            away_news=["Defender ruled out with injury"],
        )
        signal = self.agent.analyze(ctx)
        # Both injured — should be close to neutral
        self.assertAlmostEqual(signal.raw_score, 0.5, delta=0.08)


class TestEnsembleIntegration(unittest.TestCase):
    """Smoke test: run all 5 agents on the same game and check outputs."""

    def setUp(self):
        import pandas as pd
        empty_df = pd.DataFrame()
        self.agents = [
            TeamFormAgent(target_date=date(2026, 4, 8)),
            PlayerFormAgent(sk=empty_df),
            GoalieFormAgent(gl=empty_df),
            ScheduleAgent(),
            SentimentAgent(),
        ]
        # Inject mock Elo to avoid parquet load
        self.agents[0]._elo_ratings = {"BOS": 1555.0, "TOR": 1495.0}

    def test_all_agents_return_signals(self):
        ctx = MockGameContext()
        for agent in self.agents:
            signal = agent.analyze(ctx)
            self.assertIsInstance(signal, AgentSignal)
            self.assertIn(signal.pick_direction, ("home", "away", "neutral"))
            self.assertGreaterEqual(signal.raw_score, 0.0)
            self.assertLessEqual(signal.raw_score, 1.0)
            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)

    def test_all_agents_print(self):
        ctx = MockGameContext()
        for agent in self.agents:
            signal = agent.analyze(ctx)
            line   = str(signal)
            self.assertIn(agent.agent_id, line)

    def test_weighted_ensemble_score(self):
        """Simple ensemble: weighted average of signals."""
        ctx     = MockGameContext()
        signals = [a.analyze(ctx) for a in self.agents]
        weights = [a.weight for a in self.agents]

        # Weighted average raw_score
        total_w    = sum(weights)
        composite  = sum(s.raw_score * w for s, w in zip(signals, weights)) / total_w

        self.assertGreater(composite, 0.0)
        self.assertLess(composite, 1.0)
        # For BOS (home, higher Elo, better advanced stats, away B2B) → expect home lean
        self.assertGreater(composite, 0.50)


if __name__ == "__main__":
    print("=" * 65)
    print("  NHL MAS — Module 2 Agent Tests")
    print("=" * 65)
    unittest.main(verbosity=2)
