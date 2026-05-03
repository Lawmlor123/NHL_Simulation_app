"""
test_integration.py
-------------------
Module 1 → Module 2 integration tests.

Verifies that GameContext objects (from Module 1) flow correctly
into all 5 agents (Module 2) and produce sensible ensemble output.

No parquet files or live API calls required — uses GameContext objects
built directly from the real dataclasses (not mock subclasses).

Run with:
    python test_integration.py
"""

import sys
import unittest
import pandas as pd
from pathlib import Path
from datetime import date

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

# Real dataclasses (Module 1 output contract)
from game_context import GameContext, TeamContext, GoalieContext, OddsContext

# All five agents (Module 2)
from agents import (
    TeamFormAgent, PlayerFormAgent, GoalieFormAgent,
    ScheduleAgent, SentimentAgent, AgentSignal,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — build real GameContext objects (not mocks, actual dataclasses)
# ═══════════════════════════════════════════════════════════════════════════════

def make_goalie(
    name="Test Goalie", status="confirmed",
    spp5=0.915, gaa5=2.50, win5=0.60, gsaa=5.0,
) -> GoalieContext:
    return GoalieContext(
        player_id=1001, name=name, status=status,
        gaa_live=gaa5, svpct_live=spp5,
        wins_live=22, losses_live=14,
        rolling_savePct_avg3=spp5 + 0.002,
        rolling_savePct_avg5=spp5,
        rolling_savePct_avg10=spp5 - 0.002,
        rolling_gaa_avg3=gaa5 - 0.1,
        rolling_gaa_avg5=gaa5,
        rolling_win_rate_3=win5,
        rolling_win_rate_5=win5,
        rolling_win_rate_10=win5 - 0.05,
        rolling_shotsAgainst_avg5=30.0,
        gsaa_season=gsaa,
    )


def make_team(
    team="BOS", is_home=True,
    rest=2, b2b=False,
    cf=0.520, xgf=2.80, xga=2.50,
    goals5=0.27, shots5=3.1, points5=0.55, pp5=0.08,
    goals10=0.25,
    goalie_name="Home Goalie", goalie_status="confirmed",
    goalie_spp5=0.915, goalie_gsaa=5.0,
    news=None,
) -> TeamContext:
    return TeamContext(
        team=team, is_home=is_home,
        rest_days=rest, is_b2b=b2b,
        sk_goals_avg3_mean=goals5 + 0.02,
        sk_goals_avg5_mean=goals5,
        sk_goals_avg10_mean=goals10,
        sk_goals_avg20_mean=goals10 - 0.02,
        sk_shots_avg3_mean=shots5 + 0.1,
        sk_shots_avg5_mean=shots5,
        sk_shots_avg10_mean=shots5 - 0.1,
        sk_points_avg5_mean=points5,
        sk_points_avg10_mean=points5 - 0.03,
        sk_pp_goals_avg5_mean=pp5,
        sk_pp_goals_avg10_mean=pp5 - 0.01,
        sk_hits_avg5_mean=2.1,
        sk_takeaways_avg5_mean=0.9,
        sk_giveaways_avg5_mean=0.8,
        sk_plus_minus_avg5_mean=0.15 if is_home else -0.10,
        sk_goals_avg5_sum=goals5 * 18,
        sk_shots_avg5_sum=shots5 * 18,
        sk_points_avg5_sum=points5 * 18,
        sk_pp_goals_avg5_sum=pp5 * 18,
        cf_pct_last5=cf,
        ff_pct_last5=cf + 0.005,
        xg_for_last5=xgf,
        xg_against_last5=xga,
        xg_pct_last5=xgf / (xgf + xga),
        sh_attempts_last5=58.0,
        goalie=make_goalie(
            name=goalie_name, status=goalie_status,
            spp5=goalie_spp5, gsaa=goalie_gsaa,
        ),
        news_headlines=news or [],
    )


def make_context(
    home_team="BOS", away_team="TOR",
    home_rest=2, away_rest=2,
    home_b2b=False, away_b2b=False,
    home_cf=0.525, away_cf=0.490,
    home_xgf=2.85, away_xgf=2.45,
    home_goalie_spp5=0.920, away_goalie_spp5=0.900,
    home_goalie_status="confirmed", away_goalie_status="confirmed",
    home_goalie_gsaa=8.0, away_goalie_gsaa=-2.0,
    home_news=None, away_news=None,
    home_ml=-140, away_ml=120,
) -> GameContext:
    return GameContext(
        game_id=2026020500,
        game_date="2026-04-08",
        start_time="2026-04-08T23:00:00Z",
        season="20252026",
        game_type=2,
        home=make_team(
            team=home_team, is_home=True,
            rest=home_rest, b2b=home_b2b,
            cf=home_cf, xgf=home_xgf, xga=2.50,
            goalie_name=f"{home_team} Starter",
            goalie_status=home_goalie_status,
            goalie_spp5=home_goalie_spp5,
            goalie_gsaa=home_goalie_gsaa,
            news=home_news or [],
        ),
        away=make_team(
            team=away_team, is_home=False,
            rest=away_rest, b2b=away_b2b,
            cf=away_cf, xgf=away_xgf, xga=2.70,
            goalie_name=f"{away_team} Starter",
            goalie_status=away_goalie_status,
            goalie_spp5=away_goalie_spp5,
            goalie_gsaa=away_goalie_gsaa,
            news=away_news or [],
        ),
        odds=OddsContext(
            home_ml=home_ml, away_ml=away_ml,
            home_implied=100/(abs(home_ml)+100) if home_ml < 0 else 100/(home_ml+100),
            away_implied=100/(abs(away_ml)+100) if away_ml < 0 else 100/(away_ml+100),
            point_spread=-1.5,
        ),
        goalie_confirmed=(home_goalie_status in ("confirmed","likely") and
                          away_goalie_status in ("confirmed","likely")),
        has_advanced_stats=True,
        has_odds=True,
        has_news=bool(home_news or away_news),
    )


def make_all_agents() -> list:
    """Return all 5 agents with mock Elo injected (no parquet needed)."""
    team_agent   = TeamFormAgent(target_date=date(2026, 4, 8))
    player_agent = PlayerFormAgent(sk=pd.DataFrame())
    goalie_agent = GoalieFormAgent(gl=pd.DataFrame())
    sched_agent  = ScheduleAgent()
    sent_agent   = SentimentAgent()

    # Inject Elo so TeamFormAgent doesn't try to load parquet
    team_agent._elo_ratings = {
        "BOS": 1560.0, "TOR": 1490.0, "COL": 1530.0, "FLA": 1510.0,
        "EDM": 1545.0, "VAN": 1485.0, "NYR": 1520.0, "CAR": 1535.0,
    }
    return [team_agent, player_agent, goalie_agent, sched_agent, sent_agent]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestGameContextToAgents(unittest.TestCase):
    """Verify real GameContext dataclasses flow into agents without errors."""

    def setUp(self):
        self.agents = make_all_agents()
        self.ctx    = make_context()

    def test_all_agents_accept_game_context(self):
        for agent in self.agents:
            signal = agent.analyze(self.ctx)
            self.assertIsInstance(signal, AgentSignal,
                f"{agent.agent_id} did not return AgentSignal")

    def test_no_agent_raises_on_valid_context(self):
        for agent in self.agents:
            try:
                agent.analyze(self.ctx)
            except Exception as e:
                self.fail(f"{agent.agent_id} raised exception: {e}")

    def test_all_signals_have_valid_direction(self):
        valid = {"home", "away", "neutral"}
        for agent in self.agents:
            signal = agent.analyze(self.ctx)
            self.assertIn(signal.pick_direction, valid,
                f"{agent.agent_id} returned invalid direction: {signal.pick_direction}")

    def test_all_signals_have_valid_score_range(self):
        for agent in self.agents:
            signal = agent.analyze(self.ctx)
            self.assertGreaterEqual(signal.raw_score, 0.0,
                f"{agent.agent_id} raw_score below 0")
            self.assertLessEqual(signal.raw_score, 1.0,
                f"{agent.agent_id} raw_score above 1")

    def test_all_signals_have_valid_confidence_range(self):
        for agent in self.agents:
            signal = agent.analyze(self.ctx)
            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)

    def test_all_signals_have_reasoning(self):
        for agent in self.agents:
            signal = agent.analyze(self.ctx)
            self.assertIsInstance(signal.reasoning, str)
            self.assertGreater(len(signal.reasoning), 0,
                f"{agent.agent_id} returned empty reasoning")

    def test_all_signals_have_agent_id(self):
        for agent in self.agents:
            signal = agent.analyze(self.ctx)
            self.assertEqual(signal.agent_id, agent.agent_id)


class TestSignalDirectionality(unittest.TestCase):
    """Verify agents produce logically correct directional signals."""

    def setUp(self):
        self.agents = make_all_agents()

    def test_dominant_home_team_leans_home(self):
        """When home team is clearly superior on all metrics, ensemble should lean home."""
        ctx = make_context(
            home_cf=0.58, away_cf=0.42,
            home_xgf=3.5, away_xgf=1.8,
            home_goalie_spp5=0.935, away_goalie_spp5=0.875,
            home_rest=3, away_rest=1, away_b2b=True,
            home_goalie_gsaa=15.0, away_goalie_gsaa=-8.0,
        )
        # Inject matching Elo
        self.agents[0]._elo_ratings = {"BOS": 1620.0, "TOR": 1430.0}
        composite = self._ensemble_score(ctx)
        self.assertGreater(composite, 0.55,
            f"Expected clear home lean, got composite={composite:.3f}")

    def test_dominant_away_team_leans_away(self):
        """When away team is clearly superior, ensemble should lean away."""
        ctx = make_context(
            home_cf=0.42, away_cf=0.58,
            home_xgf=1.8, away_xgf=3.5,
            home_goalie_spp5=0.875, away_goalie_spp5=0.935,
            home_rest=1, away_rest=3, home_b2b=True,
            home_goalie_gsaa=-8.0, away_goalie_gsaa=15.0,
        )
        self.agents[0]._elo_ratings = {"BOS": 1430.0, "TOR": 1620.0}
        composite = self._ensemble_score(ctx)
        self.assertLess(composite, 0.45,
            f"Expected clear away lean, got composite={composite:.3f}")

    def test_equal_teams_near_neutral(self):
        """When both teams are equal on all metrics, composite should be near 0.5."""
        ctx = make_context(
            home_cf=0.500, away_cf=0.500,
            home_xgf=2.60, away_xgf=2.60,
            home_goalie_spp5=0.910, away_goalie_spp5=0.910,
            home_rest=2, away_rest=2,
            home_goalie_gsaa=0.0, away_goalie_gsaa=0.0,
        )
        self.agents[0]._elo_ratings = {"BOS": 1500.0, "TOR": 1500.0}
        composite = self._ensemble_score(ctx)
        self.assertAlmostEqual(composite, 0.5, delta=0.08,
            msg=f"Expected near-neutral, got composite={composite:.3f}")

    def _ensemble_score(self, ctx) -> float:
        """Simple weighted ensemble."""
        signals   = [a.analyze(ctx) for a in self.agents]
        weights   = [a.weight for a in self.agents]
        total_w   = sum(weights)
        return sum(s.raw_score * w for s, w in zip(signals, weights)) / total_w


class TestEdgeCases(unittest.TestCase):
    """Verify agents handle missing/partial data gracefully."""

    def setUp(self):
        self.agents = make_all_agents()

    def test_no_advanced_stats(self):
        """Agents should handle GameContext with no xG/CF% data."""
        ctx = make_context()
        ctx.home.cf_pct_last5    = None
        ctx.home.xg_for_last5    = None
        ctx.home.xg_against_last5 = None
        ctx.away.cf_pct_last5    = None
        ctx.away.xg_for_last5    = None
        ctx.away.xg_against_last5 = None
        for agent in self.agents:
            signal = agent.analyze(ctx)
            self.assertIsInstance(signal, AgentSignal)

    def test_no_goalie_data(self):
        """GoalieFormAgent should return neutral when both goalies are None."""
        from agents.goalie_form_agent import GoalieFormAgent as GFA
        agent = GFA(gl=pd.DataFrame())
        ctx   = make_context()
        ctx.home.goalie = None
        ctx.away.goalie = None
        signal = agent.analyze(ctx)
        self.assertEqual(signal.pick_direction, "neutral")
        self.assertEqual(signal.confidence, 0.0)

    def test_unconfirmed_goalies_reduce_confidence(self):
        from agents.goalie_form_agent import GoalieFormAgent as GFA
        agent_confirmed   = GFA(gl=pd.DataFrame())
        agent_unconfirmed = GFA(gl=pd.DataFrame())
        ctx_conf   = make_context(home_goalie_status="confirmed",  away_goalie_status="confirmed")
        ctx_unconf = make_context(home_goalie_status="unconfirmed", away_goalie_status="unconfirmed")
        sig_conf   = agent_confirmed.analyze(ctx_conf)
        sig_unconf = agent_unconfirmed.analyze(ctx_unconf)
        self.assertGreater(sig_conf.confidence, sig_unconf.confidence)

    def test_b2b_away_always_helps_home_schedule_agent(self):
        from agents.schedule_agent import ScheduleAgent as SA
        agent = SA()
        ctx   = make_context(home_b2b=False, away_b2b=True, home_rest=2, away_rest=1)
        signal = agent.analyze(ctx)
        self.assertGreater(signal.raw_score, 0.5)

    def test_heavy_injury_news_affects_sentiment(self):
        from agents.sentiment_agent import SentimentAgent as SentA
        agent = SentA()
        ctx = make_context(
            away_news=["Star center ruled out, suspended indefinitely, LTIR"],
            home_news=[],
        )
        signal = agent.analyze(ctx)
        self.assertGreater(signal.raw_score, 0.5)
        self.assertGreater(signal.factors.get("multiplier", 1.0), 1.0)

    def test_no_news_sentiment_is_neutral(self):
        from agents.sentiment_agent import SentimentAgent as SentA
        agent  = SentA()
        ctx    = make_context()
        signal = agent.analyze(ctx)
        self.assertEqual(signal.pick_direction, "neutral")
        self.assertAlmostEqual(signal.raw_score, 0.5, delta=0.02)

    def test_missing_odds_context_does_not_crash(self):
        ctx       = make_context()
        ctx.odds  = OddsContext()   # all None
        for agent in self.agents:
            signal = agent.analyze(ctx)
            self.assertIsInstance(signal, AgentSignal)


class TestEnsembleWeighting(unittest.TestCase):
    """Verify ensemble math is correct before Module 3 takes over."""

    def setUp(self):
        self.agents  = make_all_agents()
        self.ctx     = make_context()

    def test_weighted_average_in_range(self):
        signals  = [a.analyze(self.ctx) for a in self.agents]
        weights  = [a.weight for a in self.agents]
        total_w  = sum(weights)
        composite = sum(s.raw_score * w for s, w in zip(signals, weights)) / total_w
        self.assertGreater(composite, 0.0)
        self.assertLess(composite, 1.0)

    def test_sentiment_multiplier_is_accessible(self):
        """SentimentAgent multiplier should be in factors for meta-learner use."""
        sent   = [a for a in self.agents if a.agent_id == "sentiment"][0]
        signal = sent.analyze(self.ctx)
        self.assertIn("multiplier", signal.factors)

    def test_goalie_agent_has_highest_weight(self):
        goalie_w = next(a.weight for a in self.agents if a.agent_id == "goalie_form")
        other_w  = [a.weight for a in self.agents if a.agent_id != "goalie_form"]
        self.assertGreater(goalie_w, max(other_w),
            "GoalieFormAgent should have the highest default weight")

    def test_confidence_floor_filters_low_confidence(self):
        """Signals below confidence_floor should be flagged invalid."""
        for agent in self.agents:
            signal = agent.analyze(self.ctx)
            valid  = agent.is_signal_valid(signal)
            if signal.confidence < agent.confidence_floor:
                self.assertFalse(valid)
            else:
                self.assertTrue(valid)

    def test_valid_signals_only_ensemble(self):
        """Ensemble using only valid signals should still produce a score."""
        signals = [a.analyze(self.ctx) for a in self.agents]
        valid   = [(s, a.weight) for s, a in zip(signals, self.agents)
                   if a.is_signal_valid(s)]
        if valid:
            total_w   = sum(w for _, w in valid)
            composite = sum(s.raw_score * w for s, w in valid) / total_w
            self.assertGreater(composite, 0.0)
            self.assertLess(composite, 1.0)

    def test_print_full_signal_report(self):
        """Smoke test: print the full signal report for a sample game."""
        print("\n── Integration Signal Report: BOS vs TOR (2026-04-08) ──")
        signals   = []
        for agent in self.agents:
            signal = agent.analyze(self.ctx)
            signals.append(signal)
            valid_flag = "✅" if agent.is_signal_valid(signal) else "⚠ "
            print(f"  {valid_flag} {signal}")

        weights   = [a.weight for a in self.agents]
        total_w   = sum(weights)
        composite = sum(s.raw_score * w for s, w in zip(signals, weights)) / total_w
        direction = "HOME" if composite > 0.52 else ("AWAY" if composite < 0.48 else "NEUTRAL")
        print(f"\n  ── ENSEMBLE: {composite:.3f} → {direction} ──\n")
        self.assertIsInstance(composite, float)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Module 1 → Module 2 Integration Tests")
    print("=" * 65 + "\n")
    unittest.main(verbosity=2)
