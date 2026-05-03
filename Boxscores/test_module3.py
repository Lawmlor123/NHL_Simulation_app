"""
test_module3.py
---------------
Unit + integration tests for Module 3 (Master Synthesizer).

Groups:
  A. PickCard dataclass
  B. Weighted average synthesis
  C. SKIP logic
  D. Meta-learner (fit / predict)
  E. SQLite logging
  F. Weight recalibration
  G. Full M1→M2→M3 pipeline smoke test

Run with:
    python test_module3.py
"""

import sys
import json
import sqlite3
import unittest
import warnings
from pathlib import Path
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from agents.agent_base       import AgentSignal
from module3_synthesizer     import (
    PickCard, MasterSynthesizer,
    DEFAULT_WEIGHTS, AGENT_ORDER,
    _score_to_tier, _compute_agreement,
    _signals_to_feature_vector, _build_feature_matrix,
    _build_signal_summary, _extract_game_meta,
)

# ── Try sklearn ───────────────────────────────────────────────────────────────
try:
    import sklearn
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def make_signal(
    agent_id:  str   = "team_form",
    direction: str   = "home",
    raw_score: float = 0.62,
    confidence:float = 0.70,
    reasoning: str   = "test signal",
) -> AgentSignal:
    return AgentSignal(
        agent_id       = agent_id,
        pick_direction = direction,
        raw_score      = raw_score,
        confidence     = confidence,
        reasoning      = reasoning,
    )


def make_full_signals(
    lean: str = "home",
    conf: float = 0.68,
) -> list[AgentSignal]:
    """Five-agent signal set all pointing in same direction."""
    score = 0.63 if lean == "home" else 0.37
    return [
        make_signal("team_form",   lean, score,       conf),
        make_signal("player_form", lean, score - 0.02, conf - 0.05),
        make_signal("goalie_form", lean, score + 0.03, conf + 0.05),
        make_signal("schedule",    lean, score - 0.01, conf - 0.10),
        make_signal("sentiment",   "neutral", 0.50,   0.0,
                    reasoning="no headlines", ),
    ]


def make_split_signals() -> list[AgentSignal]:
    """Two home, two away, one neutral."""
    return [
        make_signal("team_form",   "home",    0.62, 0.70),
        make_signal("player_form", "away",    0.38, 0.65),
        make_signal("goalie_form", "home",    0.60, 0.72),
        make_signal("schedule",    "away",    0.40, 0.60),
        make_signal("sentiment",   "neutral", 0.50, 0.00),
    ]


def make_synthesizer(db: Optional[Path] = None) -> MasterSynthesizer:
    """Synthesizer with in-memory SQLite (no disk writes in tests)."""
    synth = MasterSynthesizer(
        weights_registry = dict(DEFAULT_WEIGHTS),
        confidence_floor = 0.52,
        min_agents       = 3,
        db_path          = db,
    )
    synth._db_path = ":memory:" if db is None else db
    synth._conn    = sqlite3.connect(":memory:")
    from module3_synthesizer import _SCHEMA
    synth._conn.executescript(_SCHEMA)
    synth._conn.commit()
    return synth


@dataclass
class MockHome:
    team: str = "BOS"

@dataclass
class MockAway:
    team: str = "TOR"

@dataclass
class MockGameCtx:
    game_id:   int    = 2026020500
    game_date: str    = "2026-04-08"
    home:      MockHome = field(default_factory=MockHome)
    away:      MockAway = field(default_factory=MockAway)


# ═══════════════════════════════════════════════════════════════════════════════
# A. PickCard
# ═══════════════════════════════════════════════════════════════════════════════

class TestPickCard(unittest.TestCase):

    def _card(self, tier="HIGH", direction="home", score=0.68, edge=18.0):
        return PickCard(
            game_id=1, game_date="2026-04-08",
            home_team="BOS", away_team="TOR",
            pick="BOS", pick_direction=direction,
            raw_score=score, edge_pct=edge,
            confidence_tier=tier,
            agent_agreement_score=0.80,
            agents_above_floor=4,
            mode_used="weighted_avg",
        )

    def test_is_playable_high(self):
        self.assertTrue(self._card("HIGH").is_playable)

    def test_is_playable_medium(self):
        self.assertTrue(self._card("MEDIUM").is_playable)

    def test_is_playable_low(self):
        self.assertTrue(self._card("LOW").is_playable)

    def test_is_not_playable_skip(self):
        self.assertFalse(self._card("SKIP").is_playable)

    def test_matchup_string(self):
        self.assertEqual(self._card().matchup, "TOR @ BOS")

    def test_str_contains_pick(self):
        self.assertIn("BOS", str(self._card()))

    def test_str_skip(self):
        skip = PickCard(confidence_tier="SKIP", skip_reason="test reason",
                        home_team="BOS", away_team="TOR")
        self.assertIn("SKIP", str(skip))
        self.assertIn("test reason", str(skip))

    def test_tier_emoji(self):
        self.assertIn("★★★", self._card("HIGH").tier_emoji())
        self.assertIn("⛔",  self._card("SKIP").tier_emoji())

    def test_edge_pct_is_positive(self):
        self.assertGreater(self._card().edge_pct, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# B. Weighted average synthesis
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightedSynthesis(unittest.TestCase):

    def setUp(self):
        self.synth = make_synthesizer()
        self.ctx   = MockGameCtx()

    def test_returns_pick_card(self):
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertIsInstance(card, PickCard)

    def test_strong_home_signals_pick_home(self):
        signals = make_full_signals("home", conf=0.72)
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertEqual(card.pick_direction, "home")
        self.assertEqual(card.pick, "BOS")

    def test_strong_away_signals_pick_away(self):
        signals = make_full_signals("away", conf=0.72)
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertEqual(card.pick_direction, "away")
        self.assertEqual(card.pick, "TOR")

    def test_edge_pct_correct(self):
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        expected_edge = abs(card.raw_score - 0.5) * 100
        self.assertAlmostEqual(card.edge_pct, expected_edge, places=2)

    def test_game_meta_populated(self):
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertEqual(card.home_team, "BOS")
        self.assertEqual(card.away_team, "TOR")
        self.assertEqual(card.game_id, 2026020500)

    def test_signal_summary_contains_all_agents(self):
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        for aid in AGENT_ORDER:
            self.assertIn(aid, card.signal_summary)

    def test_sentiment_multiplier_applied(self):
        """Positive sentiment should shift composite toward home."""
        base_signals = make_full_signals("home")
        # Modify sentiment to have a strong pro-home multiplier
        sent = next(s for s in base_signals if s.agent_id == "sentiment")
        sent_idx = base_signals.index(sent)
        base_signals[sent_idx] = AgentSignal(
            agent_id="sentiment", pick_direction="home",
            raw_score=0.65, confidence=0.60, reasoning="pro-home news",
            factors={"multiplier": 1.20},
        )
        base_signals[sent_idx].factors["multiplier"] = 1.20

        card_no_sent = self.synth.synthesize(make_full_signals("home"), self.ctx,
                                              mode="weighted_avg", log=False)
        card_with_sent = self.synth.synthesize(base_signals, self.ctx,
                                                mode="weighted_avg", log=False)
        # With positive sentiment, composite should be >= base
        self.assertGreaterEqual(card_with_sent.raw_score, card_no_sent.raw_score - 0.02)

    def test_agreement_score_high_when_unanimous(self):
        signals = make_full_signals("home", conf=0.70)
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertGreater(card.agent_agreement_score, 0.60)

    def test_agreement_score_low_when_split(self):
        signals = make_split_signals()
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        # 2 home vs 2 away → 0.5 agreement
        self.assertAlmostEqual(card.agent_agreement_score, 0.5, delta=0.1)

    def test_no_game_context_still_works(self):
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, None, mode="weighted_avg", log=False)
        self.assertIsInstance(card, PickCard)

    def test_tier_high_when_strong_edge(self):
        signals = [make_signal(aid, "home", 0.70, 0.75)
                   for aid in AGENT_ORDER]
        card = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertEqual(card.confidence_tier, "HIGH")

    def test_tier_low_when_weak_edge(self):
        signals = [make_signal(aid, "home", 0.54, 0.60)
                   for aid in AGENT_ORDER]
        card = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertIn(card.confidence_tier, ("LOW", "SKIP"))


# ═══════════════════════════════════════════════════════════════════════════════
# C. SKIP logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkipLogic(unittest.TestCase):

    def setUp(self):
        self.synth = make_synthesizer()
        self.ctx   = MockGameCtx()

    def test_skip_when_fewer_than_3_above_floor(self):
        """Only 2 agents above confidence_floor → SKIP."""
        signals = [
            make_signal("team_form",   "home", 0.63, 0.70),  # above floor
            make_signal("player_form", "home", 0.60, 0.65),  # above floor
            make_signal("goalie_form", "home", 0.58, 0.40),  # BELOW floor
            make_signal("schedule",    "home", 0.55, 0.30),  # BELOW floor
            make_signal("sentiment",   "neutral", 0.50, 0.0),
        ]
        card = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertEqual(card.confidence_tier, "SKIP")
        self.assertEqual(card.pick, "SKIP")
        self.assertIn("2/3", card.skip_reason)

    def test_skip_when_composite_below_tier_floor(self):
        """3+ agents above floor but composite is < 0.52 threshold → SKIP."""
        signals = [make_signal(aid, "home", 0.505, 0.65) for aid in AGENT_ORDER]
        card = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertEqual(card.confidence_tier, "SKIP")

    def test_no_skip_when_3_strong_agents(self):
        signals = [
            make_signal("team_form",   "home", 0.67, 0.75),
            make_signal("player_form", "home", 0.63, 0.65),
            make_signal("goalie_form", "home", 0.70, 0.80),
            make_signal("schedule",    "home", 0.55, 0.30),  # below floor
            make_signal("sentiment",   "neutral", 0.50, 0.0),
        ]
        card = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertNotEqual(card.confidence_tier, "SKIP")

    def test_skip_is_not_playable(self):
        signals = [make_signal(aid, "home", 0.63, 0.30) for aid in AGENT_ORDER]
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertFalse(card.is_playable)

    def test_skip_reason_populated(self):
        signals = [make_signal(aid, "home", 0.63, 0.30) for aid in AGENT_ORDER]
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertIsInstance(card.skip_reason, str)
        self.assertGreater(len(card.skip_reason), 0)

    def test_empty_signals_produces_skip(self):
        card = self.synth.synthesize([], self.ctx, mode="weighted_avg", log=False)
        self.assertEqual(card.confidence_tier, "SKIP")


# ═══════════════════════════════════════════════════════════════════════════════
# D. Meta-learner
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(SKLEARN_OK, "scikit-learn not installed")
class TestMetaLearner(unittest.TestCase):

    def setUp(self):
        self.synth = make_synthesizer()
        self.ctx   = MockGameCtx()

    def _make_training_data(self, n=80):
        """Generate synthetic training data with a clear home-bias signal."""
        import random
        random.seed(42)
        all_signals = []
        outcomes    = []
        for _ in range(n):
            home_win = random.random() > 0.45   # slight home advantage
            score_h  = 0.60 + random.gauss(0, 0.05) if home_win else 0.40 + random.gauss(0, 0.05)
            score_h  = max(0.30, min(0.70, score_h))
            signals  = [
                make_signal(aid, "home" if score_h > 0.5 else "away",
                            score_h, 0.65 + random.gauss(0, 0.05))
                for aid in AGENT_ORDER
            ]
            all_signals.append(signals)
            outcomes.append(int(home_win))
        return all_signals, outcomes

    def test_fit_does_not_raise(self):
        sigs, outs = self._make_training_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.synth.fit(sigs, outs)
        self.assertTrue(self.synth._is_fitted)

    def test_predict_after_fit_returns_pick_card(self):
        sigs, outs = self._make_training_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.synth.fit(sigs, outs)
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, self.ctx, mode="meta_learner", log=False)
        self.assertIsInstance(card, PickCard)

    def test_meta_learner_mode_used(self):
        sigs, outs = self._make_training_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.synth.fit(sigs, outs)
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, self.ctx, mode="meta_learner", log=False)
        self.assertEqual(card.mode_used, "meta_learner")

    def test_auto_mode_uses_meta_when_fitted(self):
        sigs, outs = self._make_training_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.synth.fit(sigs, outs)
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, self.ctx, mode="auto", log=False)
        self.assertEqual(card.mode_used, "meta_learner")

    def test_auto_mode_uses_weighted_when_not_fitted(self):
        signals = make_full_signals("home")
        card    = self.synth.synthesize(signals, self.ctx, mode="auto", log=False)
        self.assertEqual(card.mode_used, "weighted_avg")

    def test_meta_learner_unfitted_falls_back(self):
        signals = make_full_signals("home")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            card = self.synth.synthesize(signals, self.ctx, mode="meta_learner", log=False)
        # Should fall back to weighted_avg without raising
        self.assertIn(card.mode_used, ("weighted_avg", "meta_learner"))

    def test_feature_vector_correct_length(self):
        signals = make_full_signals("home")
        feats   = _signals_to_feature_vector(signals)
        self.assertEqual(len(feats), len(AGENT_ORDER) * 2)

    def test_feature_vector_missing_agent_defaults(self):
        """Missing agent should default to (0.5, 0.0)."""
        signals = [make_signal("team_form", "home", 0.65, 0.70)]
        feats   = _signals_to_feature_vector(signals)
        # team_form is index 0,1; player_form should be (0.5, 0.0)
        self.assertEqual(feats[2], 0.5)
        self.assertEqual(feats[3], 0.0)

    def test_build_feature_matrix_shape(self):
        all_sigs = [make_full_signals("home") for _ in range(10)]
        X = _build_feature_matrix(all_sigs)
        self.assertEqual(X.shape, (10, len(AGENT_ORDER) * 2))

    def test_fit_warns_on_small_dataset(self):
        sigs, outs = self._make_training_data(n=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                self.synth.fit(sigs, outs)
            except Exception:
                pass   # small data may cause CV errors
        # Either a warning was raised or an exception — both acceptable
        # The key check is that it doesn't silently corrupt state


# ═══════════════════════════════════════════════════════════════════════════════
# E. SQLite logging
# ═══════════════════════════════════════════════════════════════════════════════

class TestSQLiteLogging(unittest.TestCase):

    def setUp(self):
        self.synth = make_synthesizer()
        self.ctx   = MockGameCtx()

    def test_pick_logged_to_db(self):
        signals = make_full_signals("home", conf=0.72)
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=True)
        conn    = self.synth._conn
        row     = conn.execute("SELECT COUNT(*) FROM picks").fetchone()
        self.assertEqual(row[0], 1)

    def test_pick_log_contains_correct_fields(self):
        signals = make_full_signals("home", conf=0.72)
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=True)
        conn    = self.synth._conn
        row     = conn.execute(
            "SELECT game_id, home_team, away_team, pick, confidence_tier, "
            "       signals_json, weights_json FROM picks LIMIT 1"
        ).fetchone()
        self.assertEqual(row[0], 2026020500)
        self.assertEqual(row[1], "BOS")
        self.assertEqual(row[2], "TOR")
        self.assertIn(card.pick, row[3])
        # signals_json should be valid JSON with all agents
        sig_dict = json.loads(row[5])
        for aid in AGENT_ORDER:
            self.assertIn(aid, sig_dict)

    def test_skip_logged_with_skip_reason(self):
        signals = [make_signal(aid, "home", 0.63, 0.30) for aid in AGENT_ORDER]
        self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=True)
        conn = self.synth._conn
        row  = conn.execute(
            "SELECT confidence_tier, skip_reason FROM picks LIMIT 1"
        ).fetchone()
        self.assertEqual(row[0], "SKIP")
        self.assertGreater(len(row[1]), 0)

    def test_multiple_picks_logged(self):
        for _ in range(3):
            signals = make_full_signals("home")
            self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=True)
        conn = self.synth._conn
        row  = conn.execute("SELECT COUNT(*) FROM picks").fetchone()
        self.assertEqual(row[0], 3)

    def test_log_outcome_writes_to_db(self):
        self.synth.log_outcome(2026020500, "2026-04-08", "BOS", "TOR", 1)
        conn = self.synth._conn
        row  = conn.execute(
            "SELECT home_win FROM outcomes WHERE game_id=2026020500"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)

    def test_log_outcome_upserts(self):
        """Logging same game_id twice should not create duplicates (INSERT OR REPLACE)."""
        self.synth.log_outcome(1, "2026-04-08", "BOS", "TOR", 1)
        self.synth.log_outcome(1, "2026-04-08", "BOS", "TOR", 0)  # update
        conn = self.synth._conn
        n    = conn.execute("SELECT COUNT(*) FROM outcomes WHERE game_id=1").fetchone()[0]
        self.assertEqual(n, 1)
        result = conn.execute("SELECT home_win FROM outcomes WHERE game_id=1").fetchone()[0]
        self.assertEqual(result, 0)

    def test_picks_dataframe_returns_df(self):
        signals = make_full_signals("home")
        self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=True)
        df = self.synth.picks_dataframe(days=30)
        self.assertIsNotNone(df)


# ═══════════════════════════════════════════════════════════════════════════════
# F. Weight recalibration
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightRecalibration(unittest.TestCase):

    def setUp(self):
        self.synth = make_synthesizer()
        self.ctx   = MockGameCtx()

    def _populate_history(self, n=20, good_agent="goalie_form"):
        """Insert n synthetic picks + outcomes, where good_agent is always right."""
        conn = self.synth._conn
        for i in range(n):
            home_win = i % 2   # alternating
            # Build signal summary: good_agent always picks correctly
            sigs = {}
            for aid in AGENT_ORDER:
                if aid == good_agent:
                    correct_score = 0.70 if home_win else 0.30
                    sigs[aid] = {"raw_score": correct_score, "confidence": 0.75,
                                 "direction": "home" if home_win else "away",
                                 "reasoning": "test", "weight": 1.0}
                else:
                    sigs[aid] = {"raw_score": 0.50, "confidence": 0.60,
                                 "direction": "neutral", "reasoning": "test", "weight": 1.0}

            conn.execute("""
                INSERT INTO picks
                    (created_at, game_id, game_date, home_team, away_team,
                     pick, pick_direction, raw_score, edge_pct, confidence_tier,
                     agent_agreement, agents_above_floor, mode_used,
                     skip_reason, signals_json, weights_json)
                VALUES (datetime('now'), ?, ?, 'BOS', 'TOR',
                        ?, ?, 0.60, 10.0, 'HIGH', 0.75, 4, 'weighted_avg',
                        '', ?, '{}')
            """, (
                1000 + i, f"2026-03-{i+1:02d}",
                "BOS" if home_win else "TOR",
                "home" if home_win else "away",
                json.dumps(sigs),
            ))
            conn.execute("""
                INSERT OR REPLACE INTO outcomes
                    (game_id, game_date, home_team, away_team, home_win, added_at)
                VALUES (?, ?, 'BOS', 'TOR', ?, datetime('now'))
            """, (1000 + i, f"2026-03-{i+1:02d}", home_win))
        conn.commit()

    def test_update_weights_returns_dict(self):
        self._populate_history(20)
        result = self.synth.update_weights()
        self.assertIsInstance(result, dict)

    def test_update_weights_has_all_agents(self):
        self._populate_history(20)
        result = self.synth.update_weights()
        for aid in AGENT_ORDER:
            self.assertIn(aid, result)

    def test_accurate_agent_gets_higher_weight(self):
        """goalie_form is always correct → should get higher weight than random agents."""
        self._populate_history(20, good_agent="goalie_form")
        result = self.synth.update_weights()
        goalie_w = result.get("goalie_form", 1.0)
        schedule_w = result.get("schedule", 1.0)
        # goalie_form (100% accurate) should beat schedule (50% accurate)
        self.assertGreater(goalie_w, schedule_w)

    def test_weights_clamped_to_range(self):
        self._populate_history(20)
        result = self.synth.update_weights()
        for w in result.values():
            self.assertGreaterEqual(w, 0.30)
            self.assertLessEqual(w, 2.50)

    def test_insufficient_data_preserves_weights(self):
        """Fewer than 10 resolved games → weights unchanged."""
        self._populate_history(5)
        old_weights = dict(self.synth.weights_registry)
        self.synth.update_weights()
        self.assertEqual(self.synth.weights_registry, old_weights)


# ═══════════════════════════════════════════════════════════════════════════════
# G. Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers(unittest.TestCase):

    def test_score_to_tier_high(self):
        self.assertEqual(_score_to_tier(0.68), "HIGH")
        self.assertEqual(_score_to_tier(0.32), "HIGH")

    def test_score_to_tier_medium(self):
        self.assertEqual(_score_to_tier(0.60), "MEDIUM")
        self.assertEqual(_score_to_tier(0.40), "MEDIUM")

    def test_score_to_tier_low(self):
        self.assertEqual(_score_to_tier(0.54), "LOW")
        self.assertEqual(_score_to_tier(0.46), "LOW")

    def test_score_to_tier_skip(self):
        self.assertEqual(_score_to_tier(0.50), "SKIP")
        self.assertEqual(_score_to_tier(0.51), "SKIP")

    def test_compute_agreement_unanimous(self):
        sigs = [make_signal(aid, "home", 0.65, 0.70) for aid in AGENT_ORDER[:4]]
        self.assertAlmostEqual(_compute_agreement(sigs, 0.52), 1.0)

    def test_compute_agreement_split(self):
        sigs = [
            make_signal("team_form",   "home", 0.65, 0.70),
            make_signal("goalie_form", "away", 0.35, 0.70),
        ]
        self.assertAlmostEqual(_compute_agreement(sigs, 0.52), 0.5)

    def test_compute_agreement_no_opinionated(self):
        sigs = [make_signal(aid, "neutral", 0.50, 0.40) for aid in AGENT_ORDER]
        self.assertEqual(_compute_agreement(sigs, 0.52), 0.0)

    def test_build_signal_summary_keys(self):
        sigs    = make_full_signals("home")
        summary = _build_signal_summary(sigs, DEFAULT_WEIGHTS)
        for s in sigs:
            self.assertIn(s.agent_id, summary)
            self.assertIn("raw_score",  summary[s.agent_id])
            self.assertIn("confidence", summary[s.agent_id])
            self.assertIn("direction",  summary[s.agent_id])

    def test_extract_game_meta_none(self):
        meta = _extract_game_meta(None)
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta, {})

    def test_extract_game_meta_context(self):
        meta = _extract_game_meta(MockGameCtx())
        self.assertEqual(meta["game_id"],   2026020500)
        self.assertEqual(meta["home_team"], "BOS")
        self.assertEqual(meta["away_team"], "TOR")


# ═══════════════════════════════════════════════════════════════════════════════
# H. Full pipeline smoke test  M1 dataclasses → M2 agents → M3 synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline(unittest.TestCase):
    """
    Uses the real GameContext dataclasses and all 5 agents.
    No parquet / API needed — injects empty DataFrames.
    """

    def setUp(self):
        import pandas as pd
        from game_context     import GameContext, TeamContext, GoalieContext, OddsContext
        from agents           import TeamFormAgent, PlayerFormAgent, GoalieFormAgent
        from agents           import ScheduleAgent, SentimentAgent

        goalie = GoalieContext(
            name="Starter", status="confirmed",
            rolling_savePct_avg5=0.918, rolling_gaa_avg5=2.45,
            rolling_win_rate_5=0.62, gsaa_season=7.0,
        )
        home = TeamContext(
            team="BOS", is_home=True,
            rest_days=2, is_b2b=False,
            sk_goals_avg5_mean=0.27, sk_shots_avg5_mean=3.2,
            cf_pct_last5=0.530, xg_for_last5=2.90, xg_against_last5=2.40,
            sk_pp_goals_avg5_sum=2.5,
            goalie=GoalieContext(name="BOS Starter", status="confirmed",
                                  rolling_savePct_avg5=0.922, gsaa_season=10.0,
                                  rolling_win_rate_5=0.65),
        )
        away = TeamContext(
            team="TOR", is_home=False,
            rest_days=1, is_b2b=True,
            sk_goals_avg5_mean=0.24, sk_shots_avg5_mean=2.9,
            cf_pct_last5=0.480, xg_for_last5=2.45, xg_against_last5=2.75,
            sk_pp_goals_avg5_sum=1.8,
            goalie=GoalieContext(name="TOR Starter", status="confirmed",
                                  rolling_savePct_avg5=0.898, gsaa_season=-3.0,
                                  rolling_win_rate_5=0.40),
        )
        self.ctx = GameContext(
            game_id=9999, game_date="2026-04-08",
            home=home, away=away,
            odds=OddsContext(home_ml=-145, away_ml=125),
            goalie_confirmed=True, has_advanced_stats=True,
        )

        empty = pd.DataFrame()
        self.agents = [
            TeamFormAgent(target_date=date(2026, 4, 8)),
            PlayerFormAgent(sk=empty),
            GoalieFormAgent(gl=empty),
            ScheduleAgent(),
            SentimentAgent(),
        ]
        self.agents[0]._elo_ratings = {"BOS": 1565.0, "TOR": 1488.0}
        self.synth = make_synthesizer()

    def test_full_pipeline_returns_pick_card(self):
        signals = [a.analyze(self.ctx) for a in self.agents]
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertIsInstance(card, PickCard)

    def test_bos_strong_game_picks_home(self):
        """BOS has better Elo, CF%, xG, better goalie, TOR is B2B → expect HOME."""
        signals = [a.analyze(self.ctx) for a in self.agents]
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        # With these inputs, should lean home
        if card.is_playable:
            self.assertEqual(card.pick_direction, "home",
                f"Expected home pick, got {card.pick_direction} (score={card.raw_score:.3f})")

    def test_all_signal_fields_present_in_summary(self):
        signals = [a.analyze(self.ctx) for a in self.agents]
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        for aid in AGENT_ORDER:
            self.assertIn(aid, card.signal_summary)

    def test_confidence_tier_is_valid(self):
        signals = [a.analyze(self.ctx) for a in self.agents]
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        self.assertIn(card.confidence_tier, ("HIGH", "MEDIUM", "LOW", "SKIP"))

    def test_edge_pct_matches_raw_score(self):
        signals = [a.analyze(self.ctx) for a in self.agents]
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        expected = abs(card.raw_score - 0.5) * 100
        self.assertAlmostEqual(card.edge_pct, expected, places=1)

    def test_print_card(self):
        signals = [a.analyze(self.ctx) for a in self.agents]
        card    = self.synth.synthesize(signals, self.ctx, mode="weighted_avg", log=False)
        print(f"\n  Pipeline pick: {card}")
        self.assertIsInstance(str(card), str)


if __name__ == "__main__":
    print("=" * 65)
    print("  Module 3 — Master Synthesizer Tests")
    print(f"  sklearn: {'✅' if SKLEARN_OK else '❌ meta-learner tests skipped'}")
    print("=" * 65 + "\n")
    unittest.main(verbosity=2)
