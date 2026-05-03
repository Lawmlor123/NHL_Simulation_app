"""
test_module4.py
---------------
Tests for Module 4: Feedback Loop + Weight Updater

All DB operations use in-memory SQLite.
All NHL API calls are mocked via the `fetcher=` parameter.
No disk writes, no network calls.

Run:
    python test_module4.py -v
"""

import json
import sqlite3
import unittest
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch

# ── Module under test ─────────────────────────────────────────────────────────
from feedback import (
    FeedbackEngine,
    ResolveResult,
    AgentPerformance,
    TierStats,
    WeeklyReport,
    compute_roi,
    fetch_scores_for_date,
    DEFAULT_WEIGHTS,
    AGENT_ORDER,
    WEIGHT_ALPHA,
    WEIGHT_MIN,
    WEIGHT_MAX,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_engine(fetcher=None) -> FeedbackEngine:
    """FeedbackEngine backed by in-memory SQLite (no disk I/O)."""
    engine = FeedbackEngine(
        db_path=Path(":memory:"),
        weights_path=Path("/tmp/test_weights_registry.json"),
        fetcher=fetcher,
    )
    # Force in-memory connection immediately
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from feedback import _PICKS_DDL, _OUTCOMES_DDL
    conn.executescript(_PICKS_DDL)
    conn.executescript(_OUTCOMES_DDL)
    conn.commit()
    engine._conn = conn
    return engine


def _insert_pick(
    engine: FeedbackEngine,
    game_id:          int,
    game_date:        str,
    home_team:        str,
    away_team:        str,
    pick_direction:   str  = "home",
    confidence_tier:  str  = "MEDIUM",
    signals_json:     Optional[str] = None,
) -> None:
    """Helper — insert a pick into the in-memory DB."""
    if signals_json is None:
        signals_json = json.dumps({
            aid: {
                "raw_score":  0.60 if pick_direction == "home" else 0.40,
                "confidence": 0.65,
                "direction":  pick_direction,
                "reasoning":  "test",
                "weight":     DEFAULT_WEIGHTS.get(aid, 1.0),
            }
            for aid in AGENT_ORDER
        })
    engine._get_conn().execute(
        """
        INSERT INTO picks
            (created_at, game_id, game_date, home_team, away_team,
             pick, pick_direction, raw_score, edge_pct,
             confidence_tier, agent_agreement, agents_above_floor,
             mode_used, skip_reason, signals_json, weights_json)
        VALUES (datetime('now'), ?, ?, ?, ?,
                ?, ?, 0.60, 10.0,
                ?, 0.80, 4, 'weighted_avg', '', ?, '{}')
        """,
        (game_id, game_date, home_team, away_team,
         home_team if pick_direction == "home" else away_team,
         pick_direction,
         confidence_tier,
         signals_json),
    )
    engine._get_conn().commit()


def _insert_outcome(
    engine:     FeedbackEngine,
    game_id:    int,
    game_date:  str,
    home_team:  str,
    away_team:  str,
    home_win:   int,
    home_score: int = 3,
    away_score: int = 2,
) -> None:
    """Helper — insert an outcome into the in-memory DB."""
    engine._get_conn().execute(
        """
        INSERT OR REPLACE INTO outcomes
            (game_id, game_date, home_team, away_team,
             home_win, home_score, away_score, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (game_id, game_date, home_team, away_team,
         home_win, home_score, away_score),
    )
    engine._get_conn().commit()


def _make_nhl_api_response(games: list[dict]) -> dict:
    """Build a fake NHL API /score/{date} response."""
    return {"games": [
        {
            "id":          g["game_id"],
            "gameState":   g.get("gameState", "OFF"),
            "homeTeam":    {"abbrev": g["home_team"], "score": g["home_score"]},
            "awayTeam":    {"abbrev": g["away_team"], "score": g["away_score"]},
        }
        for g in games
    ]}


# ═══════════════════════════════════════════════════════════════════════════════
# TestROI
# ═══════════════════════════════════════════════════════════════════════════════

class TestROI(unittest.TestCase):

    def test_zero_picks_returns_zero(self):
        self.assertEqual(compute_roi(0, 0), 0.0)

    def test_perfect_record(self):
        # 10W-0L: net = 10*100=1000, cost = 10*110=1100, ROI = 90.9%
        roi = compute_roi(10, 0)
        self.assertAlmostEqual(roi, 90.91, delta=0.1)

    def test_zero_wins(self):
        # 0W-10L: net = -10*110=-1100, cost=1100, ROI=-100%
        roi = compute_roi(0, 10)
        self.assertAlmostEqual(roi, -100.0, delta=0.1)

    def test_breakeven_at_5263_pct(self):
        # Breakeven at 11/21 ≈ 52.38 %  (100/210 split)
        # 110W-100L: net = 110*100 - 100*110 = 0 → ROI = 0
        roi = compute_roi(110, 100)
        self.assertAlmostEqual(roi, 0.0, delta=0.5)

    def test_positive_edge(self):
        # 6W-4L at -110: net = 600-440 = 160, cost = 1100, ROI = 14.5 %
        roi = compute_roi(6, 4)
        self.assertAlmostEqual(roi, 14.55, delta=0.1)

    def test_negative_edge(self):
        # 4W-6L: net = 400-660=-260, cost=1100, ROI=-23.6%
        roi = compute_roi(4, 6)
        self.assertAlmostEqual(roi, -23.64, delta=0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# TestNHLScoreFetcher
# ═══════════════════════════════════════════════════════════════════════════════

class TestNHLScoreFetcher(unittest.TestCase):

    def _fake_fetcher(self, response: dict):
        """Return a callable that ignores URL and returns `response`."""
        def fetcher(url):
            return response
        return fetcher

    def test_parses_single_final_game(self):
        resp = _make_nhl_api_response([{
            "game_id": 2024020001, "home_team": "BOS", "away_team": "TOR",
            "home_score": 4, "away_score": 2, "gameState": "OFF",
        }])
        games = fetch_scores_for_date("2024-11-15", fetcher=self._fake_fetcher(resp))
        self.assertEqual(len(games), 1)
        g = games[0]
        self.assertEqual(g["game_id"],   2024020001)
        self.assertEqual(g["home_team"], "BOS")
        self.assertEqual(g["away_team"], "TOR")
        self.assertEqual(g["home_win"],  1)
        self.assertTrue(g["is_final"])

    def test_away_win_detection(self):
        resp = _make_nhl_api_response([{
            "game_id": 9999, "home_team": "EDM", "away_team": "VGK",
            "home_score": 1, "away_score": 3, "gameState": "OFF",
        }])
        games = fetch_scores_for_date("2024-12-01", fetcher=self._fake_fetcher(resp))
        self.assertEqual(games[0]["home_win"], 0)

    def test_non_final_game_is_not_final(self):
        resp = _make_nhl_api_response([{
            "game_id": 1, "home_team": "NYR", "away_team": "NJD",
            "home_score": 1, "away_score": 0, "gameState": "LIVE",
        }])
        games = fetch_scores_for_date("2024-11-15", fetcher=self._fake_fetcher(resp))
        self.assertFalse(games[0]["is_final"])

    def test_multiple_games_parsed(self):
        resp = _make_nhl_api_response([
            {"game_id": 1, "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 1, "gameState": "OFF"},
            {"game_id": 2, "home_team": "C", "away_team": "D",
             "home_score": 0, "away_score": 2, "gameState": "OFF"},
            {"game_id": 3, "home_team": "E", "away_team": "F",
             "home_score": 1, "away_score": 1, "gameState": "LIVE"},
        ])
        games = fetch_scores_for_date("2024-11-15", fetcher=self._fake_fetcher(resp))
        self.assertEqual(len(games), 3)
        finals = [g for g in games if g["is_final"]]
        self.assertEqual(len(finals), 2)

    def test_empty_response(self):
        games = fetch_scores_for_date("2024-07-01",
                                      fetcher=self._fake_fetcher({"games": []}))
        self.assertEqual(games, [])

    def test_fetcher_error_raises(self):
        def bad_fetcher(url):
            raise ConnectionError("network down")

        with self.assertRaises(RuntimeError):
            fetch_scores_for_date("2024-11-15", fetcher=bad_fetcher)


# ═══════════════════════════════════════════════════════════════════════════════
# TestResolveDate
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolveDate(unittest.TestCase):

    def _fetcher_for(self, games: list[dict]):
        resp = _make_nhl_api_response(games)
        def fetcher(url):
            return resp
        return fetcher

    def test_resolve_writes_outcome(self):
        engine = _make_engine()
        _insert_pick(engine, 2024020001, "2024-11-15", "BOS", "TOR")

        engine._fetcher = self._fetcher_for([{
            "game_id": 2024020001, "home_team": "BOS", "away_team": "TOR",
            "home_score": 4, "away_score": 2, "gameState": "OFF",
        }])
        result = engine.resolve_date("2024-11-15")

        self.assertEqual(result.outcomes_written, 1)
        self.assertEqual(result.picks_matched, 1)

    def test_already_logged_not_duplicated(self):
        engine = _make_engine()
        _insert_pick(engine, 1, "2024-11-15", "BOS", "TOR")
        _insert_outcome(engine, 1, "2024-11-15", "BOS", "TOR", home_win=1)

        engine._fetcher = self._fetcher_for([{
            "game_id": 1, "home_team": "BOS", "away_team": "TOR",
            "home_score": 3, "away_score": 1, "gameState": "OFF",
        }])
        result = engine.resolve_date("2024-11-15")

        self.assertEqual(result.already_logged, 1)
        self.assertEqual(result.outcomes_written, 0)

    def test_non_final_games_not_written(self):
        engine = _make_engine()
        engine._fetcher = self._fetcher_for([{
            "game_id": 5, "home_team": "TBL", "away_team": "FLA",
            "home_score": 2, "away_score": 1, "gameState": "LIVE",
        }])
        result = engine.resolve_date("2024-11-15")
        self.assertEqual(result.outcomes_written, 0)
        self.assertEqual(result.games_final, 0)

    def test_dry_run_does_not_write(self):
        engine = _make_engine()
        _insert_pick(engine, 99, "2024-11-15", "EDM", "VAN")
        engine._fetcher = self._fetcher_for([{
            "game_id": 99, "home_team": "EDM", "away_team": "VAN",
            "home_score": 5, "away_score": 2, "gameState": "OFF",
        }])
        result = engine.resolve_date("2024-11-15", dry_run=True)
        # dry_run should return result but NOT write to DB
        n = engine._get_conn().execute(
            "SELECT COUNT(*) FROM outcomes"
        ).fetchone()[0]
        self.assertEqual(n, 0)

    def test_api_error_returns_empty_result(self):
        def bad_fetcher(url):
            raise ConnectionError("down")
        engine = _make_engine(fetcher=bad_fetcher)
        result = engine.resolve_date("2024-11-15")
        self.assertEqual(result.outcomes_written, 0)
        self.assertTrue(len(result.errors) > 0)

    def test_resolve_result_str_contains_date(self):
        r = ResolveResult(game_date="2025-01-15", games_fetched=5,
                          games_final=5, outcomes_written=3, picks_matched=3)
        self.assertIn("2025-01-15", str(r))
        self.assertIn("written=3", str(r))


# ═══════════════════════════════════════════════════════════════════════════════
# TestAgentAccuracy
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentAccuracy(unittest.TestCase):

    def _seed_games(self, engine, n_games: int, pick_direction: str = "home",
                    home_win: int = 1, start_game_id: int = 1) -> None:
        """Seed n identical games: pick = pick_direction, outcome = home_win."""
        for i in range(n_games):
            gid  = start_game_id + i
            gdate = f"2024-11-{(i % 28) + 1:02d}"
            _insert_pick(engine, gid, gdate, "BOS", "TOR",
                         pick_direction=pick_direction)
            _insert_outcome(engine, gid, gdate, "BOS", "TOR",
                            home_win=home_win)

    def test_stats_returned_for_all_agents(self):
        engine = _make_engine()
        self._seed_games(engine, 10)
        stats  = engine.compute_agent_stats(window_n=20)
        ids    = {ap.agent_id for ap in stats}
        self.assertEqual(ids, set(AGENT_ORDER))

    def test_perfect_accuracy_returns_win_rate_1(self):
        """10 correct picks → win_rate=1.0 for every agent."""
        engine = _make_engine()
        self._seed_games(engine, 10, pick_direction="home", home_win=1)
        stats  = engine.compute_agent_stats(window_n=20)
        for ap in stats:
            if ap.n_picks >= 5:  # only check agents with enough data
                self.assertAlmostEqual(ap.win_rate, 1.0, places=2)

    def test_zero_accuracy_possible(self):
        """All picks wrong → win_rate=0.0."""
        engine = _make_engine()
        # picks home but away wins
        self._seed_games(engine, 10, pick_direction="home", home_win=0)
        stats  = engine.compute_agent_stats(window_n=20)
        for ap in stats:
            if ap.n_picks >= 5:
                self.assertAlmostEqual(ap.win_rate, 0.0, places=2)

    def test_50pct_accuracy_on_mixed(self):
        """5 correct + 5 wrong → win_rate ≈ 0.50."""
        engine = _make_engine()
        self._seed_games(engine, 5, pick_direction="home", home_win=1,
                         start_game_id=100)
        self._seed_games(engine, 5, pick_direction="home", home_win=0,
                         start_game_id=200)
        stats = engine.compute_agent_stats(window_n=20)
        for ap in stats:
            if ap.n_picks >= 5:
                self.assertAlmostEqual(ap.win_rate, 0.5, delta=0.05)

    def test_window_limits_sample(self):
        """compute_agent_stats(window_n=3) should only use last 3 picks."""
        engine = _make_engine()
        # 7 wrong picks dated OLDER (2024-10-xx)
        for i in range(7):
            gid   = 1 + i
            gdate = f"2024-10-{i + 1:02d}"  # older dates
            _insert_pick(engine, gid, gdate, "BOS", "TOR",
                         pick_direction="home")
            _insert_outcome(engine, gid, gdate, "BOS", "TOR", home_win=0)

        # 3 correct picks dated MORE RECENT (2024-11-xx)
        for i in range(3):
            gid   = 100 + i
            gdate = f"2024-11-{i + 1:02d}"  # newer dates
            _insert_pick(engine, gid, gdate, "BOS", "TOR",
                         pick_direction="home")
            _insert_outcome(engine, gid, gdate, "BOS", "TOR", home_win=1)

        stats3  = engine.compute_agent_stats(window_n=3)
        stats10 = engine.compute_agent_stats(window_n=10)

        # window_n=3 sees only the 3 correct → win_rate=1.0
        # window_n=10 sees all 10 → win_rate=0.3
        rate3  = next(ap.win_rate for ap in stats3  if ap.n_picks >= 3)
        rate10 = next(ap.win_rate for ap in stats10 if ap.n_picks >= 5)
        self.assertGreater(rate3, rate10)

    def test_skip_picks_excluded(self):
        """SKIP picks must not be counted in accuracy."""
        engine = _make_engine()
        # Insert 5 SKIP picks that should be ignored
        for i in range(5):
            _insert_pick(engine, 900 + i, "2024-11-01", "BOS", "TOR",
                         confidence_tier="SKIP", pick_direction="skip")
            _insert_outcome(engine, 900 + i, "2024-11-01", "BOS", "TOR",
                            home_win=1)
        stats = engine.compute_agent_stats(window_n=20)
        for ap in stats:
            self.assertEqual(ap.n_picks, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TestWeightFormula
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightFormula(unittest.TestCase):
    """Verify EMA weight update: new = (1-alpha)*old + alpha*(win_rate/0.5)."""

    def _old_weight(self) -> float:
        return DEFAULT_WEIGHTS["team_form"]  # 1.0

    def test_random_agent_weight_preserved(self):
        """50% win rate → normalised=1.0 → weight unchanged at exactly 1.0."""
        old = 1.0
        win_rate = 0.50
        norm = win_rate / 0.50
        new = (1 - WEIGHT_ALPHA) * old + WEIGHT_ALPHA * norm
        self.assertAlmostEqual(new, 1.0, places=5)

    def test_good_agent_weight_increases(self):
        """60% win rate → normalised=1.2 → weight > old."""
        old = 1.0
        norm = 0.60 / 0.50
        new = (1 - WEIGHT_ALPHA) * old + WEIGHT_ALPHA * norm
        self.assertGreater(new, old)

    def test_bad_agent_weight_decreases(self):
        """40% win rate → normalised=0.8 → weight < old."""
        old = 1.0
        norm = 0.40 / 0.50
        new = (1 - WEIGHT_ALPHA) * old + WEIGHT_ALPHA * norm
        self.assertLess(new, old)

    def test_weight_clamped_above(self):
        """Unrealistically accurate agent → weight clamped to WEIGHT_MAX."""
        engine = _make_engine()
        # 20 correct picks per agent
        for i in range(20):
            gid  = 300 + i
            _insert_pick(engine, gid, "2024-11-01", "BOS", "TOR",
                         pick_direction="home")
            _insert_outcome(engine, gid, "2024-11-01", "BOS", "TOR",
                            home_win=1)
        engine._get_conn().execute(
            "UPDATE picks SET confidence_tier='HIGH'"
        )
        engine._get_conn().commit()

        # Create a very high initial weight
        for ap in engine.compute_agent_stats(window_n=20):
            self.assertLessEqual(ap.new_weight, WEIGHT_MAX)

    def test_weight_clamped_below(self):
        """Terrible agent → weight clamped to WEIGHT_MIN."""
        engine = _make_engine()
        # Seed very high starting weight via mocked load_weights
        for i in range(20):
            gid = 400 + i
            _insert_pick(engine, gid, "2024-11-01", "COL", "ARI",
                         pick_direction="home")
            _insert_outcome(engine, gid, "2024-11-01", "COL", "ARI",
                            home_win=0)
        for ap in engine.compute_agent_stats(window_n=20):
            self.assertGreaterEqual(ap.new_weight, WEIGHT_MIN)

    def test_not_enough_data_preserves_weight(self):
        """Fewer than 5 picks per agent → weight unchanged."""
        engine = _make_engine()
        # Only 3 picks
        for i in range(3):
            _insert_pick(engine, 500 + i, "2024-11-01", "WSH", "PIT")
            _insert_outcome(engine, 500 + i, "2024-11-01", "WSH", "PIT",
                            home_win=1)
        stats = engine.compute_agent_stats(window_n=20)
        for ap in stats:
            if ap.n_picks < 5:
                self.assertEqual(ap.old_weight, ap.new_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# TestUpdateWeights
# ═══════════════════════════════════════════════════════════════════════════════

class TestUpdateWeights(unittest.TestCase):

    def test_update_weights_returns_empty_below_min_sample(self):
        """update_weights() returns {} when resolved picks < MIN_SAMPLE."""
        engine = _make_engine()
        # Only 5 resolved picks (below MIN_SAMPLE=10)
        for i in range(5):
            _insert_pick(engine, i, "2024-11-01", "BOS", "TOR")
            _insert_outcome(engine, i, "2024-11-01", "BOS", "TOR", home_win=1)
        result = engine.update_weights()
        self.assertEqual(result, {})

    def test_update_weights_returns_dict_with_all_agents(self):
        """With 10+ resolved picks, update_weights() returns all 5 agent keys."""
        engine = _make_engine()
        for i in range(15):
            _insert_pick(engine, i, "2024-11-01", "BOS", "TOR")
            _insert_outcome(engine, i, "2024-11-01", "BOS", "TOR", home_win=1)
        result = engine.update_weights()
        for aid in AGENT_ORDER:
            self.assertIn(aid, result)

    def test_accurate_agent_gets_higher_weight(self):
        """Agent with 70% accuracy should outweigh a 50% agent over time."""
        engine = _make_engine()
        # 14 correct (home picks where home wins) + 6 wrong
        for i in range(14):
            gid = 600 + i
            _insert_pick(engine, gid, "2024-11-01", "BOS", "TOR",
                         pick_direction="home")
            _insert_outcome(engine, gid, "2024-11-01", "BOS", "TOR",
                            home_win=1)
        for i in range(6):
            gid = 700 + i
            _insert_pick(engine, gid, "2024-11-01", "BOS", "TOR",
                         pick_direction="home")
            _insert_outcome(engine, gid, "2024-11-01", "BOS", "TOR",
                            home_win=0)

        stats = engine.compute_agent_stats(window_n=20)
        # All agents see same data — win_rate ≈ 0.70
        for ap in stats:
            if ap.n_picks >= 5:
                self.assertGreater(ap.new_weight, ap.old_weight * 0.9)

    def test_weights_all_valid_floats(self):
        """All returned weights should be finite floats in [WEIGHT_MIN, WEIGHT_MAX]."""
        engine = _make_engine()
        for i in range(12):
            _insert_pick(engine, i, "2024-11-01", "BOS", "TOR")
            _insert_outcome(engine, i, "2024-11-01", "BOS", "TOR", home_win=1)
        result = engine.update_weights()
        for aid, w in result.items():
            self.assertIsInstance(w, float)
            self.assertGreaterEqual(w, WEIGHT_MIN)
            self.assertLessEqual(w, WEIGHT_MAX)


# ═══════════════════════════════════════════════════════════════════════════════
# TestWeightPersistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightPersistence(unittest.TestCase):

    def test_load_weights_returns_defaults_when_no_file(self):
        engine = _make_engine()
        engine._weights_path = Path("/tmp/nonexistent_weights_abc123.json")
        weights = engine.load_weights()
        self.assertEqual(weights, DEFAULT_WEIGHTS)

    def test_persist_and_reload(self):
        """Persist weights, reload them — should get the same values."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                         mode="w") as f:
            tmp_path = Path(f.name)

        try:
            engine = _make_engine()
            engine._weights_path = tmp_path

            test_weights = {
                "team_form": 1.15, "player_form": 0.85,
                "goalie_form": 1.30, "schedule": 0.75, "sentiment": 0.55,
            }
            engine.persist_weights(test_weights)
            loaded = engine.load_weights()
            for aid in AGENT_ORDER:
                self.assertAlmostEqual(loaded[aid], test_weights[aid], places=3)
        finally:
            os.unlink(tmp_path)

    def test_load_merges_missing_agents_with_defaults(self):
        """Partial JSON (only 3 agents) should be padded with defaults."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w") as f:
            json.dump({"team_form": 1.1, "goalie_form": 1.35}, f)
            tmp_path = Path(f.name)
        try:
            engine = _make_engine()
            engine._weights_path = tmp_path
            loaded = engine.load_weights()
            self.assertAlmostEqual(loaded["team_form"], 1.1, places=3)
            self.assertAlmostEqual(loaded["goalie_form"], 1.35, places=3)
            # Missing agents filled with defaults
            self.assertEqual(loaded["player_form"], DEFAULT_WEIGHTS["player_form"])
            self.assertEqual(loaded["schedule"],    DEFAULT_WEIGHTS["schedule"])
        finally:
            os.unlink(tmp_path)

    def test_persist_only_writes_known_agents(self):
        """Unknown keys in input are silently dropped."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                         mode="w") as f:
            tmp_path = Path(f.name)
        try:
            engine = _make_engine()
            engine._weights_path = tmp_path
            dirty = dict(DEFAULT_WEIGHTS)
            dirty["rogue_agent"] = 99.9
            engine.persist_weights(dirty)
            data = json.loads(tmp_path.read_text())
            self.assertNotIn("rogue_agent", data)
            self.assertEqual(set(data.keys()), set(AGENT_ORDER))
        finally:
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# TestWeeklyReport
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeeklyReport(unittest.TestCase):

    def _seed_resolved(self, engine, n_correct, n_wrong,
                       tier="MEDIUM", start_id=1000) -> None:
        """Seed picks + outcomes: n_correct HOME picks that win, n_wrong that lose."""
        today = date.today()
        for i in range(n_correct):
            gid   = start_id + i
            gdate = (today - timedelta(days=1)).isoformat()
            _insert_pick(engine, gid, gdate, "BOS", "TOR",
                         pick_direction="home", confidence_tier=tier)
            _insert_outcome(engine, gid, gdate, "BOS", "TOR", home_win=1)

        for i in range(n_wrong):
            gid   = start_id + n_correct + i
            gdate = (today - timedelta(days=1)).isoformat()
            _insert_pick(engine, gid, gdate, "BOS", "TOR",
                         pick_direction="home", confidence_tier=tier)
            _insert_outcome(engine, gid, gdate, "BOS", "TOR", home_win=0)

    def test_report_returns_weekly_report_instance(self):
        engine = _make_engine()
        report = engine.weekly_report(weeks=1)
        self.assertIsInstance(report, WeeklyReport)

    def test_correct_win_loss_counts(self):
        engine = _make_engine()
        self._seed_resolved(engine, 7, 3)
        report = engine.weekly_report(weeks=1)
        self.assertEqual(report.total_wins,   7)
        self.assertEqual(report.total_losses, 3)
        self.assertAlmostEqual(report.win_rate, 0.70, places=2)

    def test_roi_is_positive_for_winning_record(self):
        engine = _make_engine()
        self._seed_resolved(engine, 7, 3)
        report = engine.weekly_report(weeks=1)
        self.assertGreater(report.roi_pct, 0)

    def test_roi_negative_for_losing_record(self):
        engine = _make_engine()
        self._seed_resolved(engine, 3, 7)
        report = engine.weekly_report(weeks=1)
        self.assertLess(report.roi_pct, 0)

    def test_skip_picks_not_counted_in_wins_or_losses(self):
        engine = _make_engine()
        self._seed_resolved(engine, 5, 5)
        # Add SKIP picks
        today = date.today().isoformat()
        for i in range(3):
            _insert_pick(engine, 2000 + i, today, "NYR", "NJD",
                         confidence_tier="SKIP", pick_direction="skip")
        report = engine.weekly_report(weeks=1)
        self.assertEqual(report.total_skips, 3)
        self.assertEqual(report.total_wins + report.total_losses, 10)

    def test_tier_breakdown_populated(self):
        engine = _make_engine()
        self._seed_resolved(engine, 3, 1, tier="HIGH",   start_id=3000)
        self._seed_resolved(engine, 2, 2, tier="MEDIUM", start_id=4000)
        self._seed_resolved(engine, 1, 3, tier="LOW",    start_id=5000)
        report = engine.weekly_report(weeks=1)
        self.assertIn("HIGH",   report.tier_breakdown)
        self.assertIn("MEDIUM", report.tier_breakdown)
        self.assertIn("LOW",    report.tier_breakdown)
        self.assertEqual(report.tier_breakdown["HIGH"].wins,   3)
        self.assertEqual(report.tier_breakdown["HIGH"].losses, 1)

    def test_high_tier_roi_better_than_low(self):
        engine = _make_engine()
        self._seed_resolved(engine, 4, 1, tier="HIGH",   start_id=6000)
        self._seed_resolved(engine, 1, 4, tier="LOW",    start_id=7000)
        report = engine.weekly_report(weeks=1)
        high_roi = report.tier_breakdown["HIGH"].roi_pct
        low_roi  = report.tier_breakdown["LOW"].roi_pct
        self.assertGreater(high_roi, low_roi)

    def test_unresolved_picks_counted(self):
        engine = _make_engine()
        today = date.today().isoformat()
        # Pick without matching outcome
        _insert_pick(engine, 8888, today, "BOS", "TOR")
        report = engine.weekly_report(weeks=1)
        self.assertEqual(report.unresolved, 1)

    def test_report_period_correct(self):
        engine = _make_engine()
        report = engine.weekly_report(weeks=2)
        start = date.fromisoformat(report.period_start)
        end   = date.fromisoformat(report.period_end)
        self.assertEqual((end - start).days, 14)

    def test_agent_stats_included_in_report(self):
        engine = _make_engine()
        self._seed_resolved(engine, 5, 5)
        report = engine.weekly_report(weeks=1)
        self.assertEqual(len(report.agent_stats), len(AGENT_ORDER))
        self.assertEqual(len(report.weight_updates), len(AGENT_ORDER))


# ═══════════════════════════════════════════════════════════════════════════════
# TestTierStats
# ═══════════════════════════════════════════════════════════════════════════════

class TestTierStats(unittest.TestCase):

    def test_win_rate_zero_for_empty(self):
        ts = TierStats(tier="HIGH")
        self.assertEqual(ts.win_rate, 0.0)

    def test_roi_zero_for_empty(self):
        ts = TierStats(tier="HIGH")
        self.assertEqual(ts.roi_pct, 0.0)

    def test_str_contains_tier(self):
        ts = TierStats(tier="HIGH", n=10, wins=7, losses=3)
        self.assertIn("HIGH", str(ts))
        self.assertIn("7W", str(ts))

    def test_win_rate_calculated(self):
        ts = TierStats(tier="MEDIUM", n=10, wins=6, losses=4)
        self.assertAlmostEqual(ts.win_rate, 0.60, places=2)


# ═══════════════════════════════════════════════════════════════════════════════
# TestAgentPerformance
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentPerformance(unittest.TestCase):

    def test_accuracy_score_random_is_one(self):
        ap = AgentPerformance(agent_id="test", win_rate=0.50)
        self.assertAlmostEqual(ap.accuracy_score, 1.0, places=5)

    def test_accuracy_score_good_above_one(self):
        ap = AgentPerformance(agent_id="test", win_rate=0.60)
        self.assertGreater(ap.accuracy_score, 1.0)

    def test_accuracy_score_zero_win_rate(self):
        ap = AgentPerformance(agent_id="test", win_rate=0.0)
        self.assertEqual(ap.accuracy_score, 0.0)

    def test_str_contains_agent_id(self):
        ap = AgentPerformance(
            agent_id="goalie_form", n_picks=20, n_wins=13,
            win_rate=0.65, old_weight=1.2, new_weight=1.32,
        )
        s = str(ap)
        self.assertIn("goalie_form", s)
        self.assertIn("1.200", s)
        self.assertIn("1.320", s)

    def test_up_arrow_when_weight_increases(self):
        ap = AgentPerformance(agent_id="x", n_picks=10, n_wins=6,
                              win_rate=0.60, old_weight=1.0, new_weight=1.1)
        self.assertIn("↑", str(ap))

    def test_down_arrow_when_weight_decreases(self):
        ap = AgentPerformance(agent_id="x", n_picks=10, n_wins=4,
                              win_rate=0.40, old_weight=1.0, new_weight=0.9)
        self.assertIn("↓", str(ap))


# ═══════════════════════════════════════════════════════════════════════════════
# TestFullPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline(unittest.TestCase):
    """End-to-end: resolve → update_weights → weekly_report."""

    def test_full_resolve_and_report(self):
        """Simulate a week of predictions resolved via API."""
        today     = date.today()
        game_date = (today - timedelta(days=1)).isoformat()

        # Fake NHL API response
        fake_games = [
            {"game_id": 9000 + i, "home_team": "BOS", "away_team": "TOR",
             "home_score": 3 + (i % 2), "away_score": 2 - (i % 2),
             "gameState": "OFF"}
            for i in range(5)
        ]

        def fake_fetcher(url):
            return _make_nhl_api_response(fake_games)

        engine = _make_engine(fetcher=fake_fetcher)

        # Seed matching picks
        for g in fake_games:
            _insert_pick(engine, g["game_id"], game_date, "BOS", "TOR",
                         pick_direction="home", confidence_tier="HIGH")

        # Resolve
        result = engine.resolve_date(game_date)
        self.assertEqual(result.outcomes_written, 5)
        self.assertEqual(result.picks_matched,    5)

        # Report
        report = engine.weekly_report(weeks=1)
        self.assertEqual(report.total_picks, 5)
        total_resolved = report.total_wins + report.total_losses
        self.assertEqual(total_resolved, 5)

    def test_cli_show_weights_prints_all_agents(self):
        """Smoke test: load_weights() returns all 5 agent keys."""
        engine  = _make_engine()
        weights = engine.load_weights()
        for aid in AGENT_ORDER:
            self.assertIn(aid, weights)

    def test_print_report_runs_without_error(self):
        """print_report() should not raise."""
        engine = _make_engine()
        report = engine.weekly_report(weeks=1)
        try:
            engine.print_report(report)
        except Exception as exc:
            self.fail(f"print_report raised: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    for cls in [
        TestROI,
        TestNHLScoreFetcher,
        TestResolveDate,
        TestAgentAccuracy,
        TestWeightFormula,
        TestUpdateWeights,
        TestWeightPersistence,
        TestWeeklyReport,
        TestTierStats,
        TestAgentPerformance,
        TestFullPipeline,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 62)
    print("  Module 4 — Feedback Loop + Weight Updater Tests")
    print("=" * 62)
    ran    = result.testsRun
    failed = len(result.failures) + len(result.errors)
    if failed == 0:
        print(f"  ✅  {ran} tests passed\n")
    else:
        print(f"  ❌  {failed} failures / errors  ({ran} total)\n")
        for test, tb in result.failures + result.errors:
            print(f"  FAIL: {test}")
            print(f"  {tb[:300]}\n")
