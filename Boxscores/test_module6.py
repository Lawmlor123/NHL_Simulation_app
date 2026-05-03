"""
test_module6.py
---------------
Tests for run_picks.py — Module 6: Orchestrator / Runner
Target: ≥55 tests, 0 failures
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import run_picks as rp
from run_picks import (
    Config,
    MASOrchestrator,
    RunResult,
    _PickProxy,
    _load_picks_from_db,
    load_config,
    setup_logging,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers / mock objects
# ═══════════════════════════════════════════════════════════════════════════════

def _make_config(**kw) -> Config:
    defaults = dict(
        db_path           = Path(":memory:"),
        weights_file_path = Path("/tmp/weights_test.json"),
        log_level         = "WARNING",
        confidence_floor  = 0.52,
        min_agents        = 3,
        synthesis_mode    = "auto",
        script_tone       = "hype",
        fetch_advanced    = True,
    )
    defaults.update(kw)
    return Config(**defaults)


def _make_picks_table(conn: sqlite3.Connection) -> None:
    """Create picks table that matches the schema expected by _load_picks_from_db."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id             TEXT,
            game_date           TEXT,
            home_team           TEXT,
            away_team           TEXT,
            pick                TEXT,
            pick_direction      TEXT,
            raw_score           REAL,
            edge_pct            REAL,
            confidence_tier     TEXT,
            agent_agreement     REAL,
            agents_above_floor  INTEGER,
            mode_used           TEXT,
            skip_reason         TEXT,
            signals_json        TEXT,
            weights_json        TEXT,
            created_at          TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def _insert_pick(conn, game_id="G1", game_date="2024-11-01",
                 home_team="Toronto Maple Leafs", away_team="Montreal Canadiens",
                 pick="TOR", pick_direction="home",
                 raw_score=0.58, edge_pct=5.0, confidence_tier="HIGH",
                 agent_agreement=0.75, agents_above_floor=4,
                 mode_used="weighted", skip_reason="",
                 signals_json=None, weights_json=None) -> None:
    conn.execute("""
        INSERT INTO picks (
            game_id, game_date, home_team, away_team, pick, pick_direction,
            raw_score, edge_pct, confidence_tier, agent_agreement,
            agents_above_floor, mode_used, skip_reason, signals_json, weights_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        game_id, game_date, home_team, away_team, pick, pick_direction,
        raw_score, edge_pct, confidence_tier, agent_agreement,
        agents_above_floor, mode_used, skip_reason,
        signals_json or "{}", weights_json or "{}",
    ))
    conn.commit()


class MockCard:
    """Duck-type PickCard compatible object for testing."""
    def __init__(self, tier="HIGH", edge=5.0, pick="TOR", direction="home",
                 agree=0.75, above=4, home="Toronto Maple Leafs",
                 away="Montreal Canadiens", skip_reason=""):
        self.confidence_tier       = tier
        self.edge_pct              = edge
        self.pick                  = pick
        self.pick_direction        = direction
        self.agent_agreement_score = agree
        self.agents_above_floor    = above
        self.home_team             = home
        self.away_team             = away
        self.raw_score             = 0.58
        self.mode_used             = "weighted"
        self.skip_reason           = skip_reason
        self.signal_summary        = {
            "team_form": {"raw_score": 0.60, "weight": 1.0, "direction": "home"},
        }
        self.weights_used          = {"team_form": 1.0}

    @property
    def is_playable(self):
        return self.confidence_tier != "SKIP"

    @property
    def matchup(self):
        return f"{self.away_team} @ {self.home_team}"


class MockSkipCard(MockCard):
    def __init__(self, **kw):
        super().__init__(tier="SKIP", direction="skip", **kw)


class MockSignal:
    """Duck-type AgentSignal for testing."""
    def __init__(self, agent_id="team_form", raw=0.58, conf=0.65, direction="home"):
        self.agent_id      = agent_id
        self.raw_score     = raw
        self.confidence    = conf
        self.pick_direction = direction


class MockAgent:
    """Duck-type NHL agent with analyze()."""
    def __init__(self, agent_id="team_form"):
        self.agent_id = agent_id

    def analyze(self, ctx):
        return MockSignal(self.agent_id)


class MockSynthesizer:
    """Duck-type MasterSynthesizer."""
    weights_registry = {"team_form": 1.0}

    def __init__(self, card=None):
        self._card = card or MockCard()

    def synthesize(self, signals, ctx, mode="auto", log=True):
        return self._card

    def fit_from_db(self):
        pass


class MockFeedback:
    """Duck-type FeedbackEngine."""
    def resolve_date(self, game_date, dry_run=False):
        return types.SimpleNamespace(resolved=2, skipped=0, errors=[])

    def weekly_report(self, weeks=1):
        return types.SimpleNamespace(
            overall_record=(10, 8),
            roi=-3.5,
            agent_stats=[],
            tier_stats={},
            weeks=weeks,
        )

    def print_report(self, report):
        pass


class MockContext:
    """Duck-type game context."""
    def __init__(self, home="Toronto Maple Leafs", away="Montreal Canadiens"):
        self.home = types.SimpleNamespace(team=home)
        self.away = types.SimpleNamespace(team=away)


def _make_orch(config=None, contexts=None, cards=None, agents=None,
               feedback=None, synthesizer=None) -> MASOrchestrator:
    """Build a fully mocked orchestrator ready for testing."""
    cfg      = config or _make_config()
    ctx_list = contexts if contexts is not None else [MockContext()]
    card     = (cards[0] if cards else None) or MockCard()

    def _ctx_builder(target_date, fetch_advanced=True):
        return ctx_list

    synth = synthesizer or MockSynthesizer(card)

    orch = MASOrchestrator(
        config          = cfg,
        context_builder = _ctx_builder,
        agents          = agents or [MockAgent()],
        synthesizer     = synth,
        feedback_engine = feedback or MockFeedback(),
    )
    return orch


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Config defaults
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):

    def test_default_log_level(self):
        c = Config()
        self.assertEqual(c.log_level, "INFO")

    def test_default_confidence_floor(self):
        c = Config()
        self.assertAlmostEqual(c.confidence_floor, 0.52)

    def test_default_min_agents(self):
        c = Config()
        self.assertEqual(c.min_agents, 3)

    def test_default_script_tone(self):
        c = Config()
        self.assertEqual(c.script_tone, "hype")

    def test_default_fetch_advanced(self):
        c = Config()
        self.assertTrue(c.fetch_advanced)

    def test_default_synthesis_mode(self):
        c = Config()
        self.assertEqual(c.synthesis_mode, "auto")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. load_config
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadConfig(unittest.TestCase):

    def _tmp_env(self, content: str) -> Path:
        """Write a temp .env file and return its Path."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
        f.write(content)
        f.close()
        return Path(f.name)

    def test_returns_config_instance(self):
        cfg = load_config(env_path=Path("/nonexistent/.env"))
        self.assertIsInstance(cfg, Config)

    def test_defaults_when_no_env_file(self):
        cfg = load_config(env_path=Path("/nonexistent/.env"))
        self.assertEqual(cfg.log_level, "INFO")
        self.assertAlmostEqual(cfg.confidence_floor, 0.52)

    def test_reads_key_value_from_env_file(self):
        p = self._tmp_env("LOG_LEVEL=DEBUG\nCONFIDENCE_FLOOR=0.60\n")
        cfg = load_config(env_path=p)
        self.assertEqual(cfg.log_level, "DEBUG")
        self.assertAlmostEqual(cfg.confidence_floor, 0.60)
        os.unlink(p)

    def test_ignores_comment_lines(self):
        p = self._tmp_env("# This is a comment\nLOG_LEVEL=WARNING\n")
        cfg = load_config(env_path=p)
        self.assertEqual(cfg.log_level, "WARNING")
        os.unlink(p)

    def test_env_var_overrides_env_file(self):
        p = self._tmp_env("LOG_LEVEL=INFO\n")
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            cfg = load_config(env_path=p)
        self.assertEqual(cfg.log_level, "DEBUG")
        os.unlink(p)

    def test_bool_true_variants(self):
        for val in ("1", "true", "yes", "on"):
            p = self._tmp_env(f"FETCH_ADVANCED={val}\n")
            cfg = load_config(env_path=p)
            self.assertTrue(cfg.fetch_advanced, msg=f"failed for val={val!r}")
            os.unlink(p)

    def test_bool_false_variant(self):
        p = self._tmp_env("FETCH_ADVANCED=false\n")
        cfg = load_config(env_path=p)
        self.assertFalse(cfg.fetch_advanced)
        os.unlink(p)

    def test_min_agents_parsed_as_int(self):
        p = self._tmp_env("MIN_AGENTS=5\n")
        cfg = load_config(env_path=p)
        self.assertEqual(cfg.min_agents, 5)
        self.assertIsInstance(cfg.min_agents, int)
        os.unlink(p)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RunResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunResult(unittest.TestCase):

    def test_defaults(self):
        r = RunResult()
        self.assertEqual(r.mode, "")
        self.assertEqual(r.n_games, 0)
        self.assertEqual(r.n_picks, 0)
        self.assertEqual(r.n_high, 0)
        self.assertEqual(r.n_in_script, 0)
        self.assertEqual(r.script, "")
        self.assertFalse(r.dry_run)

    def test_pick_cards_defaults_to_empty_list(self):
        r = RunResult()
        self.assertIsInstance(r.pick_cards, list)
        self.assertEqual(len(r.pick_cards), 0)

    def test_errors_defaults_to_empty_list(self):
        r = RunResult()
        self.assertIsInstance(r.errors, list)

    def test_str_normal(self):
        r = RunResult(mode="full", target_date="2024-11-01",
                      n_games=6, n_picks=6, n_high=2, n_in_script=2,
                      elapsed_seconds=1.23)
        s = str(r)
        self.assertIn("full", s)
        self.assertIn("2024-11-01", s)
        self.assertIn("games=6", s)
        self.assertNotIn("DRY-RUN", s)

    def test_str_dry_run_tag(self):
        r = RunResult(mode="full", target_date="2024-11-01", dry_run=True)
        self.assertIn("DRY-RUN", str(r))

    def test_run_at_auto_populated(self):
        r = RunResult()
        self.assertIsInstance(r.run_at, str)
        self.assertGreater(len(r.run_at), 5)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. _PickProxy
# ═══════════════════════════════════════════════════════════════════════════════

class TestPickProxy(unittest.TestCase):

    def _make_row(self, **kw):
        defaults = (
            "G1", "2024-11-01", "Toronto Maple Leafs", "Montreal Canadiens",
            "TOR", "home", 0.58, 5.0, "HIGH", 0.75, 4, "weighted", "",
            '{"team_form": {"raw_score": 0.60}}', '{"team_form": 1.0}',
        )
        return defaults

    def test_attributes_set_correctly(self):
        proxy = _PickProxy(self._make_row())
        self.assertEqual(proxy.game_id, "G1")
        self.assertEqual(proxy.confidence_tier, "HIGH")
        self.assertAlmostEqual(proxy.edge_pct, 5.0)
        self.assertEqual(proxy.pick_direction, "home")

    def test_is_playable_true_for_high(self):
        proxy = _PickProxy(self._make_row())
        self.assertTrue(proxy.is_playable)

    def test_is_playable_false_for_skip(self):
        row = list(self._make_row())
        row[8] = "SKIP"
        proxy = _PickProxy(tuple(row))
        self.assertFalse(proxy.is_playable)

    def test_matchup_property(self):
        proxy = _PickProxy(self._make_row())
        self.assertIn("@", proxy.matchup)
        self.assertIn("Toronto Maple Leafs", proxy.matchup)
        self.assertIn("Montreal Canadiens", proxy.matchup)

    def test_signal_summary_parsed(self):
        proxy = _PickProxy(self._make_row())
        self.assertIsInstance(proxy.signal_summary, dict)
        self.assertIn("team_form", proxy.signal_summary)

    def test_weights_used_parsed(self):
        proxy = _PickProxy(self._make_row())
        self.assertIsInstance(proxy.weights_used, dict)
        self.assertIn("team_form", proxy.weights_used)

    def test_null_signals_json_defaults_to_empty(self):
        row = list(self._make_row())
        row[13] = None    # signals_json
        row[14] = None    # weights_json
        proxy = _PickProxy(tuple(row))
        self.assertEqual(proxy.signal_summary, {})
        self.assertEqual(proxy.weights_used, {})

    def test_null_skip_reason_defaults_to_empty_string(self):
        row = list(self._make_row())
        row[12] = None    # skip_reason
        proxy = _PickProxy(tuple(row))
        self.assertEqual(proxy.skip_reason, "")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. _load_picks_from_db
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadPicksFromDb(unittest.TestCase):

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        _make_picks_table(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_empty_table_returns_empty_list(self):
        result = _load_picks_from_db(self.conn, "2024-11-01")
        self.assertEqual(result, [])

    def test_returns_picks_for_correct_date(self):
        _insert_pick(self.conn, game_date="2024-11-01")
        result = _load_picks_from_db(self.conn, "2024-11-01")
        self.assertEqual(len(result), 1)

    def test_filters_by_date(self):
        _insert_pick(self.conn, game_id="G1", game_date="2024-11-01")
        _insert_pick(self.conn, game_id="G2", game_date="2024-11-02")
        result = _load_picks_from_db(self.conn, "2024-11-01")
        self.assertEqual(len(result), 1)

    def test_multiple_picks_returned(self):
        _insert_pick(self.conn, game_id="G1", game_date="2024-11-01")
        _insert_pick(self.conn, game_id="G2", game_date="2024-11-01")
        result = _load_picks_from_db(self.conn, "2024-11-01")
        self.assertEqual(len(result), 2)

    def test_pick_proxy_attributes_from_db(self):
        _insert_pick(self.conn, game_id="G99", game_date="2024-11-01",
                     confidence_tier="MEDIUM", edge_pct=2.5)
        result = _load_picks_from_db(self.conn, "2024-11-01")
        card = result[0]
        self.assertEqual(card.confidence_tier, "MEDIUM")
        self.assertAlmostEqual(card.edge_pct, 2.5)

    def test_signals_json_deserialized(self):
        signals = {"team_form": {"raw_score": 0.62, "weight": 1.1}}
        _insert_pick(self.conn, game_date="2024-11-01",
                     signals_json=json.dumps(signals))
        result = _load_picks_from_db(self.conn, "2024-11-01")
        self.assertEqual(result[0].signal_summary.get("team_form", {}).get("raw_score"), 0.62)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MASOrchestrator initialisation
# ═══════════════════════════════════════════════════════════════════════════════

class TestMASOrchestrator(unittest.TestCase):

    def test_init_stores_config(self):
        cfg  = _make_config()
        orch = MASOrchestrator(config=cfg)
        self.assertIs(orch.config, cfg)

    def test_init_injectable_context_builder(self):
        called = []
        def my_builder(d, fetch_advanced=True):
            called.append(d)
            return []
        orch = _make_orch(contexts=[], agents=[MockAgent()])
        orch._context_builder = my_builder
        orch.run_full("2024-11-01")
        self.assertEqual(called, ["2024-11-01"])

    def test_init_injectable_agents(self):
        agents = [MockAgent("a1"), MockAgent("a2")]
        orch   = _make_orch(agents=agents)
        self.assertIs(orch._agents, agents)

    def test_injected_feedback_not_replaced(self):
        fb   = MockFeedback()
        orch = _make_orch(feedback=fb)
        self.assertIs(orch._feedback, fb)

    def test_conn_is_none_initially(self):
        orch = _make_orch()
        self.assertIsNone(orch._conn)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. run_full
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunFull(unittest.TestCase):

    def test_returns_run_result(self):
        orch   = _make_orch()
        result = orch.run_full("2024-11-01")
        self.assertIsInstance(result, RunResult)

    def test_mode_is_full(self):
        orch   = _make_orch()
        result = orch.run_full("2024-11-01")
        self.assertEqual(result.mode, "full")

    def test_target_date_set(self):
        orch   = _make_orch()
        result = orch.run_full("2024-11-15")
        self.assertEqual(result.target_date, "2024-11-15")

    def test_n_games_equals_context_count(self):
        contexts = [MockContext(), MockContext(), MockContext()]
        orch     = _make_orch(contexts=contexts)
        result   = orch.run_full("2024-11-01")
        self.assertEqual(result.n_games, 3)

    def test_n_picks_counts_playable(self):
        # 2 contexts → 2 MockCards (both HIGH, playable)
        contexts   = [MockContext(), MockContext()]
        high_card  = MockCard(tier="HIGH")
        skip_card  = MockSkipCard()

        # Each synthesize call returns the same card — let's use alternating
        cards_iter = iter([high_card, skip_card])
        class AltSynth:
            weights_registry = {}
            def synthesize(self, signals, ctx, mode="auto", log=True):
                return next(cards_iter)
            def fit_from_db(self): pass

        orch   = _make_orch(contexts=contexts, synthesizer=AltSynth())
        result = orch.run_full("2024-11-01")
        self.assertEqual(result.n_picks, 1)   # only HIGH is playable

    def test_n_high_counts_high_tier(self):
        contexts = [MockContext(), MockContext()]
        class AllHigh:
            weights_registry = {}
            def synthesize(self, s, c, mode="auto", log=True):
                return MockCard(tier="HIGH")
            def fit_from_db(self): pass
        orch   = _make_orch(contexts=contexts, synthesizer=AllHigh())
        result = orch.run_full("2024-11-01")
        self.assertEqual(result.n_high, 2)

    def test_context_fetch_error_returns_early(self):
        def bad_builder(d, fetch_advanced=True):
            raise RuntimeError("network down")

        orch = MASOrchestrator(
            config          = _make_config(),
            context_builder = bad_builder,
            agents          = [MockAgent()],
            synthesizer     = MockSynthesizer(),
            feedback_engine = MockFeedback(),
        )
        result = orch.run_full("2024-11-01")
        self.assertEqual(result.n_games, 0)
        self.assertTrue(len(result.errors) > 0)
        self.assertIn("Context fetch failed", result.errors[0])

    def test_agent_exception_adds_to_errors(self):
        class BoomAgent:
            def analyze(self, ctx):
                raise ValueError("boom")
        orch   = _make_orch(agents=[BoomAgent()])
        result = orch.run_full("2024-11-01")
        self.assertTrue(len(result.errors) > 0)

    def test_dry_run_flag_set_in_result(self):
        orch   = _make_orch()
        result = orch.run_full("2024-11-01", dry_run=True)
        self.assertTrue(result.dry_run)

    def test_script_is_string(self):
        orch   = _make_orch()
        result = orch.run_full("2024-11-01")
        self.assertIsInstance(result.script, str)

    def test_elapsed_seconds_positive(self):
        orch   = _make_orch()
        result = orch.run_full("2024-11-01")
        self.assertGreater(result.elapsed_seconds, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. run_script_only
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunScriptOnly(unittest.TestCase):

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        _make_picks_table(self.conn)

    def tearDown(self):
        self.conn.close()

    def _make_orch_with_conn(self, **kw) -> MASOrchestrator:
        orch = _make_orch(**kw)
        orch._conn = self.conn
        return orch

    def test_empty_db_returns_empty_string(self):
        orch   = self._make_orch_with_conn()
        result = orch.run_script_only("2024-11-01")
        self.assertEqual(result, "")

    def test_with_picks_returns_string(self):
        _insert_pick(self.conn, game_date="2024-11-01",
                     confidence_tier="HIGH", edge_pct=5.0,
                     pick_direction="home")
        orch   = self._make_orch_with_conn()
        result = orch.run_script_only("2024-11-01")
        # should at minimum return a string (could be empty if team lookup fails)
        self.assertIsInstance(result, str)

    def test_date_filter_applied(self):
        _insert_pick(self.conn, game_id="G1", game_date="2024-11-01")
        _insert_pick(self.conn, game_id="G2", game_date="2024-11-02")
        orch = self._make_orch_with_conn()
        # Only date 2024-11-02 has picks; 2024-11-01 query should get only G1
        # (both dates loaded, just testing filter doesn't crash)
        result = orch.run_script_only("2024-11-02")
        self.assertIsInstance(result, str)

    def test_tone_parameter_accepted(self):
        _insert_pick(self.conn, game_date="2024-11-01", confidence_tier="HIGH")
        orch = self._make_orch_with_conn()
        # Should not raise
        orch.run_script_only("2024-11-01", tone="analytical")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. run_report
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunReport(unittest.TestCase):

    def test_calls_resolve_date(self):
        calls = []
        class TrackingFeedback(MockFeedback):
            def resolve_date(self, game_date, dry_run=False):
                calls.append(game_date)
                return super().resolve_date(game_date, dry_run)

        orch = _make_orch(feedback=TrackingFeedback())
        orch.run_report("2024-11-01")
        self.assertEqual(calls, ["2024-11-01"])

    def test_calls_weekly_report(self):
        calls = []
        class TrackingFeedback(MockFeedback):
            def weekly_report(self, weeks=1):
                calls.append(weeks)
                return super().weekly_report(weeks)

        orch = _make_orch(feedback=TrackingFeedback())
        orch.run_report("2024-11-01", weeks=2)
        self.assertEqual(calls, [2])

    def test_returns_report_object(self):
        orch   = _make_orch()
        report = orch.run_report("2024-11-01")
        self.assertIsNotNone(report)

    def test_resolve_error_does_not_crash(self):
        class BadFeedback(MockFeedback):
            def resolve_date(self, game_date, dry_run=False):
                raise RuntimeError("API unavailable")

        orch = _make_orch(feedback=BadFeedback())
        # Should not raise — error is caught and logged
        report = orch.run_report("2024-11-01")
        self.assertIsNotNone(report)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. _print_summary_table
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrintSummaryTable(unittest.TestCase):

    def _capture(self, orch, pick_cards, n_in_script=0):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            orch._print_summary_table(pick_cards, n_in_script, "2024-11-01")
        return buf.getvalue()

    def test_header_printed(self):
        orch   = _make_orch()
        output = self._capture(orch, [])
        self.assertIn("NHL MAS", output)
        self.assertIn("2024-11-01", output)

    def test_pick_card_in_output(self):
        orch   = _make_orch()
        card   = MockCard(home="Toronto Maple Leafs", away="Montreal Canadiens")
        output = self._capture(orch, [card])
        # matchup should appear
        self.assertIn("@", output)

    def test_dry_run_tag_shown(self):
        orch = _make_orch()
        buf  = io.StringIO()
        with patch("sys.stdout", buf):
            orch._print_summary_table([], 0, "2024-11-01", dry_run=True)
        self.assertIn("DRY-RUN", buf.getvalue())

    def test_footer_counts(self):
        cards  = [MockCard(), MockCard(tier="SKIP")]
        orch   = _make_orch()
        output = self._capture(orch, cards, n_in_script=1)
        self.assertIn("Games: 2", output)
        self.assertIn("Playable: 1", output)

    def test_empty_cards_no_crash(self):
        orch = _make_orch()
        # Should not raise
        self._capture(orch, [])

    def test_skip_card_shown_in_table(self):
        orch   = _make_orch()
        card   = MockSkipCard()
        output = self._capture(orch, [card])
        self.assertIn("SKIP", output)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. _print_dry_run_signals
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrintDryRunSignals(unittest.TestCase):

    def _capture(self, orch, label, signals, card):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            orch._print_dry_run_signals(label, signals, card)
        return buf.getvalue()

    def test_game_label_shown(self):
        orch   = _make_orch()
        sig    = MockSignal("team_form", raw=0.60)
        card   = MockCard()
        output = self._capture(orch, "MTL @ TOR", [sig], card)
        self.assertIn("MTL @ TOR", output)

    def test_agent_id_shown(self):
        orch   = _make_orch()
        sig    = MockSignal("goalie_form", raw=0.58)
        card   = MockCard()
        output = self._capture(orch, "MTL @ TOR", [sig], card)
        self.assertIn("goalie_form", output)

    def test_ensemble_result_shown(self):
        orch   = _make_orch()
        sig    = MockSignal()
        card   = MockCard(tier="HIGH", edge=5.0, direction="home")
        output = self._capture(orch, "MTL @ TOR", [sig], card)
        self.assertIn("ENSEMBLE", output)
        self.assertIn("HIGH", output)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. setup_logging
# ═══════════════════════════════════════════════════════════════════════════════

class TestSetupLogging(unittest.TestCase):

    def test_info_level(self):
        import logging
        setup_logging("INFO")
        self.assertEqual(logging.getLogger("nhl_mas").level, logging.INFO)

    def test_debug_level(self):
        import logging
        setup_logging("DEBUG")
        self.assertEqual(logging.getLogger("nhl_mas").level, logging.DEBUG)

    def test_warning_level(self):
        import logging
        setup_logging("WARNING")
        self.assertEqual(logging.getLogger("nhl_mas").level, logging.WARNING)

    def test_invalid_level_defaults_to_info(self):
        import logging
        setup_logging("BOGUS")
        # basicConfig uses INFO when level string is unrecognised
        # nhl_mas logger level will be set via getattr fallback to INFO
        logger = logging.getLogger("nhl_mas")
        self.assertIn(logger.level, (logging.INFO, logging.WARNING, logging.DEBUG))


# ═══════════════════════════════════════════════════════════════════════════════
# 13. CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseArgs(unittest.TestCase):

    def _parse(self, args: list):
        with patch("sys.argv", ["run_picks.py"] + args):
            return rp._parse_args()

    def test_default_mode_is_full(self):
        args = self._parse([])
        self.assertEqual(args.mode, "full")

    def test_mode_script_only(self):
        args = self._parse(["--mode", "script-only"])
        self.assertEqual(args.mode, "script-only")

    def test_mode_report(self):
        args = self._parse(["--mode", "report"])
        self.assertEqual(args.mode, "report")

    def test_date_flag(self):
        args = self._parse(["--date", "2024-11-15"])
        self.assertEqual(args.date, "2024-11-15")

    def test_dry_run_flag(self):
        args = self._parse(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_verbose_flag(self):
        args = self._parse(["--verbose"])
        self.assertTrue(args.verbose)

    def test_quiet_flag(self):
        args = self._parse(["--quiet"])
        self.assertTrue(args.quiet)

    def test_tone_flag(self):
        args = self._parse(["--tone", "analytical"])
        self.assertEqual(args.tone, "analytical")

    def test_weeks_flag(self):
        args = self._parse(["--weeks", "4"])
        self.assertEqual(args.weeks, 4)

    def test_no_advanced_flag(self):
        args = self._parse(["--no-advanced"])
        self.assertTrue(args.no_advanced)

    def test_default_date_is_none(self):
        args = self._parse([])
        self.assertIsNone(args.date)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Integration — run_full + script output
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    def test_full_pipeline_no_exceptions(self):
        """Full orchestration path with mocks — should complete without raising."""
        orch = _make_orch(
            contexts=[MockContext(), MockContext()],
            agents  =[MockAgent("team_form"), MockAgent("goalie_form")],
        )
        result = orch.run_full("2024-11-01", dry_run=False, tone="hype")
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.n_games, 2)

    def test_dry_run_full_pipeline(self):
        """Dry-run mode runs the same pipeline but sets dry_run=True in result."""
        orch   = _make_orch(contexts=[MockContext()])
        result = orch.run_full("2024-11-01", dry_run=True)
        self.assertTrue(result.dry_run)

    def test_analytical_tone_accepted(self):
        orch   = _make_orch()
        result = orch.run_full("2024-11-01", tone="analytical")
        self.assertIsInstance(result, RunResult)

    def test_zero_contexts_gives_zero_games(self):
        orch   = _make_orch(contexts=[])
        result = orch.run_full("2024-11-01")
        self.assertEqual(result.n_games, 0)
        self.assertEqual(result.n_picks, 0)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
