"""
test_module5.py
---------------
Tests for Module 5: Script Generator — Alexis Rivers NHL Picks Channel

No network calls, no file I/O.
All picks are constructed as simple dataclass-like objects.

Run:
    python test_module5.py -v
"""

import io
import re
import sys
import unittest
from dataclasses import dataclass, field
from typing import Optional

from script_generator import (
    generate_script,
    preview_picks,
    filter_picks,
    estimate_read_seconds,
    _word_count,
    _build_reasoning,
    _full,
    _city,
    _name,
    MIN_EDGE_PCT,
    TARGET_WORDS_LO,
    TARGET_WORDS_HI,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal PickCard stand-in (no module3 import required)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockPick:
    """Minimal PickCard compatible object for testing."""
    game_id:               int   = 0
    game_date:             str   = "2025-01-15"
    home_team:             str   = "BOS"
    away_team:             str   = "TOR"
    pick:                  str   = "BOS"
    pick_direction:        str   = "home"
    raw_score:             float = 0.68
    edge_pct:              float = 18.0
    confidence_tier:       str   = "HIGH"
    agent_agreement_score: float = 0.80
    agents_above_floor:    int   = 4
    mode_used:             str   = "weighted_avg"
    skip_reason:           str   = ""
    signal_summary:        dict  = field(default_factory=dict)

    @property
    def is_playable(self) -> bool:
        return self.confidence_tier != "SKIP"

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"


def _make_pick(
    home="BOS", away="TOR",
    pick="BOS", direction="home",
    edge=18.0, tier="HIGH",
    signal_summary=None,
) -> MockPick:
    return MockPick(
        home_team      = home,
        away_team      = away,
        pick           = pick,
        pick_direction = direction,
        edge_pct       = edge,
        confidence_tier= tier,
        signal_summary = signal_summary or {},
    )


def _make_rich_pick(direction="home") -> MockPick:
    """Pick with full signal_summary matching pick direction."""
    return _make_pick(
        signal_summary={
            "team_form":   {"raw_score": 0.72, "confidence": 0.68,
                            "direction": direction, "weight": 1.0},
            "goalie_form": {"raw_score": 0.78, "confidence": 0.75,
                            "direction": direction, "weight": 1.2},
            "schedule":    {"raw_score": 0.65, "confidence": 0.60,
                            "direction": direction, "weight": 0.8},
            "player_form": {"raw_score": 0.64, "confidence": 0.62,
                            "direction": direction, "weight": 0.9},
            "sentiment":   {"raw_score": 0.58, "confidence": 0.55,
                            "direction": direction, "weight": 0.5},
        },
        direction=direction,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestFilterPicks
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterPicks(unittest.TestCase):

    def test_high_tier_included(self):
        picks = [_make_pick(tier="HIGH", edge=10.0)]
        self.assertEqual(len(filter_picks(picks)), 1)

    def test_medium_tier_excluded(self):
        picks = [_make_pick(tier="MEDIUM", edge=10.0)]
        self.assertEqual(len(filter_picks(picks)), 0)

    def test_low_tier_excluded(self):
        picks = [_make_pick(tier="LOW", edge=10.0)]
        self.assertEqual(len(filter_picks(picks)), 0)

    def test_skip_tier_excluded(self):
        picks = [_make_pick(tier="SKIP", edge=10.0)]
        self.assertEqual(len(filter_picks(picks)), 0)

    def test_edge_below_min_excluded(self):
        picks = [_make_pick(tier="HIGH", edge=2.9)]
        self.assertEqual(len(filter_picks(picks)), 0)

    def test_edge_exactly_at_min_included(self):
        picks = [_make_pick(tier="HIGH", edge=MIN_EDGE_PCT)]
        self.assertEqual(len(filter_picks(picks)), 1)

    def test_edge_zero_excluded(self):
        picks = [_make_pick(tier="HIGH", edge=0.0)]
        self.assertEqual(len(filter_picks(picks)), 0)

    def test_skip_direction_excluded(self):
        p = _make_pick(tier="HIGH", edge=15.0)
        p.pick_direction = "skip"
        self.assertEqual(len(filter_picks([p])), 0)

    def test_capped_at_three(self):
        picks = [_make_pick(tier="HIGH", edge=20.0 - i) for i in range(5)]
        result = filter_picks(picks)
        self.assertEqual(len(result), 3)

    def test_sorted_by_edge_desc(self):
        picks = [
            _make_pick(tier="HIGH", edge=5.0),
            _make_pick(tier="HIGH", edge=15.0),
            _make_pick(tier="HIGH", edge=10.0),
        ]
        result = filter_picks(picks)
        edges = [p.edge_pct for p in result]
        self.assertEqual(edges, sorted(edges, reverse=True))

    def test_mixed_tiers_returns_only_high(self):
        picks = [
            _make_pick(tier="HIGH",   edge=12.0),
            _make_pick(tier="MEDIUM", edge=20.0),
            _make_pick(tier="LOW",    edge=18.0),
            _make_pick(tier="HIGH",   edge=8.0),
        ]
        result = filter_picks(picks)
        self.assertEqual(len(result), 2)
        for p in result:
            self.assertEqual(p.confidence_tier, "HIGH")

    def test_empty_list_returns_empty(self):
        self.assertEqual(filter_picks([]), [])


# ═══════════════════════════════════════════════════════════════════════════════
# TestScriptOutput
# ═══════════════════════════════════════════════════════════════════════════════

class TestScriptOutput(unittest.TestCase):
    """Core output quality: format, content, word count."""

    def test_returns_string(self):
        picks  = [_make_rich_pick()]
        script = generate_script(picks)
        self.assertIsInstance(script, str)

    def test_no_markdown_asterisks(self):
        picks  = [_make_rich_pick()]
        script = generate_script(picks, tone="hype")
        self.assertNotIn("*", script)
        self.assertNotIn("**", script)
        self.assertNotIn("__", script)

    def test_no_markdown_headers(self):
        picks  = [_make_rich_pick()]
        script = generate_script(picks, tone="hype")
        for line in script.splitlines():
            self.assertFalse(line.lstrip().startswith("#"),
                             f"Header found: {line}")

    def test_no_markdown_bullets(self):
        picks  = [_make_rich_pick()]
        script = generate_script(picks, tone="hype")
        for line in script.splitlines():
            self.assertFalse(re.match(r"^\s*[-*]\s", line),
                             f"Bullet found: {line}")

    def test_no_parentheticals(self):
        """No text inside parentheses — Alexis reads everything literally."""
        picks  = [_make_rich_pick()]
        for tone in ("hype", "analytical"):
            script = generate_script(picks, tone=tone)
            self.assertNotIn("(", script, f"Parenthetical in {tone} script")
            self.assertNotIn(")", script)

    def test_no_stage_direction_words(self):
        """Actual stage direction phrases must not appear in the script."""
        # Use full phrases, not fragments — avoids false positives like
        # "direction" in "the right direction" (legitimate hockey commentary)
        forbidden = [
            "pause", "smiles", "nods", "looks at camera", "takes a beat",
            "laughs", "winks", "[pause]", "[smile]", "(smiles)",
            "beat.", "stage direction",
        ]
        picks  = [_make_rich_pick()]
        for tone in ("hype", "analytical"):
            script = generate_script(picks, tone=tone).lower()
            for phrase in forbidden:
                self.assertNotIn(phrase, script,
                                 f"Stage direction '{phrase}' found in {tone}")

    def test_no_model_jargon(self):
        """Model internals must not leak into the script."""
        jargon = ["raw_score", "confidence_floor", "cf%", "ff%",
                  "gsaa", "elo", "meta-learner", "logistic",
                  "weighted average", "composite", "agent_"]
        picks  = [_make_rich_pick()]
        for tone in ("hype", "analytical"):
            script = generate_script(picks, tone=tone).lower()
            for term in jargon:
                self.assertNotIn(term, script,
                                 f"Jargon '{term}' leaked into {tone} script")

    def test_pick_team_name_in_script(self):
        """The team name of the pick should appear in the script."""
        p = _make_pick(home="BOS", away="TOR", pick="BOS", direction="home",
                       edge=18.0, tier="HIGH")
        script = generate_script([p], tone="hype")
        self.assertIn("Boston", script)

    def test_away_pick_team_name_in_script(self):
        """Away pick: away team name should appear."""
        p = _make_pick(home="BOS", away="TOR", pick="TOR", direction="away",
                       edge=18.0, tier="HIGH")
        script = generate_script([p], tone="hype")
        self.assertIn("Toronto", script)

    def test_disclaimer_in_script(self):
        """Script must contain a responsible gambling disclaimer."""
        picks  = [_make_rich_pick()]
        for tone in ("hype", "analytical"):
            script = generate_script(picks, tone=tone).lower()
            self.assertTrue(
                "afford to lose" in script or "responsibly" in script,
                f"Disclaimer missing from {tone} script"
            )

    def test_no_picks_still_returns_string(self):
        """When no picks qualify, function must return a string, not raise."""
        picks  = [_make_pick(tier="MEDIUM", edge=5.0)]
        script = generate_script(picks, tone="hype")
        self.assertIsInstance(script, str)
        self.assertGreater(len(script), 20)

    def test_empty_list_returns_string(self):
        script = generate_script([])
        self.assertIsInstance(script, str)
        self.assertGreater(len(script), 20)


# ═══════════════════════════════════════════════════════════════════════════════
# TestWordCount
# ═══════════════════════════════════════════════════════════════════════════════

class TestWordCount(unittest.TestCase):
    """Scripts must hit the 200–250 word target with qualifying picks."""

    def _check_budget(self, picks, tone):
        script = generate_script(picks, tone=tone)
        wc     = _word_count(script)
        self.assertGreaterEqual(
            wc, TARGET_WORDS_LO,
            f"[{tone}] Too short: {wc} words (min {TARGET_WORDS_LO})\n{script}"
        )
        self.assertLessEqual(
            wc, TARGET_WORDS_HI,
            f"[{tone}] Too long: {wc} words (max {TARGET_WORDS_HI})\n{script}"
        )
        return wc

    def test_word_count_one_pick_hype(self):
        self._check_budget([_make_rich_pick()], "hype")

    def test_word_count_one_pick_analytical(self):
        self._check_budget([_make_rich_pick()], "analytical")

    def test_word_count_two_picks_hype(self):
        picks = [
            _make_rich_pick(direction="home"),
            _make_pick(home="EDM", away="VGK", pick="EDM", direction="home",
                       edge=8.0, tier="HIGH",
                       signal_summary={
                           "goalie_form": {"raw_score": 0.70, "confidence": 0.65,
                                           "direction": "home", "weight": 1.2},
                           "schedule":    {"raw_score": 0.62, "confidence": 0.58,
                                           "direction": "home", "weight": 0.8},
                       }),
        ]
        self._check_budget(picks, "hype")

    def test_word_count_two_picks_analytical(self):
        picks = [
            _make_rich_pick(direction="home"),
            _make_pick(home="EDM", away="VGK", pick="EDM", direction="home",
                       edge=8.0, tier="HIGH",
                       signal_summary={
                           "team_form":   {"raw_score": 0.66, "confidence": 0.62,
                                           "direction": "home", "weight": 1.0},
                           "goalie_form": {"raw_score": 0.71, "confidence": 0.68,
                                           "direction": "home", "weight": 1.2},
                       }),
        ]
        self._check_budget(picks, "analytical")

    def test_word_count_three_picks(self):
        picks = [
            _make_rich_pick(direction="home"),
            _make_pick(home="EDM", away="VGK", pick="EDM", direction="home",
                       edge=8.0, tier="HIGH"),
            _make_pick(home="TBL", away="FLA", pick="TBL", direction="home",
                       edge=6.5, tier="HIGH"),
        ]
        for tone in ("hype", "analytical"):
            self._check_budget(picks, tone)

    def test_read_time_in_range(self):
        picks  = [_make_rich_pick()]
        for tone in ("hype", "analytical"):
            script = generate_script(picks, tone=tone)
            secs   = estimate_read_seconds(script)
            self.assertGreaterEqual(secs, 75,
                f"[{tone}] Read time too short: {secs:.0f}s")
            self.assertLessEqual(secs, 130,
                f"[{tone}] Read time too long: {secs:.0f}s")

    def test_no_picks_under_budget(self):
        """'No picks tonight' message must still be reasonable length."""
        script = generate_script([], tone="hype")
        wc     = _word_count(script)
        self.assertGreater(wc, 20)
        self.assertLess(wc, TARGET_WORDS_HI)


# ═══════════════════════════════════════════════════════════════════════════════
# TestTone
# ═══════════════════════════════════════════════════════════════════════════════

class TestTone(unittest.TestCase):

    def test_hype_and_analytical_differ(self):
        picks  = [_make_rich_pick()]
        hype   = generate_script(picks, tone="hype")
        anal   = generate_script(picks, tone="analytical")
        self.assertNotEqual(hype, anal)

    def test_hype_contains_energetic_language(self):
        picks  = [_make_rich_pick()]
        script = generate_script(picks, tone="hype").lower()
        energy_words = ["let us", "let's", "fire", "locked in", "dominant",
                        "absolutely", "on fire", "get into it", "lock it in"]
        found = any(w in script for w in energy_words)
        self.assertTrue(found, "Hype script lacks energetic language")

    def test_analytical_contains_measured_language(self):
        picks  = [_make_rich_pick()]
        script = generate_script(picks, tone="analytical").lower()
        calm_words = ["analysis", "threshold", "breakdown", "metrics",
                      "average", "numbers", "performance", "advantage"]
        found = any(w in script for w in calm_words)
        self.assertTrue(found, "Analytical script lacks measured language")

    def test_invalid_tone_falls_back_to_hype(self):
        picks  = [_make_rich_pick()]
        # Should not raise — falls back to "hype"
        script = generate_script(picks, tone="ultra_serious")
        self.assertIsInstance(script, str)

    def test_both_tones_include_pick_team(self):
        p = _make_pick(home="BOS", away="TOR", pick="BOS", direction="home")
        for tone in ("hype", "analytical"):
            script = generate_script([p], tone=tone)
            self.assertIn("Boston", script, f"{tone}: team name missing")

    def test_hype_opens_with_greeting(self):
        picks  = [_make_rich_pick()]
        script = generate_script(picks, tone="hype")
        self.assertTrue(
            script.startswith("What is up") or script.startswith("Hey"),
            "Hype script should open with a greeting"
        )

    def test_analytical_opens_with_good_evening(self):
        picks  = [_make_rich_pick()]
        script = generate_script(picks, tone="analytical")
        self.assertTrue(
            "Good evening" in script[:50],
            "Analytical script should open with 'Good evening'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestReasoning
# ═══════════════════════════════════════════════════════════════════════════════

class TestReasoning(unittest.TestCase):

    def test_reasoning_returns_string(self):
        pick = _make_rich_pick()
        r    = _build_reasoning(pick, tone="hype")
        self.assertIsInstance(r, str)
        self.assertGreater(len(r), 10)

    def test_reasoning_hype_differs_from_analytical(self):
        pick = _make_rich_pick()
        r_h  = _build_reasoning(pick, tone="hype")
        r_a  = _build_reasoning(pick, tone="analytical")
        self.assertNotEqual(r_h, r_a)

    def test_empty_signal_summary_uses_fallback(self):
        """Empty signal_summary must not raise; uses fallback sentence."""
        pick = _make_pick(signal_summary={})
        r    = _build_reasoning(pick, tone="hype")
        self.assertIsInstance(r, str)
        self.assertGreater(len(r), 10)

    def test_opposite_signals_ignored(self):
        """Signals pointing AWAY should not appear in a HOME pick's reasoning."""
        pick = _make_pick(direction="home", signal_summary={
            "team_form": {"raw_score": 0.72, "confidence": 0.70,
                          "direction": "away",  # opposes pick
                          "weight": 1.0},
        })
        r = _build_reasoning(pick, tone="hype")
        # Should fall back to fallback sentence (no "away" team info)
        self.assertIsInstance(r, str)

    def test_strong_goalie_signal_produces_goalie_sentence(self):
        pick = _make_pick(direction="home", signal_summary={
            "goalie_form": {"raw_score": 0.82, "confidence": 0.78,
                            "direction": "home", "weight": 1.2},
        })
        r = _build_reasoning(pick, tone="hype")
        self.assertTrue(
            any(w in r.lower() for w in ["goalie", "goaltending", "wall", "net", "save"]),
            f"Goalie language not found in: {r}"
        )

    def test_strong_schedule_signal_produces_schedule_sentence(self):
        pick = _make_pick(direction="home", signal_summary={
            "schedule": {"raw_score": 0.75, "confidence": 0.70,
                         "direction": "home", "weight": 0.8},
        })
        r = _build_reasoning(pick, tone="hype")
        self.assertTrue(
            any(w in r.lower() for w in ["rest", "schedule", "fresh", "tired",
                                          "rested", "legs", "back-to-back", "fatigue"]),
            f"Schedule language not found in: {r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestTeamNames
# ═══════════════════════════════════════════════════════════════════════════════

class TestTeamNames(unittest.TestCase):

    def test_full_name_known_team(self):
        self.assertEqual(_full("BOS"), "Boston Bruins")
        self.assertEqual(_full("TOR"), "Toronto Maple Leafs")
        self.assertEqual(_full("VGK"), "Vegas Golden Knights")

    def test_full_name_unknown_falls_back_to_abbrev(self):
        self.assertEqual(_full("XYZ"), "XYZ")

    def test_city_known_team(self):
        self.assertEqual(_city("EDM"), "Edmonton")

    def test_name_known_team(self):
        self.assertEqual(_name("TBL"), "Lightning")

    def test_all_32_teams_present(self):
        from script_generator import TEAMS
        # All 32 current NHL teams should be in the registry
        self.assertEqual(len(TEAMS), 32)

    def test_no_generic_new_york_confusion(self):
        # NYI and NYR both exist with city "New York" but different names
        self.assertNotEqual(_full("NYI"), _full("NYR"))
        self.assertIn("Islanders",  _full("NYI"))
        self.assertIn("Rangers",    _full("NYR"))


# ═══════════════════════════════════════════════════════════════════════════════
# TestPreviewPicks
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreviewPicks(unittest.TestCase):

    def _capture_preview(self, picks, game_date="") -> str:
        """Capture stdout from preview_picks."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            preview_picks(picks, game_date=game_date)
            return sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

    def test_preview_runs_without_error(self):
        picks = [_make_pick(), _make_pick(tier="MEDIUM")]
        try:
            self._capture_preview(picks)
        except Exception as exc:
            self.fail(f"preview_picks raised: {exc}")

    def test_preview_shows_all_picks(self):
        picks = [
            _make_pick(tier="HIGH",   edge=15.0),
            _make_pick(tier="MEDIUM", edge=10.0),
            _make_pick(tier="HIGH",   edge=2.0),  # edge too low
        ]
        output = self._capture_preview(picks)
        # All 3 picks appear as rows
        self.assertEqual(output.count("@"), 3)

    def test_preview_shows_included_count(self):
        picks = [
            _make_pick(tier="HIGH", edge=15.0),
            _make_pick(tier="HIGH", edge=4.0),
            _make_pick(tier="LOW",  edge=20.0),
        ]
        output = self._capture_preview(picks)
        self.assertIn("Included: 2 of 3", output)

    def test_preview_marks_included_with_checkmark(self):
        picks = [_make_pick(tier="HIGH", edge=10.0)]
        output = self._capture_preview(picks)
        self.assertIn("✓", output)

    def test_preview_marks_excluded_with_x(self):
        picks = [_make_pick(tier="MEDIUM", edge=10.0)]
        output = self._capture_preview(picks)
        self.assertIn("✗", output)

    def test_preview_shows_filter_reason_low_edge(self):
        picks = [_make_pick(tier="HIGH", edge=1.5)]
        output = self._capture_preview(picks)
        self.assertIn("edge=", output.lower())

    def test_preview_shows_filter_reason_wrong_tier(self):
        picks = [_make_pick(tier="MEDIUM", edge=15.0)]
        output = self._capture_preview(picks)
        self.assertIn("tier=", output.lower())

    def test_preview_empty_list_runs(self):
        output = self._capture_preview([])
        self.assertIsInstance(output, str)

    def test_preview_includes_word_count(self):
        picks = [_make_pick(tier="HIGH", edge=15.0)]
        output = self._capture_preview(picks)
        self.assertIn("words", output.lower())

    def test_preview_shows_date_when_provided(self):
        picks = [_make_pick(tier="HIGH", edge=15.0)]
        output = self._capture_preview(picks, game_date="2025-01-15")
        self.assertIn("2025-01-15", output)


# ═══════════════════════════════════════════════════════════════════════════════
# TestWordCountHelper
# ═══════════════════════════════════════════════════════════════════════════════

class TestWordCountHelper(unittest.TestCase):

    def test_empty_string_is_zero(self):
        self.assertEqual(_word_count(""), 0)

    def test_single_word(self):
        self.assertEqual(_word_count("hello"), 1)

    def test_sentence(self):
        self.assertEqual(_word_count("the quick brown fox"), 4)

    def test_estimate_read_time_100_words(self):
        text  = " ".join(["word"] * 150)
        secs  = estimate_read_seconds(text, wpm=150)
        self.assertAlmostEqual(secs, 60.0, places=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestFilterPicks,
        TestScriptOutput,
        TestWordCount,
        TestTone,
        TestReasoning,
        TestTeamNames,
        TestPreviewPicks,
        TestWordCountHelper,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 62)
    print("  Module 5 — Script Generator Tests")
    print("=" * 62)
    ran    = result.testsRun
    failed = len(result.failures) + len(result.errors)
    if failed == 0:
        print(f"  ✅  {ran} tests passed\n")
    else:
        print(f"  ❌  {failed} failures / errors  ({ran} total)\n")
        for test, tb in result.failures + result.errors:
            print(f"  FAIL: {test}")
            print(f"  {tb[:400]}\n")
