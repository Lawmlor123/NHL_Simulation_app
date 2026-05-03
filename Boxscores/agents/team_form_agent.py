"""
team_form_agent.py
------------------
TeamFormAgent: team-level form analysis.

Signals derived from:
  1. Elo rating differential (computed from game_outcomes.parquet)
  2. 5v5 CF% differential (from play-by-play, Module 1)
  3. xG differential (xGF - xGA last 5 games)
  4. Home/away goal-scoring splits (is_home flag + rolling goals)
  5. Last-10 game form (sk_goals_avg10 + sk_points_avg10)

Elo ratings are computed once per date and cached to elo_cache.json
in the Boxscores folder to avoid recomputing on every call.
"""

import json
import math
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from typing import Optional

from .agent_base import NHLAgent, AgentSignal, logistic, clamp

# ── Config ────────────────────────────────────────────────────────────────────
BASE           = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
ELO_CACHE_FILE = BASE / "elo_cache.json"
OUTCOMES_PATH  = BASE / "game_outcomes.parquet"

ELO_BASE  = 1500.0
ELO_K     = 28.0      # K-factor: balances stability vs responsiveness
ELO_SCALE = 400.0     # Standard Elo scale


# ── Elo engine ────────────────────────────────────────────────────────────────

def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / ELO_SCALE))


def compute_elo_ratings(as_of_date: date) -> dict:
    """
    Compute Elo ratings for all teams from game history up to (not including)
    as_of_date. Caches result to elo_cache.json keyed by date string.

    Returns {team_abbrev: elo_float}
    """
    cache_key = str(as_of_date)

    # Try cache first
    if ELO_CACHE_FILE.exists():
        try:
            cache = json.load(open(ELO_CACHE_FILE))
            if cache_key in cache:
                return cache[cache_key]
        except (json.JSONDecodeError, KeyError):
            cache = {}
    else:
        cache = {}

    if not OUTCOMES_PATH.exists():
        return {}

    outcomes = pd.read_parquet(OUTCOMES_PATH)
    outcomes["game_date"] = pd.to_datetime(outcomes["game_date"])
    outcomes = outcomes[
        outcomes["game_date"] < pd.Timestamp(as_of_date)
    ].sort_values("game_date")

    ratings: dict[str, float] = {}

    for _, row in outcomes.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        r_h   = ratings.get(home, ELO_BASE)
        r_a   = ratings.get(away, ELO_BASE)

        e_h   = _elo_expected(r_h, r_a)
        s_h   = float(row.get("home_win", 0.5))   # 1 = home win, 0 = away win
        s_a   = 1.0 - s_h

        ratings[home] = r_h + ELO_K * (s_h - e_h)
        ratings[away] = r_a + ELO_K * (s_a - (1.0 - e_h))

    # Persist to cache
    cache[cache_key] = ratings
    try:
        json.dump(cache, open(ELO_CACHE_FILE, "w"), indent=2)
    except Exception:
        pass   # cache write failure is non-fatal

    return ratings


def elo_win_probability(elo_home: float, elo_away: float) -> float:
    """Returns expected home win probability based on Elo ratings."""
    return _elo_expected(elo_home, elo_away)


# ── Agent ─────────────────────────────────────────────────────────────────────

class TeamFormAgent(NHLAgent):
    """
    Analyzes team-level form to predict home win probability.

    Factor weights (sum to 1.0):
        Elo differential        0.35
        xG differential         0.30
        CF% differential        0.20
        Last-10 goal form       0.15
    """

    FACTOR_WEIGHTS = {
        "elo":      0.35,
        "xg":       0.30,
        "cf":       0.20,
        "form10":   0.15,
    }

    def __init__(self, weight: float = 1.0, confidence_floor: float = 0.52,
                 target_date: Optional[date] = None):
        super().__init__("team_form", weight, confidence_floor)
        self._target_date = target_date or date.today()
        self._elo_ratings: Optional[dict] = None

    def _get_elo(self) -> dict:
        if self._elo_ratings is None:
            self._elo_ratings = compute_elo_ratings(self._target_date)
        return self._elo_ratings

    def analyze(self, game_context) -> AgentSignal:
        home = game_context.home
        away = game_context.away

        factors = {}
        available = 0

        # ── Factor 1: Elo differential ───────────────────────────────────
        elo = self._get_elo()
        elo_home = elo.get(home.team, ELO_BASE)
        elo_away = elo.get(away.team, ELO_BASE)
        elo_prob = elo_win_probability(elo_home, elo_away)
        factors["elo"] = elo_prob
        factors["elo_home"] = round(elo_home, 1)
        factors["elo_away"] = round(elo_away, 1)
        available += 1

        # ── Factor 2: xG differential ────────────────────────────────────
        xg_score = 0.5   # default: no signal
        if (home.xg_for_last5 is not None and home.xg_against_last5 is not None and
                away.xg_for_last5 is not None and away.xg_against_last5 is not None):

            home_xg_net = home.xg_for_last5 - home.xg_against_last5
            away_xg_net = away.xg_for_last5 - away.xg_against_last5
            xg_diff = home_xg_net - away_xg_net    # positive = home better
            # Map ±2.0 goal differential to 0-1
            xg_score = logistic(xg_diff, scale=0.8)
            factors["xg_diff"] = round(xg_diff, 3)
            factors["xg"] = round(xg_score, 4)
            available += 1
        else:
            factors["xg"] = None

        # ── Factor 3: CF% differential ───────────────────────────────────
        cf_score = 0.5
        if home.cf_pct_last5 is not None and away.cf_pct_last5 is not None:
            cf_diff = home.cf_pct_last5 - away.cf_pct_last5   # e.g. 0.52 - 0.48 = 0.04
            # 1% CF% edge ≈ small but real advantage
            cf_score = logistic(cf_diff, scale=12.0)
            factors["cf_diff"] = round(cf_diff, 4)
            factors["cf"] = round(cf_score, 4)
            available += 1
        else:
            factors["cf"] = None

        # ── Factor 4: Last-10 goal form ───────────────────────────────────
        form10_score = 0.5
        h_goals10 = home.sk_goals_avg10_mean
        a_goals10 = away.sk_goals_avg10_mean
        if h_goals10 is not None and a_goals10 is not None:
            goal_diff = h_goals10 - a_goals10
            form10_score = logistic(goal_diff, scale=2.5)
            factors["goal_diff_10"] = round(goal_diff, 4)
            factors["form10"] = round(form10_score, 4)
            available += 1
        else:
            factors["form10"] = None

        # ── Combine factors ───────────────────────────────────────────────
        # Use weights; fall back to equal-weight if factors missing
        scores = {
            "elo":    elo_prob,
            "xg":     xg_score,
            "cf":     cf_score,
            "form10": form10_score,
        }
        weights = self.FACTOR_WEIGHTS.copy()

        # Zero-out missing factors and renormalize
        for k in list(weights.keys()):
            if factors.get(k) is None:
                weights[k] = 0.0
        total_w = sum(weights.values())
        if total_w == 0:
            return self.neutral_signal("No team form data available")

        composite = sum(scores[k] * weights[k] for k in weights) / total_w

        # ── Confidence: scales with available factors + Elo spread ───────
        elo_spread = abs(elo_home - elo_away)
        elo_conf_boost = min(elo_spread / 200.0, 0.15)   # max +0.15 from Elo spread
        base_conf = 0.50 + (available / 4) * 0.25        # 0.50–0.75 based on data
        confidence = clamp(base_conf + elo_conf_boost)

        # ── Direction ─────────────────────────────────────────────────────
        if composite > 0.52:
            direction = "home"
        elif composite < 0.48:
            direction = "away"
        else:
            direction = "neutral"

        # ── Reasoning string ──────────────────────────────────────────────
        elo_line = f"Elo {elo_home:.0f} vs {elo_away:.0f} ({elo_prob:.1%} home)"
        xg_line  = (f"xG net {factors.get('xg_diff', 'N/A')}"
                    if factors.get("xg_diff") is not None else "xG N/A")
        cf_line  = (f"CF% +{factors['cf_diff']:.3f}"
                    if factors.get("cf_diff") is not None else "CF% N/A")
        reasoning = f"{elo_line} | {xg_line} | {cf_line} | composite={composite:.3f}"

        return AgentSignal(
            agent_id       = self.agent_id,
            pick_direction = direction,
            raw_score      = round(composite, 4),
            confidence     = round(confidence, 4),
            reasoning      = reasoning,
            factors        = factors,
        )
