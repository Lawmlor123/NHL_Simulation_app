"""
player_form_agent.py
--------------------
PlayerFormAgent: individual player contribution analysis.

Signals derived from:
  1. Top-6 forward pts/60 differential (last 5 games)
  2. Power-play unit strength (pp_goals rolling sum)
  3. Line combo stability proxy (variance in scoring contribution)
  4. Injured player impact score (from news headlines keyword scoring)

The agent loads skater_features.parquet to get player-level data
(team-level aggregates in GameContext aren't granular enough for top-6 analysis).
Parquet is injected on init to avoid re-reading on every analyze() call.
"""

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .agent_base import NHLAgent, AgentSignal, logistic, clamp

BASE          = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
SKATER_PATH   = BASE / "skater_features.parquet"

# Keywords that indicate injured/absent top players — negative impact
INJURY_KEYWORDS = [
    r"\binjur",r"\bruled?\s+out\b", r"\bday[\s-]to[\s-]day\b", r"\bltir?\b",
    r"\bscratched\b", r"\bunavailable\b", r"\bmissing\b", r"\bsidelined\b",
    r"\bsurgery\b", r"\bconcussion\b",
]
# Keywords indicating good news
RETURN_KEYWORDS = [
    r"\breturns?\b", r"\bback\s+in\s+lineup\b", r"\bcleared\b",
    r"\bfull\s+practice\b", r"\bno\s+injury\b",
]


def _injury_score(headlines: list[str]) -> float:
    """
    Returns a multiplier for the team's player form signal:
      1.0 = neutral, <1.0 = injury concern, >1.0 = good injury news
    Range: approximately 0.75–1.15
    """
    if not headlines:
        return 1.0

    text  = " ".join(headlines).lower()
    neg   = sum(1 for pat in INJURY_KEYWORDS  if re.search(pat, text))
    pos   = sum(1 for pat in RETURN_KEYWORDS  if re.search(pat, text))
    net   = pos - neg
    # Each net keyword shifts multiplier by ±0.08, capped at ±0.25
    return clamp(1.0 + net * 0.08, 0.70, 1.20)


def _pts_per_60(player_row: pd.Series, window: int = 5) -> Optional[float]:
    col = f"pts_per60_avg{window}"
    if col in player_row.index:
        val = player_row[col]
        if pd.notna(val):
            return float(val)
    # Fallback: compute from points and toi
    pts_col = f"points_avg{window}"
    toi_col = f"toi_min_avg{window}"
    if pts_col in player_row.index and toi_col in player_row.index:
        pts = player_row[pts_col]
        toi = player_row[toi_col]
        if pd.notna(pts) and pd.notna(toi) and toi > 0:
            return float(pts / toi * 60)
    return None


class PlayerFormAgent(NHLAgent):
    """
    Analyzes player-level offensive contribution to predict home win probability.

    Factor weights:
        Top-6 pts/60 differential    0.45
        PP unit strength             0.35
        Injury multiplier            0.20
    """

    def __init__(
        self,
        weight:           float = 1.0,
        confidence_floor: float = 0.52,
        sk:               Optional[pd.DataFrame] = None,
    ):
        super().__init__("player_form", weight, confidence_floor)
        # Accept injected parquet or lazy-load on first analyze()
        self._sk = sk
        self._sk_loaded = sk is not None

    def _load_sk(self):
        if not self._sk_loaded:
            if SKATER_PATH.exists():
                sk = pd.read_parquet(SKATER_PATH)
                sk["game_date"] = pd.to_datetime(sk["game_date"])
                self._sk = sk
            else:
                self._sk = pd.DataFrame()
            self._sk_loaded = True

    def _team_latest(self, team: str) -> pd.DataFrame:
        """Get latest stats row per player for a given team."""
        if self._sk is None or self._sk.empty:
            return pd.DataFrame()
        team_df = self._sk[self._sk["team"] == team]
        if team_df.empty:
            return pd.DataFrame()
        return (team_df
                .sort_values(["player_id", "game_date"])
                .groupby("player_id")
                .last()
                .reset_index())

    def _top6_pts60(self, team: str, window: int = 5) -> Optional[float]:
        """
        Average pts/60 of the top-6 forwards by recent production.
        Filters to forwards (C, L, R, W) with meaningful TOI (>8 min avg).
        """
        latest = self._team_latest(team)
        if latest.empty:
            return None

        # Filter forwards with enough TOI
        toi_col = f"toi_min_avg{window}"
        pos_mask = latest["position"].isin(["C", "L", "R", "W", "F"]) if "position" in latest.columns else pd.Series(True, index=latest.index)
        toi_mask  = (latest[toi_col] >= 8.0) if toi_col in latest.columns else pd.Series(True, index=latest.index)
        fwds = latest[pos_mask & toi_mask]

        if fwds.empty:
            fwds = latest   # fallback: use all skaters

        # Compute pts/60 per player
        fwds = fwds.copy()
        fwds["_p60"] = fwds.apply(lambda r: _pts_per_60(r, window), axis=1)
        fwds = fwds.dropna(subset=["_p60"])
        if fwds.empty:
            return None

        # Top-6 by pts/60
        top6 = fwds.nlargest(6, "_p60")
        return float(top6["_p60"].mean())

    def _pp_strength(self, team_ctx) -> float:
        """
        Power-play strength score from pp_goals rolling sum.
        Returns 0-1 relative score (higher = stronger PP).
        PP goals sum last 5 typically ranges 0-10 across NHL teams.
        """
        pp = team_ctx.sk_pp_goals_avg5_sum
        if pp is None:
            return 0.5
        # Normalize: 5 PP goals per player-game rolling avg is high
        # At team level, 2.0 is avg, 4+ is excellent
        return clamp(logistic(pp - 2.0, scale=0.6))

    def analyze(self, game_context) -> AgentSignal:
        self._load_sk()

        home = game_context.home
        away = game_context.away
        factors = {}

        # ── Factor 1: Top-6 pts/60 differential ──────────────────────────
        h_p60 = self._top6_pts60(home.team)
        a_p60 = self._top6_pts60(away.team)

        p60_score = 0.5
        if h_p60 is not None and a_p60 is not None:
            p60_diff  = h_p60 - a_p60
            # ±0.5 pts/60 is a meaningful top-6 gap
            p60_score = logistic(p60_diff, scale=2.5)
            factors["home_top6_p60"] = round(h_p60, 3)
            factors["away_top6_p60"] = round(a_p60, 3)
            factors["p60_diff"]      = round(p60_diff, 3)
        else:
            # Fallback: use GameContext aggregate pts/60
            h_pts = home.sk_points_avg5_mean
            a_pts = away.sk_points_avg5_mean
            if h_pts is not None and a_pts is not None:
                p60_score = logistic(h_pts - a_pts, scale=3.0)
                factors["p60_fallback"] = True

        factors["p60_score"] = round(p60_score, 4)

        # ── Factor 2: PP unit strength ────────────────────────────────────
        h_pp = self._pp_strength(home)
        a_pp = self._pp_strength(away)
        pp_diff  = h_pp - a_pp
        pp_score = logistic(pp_diff, scale=4.0)
        factors["pp_score"] = round(pp_score, 4)
        factors["pp_diff"]  = round(pp_diff, 4)

        # ── Factor 3: Injury multiplier from news headlines ───────────────
        h_inj = _injury_score(home.news_headlines)
        a_inj = _injury_score(away.news_headlines)
        # Net injury advantage: home multiplier vs away multiplier
        # Convert to 0-1 score: 0.5 = even, >0.5 = home less injured
        inj_ratio  = h_inj / a_inj if a_inj > 0 else 1.0
        inj_score  = logistic(math.log(inj_ratio) if inj_ratio > 0 else 0, scale=3.0)
        factors["home_injury_mult"] = round(h_inj, 3)
        factors["away_injury_mult"] = round(a_inj, 3)
        factors["inj_score"]        = round(inj_score, 4)

        # ── Combine ───────────────────────────────────────────────────────
        composite = (
            0.45 * p60_score +
            0.35 * pp_score  +
            0.20 * inj_score
        )

        # Confidence: higher when parquet data available and injury news present
        has_parquet = (h_p60 is not None and a_p60 is not None)
        has_injury  = bool(home.news_headlines or away.news_headlines)
        confidence  = clamp(0.45 + (0.20 if has_parquet else 0) + (0.10 if has_injury else 0))

        direction = "home" if composite > 0.52 else ("away" if composite < 0.48 else "neutral")

        p60_str = (f"top6 p/60: {h_p60:.2f} vs {a_p60:.2f}"
                   if h_p60 and a_p60 else "top6 p/60: N/A")
        pp_str  = f"PP {home.team}:{h_pp:.2f} vs {away.team}:{a_pp:.2f}"
        inj_str = (f"injury mult {h_inj:.2f}/{a_inj:.2f}"
                   if (h_inj != 1.0 or a_inj != 1.0) else "no injury flags")

        return AgentSignal(
            agent_id       = self.agent_id,
            pick_direction = direction,
            raw_score      = round(composite, 4),
            confidence     = round(confidence, 4),
            reasoning      = f"{p60_str} | {pp_str} | {inj_str}",
            factors        = factors,
        )
