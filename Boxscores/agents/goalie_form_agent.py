"""
goalie_form_agent.py
--------------------
GoalieFormAgent: goalie matchup analysis.

Signals derived from:
  1. Starter save% -- rolling avg3/avg5 vs opponent shot volume
  2. GSAA differential -- goals saved above average (home vs away goalie)
  3. Head-to-head save% proxy -- last start vs same opponent (if in parquet)
  4. Starter confirmed flag -- unconfirmed goalies reduce confidence sharply
  5. Shot-profile adjustment -- high-volume opponent = more xGA exposure

This is typically the highest-weight agent for hockey (goalie = ~30% of outcome).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .agent_base import NHLAgent, AgentSignal, logistic, clamp

BASE          = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
GOALIE_PATH   = BASE / "goalie_features.parquet"

LEAGUE_AVG_SPP     = 0.906    # league average save% for normalization
LEAGUE_AVG_SOG_GP  = 29.5     # league average shots against per game

# Confidence penalty for unconfirmed starter status.
# Two-tier: light penalty when we still have rolling parquet form data,
# heavy penalty only when starter identity is truly unknown AND no data exists.
# Rationale: an "unknown" starter at 9am still has 3-season rolling SV% in the
# parquet -- that signal is real and shouldn't be buried below the confidence floor.
UNCONFIRMED_PENALTY_HAS_DATA = 0.04   # unknown status but rolling form data available
UNCONFIRMED_PENALTY_NO_DATA  = 0.15   # unknown status AND no rolling data at all


class GoalieFormAgent(NHLAgent):
    """
    Analyzes goalie matchup quality to predict home win probability.

    Factor weights:
        Save% differential (rolling avg5)    0.40
        GSAA differential                    0.30
        Head-to-head (same opponent)         0.15
        Shot-profile adjustment              0.15
    """

    def __init__(
        self,
        weight:           float = 1.2,    # higher default weight -- goalies matter most
        confidence_floor: float = 0.52,
        gl:               Optional[pd.DataFrame] = None,
    ):
        super().__init__("goalie_form", weight, confidence_floor)
        self._gl        = gl
        self._gl_loaded = gl is not None

    def _load_gl(self):
        if not self._gl_loaded:
            if GOALIE_PATH.exists():
                gl = pd.read_parquet(GOALIE_PATH)
                gl["game_date"] = pd.to_datetime(gl["game_date"])
                self._gl = gl
            else:
                self._gl = pd.DataFrame()
            self._gl_loaded = True

    def _get_h2h_save_pct(self, goalie_name: str, opp_team: str) -> Optional[float]:
        """
        Find average save% for this goalie specifically against opp_team.
        Uses last 3 such starts from historical parquet.
        Returns None if <2 starts found.
        """
        if self._gl is None or self._gl.empty:
            return None

        name_col = "goalie_name" if "goalie_name" in self._gl.columns else (
                   "player_name" if "player_name" in self._gl.columns else None)
        if name_col is None:
            return None

        last_name = goalie_name.strip().split()[-1] if goalie_name else ""
        if not last_name:
            return None

        opp_col = "opponent" if "opponent" in self._gl.columns else None
        if opp_col is None:
            return None

        mask = (
            self._gl[name_col].str.contains(last_name, case=False, na=False) &
            (self._gl[opp_col] == opp_team)
        )
        h2h = self._gl[mask]
        if len(h2h) < 2:
            return None

        h2h_recent = h2h.sort_values("game_date").tail(3)
        spp_col = "savePct" if "savePct" in h2h_recent.columns else (
                  "save_pctg" if "save_pctg" in h2h_recent.columns else None)
        if spp_col is None:
            return None

        vals = h2h_recent[spp_col].dropna()
        return float(vals.mean()) if len(vals) > 0 else None

    def _shot_profile_adjustment(self, goalie_side_ctx, opp_ctx) -> float:
        """
        Adjust goalie score based on opponent's shot volume.
        A goalie facing a high-volume attack has higher xGA exposure.
        Returns a 0-1 modifier centered at 0.5.
        """
        opp_shots = opp_ctx.sk_shots_avg5_mean
        if opp_shots is None:
            return 0.5

        # Above-average opponent shot volume = slight disadvantage for goalie
        shot_pressure = (opp_shots - LEAGUE_AVG_SOG_GP / 2) / (LEAGUE_AVG_SOG_GP / 2)
        # Invert: more shots against goalie = slightly worse for that goalie's team
        return clamp(0.5 - shot_pressure * 0.08)

    def _goalie_score(
        self,
        goalie_ctx,
        opp_team: str,
        opp_ctx,
        is_home: bool,
    ) -> tuple[float, dict]:
        """
        Compute a 0-1 quality score for one goalie.
        0.5 = league average.
        """
        factors = {}

        # Save% rolling avg5 (primary signal)
        spp5 = goalie_ctx.rolling_savePct_avg5
        if spp5 is None:
            spp5 = goalie_ctx.rolling_savePct_avg3
        if spp5 is None and goalie_ctx.svpct_live is not None:
            spp5 = float(goalie_ctx.svpct_live)
        factors["savePct_avg5"] = spp5

        # GSAA contribution
        gsaa = goalie_ctx.gsaa_season
        factors["gsaa"] = gsaa

        # H2H save%
        h2h_spp = self._get_h2h_save_pct(goalie_ctx.name, opp_team)
        factors["h2h_savePct"] = h2h_spp

        # Shot-profile adjustment
        shot_adj = self._shot_profile_adjustment(goalie_ctx, opp_ctx)
        factors["shot_adj"] = round(shot_adj, 4)

        # Combine
        components = []

        # Factor 1: Save% above/below league avg
        if spp5 is not None:
            spp_signal = logistic((spp5 - LEAGUE_AVG_SPP) * 100, scale=0.25)
            components.append((spp_signal, 0.40))

        # Factor 2: GSAA (per game proxy -- normalize by ~20 games)
        if gsaa is not None:
            gsaa_signal = logistic(gsaa / 15.0, scale=1.0)   # +/-15 GSAA = big spread
            components.append((gsaa_signal, 0.30))

        # Factor 3: H2H
        if h2h_spp is not None:
            h2h_signal = logistic((h2h_spp - LEAGUE_AVG_SPP) * 100, scale=0.25)
            components.append((h2h_signal, 0.15))

        # Factor 4: Shot profile
        components.append((shot_adj, 0.15))

        # Weighted average over available factors
        if not components:
            return 0.5, factors

        total_w = sum(w for _, w in components)
        score   = sum(s * w for s, w in components) / total_w
        return clamp(score), factors

    def analyze(self, game_context) -> AgentSignal:
        self._load_gl()

        home = game_context.home
        away = game_context.away
        h_goalie = home.goalie
        a_goalie = away.goalie

        if h_goalie is None and a_goalie is None:
            return self.neutral_signal("No goalie data available")

        # Use empty GoalieContext if one side is missing
        from game_context import GoalieContext as GC
        h_goalie = h_goalie or GC()
        a_goalie = a_goalie or GC()

        # ── Score each goalie ─────────────────────────────────────────────
        home_score, home_factors = self._goalie_score(h_goalie, away.team, away, is_home=True)
        away_score, away_factors = self._goalie_score(a_goalie, home.team, home, is_home=False)

        # ── Differential -> composite home-win probability ─────────────────
        # Map goalie quality spread to home advantage
        goalie_diff = home_score - away_score    # positive = home goalie better
        composite   = logistic(goalie_diff, scale=5.0)

        factors = {
            "home_goalie_score": round(home_score, 4),
            "away_goalie_score": round(away_score, 4),
            "goalie_diff":       round(goalie_diff, 4),
            "home_factors":      {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in home_factors.items()},
            "away_factors":      {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in away_factors.items()},
        }

        # ── Confidence: depends on confirmation status and data quality ───
        h_trusted = h_goalie.is_trusted
        a_trusted = a_goalie.is_trusted
        both_confirmed = h_trusted and a_trusted

        # Check whether each side has any rolling parquet data, regardless of
        # starter confirmation.  The parquet fallback (highest-TOI goalie for
        # the team) fires even when status == "unknown", so we give credit for
        # that signal rather than penalising it to below the confidence floor.
        h_has_rolling = (h_goalie.rolling_savePct_avg5 is not None or
                         h_goalie.rolling_savePct_avg3 is not None or
                         h_goalie.svpct_live is not None)
        a_has_rolling = (a_goalie.rolling_savePct_avg5 is not None or
                         a_goalie.rolling_savePct_avg3 is not None or
                         a_goalie.svpct_live is not None)

        # Raised from 0.55 -> 0.62: goalie matchup is the single strongest
        # predictor; base confidence should reflect that when data exists.
        base_conf = 0.62
        if not h_trusted:
            base_conf -= (UNCONFIRMED_PENALTY_HAS_DATA if h_has_rolling
                          else UNCONFIRMED_PENALTY_NO_DATA)
        if not a_trusted:
            base_conf -= (UNCONFIRMED_PENALTY_HAS_DATA if a_has_rolling
                          else UNCONFIRMED_PENALTY_NO_DATA)

        # Boost confidence when H2H data exists
        has_h2h = (home_factors.get("h2h_savePct") is not None or
                   away_factors.get("h2h_savePct") is not None)
        if has_h2h:
            base_conf += 0.05

        confidence = clamp(base_conf)

        direction = "home" if composite > 0.52 else ("away" if composite < 0.48 else "neutral")

        # ── Reasoning ─────────────────────────────────────────────────────
        h_spp = home_factors.get("savePct_avg5")
        a_spp = away_factors.get("savePct_avg5")
        h_spp_str = f"{h_spp:.3f}" if h_spp else "N/A"
        a_spp_str = f"{a_spp:.3f}" if a_spp else "N/A"
        conf_flag = "" if both_confirmed else " unconfirmed"
        reasoning = (
            f"{h_goalie.name}({h_goalie.status}) SV%={h_spp_str} "
            f"vs {a_goalie.name}({a_goalie.status}) SV%={a_spp_str} "
            f"| diff={goalie_diff:+.3f}{conf_flag}"
        )

        return AgentSignal(
            agent_id       = self.agent_id,
            pick_direction = direction,
            raw_score      = round(composite, 4),
            confidence     = round(confidence, 4),
            reasoning      = reasoning,
            factors        = factors,
        )
