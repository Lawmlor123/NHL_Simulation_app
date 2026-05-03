"""
sentiment_agent.py
------------------
SentimentAgent: news and headline keyword analysis.

IMPORTANT: This agent returns a MULTIPLIER, not a pick direction.
  - multiplier > 1.0 → amplify other agents' signals toward home
  - multiplier < 1.0 → amplify other agents' signals toward away
  - multiplier = 1.0 → neutral, no sentiment edge

The multiplier is applied by the meta-learner (Module 3) to adjust
the ensemble's composite score — it does not produce a standalone pick.

Signal categories scored:
  1. Injury/availability news (highest impact — key player out)
  2. Trade/roster disruption (medium impact — lineup uncertainty)
  3. Coach/system changes (low impact — scheme instability)
  4. Momentum/narrative keywords (lowest impact — media sentiment)

Current sources (available from GameContext):
  - DailyFaceoff goalie news (home_news, away_news via Module 1)

Future hook:
  - Twitter/X search API integration (stubbed as get_tweet_headlines())
  - Beat reporter RSS feeds (stubbed as get_rss_headlines())
  The scoring logic is source-agnostic — just feed it more strings.
"""

import re
from typing import Optional

from .agent_base import NHLAgent, AgentSignal, clamp

# ── Keyword dictionaries ──────────────────────────────────────────────────────
# Each entry: (regex_pattern, score_delta, description)
# score_delta: positive = good for team, negative = bad for team
# Applied to EACH team separately, then compared home vs away.

KEYWORD_RULES: list[tuple[str, float, str]] = [
    # ── Injury / absence (large negative) ────────────────────────────────
    (r"\bruled?\s+out\b",           -0.25, "ruled out"),
    (r"\bltir?\b",                  -0.20, "LTIR"),
    (r"\bsuspended\b",              -0.18, "suspension"),
    (r"\bday[\s-]to[\s-]day\b",     -0.12, "day-to-day"),
    (r"\binjur(?:ed|y)\b",          -0.10, "injury"),
    (r"\bsidelined\b",              -0.10, "sidelined"),
    (r"\bscratch(?:ed)?\b",         -0.08, "scratched"),
    (r"\bquestion(?:able)?\b",      -0.07, "questionable"),
    (r"\blimited\b",                -0.05, "limited practice"),
    (r"\bconcussion\b",             -0.15, "concussion"),
    (r"\bupper[\s-]body\b",         -0.08, "upper-body"),
    (r"\blower[\s-]body\b",         -0.08, "lower-body"),
    # ── Return from injury (positive) ─────────────────────────────────────
    (r"\breturns?\s+(?:to|from)\b", +0.15, "player returns"),
    (r"\bcleared\b",                +0.10, "cleared to play"),
    (r"\bback\s+in\s+lineup\b",     +0.12, "back in lineup"),
    (r"\bfull\s+practice\b",        +0.08, "full practice"),
    (r"\bno\s+(?:injury|issue)\b",  +0.06, "no injury concern"),
    (r"\bactivated\b",              +0.10, "activated from IR"),
    # ── Trade / roster disruption ─────────────────────────────────────────
    (r"\btrade(?:d)?\b",            -0.06, "trade/trade rumor"),
    (r"\brumor\b",                  -0.04, "trade rumor"),
    (r"\breleased\b",               -0.05, "player released"),
    (r"\bwaived\b",                 -0.05, "player waived"),
    (r"\bemergency\s+recall\b",     -0.08, "emergency call-up"),
    (r"\bcall(?:ed)?[\s-]up\b",     -0.04, "AHL call-up"),
    # ── Coaching / system changes ─────────────────────────────────────────
    (r"\bfired\b",                  -0.10, "coach fired"),
    (r"\binterim\b",                -0.08, "interim coach"),
    (r"\bcoach(?:ing)?\s+change\b", -0.07, "coaching change"),
    # ── Positive momentum ─────────────────────────────────────────────────
    (r"\bfull\s+lineup\b",          +0.06, "full lineup available"),
    (r"\bwell[\s-]rested\b",        +0.04, "well rested"),
    (r"\bstreak\b",                 +0.03, "win streak mention"),
    (r"\bconfident\b",              +0.02, "confidence reported"),
    # ── Goalie-specific (relevant since we have goalie news) ──────────────
    (r"\bstarting\s+goalie\s+(?:unconfirmed|tbd|unclear)\b",
                                    -0.12, "goalie unconfirmed"),
    (r"\bconfirmed\s+(?:starter|goalie)\b",
                                    +0.06, "goalie confirmed"),
    (r"\bback[\s-]up\s+(?:starts|starting)\b",
                                    -0.10, "backup starts"),
]


def _score_headlines(headlines: list[str]) -> tuple[float, list[str]]:
    """
    Score a list of headlines using the keyword rules.
    Returns (net_score, matched_rule_descriptions).
    net_score is unbounded; caller normalizes to a multiplier.
    """
    if not headlines:
        return 0.0, []

    text    = " ".join(str(h) for h in headlines).lower()
    total   = 0.0
    matched = []

    for pattern, delta, desc in KEYWORD_RULES:
        if re.search(pattern, text):
            total += delta
            matched.append(f"{desc}({delta:+.2f})")

    return total, matched


def _score_to_multiplier(score: float, scale: float = 0.15) -> float:
    """
    Convert a net keyword score to a multiplier in range [0.70, 1.30].
    A score of 0.0 → 1.0 (neutral).
    Positive score → > 1.0 (team looks good).
    Negative score → < 1.0 (team looks bad).
    """
    multiplier = 1.0 + score * scale
    return clamp(multiplier, 0.70, 1.30)


class SentimentAgent(NHLAgent):
    """
    Scores news headlines for each team and returns a sentiment multiplier signal.

    IMPORTANT: raw_score here means:
        > 0.5 → home team has better sentiment (amplify toward home)
        < 0.5 → away team has better sentiment (amplify toward away)
        = 0.5 → neutral

    The 'pick_direction' field is always "neutral" unless the sentiment gap
    crosses a meaningful threshold — sentiment alone rarely drives a pick.

    The `multiplier` key in signal.factors is the primary output
    for the meta-learner to consume.
    """

    def __init__(self, weight: float = 0.5, confidence_floor: float = 0.52):
        # Lower default weight — sentiment is a modifier, not a primary signal
        super().__init__("sentiment", weight, confidence_floor)

    # ── Future hooks — stub these for later integration ───────────────────
    @staticmethod
    def get_tweet_headlines(team: str, n: int = 10) -> list[str]:
        """
        STUB: Return recent tweets mentioning the team.
        Implement with Twitter/X API or third-party sentiment feed.
        """
        return []

    @staticmethod
    def get_rss_headlines(team: str) -> list[str]:
        """
        STUB: Return beat reporter RSS headlines for the team.
        Implement with feedparser or similar.
        """
        return []

    def _collect_headlines(self, team_ctx) -> list[str]:
        """Aggregate all available headline sources for a team."""
        headlines = list(team_ctx.news_headlines)   # from DailyFaceoff (Module 1)
        headlines += self.get_tweet_headlines(team_ctx.team)
        headlines += self.get_rss_headlines(team_ctx.team)
        return headlines

    def analyze(self, game_context) -> AgentSignal:
        home = game_context.home
        away = game_context.away

        h_headlines = self._collect_headlines(home)
        a_headlines = self._collect_headlines(away)

        h_score, h_matched = _score_headlines(h_headlines)
        a_score, a_matched = _score_headlines(a_headlines)

        h_mult = _score_to_multiplier(h_score)
        a_mult = _score_to_multiplier(a_score)

        # Relative multiplier: home vs away
        # > 1.0 = home team has better news context
        relative_mult = h_mult / a_mult if a_mult > 0 else 1.0

        # Convert to 0-1 raw_score (0.5 = even)
        # relative_mult of 1.15 → +0.08 raw_score
        raw_score = clamp(0.5 + (relative_mult - 1.0) * 0.4)

        factors = {
            "home_sentiment_score": round(h_score, 3),
            "away_sentiment_score": round(a_score, 3),
            "home_multiplier":      round(h_mult, 3),
            "away_multiplier":      round(a_mult, 3),
            "relative_multiplier":  round(relative_mult, 3),
            "home_matched_rules":   h_matched,
            "away_matched_rules":   a_matched,
            # Primary output for meta-learner
            "multiplier":           round(relative_mult, 3),
        }

        # Confidence: only meaningful when actual headlines found
        has_h = bool(h_headlines)
        has_a = bool(a_headlines)
        significant = abs(h_score - a_score) > 0.10

        confidence = clamp(
            (0.35 if has_h else 0.0) +
            (0.35 if has_a else 0.0) +
            (0.20 if significant else 0.0)
        )

        # Direction: usually neutral — sentiment alone rarely picks a winner
        # Only override to directional if there's a clear one-sided signal
        score_gap = abs(h_score - a_score)
        if score_gap > 0.20:
            direction = "home" if h_score > a_score else "away"
        else:
            direction = "neutral"

        # ── Reasoning ─────────────────────────────────────────────────────
        if not h_headlines and not a_headlines:
            reasoning = "No news headlines found — sentiment neutral"
        else:
            h_str = f"{home.team} sentiment={h_score:+.2f}"
            a_str = f"{away.team} sentiment={a_score:+.2f}"
            h_kw  = (", ".join(h_matched[:3]) if h_matched else "no flags")
            a_kw  = (", ".join(a_matched[:3]) if a_matched else "no flags")
            reasoning = (f"{h_str} [{h_kw}] | {a_str} [{a_kw}] "
                         f"| multiplier={relative_mult:.3f}")

        return AgentSignal(
            agent_id       = self.agent_id,
            pick_direction = direction,
            raw_score      = round(raw_score, 4),
            confidence     = round(confidence, 4),
            reasoning      = reasoning,
            factors        = factors,
        )
