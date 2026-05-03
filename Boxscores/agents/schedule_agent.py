"""
schedule_agent.py
-----------------
ScheduleAgent: situational / schedule-context analysis.

Signals derived from:
  1. Rest day delta — home rest days minus away rest days
  2. Back-to-back flag — which team(s) are on a B2B (severe fatigue penalty)
  3. Travel distance proxy — straight-line distance between team cities
  4. Altitude factor — Colorado (COL) has meaningful altitude advantage

None of these require external APIs — all computed from GameContext fields
and the built-in city coordinate table below.
"""

import math
from typing import Optional

from .agent_base import NHLAgent, AgentSignal, logistic, clamp

# ── NHL team city coordinates (lat, lon) ─────────────────────────────────────
# Used to compute travel distance as a fatigue proxy.
TEAM_COORDS: dict[str, tuple[float, float]] = {
    "ANA": (33.84, -117.92),  "BOS": (42.37, -71.06),   "BUF": (42.89, -78.88),
    "CGY": (51.04, -114.07),  "CAR": (35.80, -78.72),   "CHI": (41.88, -87.67),
    "COL": (39.75, -105.00),  "CBJ": (39.96, -82.99),   "DAL": (32.79, -96.81),
    "DET": (42.33, -83.05),   "EDM": (53.55, -113.50),  "FLA": (26.16, -80.32),
    "LAK": (34.04, -118.27),  "MIN": (44.95, -93.10),   "MTL": (45.50, -73.62),
    "NSH": (36.16, -86.78),   "NJD": (40.73, -74.17),   "NYI": (40.72, -73.59),
    "NYR": (40.75, -73.99),   "OTT": (45.30, -75.93),   "PHI": (39.90, -75.17),
    "PIT": (40.44, -80.00),   "SJS": (37.33, -121.90),  "SEA": (47.62, -122.35),
    "STL": (38.63, -90.20),   "TBL": (27.94, -82.45),   "TOR": (43.64, -79.38),
    "UTA": (40.77, -111.90),  "VAN": (49.28, -123.12),  "VGK": (36.10, -115.17),
    "WSH": (38.90, -77.02),   "WPG": (49.89, -97.14),
}

# ── Altitude: teams that play above 3000 ft (meaningful aerobic impact on visitors)
HIGH_ALTITUDE_TEAMS = {
    "COL": 5280,   # Denver — Mile High
    "UTA": 4330,   # Salt Lake City
    "CGY": 3438,   # Calgary
    "EDM": 2200,   # Edmonton (marginal, included for completeness)
}
ALTITUDE_THRESHOLD_FT = 3000   # below this, no meaningful effect
ALTITUDE_HOME_BOOST   = 0.04   # raw_score boost for home team at altitude vs sea-level visitor


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8   # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def travel_distance(from_team: str, to_team: str) -> Optional[float]:
    """Miles of travel for away team (from their home city to opponent city)."""
    a = TEAM_COORDS.get(from_team)
    b = TEAM_COORDS.get(to_team)
    if a is None or b is None:
        return None
    return haversine_miles(a[0], a[1], b[0], b[1])


class ScheduleAgent(NHLAgent):
    """
    Analyzes schedule context (rest, travel, altitude) for home win probability.

    Factor weights:
        Rest day differential    0.45
        Back-to-back penalty     0.35
        Travel distance          0.12
        Altitude factor          0.08

    Note: This agent returns relatively low base confidence because schedule
    factors are secondary to form — but they can tip close matchups.
    """

    def __init__(self, weight: float = 0.8, confidence_floor: float = 0.52):
        super().__init__("schedule", weight, confidence_floor)

    def analyze(self, game_context) -> AgentSignal:
        home = game_context.home
        away = game_context.away
        factors = {}

        # ── Factor 1: Rest day differential ──────────────────────────────
        h_rest = home.rest_days
        a_rest = away.rest_days
        rest_score = 0.5

        if h_rest is not None and a_rest is not None:
            rest_diff = h_rest - a_rest   # positive = home more rested
            factors["rest_diff"] = rest_diff
            factors["home_rest"] = h_rest
            factors["away_rest"] = a_rest
            # Each day of rest advantage is worth ~3-4% win probability
            rest_score = logistic(rest_diff, scale=0.45)
        elif h_rest is not None:
            # Only home data: neutral if away unknown
            factors["home_rest_only"] = h_rest
        elif a_rest is not None:
            factors["away_rest_only"] = a_rest

        factors["rest_score"] = round(rest_score, 4)

        # ── Factor 2: Back-to-back penalty ───────────────────────────────
        h_b2b = bool(home.is_b2b)
        a_b2b = bool(away.is_b2b)
        factors["home_b2b"] = h_b2b
        factors["away_b2b"] = a_b2b

        # B2B teams historically win ~43% of those games vs ~50% normally
        # So a B2B penalty of ~7% win probability
        b2b_adjustment = 0.0
        if h_b2b and not a_b2b:
            b2b_adjustment = -0.07   # home is tired, away is rested
        elif a_b2b and not h_b2b:
            b2b_adjustment = +0.07   # away is tired, home is rested
        # Both B2B: roughly cancel out (slight home advantage remains)
        elif h_b2b and a_b2b:
            b2b_adjustment = +0.02   # home ice advantage still exists

        b2b_score = clamp(0.5 + b2b_adjustment)
        factors["b2b_adjustment"] = b2b_adjustment
        factors["b2b_score"]      = round(b2b_score, 4)

        # ── Factor 3: Travel distance (away team fatigue) ─────────────────
        travel_score = 0.5
        dist = travel_distance(away.team, home.team)   # away team travels TO home arena
        if dist is not None:
            factors["travel_miles"] = round(dist, 0)
            # 2500+ miles is a transcontinental trip (e.g. Boston->Vegas)
            # 500 miles or less is regional (e.g. Philly->Pittsburgh)
            if dist > 2000:
                travel_score = 0.54   # slight home advantage from long away travel
            elif dist > 1000:
                travel_score = 0.52
            else:
                travel_score = 0.50
        factors["travel_score"] = round(travel_score, 4)

        # ── Factor 4: Altitude factor ─────────────────────────────────────
        altitude_score = 0.5
        home_altitude  = HIGH_ALTITUDE_TEAMS.get(home.team, 0)
        away_altitude  = HIGH_ALTITUDE_TEAMS.get(away.team, 0)

        if home_altitude >= ALTITUDE_THRESHOLD_FT and away_altitude < ALTITUDE_THRESHOLD_FT:
            # Home team plays at altitude, visitor coming from near sea level
            altitude_score = 0.5 + ALTITUDE_HOME_BOOST
            factors["altitude_boost"] = f"{home.team} home altitude advantage ({home_altitude}ft)"
        elif away_altitude >= ALTITUDE_THRESHOLD_FT and home_altitude < ALTITUDE_THRESHOLD_FT:
            # Away team is altitude-acclimated, visiting a low-altitude arena
            # Minimal effect -- don't adjust
            pass

        factors["altitude_score"] = round(altitude_score, 4)

        # ── Combine ───────────────────────────────────────────────────────
        composite = (
            0.45 * rest_score    +
            0.35 * b2b_score     +
            0.12 * travel_score  +
            0.08 * altitude_score
        )

        # ── Confidence ────────────────────────────────────────────────────
        # Rest-day data comes from the NHL API (always available) and travel
        # distance is computed from the hardcoded city-coordinate table (also
        # always available).  The old base of 0.40 meant rest + travel only
        # reached 0.48 -- perpetually below the 0.52 confidence floor on normal
        # nights.  Raised to 0.50 so the agent fires on every game where
        # rest data exists (which is virtually always).
        has_rest  = (h_rest is not None or a_rest is not None)
        has_b2b   = (h_b2b or a_b2b)
        has_dist  = dist is not None

        # Strong signal only when there's a clear B2B mismatch or big rest gap
        big_rest_diff = (h_rest is not None and a_rest is not None and
                         abs(h_rest - a_rest) >= 2)
        b2b_mismatch  = h_b2b != a_b2b

        base_conf = 0.50          # raised from 0.40 -- rest/travel data always present
        if has_rest:      base_conf += 0.05
        if b2b_mismatch:  base_conf += 0.12
        if big_rest_diff: base_conf += 0.08
        if has_dist:      base_conf += 0.03

        confidence = clamp(base_conf)

        direction = "home" if composite > 0.52 else ("away" if composite < 0.48 else "neutral")

        # ── Reasoning ─────────────────────────────────────────────────────
        parts = []
        if h_rest is not None and a_rest is not None:
            parts.append(f"rest {h_rest}d/{a_rest}d (diff {h_rest-a_rest:+d})")
        if h_b2b:
            parts.append(f"{home.team} B2B")
        if a_b2b:
            parts.append(f"{away.team} B2B")
        if dist:
            parts.append(f"travel {dist:.0f}mi")
        if altitude_score != 0.5:
            parts.append(factors.get("altitude_boost", "altitude"))

        reasoning = " | ".join(parts) if parts else "No schedule edge detected"

        return AgentSignal(
            agent_id       = self.agent_id,
            pick_direction = direction,
            raw_score      = round(composite, 4),
            confidence     = round(confidence, 4),
            reasoning      = reasoning,
            factors        = factors,
        )
