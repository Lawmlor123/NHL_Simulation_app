"""
module1_ingest.py
-----------------
Module 1: Data Ingestion Layer for the NHL Prediction MAS.

Entry point:
    contexts = build_today_contexts(date)   # returns list[GameContext]

Data sources (all free, no API keys):
    1. NHL API  — schedule, play-by-play (api-web.nhle.com)
    2. Parquet  — skater_features.parquet, goalie_features.parquet (existing pipeline)
    3. DailyFaceoff — goalie starters, moneylines, news headlines (existing scraper)

Advanced stats derived from play-by-play:
    - CF%  (Corsi For %) — all shot attempts at 5v5
    - FF%  (Fenwick For %) — unblocked shot attempts at 5v5
    - xG   (approx) — shot-distance + shot-type model, no external data needed
    - GSAA — actual saves vs expected saves based on xG against

Play-by-play results are cached to pbp_cache/ so we don't re-pull on every run.
"""

import json
import math
import sys
import time
import requests
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Path setup ──────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
PBP_CACHE = BASE / "pbp_cache"
ODDS_LOG  = BASE / "odds_history.csv"

def _ensure_dirs():
    """Create required directories lazily (called at runtime, not import time)."""
    try:
        PBP_CACHE.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass   # path may not exist in test/sandbox environments

sys.path.insert(0, str(BASE))
from game_context import (
    GameContext, TeamContext, GoalieContext, OddsContext
)

# ── Try loading the existing goalie scraper ──────────────────────────────────
try:
    from scrape_goalies_v2 import scrape_starting_goalies, get_goalie_dict
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

try:
    from manual_overrides import GOALIE_OVERRIDES
except ImportError:
    GOALIE_OVERRIDES = {}

DELAY = 0.25   # polite delay between NHL API calls

# ── xG shot-type multipliers ──────────────────────────────────────────────────
# Derived from public xG research (MoneyPuck / Manny's model references).
# Wrist shot = baseline 1.0
XG_SHOT_TYPE = {
    "wrist":      1.00,
    "snap":       1.10,
    "slap":       0.75,
    "backhand":   0.70,
    "tip-in":     1.60,
    "deflected":  1.50,
    "wrap-around":0.55,
}
XG_BASE_RATE  = 0.082   # NHL average shot-on-goal conversion ~8.2%
XG_DECAY      = 28.0    # distance decay constant (feet)
NET_X         = 89.0    # NHL net x-coordinate (attacking zone)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — NHL API helpers
# ═══════════════════════════════════════════════════════════════════════════════

def nhl_get(url: str, retries: int = 3) -> Optional[dict]:
    """GET wrapper with retry logic for the NHL API."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None   # game not found — don't retry
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1.0)
    return None


def get_schedule(target_date: date) -> list[dict]:
    """Return list of regular-season games for target_date."""
    url  = f"https://api-web.nhle.com/v1/schedule/{target_date}"
    data = nhl_get(url)
    if not data:
        return []

    games = []
    for week in data.get("gameWeek", []):
        if week["date"] == str(target_date):
            for g in week.get("games", []):
                if g.get("gameType") in (2, 3):   # regular season + playoffs
                    games.append({
                        "game_id":   g["id"],
                        "home_team": g["homeTeam"]["abbrev"],
                        "away_team": g["awayTeam"]["abbrev"],
                        "start_time": g.get("startTimeUTC", "TBD"),
                        "season":    str(g.get("season", "")),
                        "game_type": g.get("gameType", 2),
                    })
    return games


def get_team_rest(teams: list[str], as_of: date, lookback: int = 8) -> dict:
    """
    For each team, find days since last completed game.
    Reuses the same logic already in predict_today.py.
    Returns {team: {rest_days, is_b2b, last_game}}
    """
    result = {t: {"rest_days": None, "is_b2b": False, "last_game": None} for t in teams}

    for offset in range(1, lookback + 1):
        check = as_of - timedelta(days=offset)
        data  = nhl_get(f"https://api-web.nhle.com/v1/score/{check}")
        if not data:
            continue
        for g in data.get("games", []):
            if g.get("gameState") not in ("FINAL", "OFF"):
                continue
            for team in [g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]]:
                if team in result and result[team]["last_game"] is None:
                    rest = (as_of - check).days
                    result[team] = {
                        "rest_days": rest,
                        "is_b2b":    rest == 1,
                        "last_game": str(check),
                    }
        if all(v["last_game"] is not None for v in result.values()):
            break

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Play-by-Play: CF%, FF%, xG
# ═══════════════════════════════════════════════════════════════════════════════

def xg_from_shot(x: Optional[float], y: Optional[float],
                 shot_type: str = "wrist", is_goal: bool = False) -> float:
    """
    Approximate xG for a single shot using distance decay + shot-type multiplier.

    Model:
        xG = base_rate * shot_type_mult * exp(-distance / decay)

    Coordinates: NHL rink is 200ft long (-100 to +100 x-axis).
    Attacking team shoots toward x=+89 (home) or x=-89 (away).
    We use absolute distance from whichever net the shot targets.
    """
    if x is None or y is None:
        return XG_BASE_RATE   # fallback: league average

    # Distance from net center
    dist = math.sqrt((NET_X - abs(x)) ** 2 + (y ** 2))
    dist = max(dist, 1.0)   # avoid div/zero on tap-ins

    multiplier = XG_SHOT_TYPE.get(shot_type.lower().strip(), 1.0)
    xg = XG_BASE_RATE * multiplier * math.exp(-dist / XG_DECAY)
    return round(min(xg, 0.95), 6)   # cap at 95%


def parse_pbp(pbp_data: dict, home_team: str, away_team: str) -> dict:
    """
    Parse a play-by-play response into team-level CF, FF, xG tallies at 5v5.

    Situation codes: first two digits = home skaters, next two = away skaters.
    5v5 = "1515" prefix.

    Returns:
        {
          home_team: {cf, ff, xgf, xga, shots_on_goal},
          away_team: {cf, ff, xgf, xga, shots_on_goal},
        }
    """
    # Shot event type codes in the NHL API play-by-play
    SHOT_EVENTS   = {505, 506, 507, 508}   # shot-on-goal, missed, blocked, goal
    SOG_EVENTS    = {505, 506}             # shot-on-goal + goal (unblocked, on net)
    BLOCKED_EVENTS = {507}
    GOAL_EVENT    = {505}                  # goals are coded as shots that went in
    # NOTE: In the NHL v1 PBP API:
    #   505 = shot-on-goal (includes goals)
    #   506 = missed shot
    #   507 = blocked shot
    #   508 = goal (sometimes separate; handle both)

    # Revised: the actual NHL API uses typeDescKey strings, not numeric codes
    # Let's use both approaches for safety
    SHOT_KEYS   = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
    SOG_KEYS    = {"shot-on-goal", "goal"}
    BLOCKED_KEYS = {"blocked-shot"}

    stats = {
        home_team: {"cf": 0, "ff": 0, "xgf": 0.0, "xga": 0.0, "sog": 0},
        away_team: {"cf": 0, "ff": 0, "xgf": 0.0, "xga": 0.0, "sog": 0},
    }

    plays = pbp_data.get("plays", [])

    for play in plays:
        type_key  = play.get("typeDescKey", "")
        situation = play.get("situationCode", "")
        details   = play.get("details", {}) or {}

        # 5v5 filter: situationCode "1515" = 5v5 even strength
        if not situation.startswith("1515"):
            continue

        if type_key not in SHOT_KEYS:
            continue

        # Shooting team
        event_owner = details.get("eventOwnerTeamId")
        # Map team ID → abbrev using pbp header
        home_id = pbp_data.get("homeTeam", {}).get("id")
        away_id = pbp_data.get("awayTeam", {}).get("id")

        if event_owner == home_id:
            shooting_team = home_team
            defending_team = away_team
        elif event_owner == away_id:
            shooting_team = away_team
            defending_team = home_team
        else:
            continue   # can't attribute

        # Corsi = all shot attempts (SOG + missed + blocked)
        stats[shooting_team]["cf"] += 1

        # Fenwick = unblocked (SOG + missed, exclude blocked)
        is_blocked = type_key in BLOCKED_KEYS
        if not is_blocked:
            stats[shooting_team]["ff"] += 1

        # xG (only for unblocked shots on goal)
        if type_key in SOG_KEYS:
            stats[shooting_team]["sog"] += 1
            x   = details.get("xCoord")
            y   = details.get("yCoord")
            sht = details.get("shotType", "wrist") or "wrist"
            xg  = xg_from_shot(x, y, sht)
            stats[shooting_team]["xgf"]  += xg
            stats[defending_team]["xga"] += xg

    return stats


def fetch_pbp_cached(game_id: int) -> Optional[dict]:
    """
    Fetch play-by-play for a completed game. Caches to pbp_cache/{game_id}.json
    so repeat runs don't hit the API again.
    """
    _ensure_dirs()
    cache_file = PBP_CACHE / f"{game_id}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url  = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    data = nhl_get(url)
    if data:
        with open(cache_file, "w") as f:
            json.dump(data, f)
        time.sleep(DELAY)
    return data


def get_recent_game_ids(team: str, as_of_date: date,
                        n_games: int = 5, lookback_days: int = 40) -> list[int]:
    """
    Find the last n completed game IDs for a team by scanning recent schedule.
    """
    game_ids = []
    for offset in range(1, lookback_days + 1):
        check = as_of_date - timedelta(days=offset)
        data  = nhl_get(f"https://api-web.nhle.com/v1/score/{check}")
        if not data:
            continue
        for g in data.get("games", []):
            if g.get("gameState") not in ("FINAL", "OFF"):
                continue
            home = g["homeTeam"]["abbrev"]
            away = g["awayTeam"]["abbrev"]
            if team in (home, away):
                game_ids.append(g["id"])
        if len(game_ids) >= n_games:
            break
        time.sleep(DELAY)
    return game_ids[:n_games]


def get_team_advanced_stats(team: str, as_of_date: date,
                            n_games: int = 5) -> dict:
    """
    Compute rolling advanced stats for a team over their last n_games
    using the NHL play-by-play API.

    Returns:
        {cf_pct, ff_pct, xg_for, xg_against, xg_pct, sh_attempts_per_game}
    """
    print(f"    [{team}] Pulling play-by-play for last {n_games} games ...")
    game_ids = get_recent_game_ids(team, as_of_date, n_games)

    if not game_ids:
        print(f"    [{team}] No recent games found.")
        return {}

    totals = {"cf": 0, "ff": 0, "xgf": 0.0, "xga": 0.0, "sog": 0,
              "opp_cf": 0, "opp_ff": 0}
    games_found = 0

    for gid in game_ids:
        pbp = fetch_pbp_cached(gid)
        if not pbp:
            continue

        # Identify opponent
        home_abbrev = pbp.get("homeTeam", {}).get("abbrev", "")
        away_abbrev = pbp.get("awayTeam", {}).get("abbrev", "")
        if team not in (home_abbrev, away_abbrev):
            continue
        opp = away_abbrev if team == home_abbrev else home_abbrev

        game_stats = parse_pbp(pbp, home_abbrev, away_abbrev)
        t_stats   = game_stats.get(team, {})
        opp_stats = game_stats.get(opp, {})

        totals["cf"]     += t_stats.get("cf", 0)
        totals["ff"]     += t_stats.get("ff", 0)
        totals["xgf"]    += t_stats.get("xgf", 0.0)
        totals["xga"]    += t_stats.get("xga", 0.0)
        totals["sog"]    += t_stats.get("sog", 0)
        totals["opp_cf"] += opp_stats.get("cf", 0)
        totals["opp_ff"] += opp_stats.get("ff", 0)
        games_found += 1

    if games_found == 0:
        return {}

    total_cf = totals["cf"] + totals["opp_cf"]
    total_ff = totals["ff"] + totals["opp_ff"]
    total_xg = totals["xgf"] + totals["xga"]

    return {
        "cf_pct_last5":      round(totals["cf"] / total_cf, 4) if total_cf > 0 else None,
        "ff_pct_last5":      round(totals["ff"] / total_ff, 4) if total_ff > 0 else None,
        "xg_for_last5":      round(totals["xgf"] / games_found, 4),
        "xg_against_last5":  round(totals["xga"] / games_found, 4),
        "xg_pct_last5":      round(totals["xgf"] / total_xg, 4) if total_xg > 0 else None,
        "sh_attempts_last5": round(totals["cf"] / games_found, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Parquet feature helpers (existing pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_parquets():
    """Load skater and goalie feature parquets once, return as tuple."""
    sk_path = BASE / "skater_features.parquet"
    gl_path = BASE / "goalie_features.parquet"

    sk = pd.read_parquet(sk_path) if sk_path.exists() else pd.DataFrame()
    gl = pd.read_parquet(gl_path) if gl_path.exists() else pd.DataFrame()

    if not sk.empty:
        sk["game_date"] = pd.to_datetime(sk["game_date"])
    if not gl.empty:
        gl["game_date"] = pd.to_datetime(gl["game_date"])

    return sk, gl


def get_skater_features(team: str, sk: pd.DataFrame) -> dict:
    """
    Pull latest rolling skater features for a team.
    Mirrors the aggregation logic in predict_today.py.
    """
    if sk.empty:
        return {}

    rolling_cols = [c for c in sk.columns if any(
        c.endswith(s) for s in ["_avg3", "_avg5", "_avg10", "_avg20", "_season_avg"]
    )]
    sum_cols = [c for c in rolling_cols if any(
        stat in c for stat in ["goals", "points", "shots", "assists", "pp_goals"]
    )]

    team_sk = sk[sk["team"] == team]
    if team_sk.empty:
        return {}

    latest = team_sk.sort_values(["player_id", "game_date"]).groupby("player_id").last()

    out = {}
    for col in rolling_cols:
        if col in latest.columns:
            out[f"sk_{col}_mean"] = round(float(latest[col].mean()), 4)
    for col in sum_cols:
        if col in latest.columns:
            out[f"sk_{col}_sum"] = round(float(latest[col].sum()), 4)

    return out


def get_goalie_features(team: str, goalie_name: str,
                        gl: pd.DataFrame) -> tuple[dict, Optional[str]]:
    """
    Match scraped goalie name to parquet rolling features.
    Returns (rolling_feature_dict, matched_name).
    Falls back to highest-TOI goalie for team if name match fails.
    """
    if gl.empty:
        return {}, None

    starters = gl[gl.get("is_starter", pd.Series(dtype=int)) == 1] if "is_starter" in gl.columns else gl
    team_gl  = starters[starters["team"] == team] if "team" in starters.columns else pd.DataFrame()

    matched = None
    name_col = "goalie_name" if "goalie_name" in team_gl.columns else (
               "player_name" if "player_name" in team_gl.columns else None)

    # Try last-name match
    if goalie_name and goalie_name != "Unknown" and name_col:
        last = goalie_name.strip().split()[-1]
        hits = team_gl[team_gl[name_col].str.contains(last, case=False, na=False)]
        if not hits.empty:
            matched = hits.sort_values("game_date").iloc[-1]

    # Fallback: highest TOI
    if matched is None:
        fallback = gl[gl["team"] == team] if "team" in gl.columns else pd.DataFrame()
        if "toi_min" in fallback.columns and not fallback.empty:
            matched = fallback.sort_values("toi_min", ascending=False).iloc[0]

    if matched is None:
        return {}, None

    goalie_rolling = [c for c in gl.columns if c.startswith("g_")]
    feats = {}
    for col in goalie_rolling:
        if col in matched.index:
            val = matched[col]
            if pd.notna(val):
                feats[col] = round(float(val), 4)

    matched_name_col = "goalie_name" if "goalie_name" in matched.index else (
                       "player_name" if "player_name" in matched.index else None)
    matched_name = matched.get(matched_name_col) if matched_name_col else None

    return feats, matched_name


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Odds persistence (line movement foundation)
# ═══════════════════════════════════════════════════════════════════════════════

def persist_odds(games: list[dict], scraped_games: list[dict],
                 game_date: date) -> None:
    """
    Append scraped odds snapshot to odds_history.csv.
    Each row = one timestamp snapshot per game.
    Line movement can be computed later by comparing first vs latest per game_date.
    """
    rows = []
    odds_lookup = {
        (g.get("away_team"), g.get("home_team")): g
        for g in (scraped_games or [])
    }

    for game in games:
        key = (game["away_team"], game["home_team"])
        sg  = odds_lookup.get(key, {})
        rows.append({
            "snapshot_at":  datetime.utcnow().isoformat(),
            "game_date":    str(game_date),
            "game_id":      game["game_id"],
            "home_team":    game["home_team"],
            "away_team":    game["away_team"],
            "home_ml":      sg.get("home_moneyline"),
            "away_ml":      sg.get("away_moneyline"),
            "point_spread": sg.get("point_spread"),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    if ODDS_LOG.exists():
        existing = pd.read_csv(ODDS_LOG)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(ODDS_LOG, index=False)


def get_line_movement(game_id: int, game_date: date) -> dict:
    """
    Compute line movement for a game from odds_history.csv.
    Returns opening line and movement vs current if ≥2 snapshots exist.
    """
    if not ODDS_LOG.exists():
        return {}

    hist = pd.read_csv(ODDS_LOG)
    game_hist = hist[
        (hist["game_id"] == game_id) &
        (hist["game_date"] == str(game_date))
    ].sort_values("snapshot_at")

    if len(game_hist) < 2:
        return {}

    opening = game_hist.iloc[0]
    current = game_hist.iloc[-1]

    try:
        open_ml = float(opening["home_ml"])
        curr_ml = float(current["home_ml"])
        return {
            "opening_home_ml": open_ml,
            "opening_away_ml": float(opening["away_ml"]) if pd.notna(opening["away_ml"]) else None,
            "line_movement":   round(curr_ml - open_ml, 1),
        }
    except (ValueError, TypeError):
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Implied probability helper
# ═══════════════════════════════════════════════════════════════════════════════

def american_to_implied(ml) -> Optional[float]:
    try:
        ml = float(ml)
    except (ValueError, TypeError):
        return None
    if ml > 0:
        return round(100 / (ml + 100), 4)
    elif ml < 0:
        return round(abs(ml) / (abs(ml) + 100), 4)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GSAA calculation
# ═══════════════════════════════════════════════════════════════════════════════

LEAGUE_AVG_SAVE_PCT = 0.906   # NHL season average (approximate)

def compute_gsaa(gl_row: pd.Series) -> Optional[float]:
    """
    GSAA = actual_saves - expected_saves
    expected_saves = shots_against * league_avg_save_pct

    Uses season totals from the goalie parquet row.
    Returns None if data unavailable.
    """
    shots = gl_row.get("shotsAgainst") if hasattr(gl_row, "get") else None
    saves = gl_row.get("saves") if hasattr(gl_row, "get") else None
    if shots is None or saves is None:
        return None
    try:
        expected = float(shots) * LEAGUE_AVG_SAVE_PCT
        return round(float(saves) - expected, 2)
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GoalieContext builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_goalie_context(team: str, scraped: dict, gl: pd.DataFrame) -> GoalieContext:
    """
    Build a GoalieContext for one team by merging:
      - DailyFaceoff live stats (scraped)
      - goalie_features.parquet rolling stats
    """
    name   = scraped.get("name", "Unknown") or "Unknown"
    status = str(scraped.get("status", "unknown"))

    rolling_feats, matched_name = get_goalie_features(team, name, gl)

    # Try to find the matched goalie row for GSAA
    gsaa_season = None
    if not gl.empty and "team" in gl.columns:
        team_gl = gl[gl["team"] == team]
        if not team_gl.empty:
            # Use most recent season aggregates
            latest = team_gl.sort_values("game_date").iloc[-1]
            gsaa_season = compute_gsaa(latest)

    ctx = GoalieContext(
        player_id   = scraped.get("id"),
        name        = matched_name or name,
        status      = status,
        gaa_live    = _safe_float(scraped.get("gaa")),
        svpct_live  = _safe_float(scraped.get("svpct")),
        wins_live   = _safe_int(scraped.get("wins")),
        losses_live = _safe_int(scraped.get("losses")),
        # Rolling from parquet
        rolling_savePct_avg3     = rolling_feats.get("g_savePct_avg3"),
        rolling_savePct_avg5     = rolling_feats.get("g_savePct_avg5"),
        rolling_savePct_avg10    = rolling_feats.get("g_savePct_avg10"),
        rolling_gaa_avg3         = rolling_feats.get("g_goalsAgainst_avg3"),
        rolling_gaa_avg5         = rolling_feats.get("g_goalsAgainst_avg5"),
        rolling_win_rate_3       = rolling_feats.get("g_win_rate_3"),
        rolling_win_rate_5       = rolling_feats.get("g_win_rate_5"),
        rolling_win_rate_10      = rolling_feats.get("g_win_rate_10"),
        rolling_shotsAgainst_avg5 = rolling_feats.get("g_shotsAgainst_avg5"),
        gsaa_season              = gsaa_season,
    )
    return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TeamContext builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_team_context(team: str, is_home: bool, rest: dict,
                       scraped_goalie: dict, news: str,
                       sk: pd.DataFrame, gl: pd.DataFrame,
                       target_date: date,
                       fetch_advanced: bool = True) -> TeamContext:
    """
    Assemble a TeamContext for one team.
    """
    # Situational
    rest_days = rest.get("rest_days")
    is_b2b    = bool(rest.get("is_b2b", False))

    # Skater rolling features
    sk_feats = get_skater_features(team, sk)

    # Goalie
    goalie_ctx = build_goalie_context(team, scraped_goalie, gl)

    # Advanced stats from play-by-play
    adv = {}
    if fetch_advanced:
        try:
            adv = get_team_advanced_stats(team, target_date)
        except Exception as e:
            print(f"    [{team}] Advanced stats error: {e}")

    # News headlines — preserve DailyFaceoff news, don't drop it
    headlines = []
    if news and str(news).strip() and str(news).strip().lower() not in ("", "none", "nan"):
        headlines.append(str(news).strip())

    ctx = TeamContext(
        team      = team,
        is_home   = is_home,
        rest_days = rest_days,
        is_b2b    = is_b2b,

        # Skater rolling
        sk_goals_avg3_mean    = sk_feats.get("sk_goals_avg3_mean"),
        sk_goals_avg5_mean    = sk_feats.get("sk_goals_avg5_mean"),
        sk_goals_avg10_mean   = sk_feats.get("sk_goals_avg10_mean"),
        sk_goals_avg20_mean   = sk_feats.get("sk_goals_avg20_mean"),
        sk_shots_avg3_mean    = sk_feats.get("sk_shots_avg3_mean"),
        sk_shots_avg5_mean    = sk_feats.get("sk_shots_avg5_mean"),
        sk_shots_avg10_mean   = sk_feats.get("sk_shots_avg10_mean"),
        sk_points_avg5_mean   = sk_feats.get("sk_points_avg5_mean"),
        sk_points_avg10_mean  = sk_feats.get("sk_points_avg10_mean"),
        sk_pp_goals_avg5_mean = sk_feats.get("sk_pp_goals_avg5_mean"),
        sk_pp_goals_avg10_mean = sk_feats.get("sk_pp_goals_avg10_mean"),
        sk_hits_avg5_mean      = sk_feats.get("sk_hits_avg5_mean"),
        sk_takeaways_avg5_mean = sk_feats.get("sk_takeaways_avg5_mean"),
        sk_giveaways_avg5_mean = sk_feats.get("sk_giveaways_avg5_mean"),
        sk_plus_minus_avg5_mean = sk_feats.get("sk_plus_minus_avg5_mean"),
        sk_goals_avg5_sum     = sk_feats.get("sk_goals_avg5_sum"),
        sk_shots_avg5_sum     = sk_feats.get("sk_shots_avg5_sum"),
        sk_points_avg5_sum    = sk_feats.get("sk_points_avg5_sum"),
        sk_pp_goals_avg5_sum  = sk_feats.get("sk_pp_goals_avg5_sum"),

        # Advanced stats
        cf_pct_last5      = adv.get("cf_pct_last5"),
        ff_pct_last5      = adv.get("ff_pct_last5"),
        xg_for_last5      = adv.get("xg_for_last5"),
        xg_against_last5  = adv.get("xg_against_last5"),
        xg_pct_last5      = adv.get("xg_pct_last5"),
        sh_attempts_last5 = adv.get("sh_attempts_last5"),

        goalie         = goalie_ctx,
        news_headlines = headlines,
    )
    return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def build_today_contexts(target_date: Optional[date] = None,
                         fetch_advanced: bool = True) -> list[GameContext]:
    """
    Build one GameContext per game for target_date (default: today).
    This is the Module 1 entry point consumed by all downstream modules.

    Args:
        target_date:    Date to predict. Defaults to today.
        fetch_advanced: Set False to skip play-by-play (faster, no xG/CF%).

    Returns:
        list[GameContext]
    """
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    _ensure_dirs()

    print(f"\n{'='*65}")
    print(f"  MODULE 1 — Data Ingestion  [{target_date}]")
    print(f"{'='*65}\n")

    # ── 1. Schedule ─────────────────────────────────────────────────────
    print("[ 1/6 ] Fetching schedule ...")
    games = get_schedule(target_date)
    if not games:
        print("  No regular season games found.")
        return []
    print(f"  {len(games)} games: " +
          "  ".join(f"{g['away_team']}@{g['home_team']}" for g in games))

    # ── 2. Rest days ─────────────────────────────────────────────────────
    print("\n[ 2/6 ] Calculating rest days ...")
    all_teams = list({t for g in games for t in [g["home_team"], g["away_team"]]})
    rest_data = get_team_rest(all_teams, target_date)
    b2b = [t for t, v in rest_data.items() if v["is_b2b"]]
    if b2b:
        print(f"  ⚠  Back-to-back: {', '.join(sorted(b2b))}")
    else:
        print("  ✅ No back-to-back teams")

    # ── 3. Goalies + odds (multi-source: DailyFaceoff + NHL API + GoaliePost) ──
    print("\n[ 3/6 ] Scraping goalies (multi-source) ...")
    scraped_games   = []
    scraped_goalies = {}

    if SCRAPER_AVAILABLE:
        try:
            # Build game_ids_by_matchup so the NHL API source can be activated
            game_ids_by_matchup = {
                (g["away_team"], g["home_team"]): g["game_id"]
                for g in games
            }
            scraped_games = scrape_starting_goalies(
                game_ids_by_matchup=game_ids_by_matchup
            ) or []
            if scraped_games:
                scraped_goalies = get_goalie_dict(scraped_games)
                # Apply manual overrides (daily file, highest priority)
                for team, name in GOALIE_OVERRIDES.items():
                    old = scraped_goalies.get(team, {})
                    scraped_goalies[team] = {**old, "name": name, "status": "Manual"}
                confirmed = sum(1 for v in scraped_goalies.values()
                                if str(v.get("status", "")).lower() in
                                   ("confirmed", "manual"))
                likely = sum(1 for v in scraped_goalies.values()
                             if str(v.get("status", "")).lower() in
                                ("likely", "expected", "probable"))
                print(f"  ✅ {confirmed}/{len(scraped_goalies)} confirmed/manual  "
                      f"🟡 {likely} likely/probable")
                # Persist odds snapshot for line movement tracking
                persist_odds(games, scraped_games, target_date)
        except Exception as e:
            print(f"  ⚠  Multi-source goalie scrape failed: {e}")
    else:
        print("  ⚠  scrape_goalies_v2.py not found — no live goalie data")

    # Build odds lookup keyed by (away, home)
    odds_lookup = {
        (g.get("away_team"), g.get("home_team")): g
        for g in scraped_games
    }
    # Build news lookup
    news_lookup: dict[str, str] = {}
    for g in scraped_games:
        if g.get("home_team"):
            news_lookup[g["home_team"]] = g.get("home_news", "")
        if g.get("away_team"):
            news_lookup[g["away_team"]] = g.get("away_news", "")

    # ── 4. Load parquets ─────────────────────────────────────────────────
    print("\n[ 4/6 ] Loading feature parquets ...")
    sk, gl = _load_parquets()
    print(f"  Skaters: {sk.shape[0]:,} rows   Goalies: {gl.shape[0]:,} rows")

    # ── 5. Advanced stats (play-by-play) ────────────────────────────────
    if fetch_advanced:
        print(f"\n[ 5/6 ] Deriving advanced stats from NHL play-by-play API ...")
        print(f"  (CF%, FF%, xG for {len(all_teams)} teams × last 5 games)")
        print(f"  Results cached to pbp_cache/ — only new games hit the API")
    else:
        print(f"\n[ 5/6 ] Advanced stats skipped (fetch_advanced=False)")

    # ── 6. Assemble GameContext per game ─────────────────────────────────
    print(f"\n[ 6/6 ] Assembling GameContext objects ...")
    contexts = []

    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_id   = game["game_id"]

        home_scraped = scraped_goalies.get(home_team, {"name": "Unknown", "status": "unknown"})
        away_scraped = scraped_goalies.get(away_team, {"name": "Unknown", "status": "unknown"})

        odds_raw = odds_lookup.get((away_team, home_team), {})

        # Build team contexts
        home_ctx = build_team_context(
            team=home_team, is_home=True,
            rest=rest_data.get(home_team, {}),
            scraped_goalie=home_scraped,
            news=news_lookup.get(home_team, ""),
            sk=sk, gl=gl,
            target_date=target_date,
            fetch_advanced=fetch_advanced,
        )
        away_ctx = build_team_context(
            team=away_team, is_home=False,
            rest=rest_data.get(away_team, {}),
            scraped_goalie=away_scraped,
            news=news_lookup.get(away_team, ""),
            sk=sk, gl=gl,
            target_date=target_date,
            fetch_advanced=fetch_advanced,
        )

        # Build odds context
        home_ml = _safe_float(odds_raw.get("home_moneyline"))
        away_ml = _safe_float(odds_raw.get("away_moneyline"))
        movement = get_line_movement(game_id, target_date)

        odds_ctx = OddsContext(
            home_ml        = home_ml,
            away_ml        = away_ml,
            point_spread   = _safe_float(odds_raw.get("point_spread")),
            home_implied   = american_to_implied(home_ml),
            away_implied   = american_to_implied(away_ml),
            opening_home_ml = movement.get("opening_home_ml"),
            opening_away_ml = movement.get("opening_away_ml"),
            line_movement   = movement.get("line_movement"),
            scraped_at      = datetime.utcnow() if odds_raw else None,
        )

        # Data quality flags
        goalie_confirmed = (
            home_ctx.goalie is not None and home_ctx.goalie.is_trusted and
            away_ctx.goalie is not None and away_ctx.goalie.is_trusted
        )
        has_advanced = (
            home_ctx.cf_pct_last5 is not None or
            home_ctx.xg_for_last5 is not None
        )
        has_any_news = bool(home_ctx.news_headlines or away_ctx.news_headlines)

        ctx = GameContext(
            game_id    = game_id,
            game_date  = str(target_date),
            start_time = game.get("start_time", "TBD"),
            season     = game.get("season", ""),
            game_type  = game.get("game_type", 2),
            home       = home_ctx,
            away       = away_ctx,
            odds       = odds_ctx,
            goalie_confirmed   = goalie_confirmed,
            has_advanced_stats = has_advanced,
            has_odds           = home_ml is not None,
            has_news           = has_any_news,
        )
        contexts.append(ctx)
        print(f"  {ctx}")

    print(f"\n{'='*65}")
    print(f"  Module 1 complete — {len(contexts)} GameContext objects built")
    print(f"  Advanced stats: {sum(c.has_advanced_stats for c in contexts)}/{len(contexts)}")
    print(f"  Confirmed goalies: {sum(c.goalie_confirmed for c in contexts)}/{len(contexts)}")
    print(f"  Odds populated: {sum(c.has_odds for c in contexts)}/{len(contexts)}")
    print(f"{'='*65}\n")

    return contexts


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None

def _safe_int(val) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NHL MAS Module 1 — Data Ingestion")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-advanced", action="store_true",
                        help="Skip play-by-play advanced stats (faster)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    contexts = build_today_contexts(target, fetch_advanced=not args.no_advanced)

    # Quick spot-check print
    if contexts:
        print("\n── Sample GameContext dump ──")
        c = contexts[0]
        print(f"  Game:       {c.away.team} @ {c.home.team}  [{c.start_time}]")
        print(f"  Goalies:    {c.away.goalie.name} ({c.away.goalie.status}) vs "
              f"{c.home.goalie.name} ({c.home.goalie.status})")
        print(f"  Odds:       home ML {c.odds.home_ml}  away ML {c.odds.away_ml}  "
              f"spread {c.odds.point_spread}")
        if c.odds.line_movement is not None:
            print(f"  Line move:  {c.odds.line_movement:+.0f} (open {c.odds.opening_home_ml})")
        if c.home.xg_for_last5:
            print(f"  {c.home.team} xG/gm: {c.home.xg_for_last5:.2f}F / "
                  f"{c.home.xg_against_last5:.2f}A  CF%={c.home.cf_pct_last5:.3f}")
        if c.away.xg_for_last5:
            print(f"  {c.away.team} xG/gm: {c.away.xg_for_last5:.2f}F / "
                  f"{c.away.xg_against_last5:.2f}A  CF%={c.away.cf_pct_last5:.3f}")
        if c.home.news_headlines:
            print(f"  {c.home.team} news: {c.home.news_headlines[0][:80]}")
        if c.away.news_headlines:
            print(f"  {c.away.team} news: {c.away.news_headlines[0][:80]}")
