#!/usr/bin/env python
"""
Flatten NHL collector JSON files, engineer player-game features, and attach targets
from the NHL gamecenter boxscore endpoint.

Outputs (in Features/):
    - player_games.csv
    - player_games_features.csv
    - player_game_targets.csv
    - player_games_with_targets.csv
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

BASE_DIR = Path(r"C:\Users\shell\OneDrive\Documents\NHL_Player")
FEATURES_DIR = BASE_DIR / "Features"
BOX_DIR = BASE_DIR / "Boxscores"

RAW_OUTPUT = FEATURES_DIR / "player_games.csv"
FEAT_OUTPUT = FEATURES_DIR / "player_games_features.csv"
TARGET_OUTPUT = FEATURES_DIR / "player_game_targets.csv"
MERGED_OUTPUT = FEATURES_DIR / "player_games_with_targets.csv"

BOX_API_TEMPLATE = "https://api-web.nhle.com/v1/gamecenter/{game_pk}/boxscore"

HTTP_TIMEOUT = 20  # seconds
MAX_BOX_RETRIES = 5
BASE_BACKOFF = 1.0  # seconds between retry attempts
BASE_SLEEP_BETWEEN_CALLS = 0.6  # steady-state pacing to avoid 429s

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "nhl-player-pipeline/1.0 (+contact: shell)",
        "Accept": "application/json",
        "Connection": "keep-alive",
    }
)
SESSION.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
SESSION.mount("http://", HTTPAdapter(pool_connections=20, pool_maxsize=20))

SCORING_WEIGHTS = {
    "goal": 8.0,          # DraftKings-style defaults; adjust to your scoring system
    "assist": 5.0,
    "shot": 1.5,
    "block": 1.3,
    "hit": 0.5,
}

JSON_SEARCH_DIRS = [BASE_DIR, BOX_DIR]  # Look for nhl_*.json in both locations

PLAYER_COLUMNS = [
    "game_pk",
    "game_date",
    "season",
    "game_type",
    "game_state",
    "venue",
    "team_id",
    "team_abbr",
    "opponent_abbr",
    "is_home",
    "player_id",
    "player_name",
    "position",
    "shoots_catches",
    "height_in",
    "weight_lb",
    "birth_date",
    "rookie",
    "salary",
    "season_stats",
    "source_file",
    "start_time_utc",
    "start_time_local",
]
# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _coalesce(*values: Optional[Any]) -> Optional[Any]:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _safe_name(value: Any) -> str:
    if isinstance(value, dict):
        return (
            _coalesce(
                value.get("default"),
                value.get("en"),
                value.get("first"),
                next(iter(value.values()), None),
            )
            or ""
        )
    if isinstance(value, str):
        return value
    return ""


def _lookup(player: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    """Search for keys in the player dict, falling back to nested blocks."""
    for key in keys:
        if key in player and player[key] not in (None, "", []):
            return player[key]

    for nested_key in ("player", "person", "skater", "goalie", "info"):
        nested = player.get(nested_key)
        if isinstance(nested, dict):
            for key in keys:
                if key in nested and nested[key] not in (None, "", []):
                    return nested[key]
    return None


def _extract_player_core(player: Dict[str, Any]) -> Dict[str, Any]:
    person = player
    for nested_key in ("player", "person", "skater", "goalie", "info"):
        nested = player.get(nested_key)
        if isinstance(nested, dict):
            person = nested
            break
    return person


def _resolve_game_date(game: Dict[str, Any], payload: Dict[str, Any]) -> Optional[str]:
    candidates = [
        game.get("gameDate"),
        payload.get("date"),
        payload.get("gameDate"),
        game.get("gameDateISO"),
    ]
    for key in ("startTimeUTC", "startTimeLocal"):
        stamp = game.get(key)
        if isinstance(stamp, str) and len(stamp) >= 10:
            candidates.append(stamp[:10])

    result = _coalesce(*candidates)
    if isinstance(result, str) and len(result) >= 10:
        return result[:10]
    return None


def _resolve_games(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "games" in payload and isinstance(payload["games"], list):
        return payload["games"]
    if "gamesByDate" in payload:
        games: List[Dict[str, Any]] = []
        for block in payload["gamesByDate"]:
            games.extend(block.get("games", []))
        return games
    raise ValueError("No games found in payload")


def _extract_venue(game: Dict[str, Any]) -> Optional[str]:
    venue = game.get("venue")
    if isinstance(venue, dict):
        return _coalesce(venue.get("default"), venue.get("name"))
    if isinstance(venue, str):
        return venue
    return None


def _normalize_stats(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _toi_to_minutes(val: Any) -> Optional[float]:
    if val in (None, "", "00:00"):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and ":" in val:
        try:
            minutes, seconds = val.split(":")
            return int(minutes) + int(seconds) / 60.0
        except ValueError:
            return None
    return None


def _safe_numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_roster(team: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a unified list of player dicts for the given team block."""
    roster = team.get("roster")
    players: List[Dict[str, Any]] = []

    def _extend(block: Any) -> None:
        if isinstance(block, list):
            players.extend([entry for entry in block if isinstance(entry, dict)])
        elif isinstance(block, dict):
            players.extend([entry for entry in block.values() if isinstance(entry, dict)])

    if roster:
        _extend(roster)
        for key in ("forwards", "defensemen", "goalies", "skaters", "players"):
            if key in roster:
                _extend(roster[key])
    else:
        for key in ("players", "skaters", "forwards", "defensemen", "goalies"):
            if key in team:
                _extend(team[key])

    return players


# ---------------------------------------------------------------------------
# Flattening logic
# ---------------------------------------------------------------------------

def load_player_games(json_path: Path) -> pd.DataFrame:
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    games = _resolve_games(payload)
    if not games:
        raise ValueError(f"No games inside {json_path}")

    rows: List[Dict[str, Any]] = []
    for game in games:
        game_pk = _coalesce(game.get("gamePk"), game.get("id"))
        if game_pk is None:
            raise ValueError(f"Missing gamePk/id in {json_path}")

        game_date = _resolve_game_date(game, payload)
        venue = _extract_venue(game)
        game_state = game.get("gameState")
        game_type = game.get("gameType")
        season = game.get("season")

        for side in ("homeTeam", "awayTeam"):
            team = game.get(side) or {}
            opponent = game.get("homeTeam" if side == "awayTeam" else "awayTeam") or {}

            roster = _iter_roster(team)
            for player in roster:
                person = _extract_player_core(player)
                first = _safe_name(_lookup(player, ("firstName", "first_name")))
                last = _safe_name(_lookup(player, ("lastName", "last_name")))
                full_name = (first + " " + last).strip()
                if not full_name:
                    full_name = _safe_name(person.get("fullName")) or _safe_name(person.get("name"))

                rows.append(
                    {
                        "game_pk": game_pk,
                        "game_date": game_date,
                        "season": season,
                        "game_type": game_type,
                        "game_state": game_state,
                        "venue": venue,
                        "team_id": _coalesce(team.get("id"), team.get("teamId")),
                        "team_abbr": _coalesce(team.get("abbrev"), team.get("triCode"), team.get("tricode")),
                        "opponent_abbr": _coalesce(
                            opponent.get("abbrev"), opponent.get("triCode"), opponent.get("tricode")
                        ),
                        "is_home": int(side == "homeTeam"),
                        "player_id": _coalesce(
                            _lookup(player, ("playerId", "personId", "id")),
                            person.get("id"),
                        ),
                        "player_name": full_name,
                        "position": _coalesce(
                            _lookup(player, ("positionCode", "position", "pos")),
                            person.get("primaryPosition", {}).get("code"),
                        ),
                        "shoots_catches": _coalesce(
                            _lookup(player, ("shootsCatches", "shoots", "catches")),
                            person.get("shootsCatches"),
                        ),
                        "height_in": _coalesce(
                            _lookup(player, ("heightInInches", "height")),
                            person.get("heightInInches"),
                        ),
                        "weight_lb": _coalesce(
                            _lookup(player, ("weightInPounds", "weight")),
                            person.get("weightInPounds"),
                        ),
                        "birth_date": _coalesce(
                            _lookup(player, ("birthDate", "birthdate")),
                            person.get("birthDate"),
                        ),
                        "rookie": bool(_lookup(player, ("rookie", "isRookie")) or False),
                        "salary": _lookup(player, ("salary", "capHit")),
                        "season_stats": _lookup(
                            player,
                            (
                                "seasonStats",
                                "seasonTotals",
                                "stats",
                                "season",
                                "seasonRatings",
                            ),
                        ),
                        "source_file": json_path.name,
                        "start_time_utc": game.get("startTimeUTC"),
                        "start_time_local": game.get("startTimeLocal"),
                    }
                )

    df = pd.DataFrame(rows, columns=PLAYER_COLUMNS)
    if df.empty:
        print(f"[!] Warning: {json_path.name} produced zero roster rows.")

    missing_ids = df["player_id"].isna().sum()
    if missing_ids:
        print(f"[!] Warning: {missing_ids:,} rows missing player_id in {json_path.name}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

    stats_df = pd.json_normalize(df["season_stats"].apply(_normalize_stats))
    stats_df.columns = [f"season_{col}" for col in stats_df.columns]
    df = pd.concat([df.drop(columns=["season_stats"]), stats_df], axis=1)

    numeric_cols = [col for col in df.columns if col.startswith("season_")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "season_timeOnIcePerGame" in df.columns:
        df["season_timeOnIcePerGame"] = df["season_timeOnIcePerGame"].apply(_toi_to_minutes)
    if "season_powerPlayTimeOnIcePerGame" in df.columns:
        df["season_powerPlayTimeOnIcePerGame"] = df["season_powerPlayTimeOnIcePerGame"].apply(_toi_to_minutes)

    df["age_years"] = ((df["game_date"] - df["birth_date"]).dt.days / 365.25).round(2)

    position_map = {
        "L": "F",
        "LW": "F",
        "R": "F",
        "RW": "F",
        "C": "F",
        "LC": "F",
        "RC": "F",
        "D": "D",
        "LD": "D",
        "RD": "D",
        "G": "G",
        "GK": "G",
    }
    df["position_group"] = (
        df["position"]
        .fillna("")
        .str.upper()
        .map(position_map)
        .fillna("OTH")
    )

    df["shoots_catches"] = df["shoots_catches"].fillna("")
    df["shoots_left"] = (df["shoots_catches"].str.upper() == "L").astype(int)
    df["shoots_right"] = (df["shoots_catches"].str.upper() == "R").astype(int)

    df = df.sort_values(["player_id", "game_date", "game_pk"])
    df["games_played_to_date"] = df.groupby("player_id").cumcount()

    df["rest_days"] = (
        df.groupby("player_id")["game_date"]
        .diff()
        .dt.days
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )

    cumulative_cols = [
        "season_goals",
        "season_assists",
        "season_points",
        "season_shots",
        "season_hits",
        "season_blocks",
    ]
    for col in cumulative_cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("player_id")[col].transform(lambda s: s.ffill().fillna(0))
        per_game_col = col.replace("season_", "") + "_pg"
        df[per_game_col] = df.groupby("player_id")[col].diff()
        first_mask = df["games_played_to_date"] == 0
        df.loc[first_mask, per_game_col] = df.loc[first_mask, col]
        df[per_game_col] = df[per_game_col].fillna(0).clip(lower=0)

    rolling_targets = {
        "goals_pg": [3, 5, 10],
        "assists_pg": [3, 5, 10],
        "points_pg": [3, 5, 10],
        "shots_pg": [3, 5, 10],
    }
    for col, windows in rolling_targets.items():
        if col not in df.columns:
            continue
        for window in windows:
            feat_name = f"{col}_roll{window}"
            df[feat_name] = df.groupby("player_id")[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    if "season_timeOnIcePerGame" in df.columns:
        df["toi_avg"] = df["season_timeOnIcePerGame"]
    if "season_powerPlayTimeOnIcePerGame" in df.columns:
        df["pp_toi_avg"] = df["season_powerPlayTimeOnIcePerGame"]
        df["pp_usage_pct"] = (
            df["pp_toi_avg"] / df["toi_avg"]
        ).replace([np.inf, -np.inf], np.nan)

    if "birth_date" in df.columns:
        df = df.drop(columns=["birth_date"])
    return df

# ---------------------------------------------------------------------------
# Target extraction from boxscore
# ---------------------------------------------------------------------------

def fetch_boxscore(game_pk: int, session: requests.Session) -> Optional[Dict[str, Any]]:
    BOX_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = BOX_DIR / f"{game_pk}.json"

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            return json.load(fh)

    url = BOX_API_TEMPLATE.format(game_pk=game_pk)
    last_error: Optional[str] = None

    for attempt in range(MAX_BOX_RETRIES):
        try:
            resp = session.get(url, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            last_error = str(exc)
            wait = BASE_BACKOFF * (2 ** attempt)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            time.sleep(BASE_SLEEP_BETWEEN_CALLS + random.uniform(0.0, 0.8))
            return data

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(float(retry_after), BASE_BACKOFF)
                except ValueError:
                    wait = BASE_BACKOFF * (2 ** attempt)
            else:
                wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0.0, 1.5)
            print(f"  Boxscore 429 for {game_pk}: sleeping {wait:.1f}s before retry.")
            time.sleep(wait)
            continue

        if 500 <= resp.status_code < 600:
            wait = BASE_BACKOFF * (2 ** attempt)
            print(f"  Boxscore {resp.status_code} for {game_pk}: retrying in {wait:.1f}s.")
            time.sleep(wait)
            continue

        last_error = f"{resp.status_code} {resp.reason}"
        break

    print(f"  Boxscore HTTP error for {game_pk}: {last_error or 'exceeded retries'}")
    return None


def _iter_team_players(boxscore: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    players: List[Tuple[str, Dict[str, Any]]] = []

    teams_block = boxscore.get("teams")
    if isinstance(teams_block, dict):
        team_entries = [
            ("home", teams_block.get("home")),
            ("away", teams_block.get("away")),
        ]
    else:
        team_entries = [
            ("home", boxscore.get("homeTeam")),
            ("away", boxscore.get("awayTeam")),
        ]

    for label, team in team_entries:
        if not isinstance(team, dict):
            continue

        abbrev = (
            team.get("abbrev")
            or team.get("teamAbbrev")
            or team.get("triCode")
            or team.get("tricode")
        )
        player_blocks: List[Dict[str, Any]] = []

        for key in ("players", "skaters", "forwards", "defensemen", "goalies"):
            block = team.get(key)
            if isinstance(block, list):
                player_blocks.extend(block)
            elif isinstance(block, dict):
                player_blocks.extend(block.values())

        # Deduplicate by playerId
        seen: set[int] = set()
        deduped: List[Dict[str, Any]] = []
        for player in player_blocks:
            pid = (
                player.get("playerId")
                or player.get("id")
                or player.get("personId")
                or player.get("player", {}).get("id")
            )
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            deduped.append(player)

        for player in deduped:
            players.append((abbrev, player))

    return players


def _get_stat(player: Dict[str, Any], *keys: str, numeric: bool = True) -> float:
    for key in keys:
        if key in player and player[key] not in (None, "", "-"):
            return _safe_numeric(player[key]) if numeric else player[key]
    stats = player.get("stat") or player.get("stats")
    if isinstance(stats, dict):
        for key in keys:
            if key in stats and stats[key] not in (None, "", "-"):
                return _safe_numeric(stats[key]) if numeric else stats[key]
    return 0.0 if numeric else ""


def extract_targets_from_boxscore(game_pk: int, boxscore: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    stats_root = boxscore.get("playerByGameStats")
    if not isinstance(stats_root, dict):
        print(f"  {game_pk}: boxscore missing playerByGameStats block")
        return rows

    def _team_meta(side: str) -> Dict[str, Any]:
        # Side is "homeTeam" or "awayTeam"
        if side in boxscore and isinstance(boxscore[side], dict):
            return boxscore[side]
        teams_block = boxscore.get("teams")
        key = "home" if side == "homeTeam" else "away"
        if isinstance(teams_block, dict) and isinstance(teams_block.get(key), dict):
            return teams_block[key]
        return {}

    for side in ("homeTeam", "awayTeam"):
        team_meta = _team_meta(side)
        team_abbr = _coalesce(
            team_meta.get("abbrev"),
            team_meta.get("triCode"),
            team_meta.get("tricode"),
        )
        side_stats = stats_root.get(side, {})
        if not isinstance(side_stats, dict):
            continue

        skater_groups = ("forwards", "defense", "defensemen", "skaters")
        for group in skater_groups:
            players = side_stats.get(group, [])
            if not isinstance(players, list):
                continue

            for player in players:
                if not isinstance(player, dict):
                    continue
                player_id = player.get("playerId")
                if player_id is None:
                    continue

                goals = _safe_numeric(player.get("goals"))
                assists = _safe_numeric(player.get("assists"))
                points = _safe_numeric(player.get("points", goals + assists))
                shots = _safe_numeric(player.get("sog") or player.get("shotsOnGoal") or player.get("shots"))
                hits = _safe_numeric(player.get("hits"))
                blocks = _safe_numeric(player.get("blockedShots"))
                pim = _safe_numeric(player.get("penaltyMinutes") or player.get("pim"))
                toi_str = player.get("timeOnIce") or player.get("toi")
                toi_minutes = _toi_to_minutes(toi_str)

                fantasy_points = (
                    goals * SCORING_WEIGHTS["goal"]
                    + assists * SCORING_WEIGHTS["assist"]
                    + shots * SCORING_WEIGHTS["shot"]
                    + blocks * SCORING_WEIGHTS["block"]
                    + hits * SCORING_WEIGHTS["hit"]
                )

                rows.append(
                    {
                        "game_pk": game_pk,
                        "team_abbr": team_abbr,
                        "player_id": player_id,
                        "targets_goals": goals,
                        "targets_assists": assists,
                        "targets_points": points,
                        "targets_shots": shots,
                        "targets_hits": hits,
                        "targets_blocks": blocks,
                        "targets_pim": pim,
                        "targets_toi_min": toi_minutes,
                        "targets_fantasy_pts": fantasy_points,
                    }
                )

        goalies = side_stats.get("goalies", [])
        if isinstance(goalies, list):
            for goalie in goalies:
                if not isinstance(goalie, dict):
                    continue
                player_id = goalie.get("playerId")
                if player_id is None:
                    continue

                shots_against = _safe_numeric(goalie.get("shotsAgainst"))
                saves = _safe_numeric(goalie.get("saves"))
                goals_against = _safe_numeric(goalie.get("goalsAgainst"))
                toi_minutes = _toi_to_minutes(goalie.get("timeOnIce"))
                save_pct = _safe_numeric(goalie.get("savePercentage"))

                rows.append(
                    {
                        "game_pk": game_pk,
                        "team_abbr": team_abbr,
                        "player_id": player_id,
                        "targets_goals": 0.0,
                        "targets_assists": 0.0,
                        "targets_points": 0.0,
                        "targets_shots": shots_against,  # goalie-facing shots
                        "targets_hits": 0.0,
                        "targets_blocks": 0.0,
                        "targets_pim": 0.0,
                        "targets_toi_min": toi_minutes,
                        "targets_fantasy_pts": (
                            saves * SCORING_WEIGHTS["shot"]  # adjust if you score goalies differently
                        ),
                        "shots_against": shots_against,
                        "saves": saves,
                        "goals_against": goals_against,
                        "save_pct": save_pct,
                    }
                )

    print(f"  {game_pk}: built {len(rows)} target rows")
    return rows


def build_targets(df_features: pd.DataFrame) -> pd.DataFrame:
    print("game_state counts:")
    print(df_features["game_state"].value_counts(dropna=False).head(10))

    completed_states = {"OFF", "FINAL", "FINAL_OT", "FINAL_SO"}
    df_completed = df_features[df_features["game_state"].isin(completed_states)].copy()

    print(f"{len(df_completed)} player rows across {df_completed['game_pk'].nunique()} completed games.")
    if df_completed.empty:
        print("No completed games available for target generation.")
        return pd.DataFrame(columns=[
            "game_pk", "team_abbr", "player_id",
            "targets_goals", "targets_assists", "targets_shots",
            "targets_blocks", "targets_hits", "targets_points"
        ])

    unique_games = sorted(df_completed["game_pk"].unique())
    rows: List[Dict[str, Any]] = []

    print(f"Pulling boxscores for {len(unique_games)} games ...")
    with requests.Session() as session:
        for idx, game_pk in enumerate(unique_games, 1):
            boxscore = fetch_boxscore(int(game_pk), session)
            if not boxscore:
                continue
            rows.extend(extract_targets_from_boxscore(int(game_pk), boxscore))
            if idx % 5 == 0:
                time.sleep(1.0)

    if not rows:
        print("No boxscore targets were created after processing completed games.")
        return pd.DataFrame(columns=[
            "game_pk", "team_abbr", "player_id",
            "targets_goals", "targets_assists", "targets_shots",
            "targets_blocks", "targets_hits", "targets_points"
        ])

    df_targets = pd.DataFrame(rows)
    print(f"Produced {len(df_targets)} target rows from {df_targets['game_pk'].nunique()} games.")
    return df_targets
    "targets_hits", "targets_blocks", "targets_pim",
    "targets_toi_min", "targets_fantasy_pts",
        



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    json_files: List[Path] = []
    for directory in JSON_SEARCH_DIRS:
        if directory.exists():
            json_files.extend(directory.glob("nhl_*.json"))

    json_files = sorted(set(json_files))
    if not json_files:
        raise FileNotFoundError(
            f"No nhl_*.json files found in {[str(d) for d in JSON_SEARCH_DIRS]}"
        )

    frames: List[pd.DataFrame] = []
    for path in json_files:
        print(f"Flattening {path.name} ...")
        frames.append(load_player_games(path))

    df_raw = pd.concat(frames, ignore_index=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(RAW_OUTPUT, index=False)
    print(f"Wrote {len(df_raw):,} player-game rows to {RAW_OUTPUT}")

    print("Engineering features ...")
    df_feat = engineer_features(df_raw)
    df_feat.to_csv(FEAT_OUTPUT, index=False)
    print(f"Wrote {len(df_feat):,} rows with features to {FEAT_OUTPUT}")

    print("Building targets from boxscores ...")
    df_targets = build_targets(df_feat)
    df_targets.to_csv(TARGET_OUTPUT, index=False)
    print(f"Wrote {len(df_targets):,} rows to {TARGET_OUTPUT}")

    print("Merging features + targets ...")
    df_merged = df_feat.merge(
        df_targets,
        on=["game_pk", "team_abbr", "player_id"],
        how="left",
    )
    df_merged.to_csv(MERGED_OUTPUT, index=False)
    print(f"Wrote {len(df_merged):,} rows to {MERGED_OUTPUT}")


if __name__ == "__main__":
    main()