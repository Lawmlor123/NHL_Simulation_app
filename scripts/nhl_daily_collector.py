#!/usr/bin/env python3
"""
nhl_daily_collector.py

Examples:
    python nhl_daily_collector.py --date 2024-10-10 --save nhl.json
    python nhl_daily_collector.py --include-stats --max-stats 25
    python nhl_daily_collector.py --include-boxscores --start-date 2022-01-01 --end-date 2025-06-15 --save nhl_2022_2025.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCOREBOARD_URL = "https://api-web.nhle.com/v1/scoreboard/{date}"
ROSTER_URL = "https://api-web.nhle.com/v1/roster/{team}/current"
SKATER_STATS_URL = "https://api.nhle.com/stats/rest/en/skater/summary"
GOALIE_STATS_URL = "https://api.nhle.com/stats/rest/en/goalie/summary"
GAME_FEED_URL = "https://api-web.nhle.com/v1/gamecenter/{game_pk}/boxscore"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect NHL data for one date or a date range.")
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument("--date", help="Single YYYY-MM-DD (defaults to today).")
    date_group.add_argument("--start-date", help="Start of date range (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="End of date range (YYYY-MM-DD). Defaults to today if omitted.")
    parser.add_argument("--max-days", type=int, help="Safety cap on number of days when using a range.")
    parser.add_argument("--include-stats", action="store_true", help="Fetch season stats per player.")
    parser.add_argument("--include-boxscores", action="store_true", help="Fetch per-game player boxscore stats.")
    parser.add_argument("--max-stats", type=int, help="Cap the number of player stat fetches.")
    parser.add_argument("--max-stat-errors", type=int, default=10, help="Abort stat calls after this many failures.")
    parser.add_argument("--save", help="File path to dump JSON output.")
    parser.add_argument("--season", help="Explicit season string (e.g., 20242025).")
    parser.add_argument("--connect-timeout", type=float, default=5.0, help="Seconds to wait for TCP connection.")
    parser.add_argument("--read-timeout", type=float, default=20.0, help="Seconds to wait for server response.")
    parser.add_argument("--no-proxy", action="store_true", help="Ignore system proxy settings.")
    parser.add_argument("--proxy", help="Explicit proxy URL (http://host:port) for HTTP/HTTPS.")
    parser.add_argument("--verbose", action="store_true", help="Print each request URL.")
    return parser.parse_args()


def parse_date(value: str, label: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {value}") from exc


def build_date_list(args: argparse.Namespace) -> List[dt.date]:
    if args.date:
        return [parse_date(args.date, "--date")]

    start = parse_date(args.start_date, "--start-date") if args.start_date else None
    end = parse_date(args.end_date, "--end-date") if args.end_date else None

    if not start:
        return [dt.date.today()]
    if not end:
        end = dt.date.today()
    if end < start:
        raise ValueError("--end-date must be on or after --start-date.")

    days = (end - start).days + 1
    if args.max_days is not None and days > args.max_days:
        raise ValueError(
            f"Date span is {days} days which exceeds --max-days={args.max_days}. "
            "Tighten the range or raise the cap."
        )

    return [start + dt.timedelta(days=offset) for offset in range(days)]


def infer_season(game_date: dt.date) -> str:
    start_year = game_date.year if game_date.month >= 9 else game_date.year - 1
    return f"{start_year}{start_year + 1}"


def make_session(args: argparse.Namespace) -> requests.Session:
    session = requests.Session()
    if args.no_proxy:
        session.trust_env = False
        session.proxies = {}
    if args.proxy:
        session.proxies.update({"http": args.proxy, "https": args.proxy})

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "NHL-Data-Collector/0.6 (+https://github.com/your-handle)",
        "Accept": "application/json",
    })
    return session


def fetch_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeouts: Tuple[float, float],
    verbose: bool = False,
) -> Dict[str, Any]:
    if verbose:
        pretty = url if not params else f"{url}?{requests.compat.urlencode(params)}"
        print(f"[GET] {pretty}")
    resp = session.get(url, params=params, timeout=timeouts)
    resp.raise_for_status()
    return resp.json()


def get_games_for_date(scoreboard_payload: Dict[str, Any], date_str: str) -> List[Dict[str, Any]]:
    for bundle in scoreboard_payload.get("gamesByDate", []):
        if bundle.get("date") == date_str:
            return bundle.get("games", [])
    return []


def normalize_name(part: Any) -> str:
    if part is None:
        return ""
    if isinstance(part, dict):
        return part.get("default") or next(iter(part.values()), "")
    return str(part)


def resolve_player_full_name(player: Dict[str, Any]) -> str:
    if player.get("fullName"):
        return player["fullName"]
    return f"{normalize_name(player.get('firstName'))} {normalize_name(player.get('lastName'))}".strip()


def get_player_id(player: Dict[str, Any]) -> Optional[int]:
    for key in ("playerId", "id", "personId", "nhlId"):
        value = player.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def flatten_roster(roster_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    players: List[Dict[str, Any]] = []
    for group in ("forwards", "defensemen", "goalies", "injuredReserve"):
        for raw in roster_payload.get(group, []):
            pid = get_player_id(raw)
            if pid is None:
                continue
            players.append(
                {
                    "playerId": pid,
                    "fullName": resolve_player_full_name(raw),
                    "sweaterNumber": raw.get("sweaterNumber"),
                    "position": raw.get("positionCode"),
                    "shootsCatches": raw.get("shootsCatches"),
                    "heightInInches": raw.get("heightInInches"),
                    "weightInPounds": raw.get("weightInPounds"),
                    "age": raw.get("age"),
                    "isCaptain": raw.get("captain", False),
                    "isRookie": raw.get("rookie", False),
                    "raw": raw,
                }
            )
    return players


def fetch_player_stats(
    session: requests.Session,
    player_id: int,
    season: str,
    position: Optional[str],
    timeouts: Tuple[float, float],
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    stats_url = GOALIE_STATS_URL if (position or "").upper() == "G" else SKATER_STATS_URL
    params = {
        "isAggregate": "false",
        "reportType": "basic",
        "isGame": "false",
        "playerId": player_id,
        "season": season,
        "gameType": "2",
        "limit": 1,
    }
    try:
        data = fetch_json(session, stats_url, params=params, timeouts=timeouts, verbose=verbose)
    except requests.HTTPError as err:
        status = err.response.status_code if err.response else "?"
        print(f"Warning: stats fetch failed for player {player_id} (HTTP {status}): {err}", file=sys.stderr)
        return None
    except requests.RequestException as err:
        print(f"Warning: stats fetch failed for player {player_id}: {err}", file=sys.stderr)
        return None

    rows = data.get("data", [])
    return rows[0] if rows else None


def fetch_game_boxscore(
    session: requests.Session,
    game_pk: int,
    timeouts: Tuple[float, float],
    verbose: bool,
) -> Dict[str, Any]:
    url = GAME_FEED_URL.format(game_pk=game_pk)
    data = fetch_json(session, url, params=None, timeouts=timeouts, verbose=verbose)
    return data.get("liveData", {}).get("boxscore", {})


def flatten_boxscore_team(boxscore: Dict[str, Any], team_key: str) -> Dict[str, Any]:
    team_payload = boxscore.get("teams", {}).get(team_key, {})
    team_abbr = team_payload.get("team", {}).get("abbreviation")
    players_out: List[Dict[str, Any]] = []

    for player_payload in team_payload.get("players", {}).values():
        person = player_payload.get("person", {})
        stats = player_payload.get("stats", {}) or {}
        skater = stats.get("skaterStats") or {}
        goalie = stats.get("goalieStats") or {}
        position = player_payload.get("position", {}).get("abbreviation")

        players_out.append(
            {
                "playerId": person.get("id"),
                "fullName": person.get("fullName"),
                "position": position,
                "timeOnIce": skater.get("timeOnIce") or goalie.get("timeOnIce"),
                "goals": skater.get("goals"),
                "assists": skater.get("assists"),
                "points": (skater.get("goals") or 0) + (skater.get("assists") or 0),
                "shots": skater.get("shots"),
                "hits": skater.get("hits"),
                "blocks": skater.get("blocked"),
                "faceOffPct": skater.get("faceOffWinPercentage"),
                "pim": skater.get("penaltyMinutes") or goalie.get("pim"),
                "powerPlayGoals": skater.get("powerPlayGoals"),
                "powerPlayAssists": skater.get("powerPlayAssists"),
                "shortHandedGoals": skater.get("shortHandedGoals"),
                "shortHandedAssists": skater.get("shortHandedAssists"),
                "plusMinus": skater.get("plusMinus"),
                "saves": goalie.get("saves"),
                "shotsAgainst": goalie.get("shotsAgainst"),
                "savePct": goalie.get("savePercentage"),
                "powerPlaySaves": goalie.get("powerPlaySaves"),
                "powerPlayShotsAgainst": goalie.get("powerPlayShotsAgainst"),
                "shortHandedSaves": goalie.get("shortHandedSaves"),
                "shortHandedShotsAgainst": goalie.get("shortHandedShotsAgainst"),
                "evenSaves": goalie.get("evenSaves"),
                "evenShotsAgainst": goalie.get("evenShotsAgainst"),
            }
        )

    return {"team_abbr": team_abbr, "players": players_out}


def collect_for_date(
    target_date: dt.date,
    args: argparse.Namespace,
    session: requests.Session,
    roster_cache: Dict[str, List[Dict[str, Any]]],
    stats_cache: Dict[int, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    date_str = target_date.isoformat()
    season = args.season or infer_season(target_date)
    timeouts = (args.connect_timeout, args.read_timeout)

    try:
        scoreboard_payload = fetch_json(
            session,
            SCOREBOARD_URL.format(date=date_str),
            timeouts=timeouts,
            verbose=args.verbose,
        )
    except requests.RequestException as err:
        print(f"[{date_str}] Failed to fetch scoreboard: {err}", file=sys.stderr)
        return {"date": date_str, "season": season, "gameCount": 0, "games": [], "error": str(err)}

    games = get_games_for_date(scoreboard_payload, date_str)
    if not games:
        print(f"[{date_str}] No games found.")
        return {"date": date_str, "season": season, "gameCount": 0, "games": []}

    aggregated: Dict[str, Any] = {"date": date_str, "season": season, "gameCount": len(games), "games": []}
    stats_calls = 0
    stat_errors = 0

    for game in games:
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})

        for team in (home, away):
            abbrev = (team.get("abbrev") or "").upper()
            if not abbrev or abbrev in roster_cache:
                continue
            try:
                roster_payload = fetch_json(
                    session,
                    ROSTER_URL.format(team=abbrev),
                    timeouts=timeouts,
                    verbose=args.verbose,
                )
            except requests.RequestException as err:
                print(f"[{date_str}] Warning: roster fetch failed for {abbrev}: {err}", file=sys.stderr)
                roster_cache[abbrev] = []
                continue
            roster_cache[abbrev] = flatten_roster(roster_payload)

        game_entry: Dict[str, Any] = {
            "gamePk": game.get("id"),
            "season": game.get("season"),
            "gameType": game.get("gameType"),
            "startTimeUTC": game.get("startTimeUTC"),
            "venue": game.get("venue", {}).get("default"),
            "gameState": game.get("gameState"),
            "gameScheduleState": game.get("gameScheduleState"),
            "homeTeam": {
                "id": home.get("id"),
                "abbrev": home.get("abbrev"),
                "name": home.get("name", {}).get("default"),
                "record": home.get("record"),
                "score": home.get("score"),
                "roster": roster_cache.get((home.get("abbrev") or "").upper(), []),
            },
            "awayTeam": {
                "id": away.get("id"),
                "abbrev": away.get("abbrev"),
                "name": away.get("name", {}).get("default"),
                "record": away.get("record"),
                "score": away.get("score"),
                "roster": roster_cache.get((away.get("abbrev") or "").upper(), []),
            },
        }

        if args.include_stats:
            for side in ("homeTeam", "awayTeam"):
                for player in game_entry[side]["roster"]:
                    pid = player["playerId"]
                    if pid is None:
                        player["seasonStats"] = None
                        continue
                    if pid in stats_cache:
                        player["seasonStats"] = stats_cache[pid]
                        continue
                    if args.max_stats is not None and stats_calls >= args.max_stats:
                        player["seasonStats"] = None
                        continue
                    if args.max_stat_errors is not None and stat_errors >= args.max_stat_errors:
                        player["seasonStats"] = None
                        continue

                    stat_row = fetch_player_stats(
                        session,
                        pid,
                        season,
                        player.get("position"),
                        timeouts=timeouts,
                        verbose=args.verbose,
                    )
                    stats_cache[pid] = stat_row
                    player["seasonStats"] = stat_row
                    stats_calls += 1
                    if stat_row is None:
                        stat_errors += 1

        if args.include_boxscores and game_entry["gamePk"]:
            try:
                boxscore_payload = fetch_game_boxscore(
                    session,
                    game_entry["gamePk"],
                    timeouts=timeouts,
                    verbose=args.verbose,
                )
                game_entry["boxscore"] = {
                    "home": flatten_boxscore_team(boxscore_payload, "home"),
                    "away": flatten_boxscore_team(boxscore_payload, "away"),
                }
            except requests.RequestException as err:
                print(f"[{date_str}] Warning: boxscore fetch failed for game {game_entry['gamePk']}: {err}", file=sys.stderr)

        aggregated["games"].append(game_entry)

    return aggregated


def dump_output(payload: Any, save_path: Optional[str]) -> None:
    if save_path:
        path = Path(save_path).expanduser().resolve()
        path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        print(f"Wrote {path}")
    else:
        json.dump(payload, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")


def main() -> int:
    args = parse_args()
    try:
        date_list = build_date_list(args)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    session = make_session(args)
    roster_cache: Dict[str, List[Dict[str, Any]]] = {}
    stats_cache: Dict[int, Optional[Dict[str, Any]]] = {}
    bundles: List[Dict[str, Any]] = []

    for day in date_list:
        bundles.append(collect_for_date(day, args, session, roster_cache, stats_cache))

    output: Any = bundles[0] if len(bundles) == 1 else bundles
    dump_output(output, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())