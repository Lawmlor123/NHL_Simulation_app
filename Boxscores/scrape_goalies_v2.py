"""
scrape_goalies_v2.py  —  Multi-source NHL starting goalie scraper
-----------------------------------------------------------------
Sources (tried in order, merged by confidence priority):
  1. DailyFaceoff.com  — primary (Next.js JSON, rich stats)
  2. NHL API Gamecenter — secondary (api-web.nhle.com/v1/gamecenter/{id}/landing)
  3. GoaliePost.com     — tertiary (dedicated goalie-tracking site, simple HTML)

Merge rules:
  - Status priority:  Confirmed > Likely > Expected > Probable
                      > Unconfirmed > Fallback > Unknown
  - If two sources AGREE on the goalie name  → upgrade to highest status
  - If two sources DISAGREE on the goalie name → emit ⚠ CONFLICT warning,
    prefer the source with higher status; if tied, prefer DailyFaceoff
  - Stats (GAA, SV%) always come from DailyFaceoff when available

Public API (unchanged from v1 — drop-in replacement):
    games  = scrape_starting_goalies(game_ids_by_matchup=None)
    goalies = get_goalie_dict(games)

game_ids_by_matchup (optional):
    dict keyed by (away_abbrev, home_abbrev) → int game_id
    e.g. {("DET", "TBL"): 2025021001, ...}
    When provided, the NHL API source is enabled.
"""

import requests
import re
import json
import time
import logging
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional

log = logging.getLogger("nhl_mas.goalies")

# ── Status rank (higher = more trustworthy) ───────────────────────────────────
# "historical" is below "unconfirmed" — used for nhl_api_gc (goalieComparison
# season leaders), which identifies the team's best goalie this season, NOT
# necessarily today's starter.  It can fill a TBD slot but cannot override a
# DailyFaceoff name that is merely "Unconfirmed."
STATUS_RANK = {
    "confirmed":   6,
    "likely":      5,
    "expected":    4,
    "probable":    3,
    "unconfirmed": 2,
    "fallback":    1,
    "historical":  0,   # nhl_api_gc season leaders — name-fill only
    "unknown":    -1,
}

# Sources whose names CAN override a DFO name in a conflict
# (nhl_api_gc is intentionally excluded — roster-validate instead)
NAME_OVERRIDE_SOURCES = {"nhl_api_pg", "nhl_api_box", "goaliepost"}

def _rank(status: str) -> int:
    return STATUS_RANK.get(str(status).lower().strip(), -1)


# ── Team name → abbreviation ──────────────────────────────────────────────────
TEAM_ABBREV = {
    'anaheim ducks': 'ANA',
    'boston bruins': 'BOS',
    'buffalo sabres': 'BUF',
    'calgary flames': 'CGY',
    'carolina hurricanes': 'CAR',
    'chicago blackhawks': 'CHI',
    'colorado avalanche': 'COL',
    'columbus blue jackets': 'CBJ',
    'dallas stars': 'DAL',
    'detroit red wings': 'DET',
    'edmonton oilers': 'EDM',
    'florida panthers': 'FLA',
    'los angeles kings': 'LAK',
    'minnesota wild': 'MIN',
    'montreal canadiens': 'MTL',
    'montréal canadiens': 'MTL',
    'nashville predators': 'NSH',
    'new jersey devils': 'NJD',
    'new york islanders': 'NYI',
    'new york rangers': 'NYR',
    'ottawa senators': 'OTT',
    'philadelphia flyers': 'PHI',
    'pittsburgh penguins': 'PIT',
    'san jose sharks': 'SJS',
    'seattle kraken': 'SEA',
    'st. louis blues': 'STL',
    'st louis blues': 'STL',
    'tampa bay lightning': 'TBL',
    'toronto maple leafs': 'TOR',
    'utah hockey club': 'UTA',
    'utah mammoth': 'UTA',
    'vancouver canucks': 'VAN',
    'vegas golden knights': 'VGK',
    'washington capitals': 'WSH',
    'winnipeg jets': 'WPG',
}

# Reverse: abbreviation → canonical full name (for NHL API lookup)
ABBREV_TO_FULL = {v: k.title() for k, v in TEAM_ABBREV.items()}


def get_team_abbrev(team_name: str) -> Optional[str]:
    abbrev = TEAM_ABBREV.get(team_name.lower().strip())
    if abbrev is None:
        for key, val in TEAM_ABBREV.items():
            if key in team_name.lower() or team_name.lower() in key:
                return val
    return abbrev


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — DailyFaceoff
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_dailyfaceoff() -> list[dict]:
    """
    Scrape DailyFaceoff starting goalies page via embedded Next.js JSON.
    Returns list of game dicts (same schema as original scrape_starting_goalies).
    Returns [] on any failure.
    """
    url = "https://www.dailyfaceoff.com/starting-goalies/"
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0.0.0 Safari/537.36'),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Cache-Control': 'no-cache',
    }

    try:
        log.debug("DailyFaceoff: fetching %s", url)
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            log.warning("DailyFaceoff: HTTP %d", response.status_code)
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        next_data_script = soup.find('script', id='__NEXT_DATA__')

        if not next_data_script:
            for script in soup.find_all('script'):
                if script.string and '"pageProps"' in (script.string or ''):
                    if '"homeGoalieName"' in script.string:
                        next_data_script = script
                        break

        if not next_data_script or not next_data_script.string:
            log.warning("DailyFaceoff: could not find __NEXT_DATA__")
            return []

        data       = json.loads(next_data_script.string)
        games_data = data.get('props', {}).get('pageProps', {}).get('data', [])

        if not games_data:
            log.warning("DailyFaceoff: no games in JSON")
            return []

        games = []
        for game in games_data:
            home_team   = game.get('homeTeamName', 'Unknown')
            away_team   = game.get('awayTeamName', 'Unknown')
            home_abbrev = get_team_abbrev(home_team)
            away_abbrev = get_team_abbrev(away_team)

            games.append({
                'source':          'dailyfaceoff',
                'home_team':       home_abbrev,
                'away_team':       away_abbrev,
                'home_team_full':  home_team,
                'away_team_full':  away_team,
                'home_goalie':     game.get('homeGoalieName', 'TBD'),
                'away_goalie':     game.get('awayGoalieName', 'TBD'),
                'home_goalie_id':  game.get('homeGoalieId'),
                'away_goalie_id':  game.get('awayGoalieId'),
                'home_status':     game.get('homeNewsStrengthName') or 'Unconfirmed',
                'away_status':     game.get('awayNewsStrengthName') or 'Unconfirmed',
                'home_news':       game.get('homeNewsDetails', ''),
                'away_news':       game.get('awayNewsDetails', ''),
                'home_goalie_gaa':     game.get('homeGoalieGoalsAgainstAvg'),
                'away_goalie_gaa':     game.get('awayGoalieGoalsAgainstAvg'),
                'home_goalie_svpct':   game.get('homeGoalieSavePercentage'),
                'away_goalie_svpct':   game.get('awayGoalieSavePercentage'),
                'home_goalie_wins':    game.get('homeGoalieWins'),
                'away_goalie_wins':    game.get('awayGoalieWins'),
                'home_goalie_losses':  game.get('homeGoalieLosses'),
                'away_goalie_losses':  game.get('awayGoalieLosses'),
                'home_goalie_shutouts': game.get('homeGoalieShutouts'),
                'away_goalie_shutouts': game.get('awayGoalieShutouts'),
                'home_goalie_rating':  game.get('homeGoalieRating'),
                'away_goalie_rating':  game.get('awayGoalieRating'),
                'home_goalie_rank':    game.get('homeGoaliePositionRank'),
                'away_goalie_rank':    game.get('awayGoaliePositionRank'),
                'game_time':           game.get('time', ''),
                'point_spread':        game.get('pointSpread'),
                'home_moneyline':      game.get('homeTeamMoneylinePointSpread'),
                'away_moneyline':      game.get('awayTeamMoneylinePointSpread'),
            })

        log.info("DailyFaceoff: %d games scraped", len(games))
        return games

    except Exception as e:
        log.warning("DailyFaceoff scrape failed: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — NHL API Gamecenter
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_nhl_api(game_ids_by_matchup: dict) -> dict:
    """
    For each game in game_ids_by_matchup, hit the NHL API gamecenter landing
    endpoint and extract probable/confirmed starters from:
      - matchup.goalieComparison (pre-game projection)
      - matchup.teamLeaders      (sometimes holds starter)
      - gameState / situation    (if game is live/final, actual starters)

    Returns dict: { team_abbrev: {"name": str, "status": str, "source": "nhl_api"} }
    game_ids_by_matchup: { (away_abbrev, home_abbrev): game_id_int }
    """
    results = {}
    if not game_ids_by_matchup:
        return results

    for (away, home), game_id in game_ids_by_matchup.items():
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing"
        try:
            log.debug("NHL API: fetching game %d (%s @ %s)", game_id, away, home)
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                log.debug("NHL API: %d for game %d", r.status_code, game_id)
                time.sleep(0.2)
                continue

            data = r.json()

            # ── Try matchup.goalieComparison ──────────────────────────────
            matchup = data.get("matchup", {})
            gc = matchup.get("goalieComparison", {})

            if gc:
                for side, abbrev in [("homeTeam", home), ("awayTeam", away)]:
                    side_data = gc.get(side, {})
                    # The goalieComparison may have season leaders, not
                    # today's starter — only use if we find explicit fields
                    goalie_list = side_data.get("leaders", [])
                    if goalie_list:
                        g = goalie_list[0]
                        name = (g.get("firstName", {}).get("default", "") + " " +
                                g.get("lastName", {}).get("default", "")).strip()
                        if name:
                            # "Historical" — season leader, NOT today's starter.
                            # Can fill a TBD slot but cannot beat DFO Unconfirmed.
                            results[abbrev] = {
                                "name":   name,
                                "status": "Historical",
                                "source": "nhl_api_gc",
                            }

            # ── Try awayTeam / homeTeam probableGoalie ────────────────────
            for side, abbrev in [("homeTeam", home), ("awayTeam", away)]:
                side_data = data.get(side, {})
                pg = side_data.get("probableGoalie", {})
                if pg:
                    fname = pg.get("firstName", {})
                    lname = pg.get("lastName", {})
                    if isinstance(fname, dict):
                        fname = fname.get("default", "")
                    if isinstance(lname, dict):
                        lname = lname.get("default", "")
                    name = f"{fname} {lname}".strip()
                    if name:
                        # probableGoalie field is more reliable than leaders list
                        results[abbrev] = {
                            "name":   name,
                            "status": "Probable",
                            "source": "nhl_api_pg",
                        }

            # ── If game is LIVE or FINAL, pull actual starting goalie ─────
            game_state = data.get("gameState", "")
            if game_state in ("LIVE", "CRIT", "FINAL", "OFF"):
                boxscore_url = (
                    f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
                )
                bs = requests.get(boxscore_url, timeout=10)
                if bs.status_code == 200:
                    bs_data = bs.json()
                    for side, abbrev in [("homeTeam", home), ("awayTeam", away)]:
                        goalies = (bs_data.get(side, {})
                                         .get("goalies", []))
                        starters = [g for g in goalies
                                    if g.get("starter") or g.get("started")]
                        if not starters:
                            starters = goalies[:1]  # first listed = starter
                        if starters:
                            g = starters[0]
                            fname = g.get("firstName", {})
                            lname = g.get("lastName", {})
                            if isinstance(fname, dict):
                                fname = fname.get("default", "")
                            if isinstance(lname, dict):
                                lname = lname.get("default", "")
                            name = f"{fname} {lname}".strip()
                            if name:
                                results[abbrev] = {
                                    "name":   name,
                                    "status": "Confirmed",
                                    "source": "nhl_api_box",
                                }

            time.sleep(0.25)

        except Exception as e:
            log.debug("NHL API gamecenter failed for game %d: %s", game_id, e)
            time.sleep(0.25)

    log.info("NHL API: goalie data for %d teams", len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ROSTER VALIDATOR — the "agent that makes the call"
# ─────────────────────────────────────────────────────────────────────────────

# Simple cache: { team_abbrev: set_of_goalie_last_names }
_roster_cache: dict = {}

def _get_roster_goalies(team_abbrev: str) -> set:
    """
    Fetch the current NHL roster for team_abbrev and return a set of goalie
    last names (lower-cased) so we can validate conflict candidates.

    Endpoint: api-web.nhle.com/v1/roster/{team}/current
    Returns empty set on any failure (fail-open — don't block the pipeline).
    """
    if team_abbrev in _roster_cache:
        return _roster_cache[team_abbrev]

    url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            _roster_cache[team_abbrev] = set()
            return set()

        data     = r.json()
        goalies  = data.get("goalies", [])
        names    = set()
        for g in goalies:
            ln = g.get("lastName", {})
            if isinstance(ln, dict):
                ln = ln.get("default", "")
            if ln:
                names.add(ln.lower())
        _roster_cache[team_abbrev] = names
        log.debug("Roster [%s]: goalies = %s", team_abbrev, names)
        return names

    except Exception as e:
        log.debug("Roster fetch failed [%s]: %s", team_abbrev, e)
        _roster_cache[team_abbrev] = set()
        return set()


def _resolve_conflict(
    team:        str,
    dfo_name:    str,
    dfo_status:  str,
    sup_name:    str,
    sup_status:  str,
    sup_source:  str,
) -> tuple:
    """
    Intelligent conflict resolver — called when two sources disagree on the
    goalie name.  Decision logic (in order):

    1. Roster check  — if only one candidate is on the team's current roster,
                       that candidate wins regardless of source status.
    2. Status rank   — if both (or neither) are on roster, prefer the source
                       with higher status rank; DFO wins ties.
    3. Name-override guard — nhl_api_gc is never allowed to beat a real DFO name;
                             it can only fill a TBD slot (handled pre-conflict).

    Returns (winner_name, winner_source, winner_status, reasoning: str)
    """
    roster = _get_roster_goalies(team)

    dfo_last = _normalize_name(dfo_name)
    sup_last  = _normalize_name(sup_name)

    dfo_on_roster = (dfo_last in roster) if roster else None
    sup_on_roster = (sup_last in roster) if roster else None

    # ── Case 1: only one is on roster ────────────────────────────────────────
    if roster:
        if dfo_on_roster and not sup_on_roster:
            reason = (f"Roster check: {dfo_name} is on {team} roster, "
                      f"{sup_name} is NOT → DFO wins")
            log.info("  [%s] Conflict resolved by roster: %s ✅ (not %s)",
                     team, dfo_name, sup_name)
            return dfo_name, "dailyfaceoff", dfo_status, reason

        if sup_on_roster and not dfo_on_roster:
            reason = (f"Roster check: {sup_name} is on {team} roster, "
                      f"{dfo_name} is NOT → {sup_source} wins")
            log.info("  [%s] Conflict resolved by roster: %s ✅ (not %s)",
                     team, sup_name, dfo_name)
            return sup_name, sup_source, sup_status, reason

    # ── Case 2: nhl_api_gc cannot beat any real DFO name ─────────────────────
    if sup_source == "nhl_api_gc" and dfo_last:
        reason = f"nhl_api_gc (season leaders) cannot override DFO name '{dfo_name}'"
        log.info("  [%s] nhl_api_gc blocked from overriding DFO name: keeping %s",
                 team, dfo_name)
        return dfo_name, "dailyfaceoff", dfo_status, reason

    # ── Case 3: fall back to status rank ─────────────────────────────────────
    if _rank(sup_status) > _rank(dfo_status):
        reason = f"Status rank: {sup_source}={sup_status} > DFO={dfo_status}"
        return sup_name, sup_source, sup_status, reason
    else:
        reason = f"Status rank: DFO={dfo_status} >= {sup_source}={sup_status} → DFO wins"
        return dfo_name, "dailyfaceoff", dfo_status, reason


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — GoaliePost.com
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_goaliepost() -> dict:
    """
    Scrape goaliepost.com for starting goalie confirmations.
    Returns dict: { team_abbrev: {"name": str, "status": str, "source": "goaliepost"} }
    """
    url = "https://goaliepost.com/"
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0.0.0 Safari/537.36'),
        'Accept': 'text/html,application/xhtml+xml,*/*;q=0.9',
    }

    results = {}
    try:
        log.debug("GoaliePost: fetching %s", url)
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            log.warning("GoaliePost: HTTP %d", r.status_code)
            return results

        soup = BeautifulSoup(r.text, 'html.parser')

        # Strategy A: embedded JSON with "goalies" or "starters" key
        for script in soup.find_all('script'):
            text = script.string or ''
            if '"goalies"' in text or '"starters"' in text:
                try:
                    jdata = json.loads(text)
                    goalies_list = jdata.get('goalies') or jdata.get('starters') or []
                    for entry in goalies_list:
                        team   = entry.get('team') or entry.get('teamAbbrev', '')
                        name   = entry.get('goalie') or entry.get('name', '')
                        status = entry.get('status', 'Unconfirmed')
                        abbrev = get_team_abbrev(team) or team.upper()[:3]
                        if abbrev and name:
                            results[abbrev] = {
                                "name":   name,
                                "status": status.capitalize(),
                                "source": "goaliepost",
                            }
                    if results:
                        log.info("GoaliePost (JSON): %d goalies", len(results))
                        return results
                except Exception:
                    pass

        # Strategy B: HTML table rows
        rows = soup.select('table tr, .goalie-row, .starter-row, [data-goalie]')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                team_cell   = cells[0].get_text(strip=True)
                goalie_cell = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                status_cell = cells[2].get_text(strip=True) if len(cells) > 2 else 'Unconfirmed'
                abbrev = get_team_abbrev(team_cell) or (
                    team_cell.upper()[:3] if len(team_cell) <= 3 else None
                )
                if abbrev and goalie_cell and len(goalie_cell) > 3:
                    results[abbrev] = {
                        "name":   goalie_cell,
                        "status": status_cell.capitalize() if status_cell else 'Unconfirmed',
                        "source": "goaliepost",
                    }

        # Strategy C: regex over text nodes looking for confirmed/likely patterns
        if not results:
            text_blocks = soup.find_all(string=re.compile(
                r'\b(confirmed|likely|expected|probable|unconfirmed)\b',
                re.IGNORECASE
            ))
            for block in text_blocks:
                parent = block.parent
                if parent:
                    full_text = parent.get_text(' ', strip=True)
                    m = re.search(
                        r'\b([A-Z]{2,3})\b.*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
                        r'.*?\b(confirmed|likely|expected|probable|unconfirmed)\b',
                        full_text, re.IGNORECASE
                    )
                    if m:
                        abbrev = m.group(1).upper()
                        name   = m.group(2).strip()
                        status = m.group(3).capitalize()
                        if abbrev in ABBREV_TO_FULL or abbrev in TEAM_ABBREV.values():
                            results[abbrev] = {
                                "name":   name,
                                "status": status,
                                "source": "goaliepost",
                            }

        log.info("GoaliePost: %d goalies parsed", len(results))
        return results

    except Exception as e:
        log.warning("GoaliePost scrape failed: %s", e)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# MERGE — reconcile sources
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Lower-case last name for fuzzy comparison."""
    if not name:
        return ""
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def _merge_goalie_sources(dfo_games, nhl_api, goaliepost):
    """
    Merge DailyFaceoff games (primary) with supplemental sources.

    For each team in each game:
    - If names MATCH  → upgrade status to highest available
    - If names DIFFER → ⚠ CONFLICT warning, prefer higher-ranked status
    - Stats (GAA, SV%) always kept from DailyFaceoff
    """
    supplemental = {}
    for src_dict in (nhl_api, goaliepost):
        for team, info in src_dict.items():
            existing = supplemental.get(team)
            if existing is None or _rank(info["status"]) > _rank(existing["status"]):
                supplemental[team] = info

    merged = []
    for game in dfo_games:
        g = dict(game)

        for side in ("home", "away"):
            team       = g.get(f"{side}_team")
            dfo_name   = g.get(f"{side}_goalie") or ""
            dfo_status = g.get(f"{side}_status") or "Unconfirmed"

            sup = supplemental.get(team)
            if not sup:
                continue

            sup_name   = sup["name"]
            sup_status = sup["status"]
            sup_source = sup["source"]

            dfo_last = _normalize_name(dfo_name)
            sup_last  = _normalize_name(sup_name)

            if not dfo_last or not sup_last:
                if sup_last and not dfo_last:
                    g[f"{side}_goalie"] = sup_name
                    g[f"{side}_status"] = sup_status
                    log.info("  [%s] no DFO name → using %s from %s (%s)",
                             team, sup_name, sup_source, sup_status)
            elif dfo_last == sup_last:
                if _rank(sup_status) > _rank(dfo_status):
                    log.info("  [%s] %s upgraded: %s → %s (via %s)",
                             team, dfo_name, dfo_status, sup_status, sup_source)
                    g[f"{side}_status"] = sup_status
            else:
                # ── Names conflict → run the roster-validation resolver ────
                # nhl_api_gc returns season leaders (historical), not today's
                # starter.  When DFO already has a real name, nhl_api_gc can
                # never win — skip conflict resolution entirely so DFO's
                # original status is preserved untouched (no "Conflict-" wrap).
                if sup_source == "nhl_api_gc":
                    log.debug(
                        "  [%s] nhl_api_gc name '%s' ignored — DFO '%s' (%s) "
                        "takes precedence (historical data cannot override)",
                        team, sup_name, dfo_name, dfo_status,
                    )
                    continue

                winner, winner_src, winner_stat, reason = _resolve_conflict(
                    team, dfo_name, dfo_status, sup_name, sup_status, sup_source
                )

                log.warning(
                    "  ⚠ GOALIE CONFLICT [%s]: DFO=%s (%s) vs %s=%s (%s) "
                    "→ %s  [%s]",
                    team, dfo_name, dfo_status,
                    sup_source, sup_name, sup_status,
                    winner, reason,
                )
                print(
                    f"\n  ⚠️  GOALIE CONFLICT [{team}]: "
                    f"DFO={dfo_name!r} ({dfo_status})  vs  "
                    f"{sup_source}={sup_name!r} ({sup_status})"
                    f"\n      → {winner!r} from {winner_src}  [{reason}]"
                )
                # Use "Roster-Confirmed" when roster check settled it cleanly,
                # otherwise keep the Conflict- prefix so downstream sees it
                if "Roster check" in reason:
                    resolved_status = winner_stat.replace("Conflict-", "")
                    resolved_status = (
                        "Roster-Confirmed" if _rank(resolved_status) >= _rank("Probable")
                        else resolved_status
                    )
                else:
                    resolved_status = f"Conflict-{winner_stat}"

                g[f"{side}_goalie"] = winner
                g[f"{side}_status"] = resolved_status

        merged.append(g)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4 — RotoWire
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_rotowire() -> dict:
    """
    Scrape RotoWire's NHL starting goalies page.
    Returns dict: { team_abbrev: {"name": str, "status": str, "source": "rotowire"} }
    """
    url = "https://www.rotowire.com/hockey/starting-goalies.php"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/html,*/*;q=0.9",
        "Referer": "https://www.rotowire.com/",
    }
    results = {}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            log.warning("RotoWire: HTTP %d", r.status_code)
            return results

        soup = BeautifulSoup(r.text, "html.parser")

        # RotoWire renders goalie cards with class "starting-goalies__player"
        # or "is-confirmed" / "is-probable" indicators
        status_map = {
            "confirmed":  "Confirmed",
            "probable":   "Probable",
            "expected":   "Expected",
            "likely":     "Likely",
            "questionable": "Unconfirmed",
        }

        # Strategy A: JSON-LD or embedded JS data
        for script in soup.find_all("script"):
            text = script.string or ""
            if "startingGoalie" in text or "goalie" in text.lower():
                try:
                    # look for JSON block
                    m = re.search(r'(\{["\']goalie["\'].*?\})', text, re.DOTALL)
                    if m:
                        jd = json.loads(m.group(1))
                        team   = jd.get("team", "")
                        name   = jd.get("name") or jd.get("goalie", "")
                        status = jd.get("status", "Unconfirmed")
                        abbrev = get_team_abbrev(team) or team.upper()[:3]
                        if abbrev and name:
                            results[abbrev] = {
                                "name":   name,
                                "status": status_map.get(status.lower(), status.capitalize()),
                                "source": "rotowire",
                            }
                except Exception:
                    pass

        # Strategy B: HTML player cards
        for card in soup.select(".starting-goalies__player, .rw-starting-goalies-card"):
            name_el   = card.select_one(".starting-goalies__name, .rw-player-name")
            team_el   = card.select_one(".starting-goalies__team, .rw-team-abbrev")
            status_el = card.select_one(".starting-goalies__status, .rw-status")
            if not (name_el and team_el):
                continue
            name   = name_el.get_text(strip=True)
            team   = team_el.get_text(strip=True)
            status = status_el.get_text(strip=True) if status_el else "Unconfirmed"
            abbrev = get_team_abbrev(team) or (team.upper()[:3] if len(team) <= 3 else None)
            if abbrev and name and len(name) > 3:
                results[abbrev] = {
                    "name":   name,
                    "status": status_map.get(status.lower(), "Unconfirmed"),
                    "source": "rotowire",
                }

        # Strategy C: table rows
        if not results:
            for row in soup.select("table tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    team_txt   = cells[0].get_text(strip=True)
                    goalie_txt = cells[1].get_text(strip=True)
                    status_txt = cells[2].get_text(strip=True) if len(cells) > 2 else "Unconfirmed"
                    abbrev     = get_team_abbrev(team_txt)
                    if abbrev and len(goalie_txt) > 4:
                        results[abbrev] = {
                            "name":   goalie_txt,
                            "status": status_map.get(status_txt.lower(), "Unconfirmed"),
                            "source": "rotowire",
                        }

        log.info("RotoWire: %d goalies parsed", len(results))
        return results

    except Exception as e:
        log.warning("RotoWire scrape failed: %s", e)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 5 — GameDayTweets (morning skate beat reporter aggregator)
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_gamedaytweets() -> dict:
    """
    Scrape gamedaytweets.com/goalies — aggregates beat reporter tweets
    from NHL morning skates (~10-11am ET, 4-5h before puck drop).
    Returns dict: { team_abbrev: {"name": str, "status": str, "source": "gamedaytweets"} }
    """
    url = "https://www.gamedaytweets.com/goalies"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/html,*/*;q=0.9",
    }
    results = {}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            log.warning("GameDayTweets: HTTP %d", r.status_code)
            return results

        soup = BeautifulSoup(r.text, "html.parser")

        # Status keywords to detect from tweet text
        _confirmed_kws = ["will start", "gets the start", "confirmed starter",
                          "starting tonight", "confirmed to start", "in goal tonight"]
        _likely_kws    = ["expected to start", "likely starter", "morning skate",
                          "took the skate", "practice goalie", "looks like"]

        def _detect_status(text: str) -> str:
            tl = text.lower()
            if any(k in tl for k in _confirmed_kws):
                return "Confirmed"
            if any(k in tl for k in _likely_kws):
                return "Likely"
            return "Unconfirmed"

        # Strategy A: structured goalie cards
        for card in soup.select(".goalie-card, .goalie-row, [data-team], .starter-card"):
            team_el  = (card.get("data-team") or
                        (card.select_one(".team-abbrev, .team") or {}).get_text(strip=True)
                        if hasattr(card.select_one(".team-abbrev, .team"), "get_text") else "")
            name_el  = card.select_one(".goalie-name, .player-name, h3, h4")
            tweet_el = card.select_one(".tweet-text, .tweet, p")

            name   = name_el.get_text(strip=True) if name_el else ""
            tweet  = tweet_el.get_text(" ", strip=True) if tweet_el else ""
            abbrev = get_team_abbrev(str(team_el)) if team_el else None

            if not abbrev:
                # try to find team abbrev in the card text
                card_text = card.get_text(" ", strip=True)
                for abbr in ABBREV_TO_FULL:
                    if re.search(r'\b' + abbr + r'\b', card_text):
                        abbrev = abbr
                        break

            status = _detect_status(tweet or card.get_text(" ", strip=True))

            if abbrev and name and len(name) > 4:
                results[abbrev] = {
                    "name":   name,
                    "status": status,
                    "source": "gamedaytweets",
                }

        # Strategy B: regex over full page text for "X will start for TEAM"
        if not results:
            page_text = soup.get_text(" ", strip=True)
            patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+)\s+(?:will start|gets the start|confirmed starter)\s+(?:for\s+)?([A-Z]{2,3})',
                r'([A-Z]{2,3})\s+goalie[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',
            ]
            for pat in patterns:
                for m in re.finditer(pat, page_text):
                    g1, g2 = m.group(1), m.group(2)
                    # figure out which is team vs name
                    if len(g1) <= 3 and g1.isupper():
                        abbrev, name = g1, g2
                    elif len(g2) <= 3 and g2.isupper():
                        abbrev, name = g2, g1
                    else:
                        continue
                    abbrev = abbrev.upper()
                    if abbrev in ABBREV_TO_FULL and len(name) > 4:
                        results[abbrev] = {
                            "name":   name,
                            "status": "Confirmed",
                            "source": "gamedaytweets",
                        }

        log.info("GameDayTweets: %d goalies parsed", len(results))
        return results

    except Exception as e:
        log.warning("GameDayTweets scrape failed: %s", e)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 6 — NHLFantasyData (updates every 5 minutes)
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_nhlfantasydata() -> dict:
    """
    Scrape nhlfantasydata.com — lightweight goalie tracker, refreshes every 5 min.
    Returns dict: { team_abbrev: {"name": str, "status": str, "source": "nhlfantasydata"} }
    """
    urls = [
        "https://nhlfantasydata.com/goalies.html",
        "https://nhlfantasydata.com/",
        "https://nhlfantasydata.com/tomorrow.html",
    ]
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/html,*/*;q=0.9",
    }
    results = {}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            status_kw_map = {
                "confirmed": "Confirmed", "likely": "Likely",
                "probable": "Probable",  "expected": "Expected",
                "unconfirmed": "Unconfirmed",
            }

            # Strategy A: table rows (site uses simple HTML tables)
            for row in soup.select("table tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                texts = [c.get_text(strip=True) for c in cells]
                # look for team + goalie name pattern
                abbrev = None
                name   = None
                status = "Unconfirmed"
                for i, txt in enumerate(texts):
                    a = get_team_abbrev(txt) or (txt.upper()[:3] if txt.isupper() and len(txt) <= 3 else None)
                    if a and a in ABBREV_TO_FULL:
                        abbrev = a
                    # goalie name: two-word capitalized string
                    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', txt) and len(txt.split()) >= 2:
                        name = txt
                    # status keyword
                    for kw, mapped in status_kw_map.items():
                        if kw in txt.lower():
                            status = mapped
                            break
                if abbrev and name:
                    results[abbrev] = {
                        "name": name, "status": status, "source": "nhlfantasydata"
                    }

            # Strategy B: free-text pattern matching
            if not results:
                page_text = soup.get_text(" ", strip=True)
                for m in re.finditer(
                    r'\b([A-Z]{2,3})\b[^.]{0,30}?([A-Z][a-z]+\s+[A-Z][a-z]+)'
                    r'[^.]{0,40}?\b(confirmed|likely|probable|expected|unconfirmed)\b',
                    page_text, re.IGNORECASE
                ):
                    abbrev = m.group(1).upper()
                    name   = m.group(2).strip()
                    status = status_kw_map.get(m.group(3).lower(), "Unconfirmed")
                    if abbrev in ABBREV_TO_FULL and len(name) > 4:
                        results[abbrev] = {
                            "name": name, "status": status, "source": "nhlfantasydata"
                        }

            if results:
                log.info("NHLFantasyData (%s): %d goalies parsed", url, len(results))
                return results

        except Exception as e:
            log.debug("NHLFantasyData %s failed: %s", url, e)

    log.info("NHLFantasyData: 0 goalies parsed")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 7 — Inference Engine (fills remaining TBD / Unconfirmed slots)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_inference(merged_games: list, game_ids_by_matchup: dict = None) -> list:
    """
    After all external sources are merged, apply the GoalieInferenceEngine
    to any team that still has an Unconfirmed, Fallback, or TBD goalie.
    Adds status = "Inferred" (rank 3 — between Probable and Unconfirmed).

    game_ids_by_matchup is used only to detect B2B: not needed for inference
    itself, but the B2B flag comes from the scrape pipeline's rest-day data.
    """
    try:
        from goalie_inference import get_inferred_goalie
    except ImportError:
        log.warning("goalie_inference.py not found — inference skipped")
        return merged_games

    from datetime import date
    today = str(date.today())

    # Build a quick B2B lookup from the game list itself if available
    # (module1_ingest passes rest_days; here we just use the DFO game data)
    b2b_teams: set = set()
    # We can't determine B2B from inside the scraper without rest data,
    # so we set is_b2b=False conservatively (better to infer starter than backup
    # when we don't know the schedule).  module1_ingest will override if needed.

    FILL_STATUSES = {"unconfirmed", "fallback", "unknown", "tbd", ""}

    updated = 0
    for game in merged_games:
        for side in ("home", "away"):
            team       = game.get(f"{side}_team", "")
            cur_name   = game.get(f"{side}_goalie", "") or ""
            cur_status = str(game.get(f"{side}_status", "")).lower().strip()

            if cur_status not in FILL_STATUSES:
                continue   # already has a good status — leave it alone

            is_b2b = team in b2b_teams
            result = get_inferred_goalie(
                team           = team,
                game_date      = today,
                is_b2b         = is_b2b,
                current_dfo_name = cur_name,
            )

            if result is None:
                continue

            inf_name = result["name"]
            inf_conf = result["confidence"]

            # Only apply if inference confidence is meaningful
            if inf_conf < 0.55:
                log.debug("  [%s] Inference confidence too low (%.0f%%) — skipping",
                          team, inf_conf * 100)
                continue

            # If DFO already has a name, only override if inference agrees
            # (cross-validation already done inside get_inferred_goalie)
            game[f"{side}_goalie"]  = inf_name
            game[f"{side}_status"]  = "Inferred"
            updated += 1
            log.info("  [%s] Inferred starter: %s (conf=%.0f%%) — %s",
                     team, inf_name, inf_conf * 100, result["reason"])

    print(f"  [ Goalie Scraper ] Inference: {updated} slots filled with 🔮 Inferred")
    return merged_games


# Add "inferred" to STATUS_RANK if not already there
STATUS_RANK.setdefault("inferred", 3)   # between probable(3) and unconfirmed(2)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def scrape_starting_goalies(game_ids_by_matchup=None):
    """
    Main entry point — drop-in replacement for the original scrape_starting_goalies().

    Parameters
    ----------
    game_ids_by_matchup : dict, optional
        { (away_abbrev, home_abbrev): game_id_int }
        Enables the NHL API source when supplied.

    Returns
    -------
    list[dict]  — one dict per game, same schema as before.
    """
    print("  [ Goalie Scraper ] Trying DailyFaceoff ...")
    dfo_games = _scrape_dailyfaceoff()

    if not dfo_games:
        print("  [ Goalie Scraper ] DailyFaceoff returned no data — aborting")
        return []

    # Source 2: NHL API (only when game_ids provided)
    nhl_api_data = {}
    if game_ids_by_matchup:
        print("  [ Goalie Scraper ] Trying NHL API gamecenter ...")
        nhl_api_data = _scrape_nhl_api(game_ids_by_matchup)
        n_prob = sum(1 for v in nhl_api_data.values()
                     if _rank(v.get("status", "")) >= _rank("Probable"))
        print(f"  [ Goalie Scraper ] NHL API: {len(nhl_api_data)} teams "
              f"({n_prob} Probable/Confirmed)")
    else:
        print("  [ Goalie Scraper ] NHL API: skipped (no game_ids provided)")

    # Source 3: GoaliePost
    print("  [ Goalie Scraper ] Trying GoaliePost.com ...")
    gp_data = _scrape_goaliepost()
    n_gp = sum(1 for v in gp_data.values()
               if _rank(v.get("status", "")) >= _rank("Likely"))
    print(f"  [ Goalie Scraper ] GoaliePost: {len(gp_data)} teams ({n_gp} Likely+)")

    # Source 4: RotoWire starting goalies
    print("  [ Goalie Scraper ] Trying RotoWire ...")
    rw_data = _scrape_rotowire()
    n_rw = sum(1 for v in rw_data.values()
               if _rank(v.get("status", "")) >= _rank("Likely"))
    print(f"  [ Goalie Scraper ] RotoWire: {len(rw_data)} teams ({n_rw} Likely+)")

    # Source 5: GameDayTweets (morning skate aggregator — beat reporters)
    print("  [ Goalie Scraper ] Trying GameDayTweets ...")
    gdt_data = _scrape_gamedaytweets()
    n_gdt = sum(1 for v in gdt_data.values()
                if _rank(v.get("status", "")) >= _rank("Likely"))
    print(f"  [ Goalie Scraper ] GameDayTweets: {len(gdt_data)} teams ({n_gdt} Likely+)")

    # Source 6: NHLFantasyData (updates every 5 min)
    print("  [ Goalie Scraper ] Trying NHLFantasyData ...")
    nfd_data = _scrape_nhlfantasydata()
    n_nfd = sum(1 for v in nfd_data.values()
                if _rank(v.get("status", "")) >= _rank("Likely"))
    print(f"  [ Goalie Scraper ] NHLFantasyData: {len(nfd_data)} teams ({n_nfd} Likely+)")

    # Merge all external sources
    merged = _merge_goalie_sources(
        dfo_games,
        nhl_api_data,
        {**gp_data, **rw_data, **gdt_data, **nfd_data},   # supplemental pool
    )

    # Source 7: Inference Engine — fill any remaining Unconfirmed/TBD slots
    print("  [ Goalie Scraper ] Running Inference Engine ...")
    merged = _apply_inference(merged, game_ids_by_matchup)

    # Summary
    all_statuses = []
    for g in merged:
        all_statuses.extend([g.get("home_status", ""), g.get("away_status", "")])

    n_confirmed = sum(1 for s in all_statuses if _rank(s) >= _rank("Confirmed"))
    n_likely    = sum(1 for s in all_statuses
                      if _rank(s) in (_rank("Likely"), _rank("Expected"), _rank("Probable")))
    n_inferred  = sum(1 for s in all_statuses if str(s).lower() == "inferred")
    n_conflict  = sum(1 for s in all_statuses if "conflict" in str(s).lower())
    total       = len(all_statuses)

    print(f"\n  [ Goalie Scraper ] Final: {len(merged)} games | "
          f"✅ {n_confirmed} Confirmed  🟡 {n_likely} Likely/Probable  "
          f"🔮 {n_inferred} Inferred  "
          f"⚠️  {n_conflict} Conflicts  "
          f"❓ {total - n_confirmed - n_likely - n_inferred - n_conflict} Unconfirmed")

    return merged


def get_goalie_dict(games):
    """Return {team_abbrev: {name, status, gaa, svpct, ...}} — unchanged API."""
    goalies = {}
    for game in games:
        if game.get('home_team'):
            goalies[game['home_team']] = {
                'name':   game['home_goalie'],
                'status': game['home_status'],
                'gaa':    game.get('home_goalie_gaa'),
                'svpct':  game.get('home_goalie_svpct'),
                'wins':   game.get('home_goalie_wins'),
                'losses': game.get('home_goalie_losses'),
                'rating': game.get('home_goalie_rating'),
                'rank':   game.get('home_goalie_rank'),
            }
        if game.get('away_team'):
            goalies[game['away_team']] = {
                'name':   game['away_goalie'],
                'status': game['away_status'],
                'gaa':    game.get('away_goalie_gaa'),
                'svpct':  game.get('away_goalie_svpct'),
                'wins':   game.get('away_goalie_wins'),
                'losses': game.get('away_goalie_losses'),
                'rating': game.get('away_goalie_rating'),
                'rank':   game.get('away_goalie_rank'),
            }
    return goalies


def display_goalies(games):
    """Pretty-print goalie matchups — same signature as original."""
    print("\n" + "=" * 78)
    print(f"  NHL STARTING GOALIES — {datetime.now().strftime('%B %d, %Y')}")
    print(f"  Sources: DailyFaceoff · NHL API · GoaliePost")
    print("=" * 78)

    confirmed_count = 0
    likely_count    = 0
    conflict_count  = 0

    def status_icon(status):
        s = str(status).lower()
        if s == 'confirmed':                            return '✅'
        if s in ('likely', 'expected', 'probable'):     return '🟡'
        if 'conflict' in s:                             return '⚠️ '
        return '❓'

    def goalie_line(name, status, gaa, svpct, w, l, rating, rank):
        parts = f"{name:<26} [{status}]"
        stats = []
        try:
            if gaa   is not None: stats.append(f"GAA:{float(gaa):.2f}")
            if svpct is not None: stats.append(f"SV%:{float(svpct):.3f}")
            if w is not None and l is not None: stats.append(f"W-L:{w}-{l}")
            if rating is not None: stats.append(f"Rtg:{float(rating):.1f}")
            if rank   is not None: stats.append(f"#{rank}")
        except (ValueError, TypeError):
            pass
        return parts + ("  " + " | ".join(stats) if stats else "")

    for i, game in enumerate(games, 1):
        away        = game.get('away_team', '???')
        home        = game.get('home_team', '???')
        away_goalie = game.get('away_goalie', 'TBD')
        home_goalie = game.get('home_goalie', 'TBD')
        away_status = str(game.get('away_status', 'Unknown')).strip()
        home_status = str(game.get('home_status', 'Unknown')).strip()

        away_icon = status_icon(away_status)
        home_icon = status_icon(home_status)

        if 'conflict' in away_status.lower(): conflict_count += 1
        if 'conflict' in home_status.lower(): conflict_count += 1
        if away_status.lower() == 'confirmed':                          confirmed_count += 1
        elif away_status.lower() in ('likely', 'expected', 'probable'): likely_count    += 1
        if home_status.lower() == 'confirmed':                          confirmed_count += 1
        elif home_status.lower() in ('likely', 'expected', 'probable'): likely_count    += 1

        game_time = game.get('game_time', '')
        away_ml   = game.get('away_moneyline', '')
        home_ml   = game.get('home_moneyline', '')
        odds_parts = []
        if away_ml: odds_parts.append(f"{away} ML:{away_ml}")
        if home_ml: odds_parts.append(f"{home} ML:{home_ml}")
        odds_str = f"  ({' | '.join(odds_parts)})" if odds_parts else ""

        print(f"\n  Game {i}: {away} @ {home}  [{game_time} ET]{odds_str}")
        print(f"    {away_icon} {away:>3}  " + goalie_line(
            away_goalie, away_status,
            game.get('away_goalie_gaa'), game.get('away_goalie_svpct'),
            game.get('away_goalie_wins'), game.get('away_goalie_losses'),
            game.get('away_goalie_rating'), game.get('away_goalie_rank')))
        print(f"    {home_icon} {home:>3}  " + goalie_line(
            home_goalie, home_status,
            game.get('home_goalie_gaa'), game.get('home_goalie_svpct'),
            game.get('home_goalie_wins'), game.get('home_goalie_losses'),
            game.get('home_goalie_rating'), game.get('home_goalie_rank')))

    total = len(games) * 2
    print(f"\n{'=' * 78}")
    print(f"  ✅ Confirmed:        {confirmed_count}/{total}")
    print(f"  🟡 Likely/Probable:  {likely_count}/{total}")
    print(f"  ⚠️  Conflicts:        {conflict_count}/{total}")
    print(f"  ❓ Unconfirmed:      {total - confirmed_count - likely_count - conflict_count}/{total}")
    print(f"  Games: {len(games)}")
    print(f"{'=' * 78}")
    return games


# ─────────────────────────────────────────────────────────────────────────────
# CLI / standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    print("=" * 78)
    print("  DailyFaceoff + NHL API + GoaliePost  Multi-Source Scraper")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    games = scrape_starting_goalies()

    if games:
        display_goalies(games)
        goalie_dict = get_goalie_dict(games)
        print(f"\n Quick reference ({len(goalie_dict)} teams):")
        for team, info in sorted(goalie_dict.items()):
            s = str(info['status']).lower()
            icon = ('OK' if s in ('confirmed','roster-confirmed') else
                    'LK' if s in ('likely','expected','probable') else
                    'CF' if 'conflict' in s else '??')
            gaa   = f" GAA:{float(info['gaa']):.2f}"   if info.get('gaa')   else ""
            svpct = f" SV%:{float(info['svpct']):.3f}" if info.get('svpct') else ""
            print(f"   [{icon}] {team}: {info['name']:<26} {info['status']}{gaa}{svpct}")
    else:
        print("\n  Failed to scrape goalies from all sources")
        sys.exit(1)
