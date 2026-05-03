# ============================================================
# SECTION 3: SEASON SIMULATION
# ============================================================

import csv
from datetime import datetime

# Section 2 (simulation engine)
from Sec2_Simengine import simulate_result

def parse_date(date_str):
    return datetime.strptime(date_str, "%m/%d/%Y").date()

# --- Load master schedule ---
def load_master_schedule(csv_file):
    schedule_by_date = {}
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        cleaned = [h.lower().replace("\ufeff","").replace("ï»¿","").strip() for h in reader.fieldnames]
        field_map = dict(zip(cleaned, reader.fieldnames))
        for row in reader:
            date_str = row[field_map["date"]].strip()
            visitor = row[field_map["visitor"]].strip()
            home = row[field_map["home"]].strip()
            date = parse_date(date_str)
            if date not in schedule_by_date:
                schedule_by_date[date] = []
            schedule_by_date[date].append((home, visitor))
    return schedule_by_date

# --- Streak updater ---
def update_streak(team, outcome, streak_state, season_streaks):
    curr_type = streak_state[team]["current_type"]
    curr_len = streak_state[team]["length"]
    if curr_type == outcome:
        streak_state[team]["length"] += 1
    else:
        if curr_type in ("W","L","OT") and curr_len > 0:
            # record this streak length
            season_streaks[team][curr_type].append(curr_len)
            # update maxima
            if curr_type == "W":
                season_streaks[team]["maxW"] = max(season_streaks[team]["maxW"], curr_len)
            elif curr_type == "L":
                season_streaks[team]["maxL"] = max(season_streaks[team]["maxL"], curr_len)
            elif curr_type == "OT":
                season_streaks[team]["maxOT"] = max(season_streaks[team]["maxOT"], curr_len)
        streak_state[team]["current_type"] = outcome
        streak_state[team]["length"] = 1

# --- Full League Simulation ---
def simulate_full_league(schedule_by_date, verbose=False, track_shots=False):
    # lazy imports to avoid circular deps
    from Sec5_endstats import team_stats, team_rosters, goalies
    from Sec1_Core_Inj import (
        injuries,
        season_injury_impact,
        update_injuries,
        choose_goalie,
    )

    global season_injury_impact
    season_injury_impact = {}
    standings = {team: {"W":0,"L":0,"OT":0,"PTS":0} for team in team_stats.keys()}
    season_stats = {team: {"GF":0,"GA":0,"SF":0,"SA":0} for team in team_stats.keys()}
    season_logs = {}
    # 🔽 streak lists (for distribution) + maxima (for legacy)
    season_streaks = {team: {"W":[],"L":[],"OT":[],"maxW":0,"maxL":0,"maxOT":0} for team in team_stats.keys()}
    season_notes = {}
    streak_state = {team: {"current_type":None,"length":0} for team in team_stats.keys()}
    game_history = {}
    injuries.clear()

    for date in sorted(schedule_by_date.keys()):
        for home, visitor in schedule_by_date[date]:
            # Ensure defaults exist
            for t in (home, visitor):
                if t not in team_stats:
                    team_stats[t] = {"GF":3.0,"GA":3.0,"PP":20.0,"PK":80.0}
                if t not in goalies:
                    goalies[t] = {
                        "starter":{"SV":0.905,"name":"Generic Starter"},
                        "backup":{"SV":0.895,"name":"Generic Backup"},
                    }
                if t not in team_rosters:
                    team_rosters[t] = []
                if t not in standings:
                    standings[t] = {"W":0,"L":0,"OT":0,"PTS":0}
                if t not in season_stats:
                    season_stats[t] = {"GF":0,"GA":0,"SF":0,"SA":0}
                if t not in season_streaks:
                    season_streaks[t] = {"W":[],"L":[],"OT":[],"maxW":0,"maxL":0,"maxOT":0}
                if t not in streak_state:
                    streak_state[t] = {"current_type":None,"length":0}

            # Injuries + goalie choice
            update_injuries(home)
            update_injuries(visitor)
            h_g = choose_goalie(home, date, game_history)
            a_g = choose_goalie(visitor, date, game_history)

            # Play the game
            result, hs, vs, hshots, ashots, log, note = simulate_result(
                home, visitor, h_g, a_g, date, game_history
            )
            season_notes[(date,home,visitor)] = note

            # Add to history
            game_history.setdefault(home, []).append(date)
            game_history.setdefault(visitor, []).append(date)

            # Update standings and streaks
            if result == "H":
                standings[home]["W"] += 1; standings[home]["PTS"] += 2
                standings[visitor]["L"] += 1
                update_streak(home,"W",streak_state,season_streaks)
                update_streak(visitor,"L",streak_state,season_streaks)
            elif result == "A":
                standings[visitor]["W"] += 1; standings[visitor]["PTS"] += 2
                standings[home]["L"] += 1
                update_streak(visitor,"W",streak_state,season_streaks)
                update_streak(home,"L",streak_state,season_streaks)
            elif result == "OTW":
                standings[home]["W"] += 1; standings[home]["PTS"] += 2
                standings[visitor]["OT"] += 1; standings[visitor]["PTS"] += 1
                update_streak(home,"W",streak_state,season_streaks)
                update_streak(visitor,"OT",streak_state,season_streaks)
            else:  # OTL
                standings[visitor]["W"] += 1; standings[visitor]["PTS"] += 2
                standings[home]["OT"] += 1; standings[home]["PTS"] += 1
                update_streak(visitor,"W",streak_state,season_streaks)
                update_streak(home,"OT",streak_state,season_streaks)

            # Shot/goal stats
            if track_shots:
                season_stats[home]["GF"] += hs
                season_stats[home]["GA"] += vs
                season_stats[home]["SF"] += hshots
                season_stats[home]["SA"] += ashots
                season_stats[visitor]["GF"] += vs
                season_stats[visitor]["GA"] += hs
                season_stats[visitor]["SF"] += ashots
                season_stats[visitor]["SA"] += hshots
                season_logs[(date,home,visitor)] = log

            if verbose:
                print(f"{date}: {home} vs {visitor} → {hs}-{vs} ({result}), Shots {hshots}-{ashots}")

    # finalize streaks
    for team in streak_state:
        curr_type = streak_state[team]["current_type"]
        curr_len = streak_state[team]["length"]
        if curr_type and curr_len>0:
            season_streaks[team][curr_type].append(curr_len)
            if curr_type=="W":
                season_streaks[team]["maxW"] = max(season_streaks[team]["maxW"], curr_len)
            elif curr_type=="L":
                season_streaks[team]["maxL"] = max(season_streaks[team]["maxL"], curr_len)
            elif curr_type=="OT":
                season_streaks[team]["maxOT"] = max(season_streaks[team]["maxOT"], curr_len)

    # 🔽 return includes streak arrays + maxima
    return (standings, season_stats, season_logs, season_streaks, season_notes) if track_shots else standings

# --- explicit exports ---
__all__ = ["simulate_full_league", "load_master_schedule"]

# ========== END OF SECTION 3 =================================