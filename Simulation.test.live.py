import numpy as np
import csv
import random
import datetime

# --- Constants ---
HOME_ADVANTAGE = 0.25
LEAGUE_AVG_SV = 0.905
SHOTS_PER_GAME = 30
BACKUP_RANDOM_CHANCE = 0.30
INJURY_PROB = 0.02
INJURY_LENGTH_RANGE = (3, 10)

# --- Global injury tracking ---
injuries = {}
season_injury_impact = {}

# --- xg model ---
def predict_xg(features):
    """
    Simplified expected goals (xg) stub until a trained model is used.
    """
    base = max(0.01, (60 - features["distance_ft"]) / 60.0)
    angle_factor = (60 - features["angle_deg"]) / 60.0
    type_factor = 1.0 if features["shot_type"] == "wrist" else 0.9
    rebound_factor = 1.25 if features["rebound"] else 1.0
    rush_factor = 1.15 if features["rush"] else 1.0
    prob = base * angle_factor * type_factor * rebound_factor * rush_factor
    return round(min(prob, 0.9), 3)  # cap to 90%

# --- Date handling ---
def parse_date(date_str):
    """Convert mm/dd/yyyy string to datetime.date."""
    return datetime.datetime.strptime(date_str, "%m/%d/%Y").date()

# --- Injury system ---
def apply_injury_adjustments(team, gf, ga):
    if team not in injuries:
        return gf, ga
    impact_total = 0.0
    for player, games_left in injuries[team].items():
        if games_left > 0:
            p = injury_profiles.get(player)
            if p:
                if p["role"] == "forward":
                    gf -= p["impact"]
                    impact_total += p["impact"]
                elif p["role"] == "defense":
                    ga += p["impact"]
                    impact_total += p["impact"]
    season_injury_impact[team] = season_injury_impact.get(team, 0.0) + impact_total
    return gf, ga

def update_injuries(team):
    if team not in injuries:
        injuries[team] = {}
    for player in list(injuries[team].keys()):
        if injuries[team][player] > 0:
            injuries[team][player] -= 1
    if random.random() < INJURY_PROB:
        candidates = [p for p in injury_profiles if p in team_rosters.get(team, [])]
        if candidates:
            player = random.choice(candidates)
            if injuries[team].get(player, 0) <= 0:
                injuries[team][player] = random.randint(*INJURY_LENGTH_RANGE)

# --- Fatigue/rest adjustments ---
def calc_rest_adjustment(team, date, game_history):
    if team not in game_history or not game_history[team]:
        return 0.0
    days_since = (date - game_history[team][-1]).days
    adj = 0.0
    if days_since == 1:
        adj -= 0.20
        if len(game_history[team]) >= 3 and (date - game_history[team][-3]).days <= 4:
            adj -= 0.10
        if len(game_history[team]) >= 4 and (date - game_history[team][-4]).days <= 6:
            adj -= 0.10
    elif days_since >= 4:
        adj += 0.10
    return adj

# --- Goalie selection ---
def choose_goalie(team, date, game_history):
    if team not in game_history or not game_history[team]:
        return "starter"
    days_since = (date - game_history[team][-1]).days
    if days_since == 1:
        return "backup"
    return "backup" if random.random() < BACKUP_RANDOM_CHANCE else "starter"

# --- Core sim ---
def simulate_game(h, a):
    return np.random.poisson(max(h,0.1)), np.random.poisson(max(a,0.1))

# --- Shot-by-shot simulation with xg + shooter multipliers ---
def simulate_game_shots(h_exp, a_exp, h_goalie, a_goalie, h_team, a_team):
    play_log = []

    def pick_shooter(team):
        roster = team_rosters.get(team, [])
        if not roster:
            return {"name": "Generic Player", "factor": 1.0}
        shot = random.choice(roster)
        if isinstance(shot, str):
            return {"name": shot, "factor": 1.0}
        return shot

    h_shots = int(np.random.normal(SHOTS_PER_GAME, 5) * (h_exp / 3.0))
    a_shots = int(np.random.normal(SHOTS_PER_GAME, 5) * (a_exp / 3.0))
    h_shots = max(15, h_shots)
    a_shots = max(15, a_shots)

    h_goalie_sv = goalies[a_team][h_goalie]["SV"]
    a_goalie_sv = goalies[h_team][a_goalie]["SV"]

    h_ev_shots = int(h_shots * 0.85)
    a_ev_shots = int(a_shots * 0.85)
    h_pp_shots = h_shots - h_ev_shots
    a_pp_shots = a_shots - a_ev_shots

    h_pp_boost = (team_stats[h_team]["PP"] - team_stats[a_team]["PK"]) / 200.0
    a_pp_boost = (team_stats[a_team]["PP"] - team_stats[h_team]["PK"]) / 200.0

    h_goals = 0
    a_goals = 0

    # --- Home EV ---
    for _ in range(h_ev_shots):
        shooter = pick_shooter(h_team)
        features = {
            "distance_ft": np.random.uniform(5, 60),
            "angle_deg": np.random.uniform(0, 60),
            "shot_type": random.choice(["wrist","slap","backhand"]),
            "rebound": random.random() < 0.1,
            "rush": random.random() < 0.15,
            "strength": "5v5"
        }
        xg = predict_xg(features) * shooter.get("factor",1.0)
        prob_goal = xg * (1 - a_goalie_sv)
        is_goal = np.random.rand() < prob_goal
        if is_goal: h_goals += 1
        play_log.append({"team":h_team,"type":"EV","result":"GOAL" if is_goal else "MISS",
                         "goalie":goalies[a_team][h_goalie]["name"],
                         "shooter":shooter["name"],"xg":xg,"prob_goal":prob_goal})

    # --- Home PP ---
    for _ in range(h_pp_shots):
        shooter = pick_shooter(h_team)
        features = {
            "distance_ft": np.random.uniform(5, 60),
            "angle_deg": np.random.uniform(0, 60),
            "shot_type": random.choice(["wrist","slap","backhand"]),
            "rebound": random.random() < 0.1,
            "rush": random.random() < 0.15,
            "strength": "PP"
        }
        xg = predict_xg(features) * shooter.get("factor",1.0) * (1 + h_pp_boost)
        prob_goal = xg * (1 - a_goalie_sv)
        is_goal = np.random.rand() < prob_goal
        if is_goal: h_goals += 1
        play_log.append({"team":h_team,"type":"PP","result":"GOAL" if is_goal else "MISS",
                         "goalie":goalies[a_team][h_goalie]["name"],
                         "shooter":shooter["name"],"xg":xg,"prob_goal":prob_goal})

    # --- Away EV ---
    for _ in range(a_ev_shots):
        shooter = pick_shooter(a_team)
        features = {
            "distance_ft": np.random.uniform(5, 60),
            "angle_deg": np.random.uniform(0, 60),
            "shot_type": random.choice(["wrist","slap","backhand"]),
            "rebound": random.random() < 0.1,
            "rush": random.random() < 0.15,
            "strength": "5v5"
        }
        xg = predict_xg(features) * shooter.get("factor",1.0)
        prob_goal = xg * (1 - h_goalie_sv)
        is_goal = np.random.rand() < prob_goal
        if is_goal: a_goals += 1
        play_log.append({"team":a_team,"type":"EV","result":"GOAL" if is_goal else "MISS",
                         "goalie":goalies[h_team][a_goalie]["name"],
                         "shooter":shooter["name"],"xg":xg,"prob_goal":prob_goal})

    # --- Away PP ---
    for _ in range(a_pp_shots):
        shooter = pick_shooter(a_team)
        features = {
            "distance_ft": np.random.uniform(5, 60),
            "angle_deg": np.random.uniform(0, 60),
            "shot_type": random.choice(["wrist","slap","backhand"]),
            "rebound": random.random() < 0.1,
            "rush": random.random() < 0.15,
            "strength": "PP"
        }
        xg = predict_xg(features) * shooter.get("factor",1.0) * (1 + a_pp_boost)
        prob_goal = xg * (1 - h_goalie_sv)
        is_goal = np.random.rand() < prob_goal
        if is_goal: a_goals += 1
        play_log.append({"team":a_team,"type":"PP","result":"GOAL" if is_goal else "MISS",
                         "goalie":goalies[h_team][a_goalie]["name"],
                         "shooter":shooter["name"],"xg":xg,"prob_goal":prob_goal})
        # --- Pulled goalie EN ---
    if abs(h_goals - a_goals) <= 2:
        if h_goals < a_goals:
            shooter = pick_shooter(a_team)
            features = {
                "distance_ft": np.random.uniform(60, 200),
                "angle_deg": 0,
                "shot_type": "EN",
                "rebound": False,
                "rush": False,
                "strength": "EN"
            }
            xg = predict_xg(features) * shooter.get("factor",1.0)
            prob_goal = min(0.9, xg + 0.1)
            is_goal = np.random.rand() < prob_goal
            if is_goal: a_goals += 1
            play_log.append({"team":a_team,"type":"EN","result":"GOAL" if is_goal else "MISS",
                             "goalie":"Empty Net","shooter":shooter["name"],
                             "xg":xg,"prob_goal":prob_goal})
        elif a_goals < h_goals:
            shooter = pick_shooter(h_team)
            features = {
                "distance_ft": np.random.uniform(60, 200),
                "angle_deg": 0,
                "shot_type": "EN",
                "rebound": False,
                "rush": False,
                "strength": "EN"
            }
            xg = predict_xg(features) * shooter.get("factor",1.0)
            prob_goal = min(0.9, xg + 0.1)
            is_goal = np.random.rand() < prob_goal
            if is_goal: h_goals += 1
            play_log.append({"team":h_team,"type":"EN","result":"GOAL" if is_goal else "MISS",
                             "goalie":"Empty Net","shooter":shooter["name"],
                             "xg":xg,"prob_goal":prob_goal})

    return h_goals, a_goals, h_shots, a_shots, play_log

# --- Special Teams & Goalie adjustments ---
def adjust_for_special_teams(team, opp):
    return (team_stats[team]["PP"] - team_stats[opp]["PK"] +
            team_stats[team]["PK"] - team_stats[opp]["PP"]) / 200.0

def adjust_for_goalie(team, goalie):
    g = goalies.get(team, {}).get(goalie)
    if not g: return 0.0
    return (LEAGUE_AVG_SV - g["SV"]) * SHOTS_PER_GAME

# --- Collect Adjustments for Doctor’s Notes ---
def collect_adjustments(team, opp, goalie, date=None, games=None):
    """Return dict of all adjustments applied to a team's GF expectation."""
    adj = {"injury":0.0,"fatigue":0.0,"goalie":0.0,"special_teams":0.0,"rest":0.0}
    if team in injuries:
        total = 0.0
        for player,games_left in injuries[team].items():
            if games_left > 0 and player in injury_profiles:
                impact = injury_profiles[player]["impact"]
                total -= impact
        adj["injury"] = total
    adj["goalie"] = -adjust_for_goalie(team, goalie)
    adj["special_teams"] = adjust_for_special_teams(team, opp)
    if date and games:
        adj["rest"] = calc_rest_adjustment(team, date, games)
        adj["fatigue"] = adj["rest"] if adj["rest"] < 0 else 0
    return adj

# --- Generate Doctor's Note ---
def generate_doctors_note(date, home, away, h_adj, a_adj, proj_home, proj_away, ot_prob):
    note = []
    note.append(f"\nDoctor’s Note – {home} vs {away} – {date}")
    for team, adj in [(home,h_adj),(away,a_adj)]:
        note.append(f"{team}:")
        if abs(adj["injury"])>0: note.append(f"  - Injuries adj: {adj['injury']:+.2f} GF")
        if abs(adj["rest"])>0: note.append(f"  - Rest/Fatigue adj: {adj['rest']:+.2f} GF")
        if abs(adj["goalie"])>0: note.append(f"  - Goalie adj: {adj['goalie']:+.2f} GF")
        if abs(adj["special_teams"])>0: note.append(f"  - Special teams adj: {adj['special_teams']:+.2f} GF")
    note.append(f"Summary Diagnosis: {home} proj {proj_home:.1f} GF, {away} proj {proj_away:.1f} GF")
    note.append(f"Overtime likelihood: {ot_prob:.1f}%")
    return "\n".join(note)

# --- Simulate one result ---
def simulate_result(h_team, a_team, h_goalie, a_goalie, date=None, games=None):
    h_GF,h_GA = team_stats[h_team]["GF"], team_stats[h_team]["GA"]
    a_GF,a_GA = team_stats[a_team]["GF"], team_stats[a_team]["GA"]

    h_GF,h_GA = apply_injury_adjustments(h_team,h_GF,h_GA)
    a_GF,a_GA = apply_injury_adjustments(a_team,a_GF,a_GA)

    h_exp=(h_GF+a_GA)/2
    a_exp=(a_GF+h_GA)/2

    h_exp += HOME_ADVANTAGE + adjust_for_special_teams(h_team,a_team)
    a_exp += adjust_for_special_teams(a_team,h_team)

    h_exp -= adjust_for_goalie(h_team,h_goalie)
    a_exp -= adjust_for_goalie(a_team,a_goalie)

    if date and games:
        h_exp += calc_rest_adjustment(h_team,date,games)
        a_exp += calc_rest_adjustment(a_team,date,games)

    hs, as_, h_shots, a_shots, log = simulate_game_shots(h_exp,a_exp,h_goalie,a_goalie,h_team,a_team)

    result = "H" if hs>as_ else "A" if as_>hs else ("OTW" if random.choice(["H","A"])=="H" else "OTL")

    h_adj = collect_adjustments(h_team, a_team, h_goalie, date, games)
    a_adj = collect_adjustments(a_team, h_team, a_goalie, date, games)
    ot_prob = 100*(0.12)  # approx. league avg
    note = generate_doctors_note(date or "N/A", h_team, a_team, h_adj, a_adj, h_exp, a_exp, ot_prob)

    return result, hs, as_, h_shots, a_shots, log, note

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
            season_streaks[team][curr_type].append(curr_len)
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
    global season_injury_impact
    season_injury_impact = {}
    standings = {team: {"W":0,"L":0,"OT":0,"PTS":0} for team in team_stats.keys()}
    season_stats = {team: {"GF":0,"GA":0,"SF":0,"SA":0} for team in team_stats.keys()}
    season_logs = {}
    season_streaks = {team: {"W":[],"L":[],"OT":[],"maxW":0,"maxL":0,"maxOT":0} for team in team_stats.keys()}
    season_notes = {}
    streak_state = {team: {"current_type":None,"length":0} for team in team_stats.keys()}
    game_history = {}
    injuries.clear()

    for date in sorted(schedule_by_date.keys()):
        for home, visitor in schedule_by_date[date]:
            for t in (home, visitor):
                if t not in team_stats:
                    team_stats[t] = {"GF":3.0,"GA":3.0,"PP":20.0,"PK":80.0}
                if t not in goalies:
                    goalies[t] = {
                        "starter":{"SV":LEAGUE_AVG_SV,"name":"Generic Starter"},
                        "backup":{"SV":LEAGUE_AVG_SV-0.01,"name":"Generic Backup"},
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

            update_injuries(home)
            update_injuries(visitor)
            h_g = choose_goalie(home, date, game_history)
            a_g = choose_goalie(visitor, date, game_history)
            result, hs, vs, hshots, ashots, log, note = simulate_result(home, visitor, h_g, a_g, date, game_history)

            season_notes[(date,home,visitor)] = note
            game_history.setdefault(home, []).append(date)
            game_history.setdefault(visitor, []).append(date)

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
            else:
                standings[visitor]["W"] += 1; standings[visitor]["PTS"] += 2
                standings[home]["OT"] += 1; standings[home]["PTS"] += 1
                update_streak(visitor,"W",streak_state,season_streaks)
                update_streak(home,"OT",streak_state,season_streaks)

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

    return (standings, season_stats, season_logs, season_streaks, season_notes) if track_shots else standings
# --- Monte Carlo League ---
def monte_carlo_league(schedule_by_date, runs=500, debug=False):
    final_points = {team: [] for team in team_stats.keys()}
    for i in range(runs):
        standings = simulate_full_league(schedule_by_date, verbose=False)
        for team, rec in standings.items():
            final_points[team].append(rec["PTS"])
        if (i+1) % (runs//10 or 1) == 0 or i+1 == runs:
            pct = (i+1)/runs*100
            print(f"\rMonte Carlo progress: {i+1}/{runs} ({pct:.0f}%)", end="", flush=True)
    print()

    results = {}
    for team, pts in final_points.items():
        arr = np.array(pts)
        results[team] = {
            "avg": round(np.mean(arr),1),
            "median": round(np.median(arr),1),
            "p25": round(np.percentile(arr,25),1),
            "p75": round(np.percentile(arr,75),1),
            "std": round(np.std(arr),2)
        }
    return results

# --- Head-to-head matchup ---
def simulate_matchup_probs(team1, team2, runs=500, date=None):
    results = {"team1_wins":0,"team2_wins":0,"team1_OT":0,"team2_OT":0}
    for _ in range(runs):
        h_g = choose_goalie(team1, date or datetime.date.today(), {})
        a_g = choose_goalie(team2, date or datetime.date.today(), {})
        result, hs, as_, *_ = simulate_result(team1, team2, h_g, a_g, date or datetime.date.today(), {})
        if result == "H":
            results["team1_wins"] += 1
        elif result == "A":
            results["team2_wins"] += 1
        elif result == "OTW":
            results["team1_wins"] += 1
            results["team2_OT"] += 1
        else:
            results["team2_wins"] += 1
            results["team1_OT"] += 1
    for k in results:
        results[k] = round(results[k]/runs*100,1)
    return results

# --- Injury profiles ---
injury_profiles = {
    "David Pastrnak":{"role":"forward","impact":0.2},
    "Charlie McAvoy":{"role":"defense","impact":0.2},
    "Connor McDavid":{"role":"forward","impact":0.35},
    "Leon Draisaitl":{"role":"forward","impact":0.3},
    "Auston Matthews":{"role":"forward","impact":0.25},
    "Morgan Rielly":{"role":"defense","impact":0.15},
    "Matthew Tkachuk":{"role":"forward","impact":0.25},
    "Aleksander Barkov":{"role":"forward","impact":0.2},
    "Sidney Crosby":{"role":"forward","impact":0.25},
    "Alex Ovechkin":{"role":"forward","impact":0.25},
    "Nathan MacKinnon":{"role":"forward","impact":0.3},
    "Cale Makar":{"role":"defense","impact":0.25},
    "Kirill Kaprizov":{"role":"forward","impact":0.25},
    "Jack Hughes":{"role":"forward","impact":0.25},
    "Artemi Panarin":{"role":"forward","impact":0.25},
    "Adam Fox":{"role":"defense","impact":0.2},
    "Nikita Kucherov":{"role":"forward","impact":0.3},
    "Victor Hedman":{"role":"defense","impact":0.2}
}

# --- Team rosters (with factors optional) ---
team_rosters = {
    "Boston Bruins":[{"name":"David Pastrnak","factor":1.25},{"name":"Charlie McAvoy","factor":0.95}],
    "Toronto Maple Leafs":[{"name":"Auston Matthews","factor":1.25},{"name":"Morgan Rielly","factor":0.95}],
    "Tampa Bay Lightning":[{"name":"Nikita Kucherov","factor":1.25},{"name":"Victor Hedman","factor":0.95}],
    "Colorado Avalanche":[{"name":"Nathan MacKinnon","factor":1.25},{"name":"Cale Makar","factor":0.95}],
    "Edmonton Oilers":[{"name":"Connor McDavid","factor":1.3},{"name":"Leon Draisaitl","factor":1.2}],
    "New York Rangers":[{"name":"Artemi Panarin","factor":1.2},{"name":"Adam Fox","factor":0.95}],
    "New Jersey Devils":["Jack Hughes","Dougie Hamilton"],
    "Ottawa Senators":["Brady Tkachuk","Thomas Chabot"],
    "Pittsburgh Penguins":["Sidney Crosby","Kris Letang"],
    "Washington Capitals":["Alex Ovechkin","John Carlson"],
    "Vancouver Canucks":["Elias Pettersson","Quinn Hughes"],
    "Florida Panthers":["Matthew Tkachuk","Aleksander Barkov"],
    "Minnesota Wild":["Kirill Kaprizov"],
    "Nashville Predators":["Filip Forsberg","Roman Josi"],
    "Vegas Golden Knights":["Jack Eichel","Mark Stone"],
    "Los Angeles Kings":["Anze Kopitar","Drew Doughty"],
    "Columbus Blue Jackets":["Johnny Gaudreau","Zach Werenski"],
    "Arizona Coyotes":["Clayton Keller"],
    "Buffalo Sabres":["Rasmus Dahlin"],
    "Winnipeg Jets":["Kyle Connor","Josh Morrissey"],
    "San Jose Sharks":["Logan Couture","Tomas Hertl"],
    "St. Louis Blues":["Robert Thomas"],
    "Calgary Flames":["Jonathan Huberdeau"],
    "Chicago Blackhawks":["Connor Bedard"],
    "Detroit Red Wings":["Dylan Larkin","Moritz Seider"],
    "Dallas Stars":["Jason Robertson","Miro Heiskanen"],
    "Seattle Kraken":["Matty Beniers"],
    "Philadelphia Flyers":["Travis Konecny"],
    "Montreal Canadiens":["Cole Caufield","Nick Suzuki"],
    "Anaheim Ducks":["Trevor Zegras","Mason McTavish"],
    "Carolina Hurricanes":["Sebastian Aho","Andrei Svechnikov"],
    "New York Islanders":["Mathew Barzal"],
    "Utah Mammoth":[]
}

# --- Team stats ---
team_stats = {
    "Boston Bruins":{"GF":3.24,"GA":2.64,"PP":23.3,"PK":82.6},
    "Toronto Maple Leafs":{"GF":3.63,"GA":3.01,"PP":26.8,"PK":76.9},
    "Tampa Bay Lightning":{"GF":3.5,"GA":3.3,"PP":28.6,"PK":79.0},
    "Colorado Avalanche":{"GF":3.35,"GA":2.99,"PP":25.2,"PK":79.6},
    "Edmonton Oilers":{"GF":3.96,"GA":3.2,"PP":29.5,"PK":79.5},
    "New York Rangers":{"GF":3.3,"GA":2.7,"PP":25.2,"PK":82.0},
    "New Jersey Devils":{"GF":3.4,"GA":3.1,"PP":21.9,"PK":81.5},
    "Ottawa Senators":{"GF":3.2,"GA":3.4,"PP":20.4,"PK":79.0},
    "Pittsburgh Penguins":{"GF":3.2,"GA":3.1,"PP":21.7,"PK":80.2},
    "Washington Capitals":{"GF":2.9,"GA":3.2,"PP":21.3,"PK":80.1},
    "Vancouver Canucks":{"GF":3.4,"GA":2.9,"PP":23.7,"PK":79.5},
    "Florida Panthers":{"GF":3.5,"GA":3.0,"PP":25.5,"PK":78.2},
    "Minnesota Wild":{"GF":3.2,"GA":2.9,"PP":22.0,"PK":81.1},
    "Nashville Predators":{"GF":3.0,"GA":3.0,"PP":19.3,"PK":80.0},
    "Vegas Golden Knights":{"GF":3.4,"GA":2.8,"PP":21.4,"PK":79.2},
    "Los Angeles Kings":{"GF":3.2,"GA":3.1,"PP":20.0,"PK":78.1},
    "Columbus Blue Jackets":{"GF":3.0,"GA":3.6,"PP":19.8,"PK":76.3},
    "Arizona Coyotes":{"GF":2.8,"GA":3.2,"PP":18.9,"PK":77.1},
    "Buffalo Sabres":{"GF":3.1,"GA":3.3,"PP":23.4,"PK":78.9},
    "Winnipeg Jets":{"GF":3.1,"GA":2.6,"PP":21.2,"PK":80.9},
    "San Jose Sharks":{"GF":2.7,"GA":3.6,"PP":18.7,"PK":75.0},
    "St. Louis Blues":{"GF":3.0,"GA":3.3,"PP":19.2,"PK":76.5},
    "Calgary Flames":{"GF":3.0,"GA":3.2,"PP":20.9,"PK":78.8},
    "Chicago Blackhawks":{"GF":2.7,"GA":3.7,"PP":17.2,"PK":76.0},
    "Detroit Red Wings":{"GF":3.0,"GA":3.3,"PP":21.1,"PK":78.6},
    "Dallas Stars":{"GF":3.4,"GA":2.7,"PP":27.0,"PK":82.2},
    "Seattle Kraken":{"GF":3.1,"GA":2.9,"PP":20.5,"PK":79.3},
    "Philadelphia Flyers":{"GF":2.8,"GA":3.0,"PP":17.1,"PK":78.2},
    "Montreal Canadiens":{"GF":2.9,"GA":3.6,"PP":16.9,"PK":72.9},
    "Anaheim Ducks":{"GF":2.7,"GA":3.5,"PP":17.3,"PK":75.6},
    "Carolina Hurricanes":{"GF":3.3,"GA":2.5,"PP":23.0,"PK":84.4},
    "New York Islanders":{"GF":2.9,"GA":2.8,"PP":18.5,"PK":81.2},
    "Utah Mammoth":{"GF":3.0,"GA":3.0,"PP":20.0,"PK":80.0}
}

# --- Goalies ---
goalies = {
    "Boston Bruins":{"starter":{"SV":0.918,"name":"Jeremy Swayman"},"backup":{"SV":0.902,"name":"Brandon Bussi"}},
    "Toronto Maple Leafs":{"starter":{"SV":0.912,"name":"Ilya Samsonov"},"backup":{"SV":0.898,"name":"Joseph Woll"}},
    "Tampa Bay Lightning":{"starter":{"SV":0.915,"name":"Andrei Vasilevskiy"},"backup":{"SV":0.899,"name":"Jonas Johansson"}},
    "Colorado Avalanche":{"starter":{"SV":0.910,"name":"Alexandar Georgiev"},"backup":{"SV":0.899,"name":"Justus Annunen"}},
    "Edmonton Oilers":{"starter":{"SV":0.913,"name":"Stuart Skinner"},"backup":{"SV":0.892,"name":"Calvin Pickard"}},
    "New York Rangers":{"starter":{"SV":0.919,"name":"Igor Shesterkin"},"backup":{"SV":0.900,"name":"Jonathan Quick"}},
    "New Jersey Devils":{"starter":{"SV":0.907,"name":"Vitek Vanecek"},"backup":{"SV":0.897,"name":"Akira Schmid"}},
    "Ottawa Senators":{"starter":{"SV":0.905,"name":"Joonas Korpisalo"},"backup":{"SV":0.893,"name":"Anton Forsberg"}},
    "Pittsburgh Penguins":{"starter":{"SV":0.913,"name":"Tristan Jarry"},"backup":{"SV":0.899,"name":"Alex Nedeljkovic"}},
    "Washington Capitals":{"starter":{"SV":0.909,"name":"Darcy Kuemper"},"backup":{"SV":0.895,"name":"Charlie Lindgren"}},
    "Vancouver Canucks":{"starter":{"SV":0.920,"name":"Thatcher Demko"},"backup":{"SV":0.901,"name":"Casey DeSmith"}},
    "Florida Panthers":{"starter":{"SV":0.913,"name":"Sergei Bobrovsky"},"backup":{"SV":0.898,"name":"Anthony Stolarz"}},
    "Minnesota Wild":{"starter":{"SV":0.912,"name":"Filip Gustavsson"},"backup":{"SV":0.901,"name":"Marc-Andre Fleury"}},
    "Nashville Predators":{"starter":{"SV":0.916,"name":"Juuse Saros"},"backup":{"SV":0.900,"name":"Kevin Lankinen"}},
    "Vegas Golden Knights":{"starter":{"SV":0.909,"name":"Logan Thompson"},"backup":{"SV":0.902,"name":"Adin Hill"}},
    "Los Angeles Kings":{"starter":{"SV":0.907,"name":"Cam Talbot"},"backup":{"SV":0.898,"name":"Pheonix Copley"}},
    "Columbus Blue Jackets":{"starter":{"SV":0.902,"name":"Elvis Merzlikins"},"backup":{"SV":0.892,"name":"Spencer Martin"}},
    "Arizona Coyotes":{"starter":{"SV":0.908,"name":"Karel Vejmelka"},"backup":{"SV":0.895,"name":"Connor Ingram"}},
    "Buffalo Sabres":{"starter":{"SV":0.906,"name":"Ukko-Pekka Luukkonen"},"backup":{"SV":0.894,"name":"Devon Levi"}},
    "Winnipeg Jets":{"starter":{"SV":0.920,"name":"Connor Hellebuyck"},"backup":{"SV":0.899,"name":"Laurent Brossoit"}},
    "San Jose Sharks":{"starter":{"SV":0.898,"name":"Kaapo Kahkonen"},"backup":{"SV":0.890,"name":"Mackenzie Blackwood"}},
    "St. Louis Blues":{"starter":{"SV":0.909,"name":"Jordan Binnington"},"backup":{"SV":0.896,"name":"Joel Hofer"}},
    "Calgary Flames":{"starter":{"SV":0.910,"name":"Jacob Markstrom"},"backup":{"SV":0.898,"name":"Dan Vladar"}},
    "Chicago Blackhawks":{"starter":{"SV":0.907,"name":"Petr Mrazek"},"backup":{"SV":0.892,"name":"Arvid Soderblom"}},
    "Detroit Red Wings":{"starter":{"SV":0.905,"name":"Ville Husso"},"backup":{"SV":0.894,"name":"James Reimer"}},
    "Dallas Stars":{"starter":{"SV":0.917,"name":"Jake Oettinger"},"backup":{"SV":0.902,"name":"Scott Wedgewood"}},
    "Seattle Kraken":{"starter":{"SV":0.907,"name":"Philipp Grubauer"},"backup":{"SV":0.897,"name":"Joey Daccord"}},
    "Philadelphia Flyers":{"starter":{"SV":0.910,"name":"Carter Hart"},"backup":{"SV":0.896,"name":"Samuel Ersson"}},
    "Montreal Canadiens":{"starter":{"SV":0.903,"name":"Jake Allen"},"backup":{"SV":0.893,"name":"Sam Montembeault"}},
    "Anaheim Ducks":{"starter":{"SV":0.906,"name":"John Gibson"},"backup":{"SV":0.895,"name":"Lukas Dostal"}},
    "Carolina Hurricanes":{"starter":{"SV":0.916,"name":"Frederik Andersen"},"backup":{"SV":0.905,"name":"Antti Raanta"}},
    "New York Islanders":{"starter":{"SV":0.918,"name":"Ilya Sorokin"},"backup":{"SV":0.907,"name":"Semyon Varlamov"}},
    "Utah Mammoth":{"starter":{"SV":0.905,"name":"Generic Starter"},"backup":{"SV":0.895,"name":"Generic Backup"}}
}

# --- Run ---
if __name__=="__main__":
    master_schedule = load_master_schedule("master_schedule.csv")
    print("\n=== FULL NHL SEASON SIMULATION (with play-by-play logging) ===")
    standings, season_stats, season_logs, season_streaks, season_notes = simulate_full_league(master_schedule, verbose=False, track_shots=True)

    print("\n=== Final Standings (1 sim) ===")
    sorted_standings = sorted(standings.items(), key=lambda x: x[1]["PTS"], reverse=True)
    for team, rec in sorted_standings:
        print(f"{team}: {rec['W']}-{rec['L']}-{rec['OT']} ({rec['PTS']} pts)")

    print("\n=== Per-Team Season Shot Averages & Gambler Values ===")
    ot_rates = []
    for team, stats in season_stats.items():
        games = standings[team]["W"] + standings[team]["L"] + standings[team]["OT"]
        if games == 0: continue
        sf = stats["SF"]/games
        sa = stats["SA"]/games
        sh_pct = (stats["GF"]/stats["SF"]*100 if stats["SF"]>0 else 0)
        sv_pct = (1 - stats["GA"]/stats["SA"] if stats["SA"]>0 else 0)
        ot_pct = standings[team]["OT"] / games * 100
        diff_pg = (stats["GF"] - stats["GA"]) / games
        pace = sf + sa
        st_strength = team_stats[team]["PP"] + team_stats[team]["PK"]
        mov = (stats["GF"] - stats["GA"]) / games
        inj_factor = (season_injury_impact.get(team, 0.0) / games) if games > 0 else 0.0

        close_games = 0
        fatigue_flags = 0
        for (date,h,a), note in season_notes.items():
            if team in (h,a):
                gf = season_stats[team]["GF"]; ga = season_stats[team]["GA"]
                if abs(gf-ga)==1: close_games+=1
                if "Fatigue" in note: fatigue_flags+=1

        close_game_pct = (close_games/games*100) if games>0 else 0
        fatigue_pct = (fatigue_flags/games*100) if games>0 else 0

        one_goal_games = 0
        for (date,h,a), note in season_notes.items():
            if team in (h,a):
                gf = season_stats[team]["GF"]; ga = season_stats[team]["GA"]
                if abs(gf-ga) == 1:
                    one_goal_games += 1
        one_goal_pct = (one_goal_games/games*100) if games>0 else 0

        top_tier = {"Boston Bruins","Colorado Avalanche","Edmonton Oilers","Toronto Maple Leafs",
                    "New York Rangers","Dallas Stars","Florida Panthers","Carolina Hurricanes"}
        h2h_diff = []
        for (date,h,a), note in season_notes.items():
            if team in (h,a):
                opp = a if team==h else h
                if opp in top_tier:
                    gf = season_stats[team]["GF"]; ga = season_stats[team]["GA"]
                    h2h_diff.append(gf-ga)
        h2h_delta = (np.mean(h2h_diff) if h2h_diff else 0)

        ot_rates.append((standings[team]["OT"], games))
        print(
            f"{team:20s}  SF/G: {sf:.1f}  SA/G: {sa:.1f}  Sh%: {sh_pct:.1f}%  Sv%: {sv_pct:.3f}  "
              f"OT%: {ot_pct:.1f}%  Diff/G: {diff_pg:+.2f}  Pace/G: {pace:.1f}  ST%: {st_strength:.1f}  "
              f"MOV: {mov:+.2f}  InjAdj: {inj_factor:+.2f}  CloseG%: {close_game_pct:.1f}%  "
              f"FatigueG%: {fatigue_pct:.1f}%  1G%: {one_goal_pct:.1f}%  H2HΔ(top8): {h2h_delta:+.2f}"
        )

    print("\n=== Win/Loss/OT Streaks (per team) ===")
    for team, streaks in season_streaks.items():
        print(f"{team:20s}  MaxW:{streaks['maxW']}  MaxL:{streaks['maxL']}  MaxOT:{streaks['maxOT']}")

    total_ot = sum(ot for ot,g in ot_rates)
    total_games = sum(g for ot,g in ot_rates)
    league_avg_ot = (total_ot / total_games * 100) if total_games > 0 else 0
    print(f"\nNHL Average OT%: {league_avg_ot:.1f}%")

    if season_logs:
        first_game = next(iter(season_logs.keys()))
        print("\n=== Sample Play-by-Play (first 10 events) ===")
        print(f"Game: {first_game}")
        for event in season_logs[first_game][:10]:
            print(event)

    if season_notes:
        key = next(iter(season_notes.keys()))
        print("\n=== Sample Doctor’s Note ===")
        print(season_notes[key])

    avg_points = monte_carlo_league(master_schedule, runs=300, debug=False)
    print("\n=== Monte Carlo Results Ready for UX (buckets available) ===")

    print("\n=== Head-to-Head Probability Demo ===")
    matchup = simulate_matchup_probs("Boston Bruins","Toronto Maple Leafs", runs=300)
    print("Boston vs Toronto (300 sims):", matchup)