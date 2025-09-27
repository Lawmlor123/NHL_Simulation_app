# ============================================================
# SECTION 2: SIMULATION ENGINE (SHOTS & RESULTS)
# ============================================================

import random
import numpy as np

# =========================
# IMPORTS FROM OTHER SECTIONS
# =========================
# NOTE: to avoid circular imports, Sec5 data are pulled lazily inside functions.
from Sec1_Core_Inj import apply_injury_adjustments, injuries, season_injury_impact
from Sec1_Core_Inj import predict_xg as _predict_xg

# =========================
# SECTION 2 CONSTANTS
# =========================
HOME_ADVANTAGE = 0.25      # home ice edge (approx goals)
LEAGUE_AVG_SV = 0.910      # league-average save percentage
SHOTS_PER_GAME = 30        # avg shots per team per game

# --- Core sim ---
def simulate_game(h, a):
    return np.random.poisson(max(h,0.1)), np.random.poisson(max(a,0.1))

# --- xg model wrapper ---
def predict_xg(features):
    """Wrapper around Sec1 predict_xg to ensure strength is numeric."""
    val = features.get("strength")
    try:
        features["strength"] = float(val)
    except (TypeError, ValueError):
        pass
    return _predict_xg(features)

# --- Shot-by-shot simulation with xg + shooter multipliers ---
def simulate_game_shots(h_exp, a_exp, h_goalie, a_goalie, h_team, a_team):
    # lazy import to avoid circular dependency
    from Sec5_endstats import team_stats, goalies, team_rosters

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

    # Home EV
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

    # Home PP
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

    # Away EV
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

    # Away PP
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

    # Pulled goalie EN
    if abs(h_goals - a_goals) <= 2:
        if h_goals < a_goals:
            shooter = pick_shooter(a_team)
            features = {"distance_ft": np.random.uniform(60, 200),
                        "angle_deg": 0, "shot_type": "EN",
                        "rebound": False, "rush": False, "strength": "EN"}
            xg = predict_xg(features) * shooter.get("factor",1.0)
            prob_goal = min(0.9, xg + 0.1)
            is_goal = np.random.rand() < prob_goal
            if is_goal: a_goals += 1
            play_log.append({"team":a_team,"type":"EN","result":"GOAL" if is_goal else "MISS",
                             "goalie":"Empty Net","shooter":shooter["name"],
                             "xg":xg,"prob_goal":prob_goal})
        elif a_goals < h_goals:
            shooter = pick_shooter(h_team)
            features = {"distance_ft": np.random.uniform(60, 200),
                        "angle_deg": 0, "shot_type": "EN",
                        "rebound": False, "rush": False, "strength": "EN"}
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
    from Sec5_endstats import team_stats
    return (team_stats[team]["PP"] - team_stats[opp]["PK"] +
            team_stats[team]["PK"] - team_stats[opp]["PP"]) / 200.0

def adjust_for_goalie(team, goalie):
    from Sec5_endstats import goalies
    g = goalies.get(team, {}).get(goalie)
    if not g: return 0.0
    return (LEAGUE_AVG_SV - g["SV"]) * SHOTS_PER_GAME

# --- Rest Adjustment (from Golden Script) ---
def calc_rest_adjustment(team, date, game_history):
    if team not in game_history or not game_history[team]:
        return 0.0
    days_since = (date - game_history[team][-1]).days
    adj = 0.0
    if days_since == 1:
        adj -= 0.20
        if len(game_history[team]) >= 3 and (date - game_history[team][-3]).days == 3:
            adj -= 0.10
        if len(game_history[team]) >= 4 and (date - game_history[team][-4]).days == 4:
            adj -= 0.10
    elif days_since >= 4:
        adj += 0.10
    return adj

# --- Collect Adjustments for Doctor’s Notes ---
def collect_adjustments(team, opp, goalie, date=None, games=None):
    from Sec5_endstats import injury_profiles
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
    from Sec5_endstats import team_stats
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

# ========== END OF SECTION 2 =================================