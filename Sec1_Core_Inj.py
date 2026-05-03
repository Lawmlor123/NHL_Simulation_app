# ============================================================
# SECTION 1: CORE CONFIG + INJURY/FATIGUE SYSTEMS
# ============================================================

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
    # lazy import to avoid circular dependency
    from Sec5_endstats import injury_profiles

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
    # lazy import to avoid circular dependency
    from Sec5_endstats import injury_profiles, team_rosters

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

# --- Explicit exports ---
__all__ = [
    "injuries",
    "season_injury_impact",
    "apply_injury_adjustments",
    "update_injuries",
    "calc_rest_adjustment",
    "choose_goalie",
    "predict_xg",
    "parse_date"
]

# ========== END OF SECTION 1 =================================