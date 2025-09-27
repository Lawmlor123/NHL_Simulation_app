import numpy as np
import csv
import random

HOME_ADVANTAGE = 0.25
LEAGUE_AVG_SV = 0.905
SHOTS_PER_GAME = 30

# --- Simulation engine ---
def simulate_game(home_expected_goals, away_expected_goals):
    home_score = np.random.poisson(max(home_expected_goals, 0.1))
    away_score = np.random.poisson(max(away_expected_goals, 0.1))
    return home_score, away_score

def adjust_for_special_teams(team_name, opponent):
    team_PP = team_stats[team_name].get("PP", 20.0)
    team_PK = team_stats[team_name].get("PK", 80.0)
    opp_PP = team_stats[opponent].get("PP", 20.0)
    opp_PK = team_stats[opponent].get("PK", 80.0)
    pp_edge = team_PP - opp_PK
    pk_edge = team_PK - opp_PP
    return (pp_edge + pk_edge) / 200.0

def adjust_for_goalie(team_name, goalie_name):
    goalie = goalies.get(team_name, {}).get(goalie_name)
    if not goalie:
        return 0.0
    goalie_sv = goalie["SV"]
    edge = (LEAGUE_AVG_SV - goalie_sv) * SHOTS_PER_GAME
    return edge

def simulate_result(home_team, away_team, home_goalie="starter", away_goalie="starter"):
    home_GF = team_stats[home_team]["GF"]
    home_GA = team_stats[home_team]["GA"]
    away_GF = team_stats[away_team]["GF"]
    away_GA = team_stats[away_team]["GA"]

    home_expected = (home_GF + away_GA) / 2
    away_expected = (away_GF + home_GA) / 2

    # Factors
    home_expected += HOME_ADVANTAGE
    home_expected += adjust_for_special_teams(home_team, away_team)
    away_expected += adjust_for_special_teams(away_team, home_team)
    home_expected -= adjust_for_goalie(home_team, home_goalie)
    away_expected -= adjust_for_goalie(away_team, away_goalie)

    home_score, away_score = simulate_game(home_expected, away_expected)

    if home_score > away_score:
        return "H", home_score, away_score
    elif away_score > home_score:
        return "A", home_score, away_score
    else:
        winner = random.choice(["H", "A"])
        return ("OTW" if winner == "H" else "OTL"), home_score, away_score

# --- Team stats ---
team_stats = {
    "Boston Bruins": {"GF": 3.24, "GA": 2.64, "PP": 23.3, "PK": 82.6},
    "Toronto Maple Leafs": {"GF": 3.63, "GA": 3.01, "PP": 26.8, "PK": 76.9},
}

goalies = {
    "Boston Bruins": {
        "starter": {"SV": 0.918, "name": "Jeremy Swayman"},
        "backup": {"SV": 0.902, "name": "Linus Ullmark"},
    },
    "Toronto Maple Leafs": {
        "starter": {"SV": 0.912, "name": "Ilya Samsonov"},
        "backup": {"SV": 0.898, "name": "Joseph Woll"},
    },
}

# --- Goalie test helper ---
def goalie_vs_goalie(team, opponent, goalie_choice, sims=2000):
    results = [simulate_result(team, opponent, home_goalie=goalie_choice, away_goalie="starter")
               for _ in range(sims)]
    wins = sum(1 for r,_,_ in results if r in ["H","OTW"])
    avg_margin = np.mean([hs - as_ for _,hs,as_ in results])
    return wins/sims*100, avg_margin

# --- Runner ---
if __name__ == "__main__":
    print("=== Goalie Impact Test: Bruins vs Leafs (2000 sims each) ===\n")
    wr_sway, margin_sway = goalie_vs_goalie("Boston Bruins", "Toronto Maple Leafs", "starter")
    wr_ull, margin_ull = goalie_vs_goalie("Boston Bruins", "Toronto Maple Leafs", "backup")

    print(f"Swayman ({goalies['Boston Bruins']['starter']['SV']:.3f} SV): "
          f"Win% = {wr_sway:.1f}%, Avg margin = {margin_sway:+.2f}")
    print(f"Ullmark ({goalies['Boston Bruins']['backup']['SV']:.3f} SV): "
          f"Win% = {wr_ull:.1f}%, Avg margin = {margin_ull:+.2f}")