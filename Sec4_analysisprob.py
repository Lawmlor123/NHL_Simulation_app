# ============================================================
# SECTION 4: ANALYSIS / PROBABILITIES
# ============================================================

import numpy as np
import datetime
from collections import defaultdict, Counter

# ============================================================
# Helper: bucket streak lengths into threshold labels
# ============================================================
def _compute_probabilities(streak_values, thresholds, label_prefix):
    """
    Compute probability (%) of hitting thresholds in streak_values.
    Returns a dict with keys like 'win_3+' or 'loss_5+'.
    """
    probs = {}
    arr = np.array(streak_values, dtype=float)
    for t in thresholds:
        key = f"{label_prefix}_{t}+"
        if len(arr) > 0:
            probs[key] = round(np.mean(arr >= t) * 100, 1)
        else:
            probs[key] = 0.0
    return probs

# ============================================================
# Monte Carlo League Simulation
# ============================================================
def monte_carlo_league(schedule_by_date, runs=500, debug=False, seed=None):
    """
    Run Monte Carlo simulations of a full season.
    
    Args:
        schedule_by_date (dict): game schedule grouped by date
        runs (int): number of Monte Carlo simulations to run
        debug (bool): if True, prints extra debug info
        seed (int or None): optional deterministic seed for np.random. 
            If provided, ensures reproducible results across runs.

    Returns:
        dict: keyed by team with summary info:
            - avg/median/p25/p75/std of points
            - streak_probs: dict with win/loss/ot probabilities
            - playoff_pct: % of runs finishing top-8 in points
    """
    # optional deterministic seed
    if seed is not None:
        np.random.seed(seed)

    # lazy imports
    from Sec5_endstats import team_stats
    from Sec3_seasim import simulate_full_league

    # container for results
    final_points = {team: [] for team in team_stats.keys()}
    streak_collections = {
        team: {"maxW": [], "maxL": [], "maxOT": []}
        for team in team_stats.keys()
    }
    playoff_counts = {team: 0 for team in team_stats.keys()}

    for i in range(runs):
        # simulate one season
        standings, _, _, season_streaks, _ = simulate_full_league(
            schedule_by_date, verbose=False, track_shots=True
        )
        # collect points
        team_points = {team: rec["PTS"] for team, rec in standings.items()}
        for team, pts in team_points.items():
            final_points[team].append(pts)
        # collect streak maxima for each team
        for team, s in season_streaks.items():
            streak_collections[team]["maxW"].append(s["maxW"])
            streak_collections[team]["maxL"].append(s["maxL"])
            streak_collections[team]["maxOT"].append(s["maxOT"])
        # count playoff appearances (top-8 by points)
        ranked = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
        top8 = {team for team, _ in ranked[:8]}
        for team in top8:
            playoff_counts[team] += 1
        # progress bar print
        if (i+1) % (runs//10 or 1) == 0 or i+1 == runs:
            pct = (i+1) / runs * 100
            print(f"\rMonte Carlo progress: {i+1}/{runs} ({pct:.0f}%)",
                  end="", flush=True)
    print()

    # aggregate results
    results = {}
    for team, pts in final_points.items():
        arr = np.array(pts, dtype=float)
        res = {
            "avg": float(round(np.mean(arr), 1)),
            "median": float(round(np.median(arr), 1)),
            "p25": float(round(np.percentile(arr, 25), 1)),
            "p75": float(round(np.percentile(arr, 75), 1)),
            "std": float(round(np.std(arr), 2)),
            "playoff_pct": round(playoff_counts[team] / runs * 100, 1)
        }
        # streak probability buckets
        res["streak_probs"] = {}
        res["streak_probs"].update(
            _compute_probabilities(streak_collections[team]["maxW"], [3,5,7], "win")
        )
        res["streak_probs"].update(
            _compute_probabilities(streak_collections[team]["maxL"], [3,5], "loss")
        )
        res["streak_probs"].update(
            _compute_probabilities(streak_collections[team]["maxOT"], [2,3,4], "ot")
        )
        results[team] = res
    return results

# ============================================================
# Printer: Monte Carlo summary
# ============================================================
def print_monte_carlo_results(results):
    """
    Pretty-print Monte Carlo outputs in a standings-style order by avg points.
    Displays points summary and selected streak probabilities.
    """
    print("\n=== Monte Carlo Results (avg pts order) ===")
    sorted_res = sorted(results.items(), key=lambda x: x[1]["avg"], reverse=True)
    for team, stats in sorted_res:
        line = (
            f"{team:20s}  avg:{stats['avg']:.1f}  median:{stats['median']:.1f}  "
            f"p25:{stats['p25']:.1f}  p75:{stats['p75']:.1f}  std:{stats['std']:.2f}  "
            f"Playoff%:{stats['playoff_pct']:.1f}"
        )
        print(line)
        sp = stats.get("streak_probs", {})
        if sp:
            # pick a few headline buckets for display
            w5 = sp.get("win_5+", 0.0)
            l3 = sp.get("loss_3+", 0.0)
            o2 = sp.get("ot_2+", 0.0)
            print(f"   Win≥5:{w5}%  Loss≥3:{l3}%  OT≥2:{o2}%")

# ============================================================
# Head-to-head matchup simulation
# ============================================================
def simulate_matchup_probs(team1, team2, runs=500, date=None, scoreline_top_n=5):
    """
    Run Monte Carlo style H2H between two teams.

    Returns dict with:
        - % chances of each outcome (W/L/OT split)
        - avg_margin: average (team1 goals - team2 goals) across sims
                      >0 means team1 favored, <0 means team2 favored
        - score_dist: dict of Top N most common final scorelines with %
        - close_games: grouped stats like one-goal frequency and OT%
    """
    # lazy imports
    from Sec1_Core_Inj import choose_goalie
    from Sec2_Simengine import simulate_result

    results = {"team1_wins":0,"team2_wins":0,"team1_OT":0,"team2_OT":0}
    margins = []
    score_counter = Counter()
    one_goal_count = 0
    ot_count = 0

    for _ in range(runs):
        h_g = choose_goalie(team1, date or datetime.date.today(), {})
        a_g = choose_goalie(team2, date or datetime.date.today(), {})
        result, hs, as_, *_ = simulate_result(
            team1, team2, h_g, a_g, date or datetime.date.today(), {}
        )

        # track scoreline
        scoreline = f"{hs}-{as_}"
        score_counter[scoreline] += 1

        # collect margin
        margins.append(hs - as_)

        # close game check
        if abs(hs - as_) == 1:
            one_goal_count += 1

        if result == "H":
            results["team1_wins"] += 1
        elif result == "A":
            results["team2_wins"] += 1
        elif result == "OTW":
            results["team1_wins"] += 1
            results["team2_OT"] += 1
            ot_count += 1
        else:
            results["team2_wins"] += 1
            results["team1_OT"] += 1
            ot_count += 1

    # convert counts to %
    for k in results:
        results[k] = round(results[k]/runs*100, 1)

    # add average margin
    results["avg_margin"] = float(round(np.mean(margins), 2)) if margins else 0.0

    # score distribution (Top N)
    most_common = score_counter.most_common(scoreline_top_n)
    results["score_dist"] = {
        s: round(c/runs*100, 1) for s, c in most_common
    }

    # close games grouping
    results["close_games"] = {
        "one_goal_pct": round(one_goal_count / runs * 100, 1),
        "ot_pct": round(ot_count / runs * 100, 1)
    }

    return results

# ============================================================
# Explicit exports
# ============================================================
__all__ = [
    "monte_carlo_league",
    "print_monte_carlo_results",
    "simulate_matchup_probs",
]

# ========== END OF SECTION 4 =================================