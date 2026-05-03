# simulation.py

# Import your section modules
from Sec3_seasim import simulate_full_league, load_master_schedule
from Sec4_analysisprob import monte_carlo_league, simulate_matchup_probs
from Sec5_endstats import (
    print_team_averages,
    print_streaks,
    print_sample_playbyplay,
    print_sample_doctors_note,
    print_monte_carlo_playoff_odds
)

def run_simulation(team_a: str = "Boston Bruins", team_b: str = "Toronto Maple Leafs", runs: int = 100):
    """
    Run a simulation season + Monte Carlo + head-to-head matchup.
    Returns a dictionary of results that can be returned as JSON via Flask.
    """

    # Load season schedule
    master_schedule = load_master_schedule("master_schedule.csv")

    # Run full season with stats tracking
    standings, season_stats, season_logs, season_streaks, season_notes = simulate_full_league(
        master_schedule,
        verbose=False,
        track_shots=True
    )

    # Monte Carlo sim
    avg_points = monte_carlo_league(master_schedule, runs=runs)

    # Head-to-head matchups
    matchup = simulate_matchup_probs(team_a, team_b)

    # Package results into dictionary
    return {
        "final_standings": {
            team: rec for team, rec in sorted(standings.items(), key=lambda x: x[1]["PTS"], reverse=True)
        },
        "streaks": season_streaks,
        "team_stats": season_stats,
        "monte_carlo_results": avg_points,
        "head_to_head": matchup
    }

# Script mode for local testing
if __name__ == "__main__":
    results = run_simulation()
    print(results)