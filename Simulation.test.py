# Simulation.test.py

import datetime

# Bring in your section modules
from Sec3_seasim import simulate_full_league, load_master_schedule
from Sec4_analysisprob import monte_carlo_league, simulate_matchup_probs, print_monte_carlo_results
from Sec5_endstats import (
    print_team_averages,
    print_streaks,
    print_sample_playbyplay,
    print_sample_doctors_note
)

if __name__ == "__main__":
    master_schedule = load_master_schedule("master_schedule.csv")

    # Run a single full season WITH shots tracking
    standings, season_stats, season_logs, season_streaks, season_notes = simulate_full_league(
        master_schedule,
        verbose=False,
        track_shots=True
    )

    # 1. Final standings
    print("\n=== Final Standings (1 sim) ===")
    for team, rec in sorted(standings.items(), key=lambda x: x[1]["PTS"], reverse=True):
        print(f"{team}: {rec['W']}-{rec['L']}-{rec['OT']} ({rec['PTS']} pts)")

    # 2. Team averages & gambler values
    print_team_averages(season_stats)

    # 3. Win/Loss/OT streaks
    print_streaks(season_streaks)

    # 4. Sample play-by-play + Doctor’s note
    print_sample_playbyplay(season_logs)
    print_sample_doctors_note(season_notes)

    # 5. Monte Carlo probabilities
    print("\n=== Monte Carlo Results ===")
    avg_points = monte_carlo_league(master_schedule, runs=100)
    print_monte_carlo_results(avg_points)

    # 6. Head-to-head demo
    print("\n=== Head-to-Head Probability Demo ===")
    matchup = simulate_matchup_probs("Boston Bruins", "Toronto Maple Leafs", runs=100)
    print("Boston vs Toronto:", matchup)