import random

# --- Simplified team data (sample) ---
sample_standings = {
    "Boston Bruins": 112,
    "Toronto Maple Leafs": 105,
    "Tampa Bay Lightning": 100,
    "Florida Panthers": 98,
    "New York Rangers": 110,
    "Carolina Hurricanes": 108,
    "New Jersey Devils": 102,
    "Washington Capitals": 95,
    "Edmonton Oilers": 115,
    "Colorado Avalanche": 106,
    "Dallas Stars": 104,
    "Vegas Golden Knights": 99,
    "Los Angeles Kings": 101,
    "Winnipeg Jets": 97,
    "Minnesota Wild": 94,
    "Vancouver Canucks": 92,
}

# --- Conferences ---
EAST_TEAMS = [
    "Boston Bruins","Toronto Maple Leafs","Tampa Bay Lightning",
    "Florida Panthers","New York Rangers","Carolina Hurricanes",
    "New Jersey Devils","Washington Capitals"
]
WEST_TEAMS = [t for t in sample_standings if t not in EAST_TEAMS]

# --- Series Simulation ---
def simulate_series(team1, team2, best_of=7):
    """Simulate best-of-seven series and return winner + score line."""
    wins = {team1: 0, team2: 0}
    needed = best_of // 2 + 1
    while wins[team1] < needed and wins[team2] < needed:
        chance1 = sample_standings[team1] / (sample_standings[team1] + sample_standings[team2])
        if random.random() < chance1:
            wins[team1] += 1
        else:
            wins[team2] += 1
    winner = team1 if wins[team1] > wins[team2] else team2
    loser = team2 if winner == team1 else team1
    return winner, f"{winner} defeats {loser} {wins[winner]}–{wins[loser]}"

# --- Bracket Simulation ---
def run_playoffs(standings):
    east = sorted([t for t in standings if t in EAST_TEAMS],
                  key=lambda x: standings[x], reverse=True)[:8]
    west = sorted([t for t in standings if t in WEST_TEAMS],
                  key=lambda x: standings[x], reverse=True)[:8]

    print("EAST seeds:", east)
    print("WEST seeds:", west)

    # Round 1
    east_r1 = [simulate_series(east[i], east[-(i+1)]) for i in range(4)]
    west_r1 = [simulate_series(west[i], west[-(i+1)]) for i in range(4)]
    print("\n--- Round 1 ---")
    for _, line in east_r1 + west_r1:
        print(line)

    # Round 2
    east_r2 = [simulate_series(east_r1[i][0], east_r1[-(i+1)][0]) for i in range(2)]
    west_r2 = [simulate_series(west_r1[i][0], west_r1[-(i+1)][0]) for i in range(2)]
    print("\n--- Round 2 ---")
    for _, line in east_r2 + west_r2:
        print(line)

    # Conference Finals
    east_final = simulate_series(east_r2[0][0], east_r2[1][0])
    west_final = simulate_series(west_r2[0][0], west_r2[1][0])
    print("\n--- Conference Finals ---")
    print(east_final[1])
    print(west_final[1])

    # Stanley Cup Final
    cup_final = simulate_series(east_final[0], west_final[0])
    print("\n--- Stanley Cup Final ---")
    print(cup_final[1])
    print("\n=== Stanley Cup Champion:", cup_final[0], "===")

# --- Run ---
if __name__ == "__main__":
    run_playoffs(sample_standings)