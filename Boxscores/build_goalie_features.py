"""
build_goalie_features.py
1. Determine game winners from goalie W + skater goal diff
2. Build goalie rolling features
3. Save both outputs
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")

# ── Load data ──────────────────────────────────────────────
print("Loading data ...")
sk = pd.read_parquet(BASE / "skater_features.parquet")
gl = pd.read_parquet(BASE / "goalie_boxscores_raw.parquet")

# ── 1. RESOLVE GAME WINNERS ───────────────────────────────
print("\n=== Resolving game winners ===")

# Method A: goalie wins
goalie_wins = gl[gl["wins"] == 1][["gameId", "teamAbbrev"]].rename(
    columns={"gameId": "game_pk", "teamAbbrev": "winning_team_goalie"}
).drop_duplicates(subset="game_pk")

# Method B: skater goal differential (non-tied games)
game_goals = sk.groupby(["game_pk", "team", "is_home"]).agg(
    goals=("goals", "sum")
).reset_index()

home = game_goals[game_goals["is_home"] == 1][["game_pk", "team", "goals"]].rename(
    columns={"team": "home_team", "goals": "home_goals"})
away = game_goals[game_goals["is_home"] == 0][["game_pk", "team", "goals"]].rename(
    columns={"team": "away_team", "goals": "away_goals"})

games = home.merge(away, on="game_pk")
games["goal_diff"] = games["home_goals"] - games["away_goals"]
games["winning_team_goals"] = np.where(
    games["goal_diff"] > 0, games["home_team"],
    np.where(games["goal_diff"] < 0, games["away_team"], None)
)

# Merge both methods
games = games.merge(goalie_wins, on="game_pk", how="left")
games["winner"] = games["winning_team_goalie"].fillna(games["winning_team_goals"])

# Add game_date from skater data
game_dates = sk.groupby("game_pk")["game_date"].first().reset_index()
games = games.merge(game_dates, on="game_pk", how="left")

# Add is_home_win flag
games["home_win"] = (games["winner"] == games["home_team"]).astype(int)

resolved = games["winner"].notna().sum()
total = len(games)
print(f"Total games:    {total:,}")
print(f"Resolved:       {resolved:,} ({resolved/total:.1%})")
print(f"Unresolved:     {total - resolved:,}")
print(f"Home win rate:  {games.loc[games['winner'].notna(), 'home_win'].mean():.1%}")

# Save game outcomes
games_out = games[["game_pk", "game_date", "home_team", "away_team",
                   "home_goals", "away_goals", "winner", "home_win"]].copy()
games_out = games_out[games_out["winner"].notna()].reset_index(drop=True)
games_out.to_parquet(BASE / "game_outcomes.parquet", index=False)
print(f"Saved: game_outcomes.parquet  ({len(games_out):,} games)")

# ── 2. BUILD GOALIE FEATURES ──────────────────────────────
print("\n=== Building goalie features ===")

# Clean up goalie data
gl = gl.rename(columns={
    "gameId": "game_pk",
    "gameDate": "game_date",
    "goalieFullName": "goalie_name",
    "teamAbbrev": "team",
    "opponentTeamAbbrev": "opponent",
    "homeRoad": "home_road",
})

gl["game_date"] = pd.to_datetime(gl["game_date"])
gl["is_home"] = (gl["home_road"] == "H").astype(int)
gl["is_starter"] = gl["gamesStarted"].fillna(0).astype(int)

# Convert TOI from seconds to minutes
gl["toi_min"] = gl["timeOnIce"] / 60.0

# Sort
gl = gl.sort_values(["playerId", "game_date", "game_pk"]).reset_index(drop=True)

# Game number per goalie
gl["game_num"] = gl.groupby("playerId").cumcount() + 1

# Rolling features for goalies
GOALIE_STATS = ["goalsAgainst", "saves", "shotsAgainst", "savePct", "toi_min"]
WINDOWS = [3, 5, 10]

print(f"  Stats: {GOALIE_STATS}")
print(f"  Windows: {WINDOWS}")

new_cols = 0
for stat in GOALIE_STATS:
    for w in WINDOWS:
        col = f"g_{stat}_avg{w}"
        gl[col] = (
            gl.groupby("playerId")[stat]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        new_cols += 1

    # Season average
    col = f"g_{stat}_season_avg"
    gl[col] = (
        gl.groupby("playerId")[stat]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    new_cols += 1

# Win rate rolling
for w in WINDOWS:
    gl[f"g_win_rate_{w}"] = (
        gl.groupby("playerId")["wins"]
        .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    )
    new_cols += 1

gl[f"g_win_rate_season"] = (
    gl.groupby("playerId")["wins"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)
new_cols += 1

print(f"  Added {new_cols} rolling columns")

# Mark starters
starters = gl[gl["is_starter"] == 1].copy()
print(f"\n  Total goalie rows: {len(gl):,}")
print(f"  Starter rows:      {len(starters):,}")
print(f"  Unique goalies:    {gl['playerId'].nunique()}")

# Save
gl.to_parquet(BASE / "goalie_features.parquet", index=False)
print(f"\n  Saved: goalie_features.parquet ({gl.shape[0]} rows x {gl.shape[1]} cols)")

# ── 3. SUMMARY ─────────────────────────────────────────────
print("\n=======================================================")
print("FILES CREATED:")
print(f"  game_outcomes.parquet   - {len(games_out):,} games with winners")
print(f"  goalie_features.parquet - {gl.shape[0]:,} rows x {gl.shape[1]:,} cols")
print("=======================================================")

# Spot check: goalie rolling features
star_goalie = gl[gl["goalie_name"].str.contains("Vasilevskiy", na=False)].head(8)
if len(star_goalie) > 0:
    print(f"\nSpot check - Vasilevskiy:")
    print(star_goalie[["game_date", "goalsAgainst", "savePct",
                        "g_goalsAgainst_avg3", "g_savePct_avg3",
                        "g_win_rate_5", "wins"]].to_string())