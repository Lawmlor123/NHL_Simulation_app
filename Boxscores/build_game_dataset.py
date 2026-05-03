"""
build_game_dataset.py
Assemble game-level features: team skater aggregates + starting goalie stats
Target: home_win (1/0)
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")

print("Loading data ...")
sk = pd.read_parquet(BASE / "skater_features.parquet")
gl = pd.read_parquet(BASE / "goalie_features.parquet")
outcomes = pd.read_parquet(BASE / "game_outcomes.parquet")

print(f"  Skaters:  {sk.shape}")
print(f"  Goalies:  {gl.shape}")
print(f"  Games:    {outcomes.shape}")

# ── 1. SKATER AGGREGATION BY TEAM PER GAME ────────────────
print("\n=== Aggregating skater features by team ===")

# Rolling columns we want to aggregate
rolling_cols = [c for c in sk.columns if any(
    c.endswith(suf) for suf in ["_avg3", "_avg5", "_avg10", "_avg20", "_season_avg"]
)]
print(f"  Rolling columns to aggregate: {len(rolling_cols)}")

# Also include raw game counts for context
sk["toi_weight"] = sk["toi_min"].fillna(0)

# Aggregate: mean of rolling features for all skaters on a team in a game
# This gives us "average team player form"
team_agg_mean = (
    sk.groupby(["game_pk", "team"])[rolling_cols]
    .mean()
    .reset_index()
)

# Also: sum of rolling features (total team offensive firepower)
sum_cols = [c for c in rolling_cols if any(
    stat in c for stat in ["goals", "points", "shots", "assists", "pp_goals"]
)]
team_agg_sum = (
    sk.groupby(["game_pk", "team"])[sum_cols]
    .sum()
    .reset_index()
)

# Rename: mean gets _mean, sum gets _sum
team_agg_mean = team_agg_mean.rename(
    columns={c: f"sk_{c}_mean" for c in rolling_cols}
)
team_agg_sum = team_agg_sum.rename(
    columns={c: f"sk_{c}_sum" for c in sum_cols}
)

# Merge mean + sum
team_agg = team_agg_mean.merge(team_agg_sum, on=["game_pk", "team"])

# Also add: roster size (players who played), total TOI
roster_info = sk.groupby(["game_pk", "team"]).agg(
    roster_size=("player_id", "nunique"),
    team_toi=("toi_min", "sum"),
).reset_index()

team_agg = team_agg.merge(roster_info, on=["game_pk", "team"])

print(f"  Team aggregation shape: {team_agg.shape}")

# ── 2. STARTING GOALIE FEATURES ───────────────────────────
print("\n=== Attaching starting goalie features ===")

# Use starters only
starters = gl[gl["is_starter"] == 1].copy()

goalie_rolling = [c for c in gl.columns if c.startswith("g_")]
goalie_keep = ["game_pk", "team", "goalie_name", "playerId"] + goalie_rolling

starters_slim = starters[goalie_keep].copy()
starters_slim = starters_slim.rename(columns={"playerId": "goalie_id"})

# Handle rare case of two starters listed (use the one with more TOI)
starters_slim = starters_slim.drop_duplicates(subset=["game_pk", "team"], keep="first")

print(f"  Starter rows: {len(starters_slim):,}")
print(f"  Goalie rolling cols: {len(goalie_rolling)}")

# ── 3. MERGE INTO GAME-LEVEL DATASET ──────────────────────
print("\n=== Building game-level dataset ===")

# Split into home and away
home_sk = team_agg.merge(
    outcomes[["game_pk", "home_team"]].rename(columns={"home_team": "team"}),
    on=["game_pk", "team"]
)
away_sk = team_agg.merge(
    outcomes[["game_pk", "away_team"]].rename(columns={"away_team": "team"}),
    on=["game_pk", "team"]
)

# Rename columns with home_/away_ prefix
home_cols = {c: f"home_{c}" for c in home_sk.columns if c not in ["game_pk", "team"]}
away_cols = {c: f"away_{c}" for c in away_sk.columns if c not in ["game_pk", "team"]}

home_sk = home_sk.rename(columns=home_cols).drop(columns=["team"])
away_sk = away_sk.rename(columns=away_cols).drop(columns=["team"])

# Same for goalies
home_gl = starters_slim.merge(
    outcomes[["game_pk", "home_team"]].rename(columns={"home_team": "team"}),
    on=["game_pk", "team"]
)
away_gl = starters_slim.merge(
    outcomes[["game_pk", "away_team"]].rename(columns={"away_team": "team"}),
    on=["game_pk", "team"]
)

home_gl_cols = {c: f"home_gl_{c}" for c in home_gl.columns if c not in ["game_pk", "team"]}
away_gl_cols = {c: f"away_gl_{c}" for c in away_gl.columns if c not in ["game_pk", "team"]}

home_gl = home_gl.rename(columns=home_gl_cols).drop(columns=["team"])
away_gl = away_gl.rename(columns=away_gl_cols).drop(columns=["team"])

# Merge everything
game_df = outcomes[["game_pk", "game_date", "home_team", "away_team", "home_win"]].copy()
game_df = game_df.merge(home_sk, on="game_pk", how="left")
game_df = game_df.merge(away_sk, on="game_pk", how="left")
game_df = game_df.merge(home_gl, on="game_pk", how="left")
game_df = game_df.merge(away_gl, on="game_pk", how="left")

# ── 4. ADD DIFFERENTIAL FEATURES ──────────────────────────
print("  Adding differential features ...")

# Key differentials: home minus away
diff_pairs = [
    ("sk_goals_avg5_mean", "Offensive form"),
    ("sk_points_avg5_mean", "Point production"),
    ("sk_shots_avg5_mean", "Shot volume"),
    ("sk_goals_avg10_mean", "Goals trend"),
    ("sk_points_avg10_mean", "Points trend"),
    ("sk_goals_avg5_sum", "Team goal power"),
    ("sk_points_avg5_sum", "Team point power"),
    ("sk_shots_avg5_sum", "Team shot volume"),
]

diff_count = 0
for col, _ in diff_pairs:
    home_col = f"home_{col}"
    away_col = f"away_{col}"
    if home_col in game_df.columns and away_col in game_df.columns:
        game_df[f"diff_{col}"] = game_df[home_col] - game_df[away_col]
        diff_count += 1

# Goalie differentials
for col in ["g_savePct_avg5", "g_goalsAgainst_avg5", "g_win_rate_5", "g_savePct_avg10", "g_win_rate_season"]:
    hc = f"home_gl_{col}"
    ac = f"away_gl_{col}"
    if hc in game_df.columns and ac in game_df.columns:
        game_df[f"diff_{col}"] = game_df[hc] - game_df[ac]
        diff_count += 1

print(f"  Added {diff_count} differential features")

# ── 5. CLEANUP AND SAVE ───────────────────────────────────
# Drop rows with missing skater/goalie data
before = len(game_df)
game_df = game_df.dropna(subset=["home_roster_size", "away_roster_size"]).reset_index(drop=True)
after = len(game_df)
print(f"\n  Dropped {before - after} games missing skater data")

# Count goalie coverage
goalie_missing = game_df["home_gl_goalie_name"].isna().sum()
print(f"  Games missing home goalie: {goalie_missing}")
goalie_missing = game_df["away_gl_goalie_name"].isna().sum()
print(f"  Games missing away goalie: {goalie_missing}")

# Save
out = BASE / "game_dataset.parquet"
game_df.to_parquet(out, index=False)

# ── 6. SUMMARY ─────────────────────────────────────────────
feature_cols = [c for c in game_df.columns if c not in [
    "game_pk", "game_date", "home_team", "away_team", "home_win",
    "home_gl_goalie_name", "away_gl_goalie_name", "home_gl_goalie_id", "away_gl_goalie_id"
]]

print(f"\n=======================================================")
print(f"SAVED: {out.name}")
print(f"Shape:    {game_df.shape[0]:,} games x {game_df.shape[1]} columns")
print(f"Features: {len(feature_cols)} numeric features")
print(f"Target:   home_win (mean={game_df['home_win'].mean():.3f})")
print(f"Seasons:  {game_df['game_date'].min()} to {game_df['game_date'].max()}")
print(f"=======================================================")

# Show a sample game
print(f"\nSample game:")
row = game_df.iloc[100]
print(f"  {row['game_date']}: {row['home_team']} vs {row['away_team']}")
print(f"  Home win: {row['home_win']}")
print(f"  Home goalie: {row.get('home_gl_goalie_name', 'N/A')}")
print(f"  Away goalie: {row.get('away_gl_goalie_name', 'N/A')}")
print(f"  diff_sk_goals_avg5_mean: {row.get('diff_sk_goals_avg5_mean', 'N/A'):.4f}")
print(f"  diff_g_savePct_avg5:     {row.get('diff_g_savePct_avg5', 'N/A'):.4f}")