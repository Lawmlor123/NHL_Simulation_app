"""
build_game_dataset_v2.py
Game-level features with fixed goalie selection (max TOI per team per game)
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

rolling_cols = [c for c in sk.columns if any(
    c.endswith(suf) for suf in ["_avg3", "_avg5", "_avg10", "_avg20", "_season_avg"]
)]

# Mean of rolling features (average player form)
team_agg_mean = (
    sk.groupby(["game_pk", "team"])[rolling_cols]
    .mean()
    .reset_index()
)

# Sum of key offensive rolling features (total team firepower)
sum_cols = [c for c in rolling_cols if any(
    stat in c for stat in ["goals", "points", "shots", "assists", "pp_goals"]
)]
team_agg_sum = (
    sk.groupby(["game_pk", "team"])[sum_cols]
    .sum()
    .reset_index()
)

team_agg_mean = team_agg_mean.rename(columns={c: f"sk_{c}_mean" for c in rolling_cols})
team_agg_sum = team_agg_sum.rename(columns={c: f"sk_{c}_sum" for c in sum_cols})

team_agg = team_agg_mean.merge(team_agg_sum, on=["game_pk", "team"])

roster_info = sk.groupby(["game_pk", "team"]).agg(
    roster_size=("player_id", "nunique"),
    team_toi=("toi_min", "sum"),
).reset_index()

team_agg = team_agg.merge(roster_info, on=["game_pk", "team"])
print(f"  Team aggregation shape: {team_agg.shape}")

# ── 2. GOALIE SELECTION: MAX TOI PER TEAM PER GAME ────────
print("\n=== Selecting starting goalie by max TOI ===")

goalie_rolling = [c for c in gl.columns if c.startswith("g_")]

# Sort by TOI descending, take first per team per game
gl_sorted = gl.sort_values("toi_min", ascending=False)
starters = gl_sorted.drop_duplicates(subset=["game_pk", "team"], keep="first").copy()

goalie_keep = ["game_pk", "team", "goalie_name", "playerId"] + goalie_rolling
starters_slim = starters[goalie_keep].rename(columns={"playerId": "goalie_id"})

# Verify: should be max 2 per game now
per_game = starters_slim.groupby("game_pk").size()
print(f"  Goalies per game distribution:")
print(f"    {per_game.value_counts().sort_index().to_dict()}")
print(f"  Total starter rows: {len(starters_slim):,}")

# ── 2b. SITUATIONAL FEATURES: REST DAYS + BACK-TO-BACK ───
print("\n=== Adding rest days and back-to-back features ===")

# Build long-form schedule: one row per team per game
outcomes["game_date"] = pd.to_datetime(outcomes["game_date"])

home_sched = outcomes[["game_pk", "game_date", "home_team"]].rename(
    columns={"home_team": "team"})
away_sched = outcomes[["game_pk", "game_date", "away_team"]].rename(
    columns={"away_team": "team"})
team_schedule = pd.concat([home_sched, away_sched], ignore_index=True)
team_schedule = team_schedule.sort_values(["team", "game_date"]).reset_index(drop=True)

# Days since last game for each team
team_schedule["prev_game_date"] = team_schedule.groupby("team")["game_date"].shift(1)
team_schedule["rest_days"] = (
    team_schedule["game_date"] - team_schedule["prev_game_date"]
).dt.days
team_schedule["is_b2b"] = (team_schedule["rest_days"] == 1).astype(int)

# Join back as home and away rest
home_rest = team_schedule.merge(
    outcomes[["game_pk", "home_team"]].rename(columns={"home_team": "team"}),
    on=["game_pk", "team"]
)[["game_pk", "rest_days", "is_b2b"]].rename(
    columns={"rest_days": "home_rest_days", "is_b2b": "home_b2b"})

away_rest = team_schedule.merge(
    outcomes[["game_pk", "away_team"]].rename(columns={"away_team": "team"}),
    on=["game_pk", "team"]
)[["game_pk", "rest_days", "is_b2b"]].rename(
    columns={"rest_days": "away_rest_days", "is_b2b": "away_b2b"})

outcomes = outcomes.merge(home_rest, on="game_pk", how="left")
outcomes = outcomes.merge(away_rest, on="game_pk", how="left")

# Rest days differential (positive = home team more rested)
outcomes["rest_days_diff"] = outcomes["home_rest_days"] - outcomes["away_rest_days"]

b2b_games = outcomes["home_b2b"].fillna(0).sum() + outcomes["away_b2b"].fillna(0).sum()
print(f"  rest_days_diff range: {outcomes['rest_days_diff'].min():.0f} to "
      f"{outcomes['rest_days_diff'].max():.0f}  (mean: {outcomes['rest_days_diff'].mean():.2f})")
print(f"  Back-to-back appearances: {int(b2b_games):,}")

# ── 3. BUILD GAME-LEVEL DATASET ───────────────────────────
print("\n=== Building game-level dataset ===")

# HOME skaters
home_sk = team_agg.merge(
    outcomes[["game_pk", "home_team"]].rename(columns={"home_team": "team"}),
    on=["game_pk", "team"]
)
home_cols = {c: f"home_{c}" for c in home_sk.columns if c not in ["game_pk", "team"]}
home_sk = home_sk.rename(columns=home_cols).drop(columns=["team"])

# AWAY skaters
away_sk = team_agg.merge(
    outcomes[["game_pk", "away_team"]].rename(columns={"away_team": "team"}),
    on=["game_pk", "team"]
)
away_cols = {c: f"away_{c}" for c in away_sk.columns if c not in ["game_pk", "team"]}
away_sk = away_sk.rename(columns=away_cols).drop(columns=["team"])

# HOME goalie
home_gl = starters_slim.merge(
    outcomes[["game_pk", "home_team"]].rename(columns={"home_team": "team"}),
    on=["game_pk", "team"]
)
home_gl_cols = {c: f"home_gl_{c}" for c in home_gl.columns if c not in ["game_pk", "team"]}
home_gl = home_gl.rename(columns=home_gl_cols).drop(columns=["team"])

# AWAY goalie
away_gl = starters_slim.merge(
    outcomes[["game_pk", "away_team"]].rename(columns={"away_team": "team"}),
    on=["game_pk", "team"]
)
away_gl_cols = {c: f"away_gl_{c}" for c in away_gl.columns if c not in ["game_pk", "team"]}
away_gl = away_gl.rename(columns=away_gl_cols).drop(columns=["team"])

# Assemble — include situational columns from outcomes
sit_cols = ["game_pk", "game_date", "home_team", "away_team", "home_win",
            "home_rest_days", "away_rest_days", "rest_days_diff",
            "home_b2b", "away_b2b"]
game_df = outcomes[[c for c in sit_cols if c in outcomes.columns]].copy()
game_df = game_df.merge(home_sk, on="game_pk", how="left")
game_df = game_df.merge(away_sk, on="game_pk", how="left")
game_df = game_df.merge(home_gl, on="game_pk", how="left")
game_df = game_df.merge(away_gl, on="game_pk", how="left")

# ── 4. DIFFERENTIAL FEATURES ──────────────────────────────
print("  Adding differential features ...")

diff_count = 0

# Skater differentials
for col in rolling_cols:
    for agg in ["mean", "sum"]:
        hc = f"home_sk_{col}_{agg}"
        ac = f"away_sk_{col}_{agg}"
        if hc in game_df.columns and ac in game_df.columns:
            game_df[f"diff_sk_{col}_{agg}"] = game_df[hc] - game_df[ac]
            diff_count += 1

# Goalie differentials
for col in goalie_rolling:
    hc = f"home_gl_{col}"
    ac = f"away_gl_{col}"
    if hc in game_df.columns and ac in game_df.columns:
        game_df[f"diff_{col}"] = game_df[hc] - game_df[ac]
        diff_count += 1

# Roster size differential
if "home_roster_size" in game_df.columns:
    game_df["diff_roster_size"] = game_df["home_roster_size"] - game_df["away_roster_size"]
    diff_count += 1

print(f"  Added {diff_count} differential features")

# ── 5. COVERAGE REPORT ────────────────────────────────────
print("\n=== Coverage Report ===")
print(f"  Total games: {len(game_df):,}")

has_home_sk = game_df["home_roster_size"].notna().sum()
has_away_sk = game_df["away_roster_size"].notna().sum()
has_home_gl = game_df["home_gl_goalie_name"].notna().sum()
has_away_gl = game_df["away_gl_goalie_name"].notna().sum()
has_all = (
    game_df["home_roster_size"].notna() &
    game_df["away_roster_size"].notna() &
    game_df["home_gl_goalie_name"].notna() &
    game_df["away_gl_goalie_name"].notna()
).sum()

print(f"  Has home skaters: {has_home_sk:,} ({has_home_sk/len(game_df):.1%})")
print(f"  Has away skaters: {has_away_sk:,} ({has_away_sk/len(game_df):.1%})")
print(f"  Has home goalie:  {has_home_gl:,} ({has_home_gl/len(game_df):.1%})")
print(f"  Has away goalie:  {has_away_gl:,} ({has_away_gl/len(game_df):.1%})")
print(f"  Has ALL data:     {has_all:,} ({has_all/len(game_df):.1%})")

# ── 6. SAVE ───────────────────────────────────────────────
out = BASE / "game_dataset.parquet"
game_df.to_parquet(out, index=False)

feature_cols = [c for c in game_df.columns if c not in [
    "game_pk", "game_date", "home_team", "away_team", "home_win",
    "home_gl_goalie_name", "away_gl_goalie_name", "home_gl_goalie_id", "away_gl_goalie_id"
]]

print(f"\n=======================================================")
print(f"SAVED: {out.name}")
print(f"Shape:    {game_df.shape[0]:,} games x {game_df.shape[1]} columns")
print(f"Features: {len(feature_cols)} numeric features")
print(f"Target:   home_win (mean={game_df['home_win'].mean():.3f})")
print(f"=======================================================")