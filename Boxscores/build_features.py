"""
build_features.py
Takes cleaned skater/goalie parquet files and builds
rolling features for prediction modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")

# ── 1. LOAD & FILTER ───────────────────────────────────────────────
print("Loading data ...")
sk = pd.read_parquet(BASE / "skater_stats.parquet")
sk = sk[sk["game_type"].isin([2, 3])].copy()

# ── 2. PARSE TOI TO MINUTES ────────────────────────────────────────
def toi_to_minutes(toi_str):
    try:
        parts = str(toi_str).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except:
        return 0.0

sk["toi_min"] = sk["toi"].apply(toi_to_minutes)
print(f"TOI range: {sk['toi_min'].min():.1f} - {sk['toi_min'].max():.1f} minutes")

# ── 3. PARSE GAME DATE & SORT ──────────────────────────────────────
sk["game_date"] = pd.to_datetime(sk["game_date"])
sk = sk.sort_values(["player_id", "game_date", "game_pk"]).reset_index(drop=True)

# ── 4. REST DAYS ───────────────────────────────────────────────────
sk["prev_game_date"] = sk.groupby("player_id")["game_date"].shift(1)
sk["rest_days"] = (sk["game_date"] - sk["prev_game_date"]).dt.days
sk["rest_days"] = sk["rest_days"].clip(upper=14).fillna(7)  # cap at 14, default 7 for first game

# ── 5. GAME NUMBER (within season) ─────────────────────────────────
sk["game_num"] = sk.groupby(["player_id", "season"]).cumcount() + 1

# ── 6. ROLLING FEATURES ────────────────────────────────────────────
print("Building rolling features ...")

ROLL_COLS = [
    "goals", "assists", "points", "shots", "hits",
    "blocked_shots", "pim", "pp_goals", "toi_min",
    "shifts", "giveaways", "takeaways", "plus_minus"
]

WINDOWS = [3, 5, 10, 20]

# Group by player, shift so we don't leak the current game
grouped = sk.groupby("player_id")

for col in ROLL_COLS:
    shifted = grouped[col].shift(1)  # exclude current game
    for w in WINDOWS:
        roll = shifted.rolling(window=w, min_periods=1)
        sk[f"{col}_avg{w}"] = roll.mean().round(4)
    # Season-long average
    sk[f"{col}_season_avg"] = grouped[col].apply(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).round(4).values

print(f"  Added {len(ROLL_COLS) * (len(WINDOWS) + 1)} rolling columns")

# ── 7. SHOOTING PERCENTAGE (rolling) ───────────────────────────────
for w in WINDOWS:
    shot_col = f"shots_avg{w}"
    goal_col = f"goals_avg{w}"
    sk[f"sh_pct_avg{w}"] = np.where(
        sk[shot_col] > 0,
        (sk[goal_col] / sk[shot_col]).round(4),
        0.0
    )

# ── 8. POINTS PER 60 (rolling) ─────────────────────────────────────
for w in WINDOWS:
    toi_col = f"toi_min_avg{w}"
    pts_col = f"points_avg{w}"
    sk[f"pts_per60_avg{w}"] = np.where(
        sk[toi_col] > 0,
        (sk[pts_col] / sk[toi_col] * 60).round(4),
        0.0
    )

# ── 9. POSITION ENCODING ───────────────────────────────────────────
sk["is_center"]  = (sk["position"] == "C").astype(int)
sk["is_wing"]    = sk["position"].isin(["L", "R", "W"]).astype(int)
sk["is_defense"] = (sk["position"] == "D").astype(int)

# ── 10. HOME/AWAY ──────────────────────────────────────────────────
sk["is_home"] = sk["is_home"].astype(int)

# ── 11. DAYS INTO SEASON ───────────────────────────────────────────
season_starts = sk.groupby("season")["game_date"].transform("min")
sk["days_into_season"] = (sk["game_date"] - season_starts).dt.days

# ── 12. DROP ROWS WITH NO HISTORY ──────────────────────────────────
# First game of each player has no rolling data - keep them but flag
sk["has_history"] = (sk.groupby("player_id").cumcount() > 0).astype(int)

# ── 13. SAVE ────────────────────────────────────────────────────────
OUT = BASE / "skater_features.parquet"
sk.to_parquet(OUT, index=False)

print(f"\n{'='*55}")
print(f"SAVED: {OUT.name}")
print(f"Shape: {sk.shape[0]:,} rows x {sk.shape[1]} columns")
print(f"Players: {sk['player_id'].nunique():,}")
print(f"Seasons: {sorted(sk['season'].unique())}")
print(f"\nSample columns:")
print(f"  {[c for c in sk.columns if 'goals' in c]}")
print(f"\nFirst row rolling features:")
sample = sk[sk["has_history"] == 1].iloc[0]
for w in WINDOWS:
    print(f"  goals_avg{w}={sample[f'goals_avg{w}']:.3f}  "
          f"points_avg{w}={sample[f'points_avg{w}']:.3f}  "
          f"shots_avg{w}={sample[f'shots_avg{w}']:.3f}")
print(f"{'='*55}")