#!/usr/bin/env python
"""
model_prep.py

Loads the merged player-game dataset, performs basic inspection/cleaning,
and saves a Parquet version for downstream modeling.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\shell\OneDrive\Documents\NHL_Player")
INPUT_PATH = BASE_DIR / "Features" / "player_games_with_targets.csv"
OUTPUT_PATH = BASE_DIR / "Features" / "player_games_prepped.parquet"

# Update to whichever target you plan to model first.
TARGET_COL = "targets_fantasy_pts"

SAMPLE_STATS_COLS = ["toi", "shots", "goals", "assists", "blocks",
                     "hits", "pim", "targets_fantasy_pts"]

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Ensure datetime ordering
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        missing_dates = df["game_date"].isna().sum()
        if missing_dates:
            print(f"[!] {missing_dates:,} rows have invalid dates.")
        df = df.sort_values("game_date")
    else:
        print("[!] Column 'game_date' missing—add it before continuing.")

    # Quick preview
    print("\nHead (5 rows):")
    print(df.head().to_string(index=False))

    # Dtype summary
    print("\nData types:")
    print(df.dtypes)

    # Column overview
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    # Null counts (top 25)
    print("\nNull counts (top 25):")
    nulls = df.isna().sum().sort_values(ascending=False)
    print(nulls.head(25))

    # Target coverage
    if TARGET_COL in df.columns:
        target_non_null = df[TARGET_COL].notna().sum()
        target_null = len(df) - target_non_null
        print(f"\nTarget '{TARGET_COL}': {target_non_null:,} rows with values, "
              f"{target_null:,} rows missing.")
    else:
        print(f"\n[!] Target column '{TARGET_COL}' not found—update TARGET_COL.")

    # Descriptive stats for selected columns (if present)
    existing_stats_cols = [c for c in SAMPLE_STATS_COLS if c in df.columns]
    if existing_stats_cols:
        print("\nDescriptive stats:")
        print(df[existing_stats_cols].describe().T)
    else:
        print("\n(no sample statistic columns found to describe)")

    # Save cleaned dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved prepped data to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()