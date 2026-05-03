#!/usr/bin/env python
"""
split_datasets.py

Chronologically split the prepped player-game dataset into train/validation/test
parquet files so downstream models avoid look-ahead leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronological train/val/test split for NHL player data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../Features/player_games_prepped.parquet"),
        help="Path to the prepped parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Features"),
        help="Directory where split parquet files will be written.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="targets_fantasy_pts",
        help="Target column used to filter rows before splitting.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (must sum to 1.0).",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=None,
        help="Optional minimum season (inclusive) to keep before splitting.",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=None,
        help="Optional maximum season (inclusive) to keep before splitting.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="player_games",
        help="Filename prefix for the output parquet files.",
    )
    return parser.parse_args()


def validate_ratios(ratios: Tuple[float, float, float]) -> Tuple[float, float, float]:
    total = sum(ratios)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={total}).")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"All split ratios must be positive, got {ratios}.")
    return ratios


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {path}")
    return df


def filter_dataset(
    df: pd.DataFrame,
    target_col: str,
    min_season: int | None,
    max_season: int | None,
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    mask = df[target_col].notna()
    if min_season is not None:
        mask &= df["season"] >= min_season
    if max_season is not None:
        mask &= df["season"] <= max_season

    df_filtered = df.loc[mask].copy()
    print(f"Filtered to {len(df_filtered):,} rows with non-null '{target_col}'.")
    return df_filtered


def chronological_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float],
    sort_cols: Tuple[str, ...] = ("game_date", "game_pk", "player_id"),
) -> Dict[str, pd.DataFrame]:
    df_sorted = df.sort_values(list(sort_cols)).reset_index(drop=True)

    n = len(df_sorted)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # ensure all rows accounted for due to rounding
    n_test = n - n_train - n_val

    splits = {
        "train": df_sorted.iloc[:n_train].reset_index(drop=True),
        "val": df_sorted.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": df_sorted.iloc[n_train + n_val :].reset_index(drop=True),
    }

    print(
        f"Split counts → train: {len(splits['train']):,}, "
        f"val: {len(splits['val']):,}, test: {len(splits['test']):,}"
    )
    return splits


def save_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split_name, df_split in splits.items():
        path = output_dir / f"{prefix}_{split_name}.parquet"
        df_split.to_parquet(path, index=False)
        manifest[split_name] = str(path.resolve())
        print(f"Saved {split_name} ({len(df_split):,} rows) → {path}")
    return manifest


def write_summary(
    manifest: Dict[str, str],
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    ratios: Tuple[float, float, float],
    output_dir: Path,
    prefix: str,
) -> None:
    summary = {
        "ratios": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "counts": {
            "total_rows_loaded": int(len(df_full)),
            "rows_with_target": int(len(df_filtered)),
            "train": int(df_filtered[df_filtered.index.isin(df_filtered.index[: int(len(df_filtered) * ratios[0])])].shape[0]),
        },
        "outputs": manifest,
    }
    summary_path = output_dir / f"{prefix}_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved split summary → {summary_path}")


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(tuple(args.ratios))

    df_full = load_dataset(args.input)
    df_filtered = filter_dataset(df_full, args.target_col, args.min_season, args.max_season)
    if df_filtered.empty:
        raise ValueError("No rows remain after filtering. Check your target column and season filters.")

    splits = chronological_split(df_filtered, ratios)
    manifest = save_splits(splits, args.output_dir, args.prefix)
    write_summary(manifest, df_full, df_filtered, ratios, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()