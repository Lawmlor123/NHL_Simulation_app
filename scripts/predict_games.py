#!/usr/bin/env python
"""
predict_games.py

Generate fantasy-point projections for NHL games using the trained LightGBM model
plus metadata saved by train_model.py. Prints a preview to stdout and saves the
full table to CSV (or Parquet if you ask for it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import lightgbm as lgb
import pandas as pd

TARGET_COL = "targets_fantasy_pts"
PARQUET_SUFFIXES = {".parquet", ".pq"}
CSV_SUFFIXES = {".csv", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightGBM fantasy-point predictions for upcoming games.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../Features/player_games_upcoming.parquet"),
        help="Feature file containing upcoming-game rows (Parquet or CSV).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../Models/lgbm_fantasy_pts.txt"),
        help="Path to the trained LightGBM model file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("../Models/model_metadata.json"),
        help="Metadata JSON with feature names, categorical columns, etc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Outputs/projections.csv"),
        help="Where to save the projection table (extension chooses format: .csv or .parquet).",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Number of rows to print to the console preview (after sorting).",
    )
    return parser.parse_args()


def read_any_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        return pd.read_parquet(path)
    if suffix in CSV_SUFFIXES:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format '{suffix}'. Use CSV or Parquet.")


def write_any_frame(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        df.to_parquet(path, index=False)
    elif suffix in CSV_SUFFIXES:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format '{suffix}'. Use CSV or Parquet.")


def load_upcoming(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Upcoming feature file not found: {path}")
    df = read_any_frame(path)
    print(f"Loaded {len(df):,} rows from {path}")
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    return df


def load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if "features" not in meta:
        raise KeyError("Metadata missing 'features'. Re-train model to regenerate metadata.")
    return meta


def prepare_feature_frame(
    raw_df: pd.DataFrame,
    feature_names: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    df = raw_df.copy()

    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        print(f"Adding {len(missing_cols)} missing features with zeros: {missing_cols}")
        for col in missing_cols:
            df[col] = 0.0

    extra_cols = sorted(set(df.columns) - set(feature_names))
    if extra_cols:
        print(f"Dropping {len(extra_cols)} unused columns: {extra_cols}")

    X = df[feature_names].copy()

    cat_cols = [col for col in categorical_features if col in X.columns]
    for col in cat_cols:
        X[col] = X[col].astype("category")

    numeric_cols = [col for col in X.columns if col not in cat_cols]
    X[numeric_cols] = X[numeric_cols].fillna(0.0)

    return X


def run_predictions(args: argparse.Namespace) -> None:
    upcoming_df = load_upcoming(args.input)
    metadata = load_metadata(args.metadata)

    feature_names = metadata["features"]
    categorical_features = metadata.get("categorical_features", [])

    X_upcoming = prepare_feature_frame(upcoming_df, feature_names, categorical_features)

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    model = lgb.Booster(model_file=str(args.model))
    print(f"Loaded model from {args.model}")

    preds = model.predict(
        X_upcoming,
        num_iteration=metadata.get("best_iteration"),
    )

    output_df = upcoming_df.copy()
    output_df["projection_fantasy_pts"] = preds
    output_df = output_df.sort_values(
        ["game_date", "game_pk", "projection_fantasy_pts"],
        ascending=[True, True, False],
    )

    # Console preview
    preview_rows = max(1, args.preview_rows)
    print("\nPreview (top rows after sorting):")
    print(output_df.head(preview_rows).to_string(index=False))

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_any_frame(output_df, output_path)

    print(f"\nSaved {len(output_df):,} projections → {output_path}")


def main() -> None:
    args = parse_args()
    run_predictions(args)


if __name__ == "__main__":
    main()