#!/usr/bin/env python
"""
Train a LightGBM regressor to predict NHL fantasy points.

Usage example:
    python train_model.py \
        --train ../Features/player_games_train.parquet \
        --val   ../Features/player_games_val.parquet   \
        --test  ../Features/player_games_test.parquet  \
        --history ../Features/game_history_2022_2025.parquet \
        --output-dir ../Models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "targets_fantasy_pts"

RESERVED_COLS = {
    TARGET,
    "game_date",
    "game_pk",
    "game_state",
    "game_type",
    "opponent_abbr",
    "player_id",
    "player_name",
    "shoots_catches",
    "source_file",
    "start_time_local",
    "start_time_utc",
    "team_abbr",
    "team_id",
    "venue",
    # History-derived columns we do NOT want treated as features
    "season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}

FEATURES: List[str] = []
CATEGORICAL_FEATURES: List[str] = []

PARQUET_SUFFIXES = {".parquet", ".pq"}
CSV_SUFFIXES = {".csv", ".txt"}
HISTORY_KEYS = ["game_date", "game_pk"]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        return pd.read_parquet(path)
    if suffix in CSV_SUFFIXES:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or Parquet.")


def load_history(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    history = read_table(path).copy()
    missing = set(HISTORY_KEYS) - set(history.columns)
    if missing:
        raise KeyError(f"History file missing required columns: {missing}")

    history["game_date"] = pd.to_datetime(history["game_date"]).dt.strftime("%Y-%m-%d")
    history["game_pk"] = (
        pd.to_numeric(history["game_pk"], errors="raise").astype("int64")
    )

    history = history.drop_duplicates(subset=HISTORY_KEYS, keep="last")
    print(f"Loaded history table with {len(history):,} games from {path}")
    return history


def merge_with_history(
    df: pd.DataFrame, history: Optional[pd.DataFrame], split_label: str
) -> pd.DataFrame:
    if history is None:
        return df

    missing_keys = set(HISTORY_KEYS) - set(df.columns)
    if missing_keys:
        raise KeyError(
            f"Split '{split_label}' missing merge keys {missing_keys} required for history join."
        )

    history_cols = [
        col for col in history.columns if col not in df.columns or col in HISTORY_KEYS
    ]
    history_subset = history[history_cols]

    before = len(df)
    merged = df.merge(
        history_subset,
        on=HISTORY_KEYS,
        how="inner",
        validate="many_to_one",
    )
    dropped = before - len(merged)
    if dropped:
        print(
            f"{split_label.title()} split: dropped {dropped} row(s) with no matching history."
        )

    return merged


def infer_features(df: pd.DataFrame) -> List[str]:
    return sorted(
        col
        for col in df.columns
        if col not in RESERVED_COLS and not (col.startswith("targets_") and col != TARGET)
    )


def load_split(path: Path, history: Optional[pd.DataFrame], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data split: {path}")
    df = read_table(path)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in {path}")

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="raise").astype("int64")

    df = merge_with_history(df, history, label)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")

    global CATEGORICAL_FEATURES
    CATEGORICAL_FEATURES = cat_cols

    return X


def extract_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = prepare_features(df)
    y = df[TARGET]
    return X, y


def evaluate_split(name: str, model: lgb.Booster, X: pd.DataFrame, y: pd.Series) -> None:
    preds = model.predict(X, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    print(f"{name:<5} | MAE = {mae: .4f} | RMSE = {rmse: .4f}")


def train(args: argparse.Namespace) -> None:
    global FEATURES

    history_df = load_history(args.history)

    train_df = load_split(Path(args.train), history_df, "train")
    FEATURES = infer_features(train_df)
    print(f"Using {len(FEATURES)} features: {FEATURES}")

    val_df = load_split(Path(args.val), history_df, "val")
    test_df = load_split(Path(args.test), history_df, "test")

    X_train, y_train = extract_xy(train_df)
    X_val, y_val = extract_xy(val_df)
    X_test, y_test = extract_xy(test_df)

    if CATEGORICAL_FEATURES:
        print(f"Categorical features: {CATEGORICAL_FEATURES}")

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_set,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )

    params = {
        "objective": "regression",
        "metric": ["l1", "l2"],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
    }

    early_stop_cb = lgb.early_stopping(
        stopping_rounds=args.early_stop,
        verbose=True,
    )

    model = lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        num_boost_round=args.num_rounds,
        callbacks=[early_stop_cb],
    )

    print("\nEvaluation summary:")
    evaluate_split("train", model, X_train, y_train)
    evaluate_split("val", model, X_val, y_val)
    evaluate_split("test", model, X_test, y_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "lgbm_fantasy_pts.txt"
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    metadata = {
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "best_iteration": model.best_iteration,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    metadata_path = output_dir / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LightGBM fantasy-point model.")
    parser.add_argument("--train", default="../Features/player_games_train.parquet")
    parser.add_argument("--val", default="../Features/player_games_val.parquet")
    parser.add_argument("--test", default="../Features/player_games_test.parquet")
    parser.add_argument(
        "--history",
        type=Path,
        help="Flattened NHL game-history table (CSV or Parquet) to merge before training.",
    )
    parser.add_argument("--output-dir", default="../Models", type=Path)
    parser.add_argument("--num-rounds", type=int, default=5000, help="Max boosting rounds.")
    parser.add_argument("--early-stop", type=int, default=200, help="Early stopping patience.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)