"""
retrain.py
Full model retrain pipeline with new situational features.

Runs in order:
  1. collect_and_flatten.py     — latest skater box scores
  2. build_features.py          — skater rolling features
  3. pull_goalies.py            — latest goalie box scores
  4. build_goalie_features.py   — goalie rolling features + game_outcomes.parquet
  5. build_game_dataset_v2.py   — rebuild training dataset (now includes rest_days, b2b)
  6. train_model_v2.py          — retrain XGBoost + recalibrate

After this runs:
  - xgb_game_model_v2.json      (new model)
  - game_calibrator.pkl         (new calibrator, trained on 2024-25 test set raw outputs)
  - model_features_v2.txt       (updated feature list — now includes situational features)

Then run:
  python predict_today.py       — uses new model automatically

Usage:
  python retrain.py             — full retrain (takes ~5-10 mins)
  python retrain.py --dataset-only   — only steps 1-5, skip training
  python retrain.py --train-only     — only steps 5-6, skip data collection
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")

dataset_only = "--dataset-only" in sys.argv
train_only   = "--train-only"   in sys.argv

sep = "=" * 60

print(f"\n{sep}")
print(f"  NHL MODEL RETRAIN — {date.today()}")
print(f"  New features: rest_days_diff, home_b2b, away_b2b,")
print(f"                home_rest_days, away_rest_days")
print(sep)

# ── Step definitions ─────────────────────────────────────────────────────────
data_steps = [
    ("Pulling skater box scores",    "collect_and_flatten.py"),
    ("Building skater features",     "build_features.py"),
    ("Pulling goalie box scores",    "pull_goalies.py"),
    ("Building goalie features",     "build_goalie_features.py"),
    ("Rebuilding game dataset",      "build_game_dataset_v2.py"),
]

train_steps = [
    ("Training model + calibrator",  "train_model_v2.py"),
]

steps_to_run = []
if not train_only:
    steps_to_run += data_steps
elif train_only:
    # Still run dataset rebuild so new features are included
    steps_to_run += [data_steps[-1]]   # build_game_dataset_v2.py only

if not dataset_only:
    steps_to_run += train_steps

# ── Run ──────────────────────────────────────────────────────────────────────
for i, (desc, script) in enumerate(steps_to_run, 1):
    print(f"\n{'─'*40}")
    print(f"  [{i}/{len(steps_to_run)}] {desc} ...")
    print(f"{'─'*40}")

    result = subprocess.run(
        [sys.executable, str(BASE / script)],
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n  ❌ ERROR in {script}. Stopping.")
        print(f"     Fix the error above and re-run with --train-only to skip data collection.")
        sys.exit(1)

print(f"\n{sep}")
if not dataset_only:
    print(f"  ✅ RETRAIN COMPLETE")
    print(f"  New model:      xgb_game_model_v2.json")
    print(f"  New calibrator: game_calibrator.pkl")
    print(f"  New features:   model_features_v2.txt")
    print(f"\n  Next step: python predict_today.py")
else:
    print(f"  ✅ DATASET REBUILT (no training)")
    print(f"  Run 'python retrain.py --train-only' to train on new dataset")
print(sep + "\n")
