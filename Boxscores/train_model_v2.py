"""
train_model_v2.py
XGBoost classifier: predict home_win (regular season only, no leakage)
Includes isotonic regression calibration on the 2024-25 test set.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")

print("Loading game dataset ...")
df = pd.read_parquet(BASE / "game_dataset.parquet")

# ── 1. FILTER: REGULAR SEASON ONLY ────────────────────────
df["game_type"] = df["game_pk"].astype(str).str[4:6]
df = df[df["game_type"] == "02"].copy()
print(f"  Regular season games: {len(df):,}")

# ── 2. REMOVE LEAKY FEATURES ──────────────────────────────
# These are computed FROM the game we're predicting
leaky = [
    "home_team_toi", "away_team_toi",
    "home_roster_size", "away_roster_size",
    "diff_roster_size",
]

exclude = [
    "game_pk", "game_date", "home_team", "away_team", "home_win",
    "home_gl_goalie_name", "away_gl_goalie_name",
    "home_gl_goalie_id", "away_gl_goalie_id",
    "game_type",
] + leaky

feature_cols = [c for c in df.columns if c not in exclude]
feature_cols = [c for c in feature_cols if df[c].dtype in ["float64", "float32", "int64", "int32"]]

print(f"  Features: {len(feature_cols)} (removed {len(leaky)} leaky)")

X = df[feature_cols]
y = df["home_win"]

# ── 3. TIME-BASED TRAIN/TEST SPLIT ────────────────────────
df["game_date"] = pd.to_datetime(df["game_date"])
train_mask = df["game_date"] < "2024-10-01"
test_mask = df["game_date"] >= "2024-10-01"

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"\n  Train: {len(X_train):,} games ({y_train.mean():.3f} home win rate)")
print(f"  Test:  {len(X_test):,} games ({y_test.mean():.3f} home win rate)")

# ── 4. TRAIN XGBOOST ──────────────────────────────────────
print("\n=== Training XGBoost ===")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.2,
    min_child_weight=20,
    reg_alpha=2.0,
    reg_lambda=10.0,
    gamma=1.0,
    scale_pos_weight=1.0,
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50,
)

# ── 5. EVALUATE (raw, uncalibrated) ───────────────────────
print("\n=== Evaluation on TEST set (2024-25 regular season) ===")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
ll = log_loss(y_test, y_pred_proba)
brier_raw = brier_score_loss(y_test, y_pred_proba)
baseline = max(y_test.mean(), 1 - y_test.mean())

print(f"\n  Accuracy:  {acc:.4f} ({acc:.1%})")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"  Log Loss:  {ll:.4f}")
print(f"  Brier:     {brier_raw:.4f} (raw, uncalibrated)")
print(f"  Baseline:  {baseline:.4f} (always pick majority)")
print(f"  Lift:      +{(acc - baseline)*100:.1f} percentage points over baseline")

print(f"\n{classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win'])}")

# ── 5b. FIT ISOTONIC CALIBRATOR ON TEST SET ───────────────
print("\n=== Fitting Isotonic Calibration on Test Set ===")
calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
calibrator.fit(y_pred_proba, y_test.values)

y_cal_proba = calibrator.predict(y_pred_proba)
brier_cal = brier_score_loss(y_test, y_cal_proba)
ll_cal = log_loss(y_test, y_cal_proba)
print(f"  Brier before calibration: {brier_raw:.4f}")
print(f"  Brier after calibration:  {brier_cal:.4f}  (improvement: {brier_raw - brier_cal:+.4f})")
print(f"  LogLoss after calibration: {ll_cal:.4f}")

# ── 6. MONTHLY PERFORMANCE ────────────────────────────────
print("=== Monthly Accuracy ===")
test_df = df[test_mask].copy()
# Use calibrated probabilities for predictions
test_df["cal_prob"] = y_cal_proba
test_df["pred"] = (test_df["cal_prob"] >= 0.5).astype(int)
test_df["correct"] = (test_df["pred"] == test_df["home_win"]).astype(int)
test_df["month"] = test_df["game_date"].dt.to_period("M")

monthly = test_df.groupby("month").agg(
    games=("correct", "count"),
    accuracy=("correct", "mean"),
).reset_index()

for _, row in monthly.iterrows():
    bar = "█" * int(row["accuracy"] * 40)
    print(f"  {row['month']}: {row['games']:3.0f} games, {row['accuracy']:.1%} {bar}")

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────
print("\n=== Top 30 Features ===")
importance = pd.Series(
    model.feature_importances_, index=feature_cols
).sort_values(ascending=False)

for i, (feat, imp) in enumerate(importance.head(30).items()):
    print(f"  {i+1:2d}. {feat:55s} {imp:.4f}")

# ── 8. CONFIDENCE BUCKETS (calibrated probabilities) ──────
print("\n=== Confidence Buckets (calibrated) ===")
test_results = pd.DataFrame({
    "prob_home_win": y_cal_proba,
    "actual": y_test.values,
})

bins = [0, 0.35, 0.45, 0.55, 0.65, 1.0]
labels = ["Strong Away", "Lean Away", "Toss-up", "Lean Home", "Strong Home"]
test_results["bucket"] = pd.cut(test_results["prob_home_win"], bins=bins, labels=labels)

for label in labels:
    subset = test_results[test_results["bucket"] == label]
    if len(subset) > 0:
        actual_hr = subset["actual"].mean()
        calibration = abs(subset["prob_home_win"].mean() - actual_hr)
        print(f"  {label:12s}: {len(subset):4d} games, "
              f"pred {subset['prob_home_win'].mean():.1%}, "
              f"actual {actual_hr:.1%}, "
              f"calibration error {calibration:.1%}")

# ── 9. HIGH CONFIDENCE PICKS ──────────────────────────────
print("\n=== High Confidence Picks (calibrated >65% or <35%) ===")
high_conf = test_results[
    (test_results["prob_home_win"] > 0.65) | (test_results["prob_home_win"] < 0.35)
]
if len(high_conf) > 0:
    hc_acc = (
        ((high_conf["prob_home_win"] > 0.5) & (high_conf["actual"] == 1)) |
        ((high_conf["prob_home_win"] < 0.5) & (high_conf["actual"] == 0))
    ).mean()
    print(f"  Games:    {len(high_conf):,} ({len(high_conf)/len(test_results):.1%} of all)")
    print(f"  Accuracy: {hc_acc:.1%}")

# ── 10. SAVE ──────────────────────────────────────────────
model_path = BASE / "xgb_game_model_v2.json"
model.save_model(str(model_path))

feat_path = BASE / "model_features_v2.txt"
with open(feat_path, "w") as f:
    for c in feature_cols:
        f.write(c + "\n")

cal_path = BASE / "game_calibrator.pkl"
with open(cal_path, "wb") as f:
    pickle.dump(calibrator, f)

print(f"\nModel saved:      {model_path}")
print(f"Features saved:   {feat_path}")
print(f"Calibrator saved: {cal_path}")
print("\n=== DONE ===")