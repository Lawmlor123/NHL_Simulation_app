"""
train_model.py
XGBoost classifier: predict home_win
Time-based split: train on seasons 1-2, test on season 3
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
import xgboost as xgb

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")

print("Loading game dataset ...")
df = pd.read_parquet(BASE / "game_dataset.parquet")
print(f"  Shape: {df.shape}")

# ── 1. DEFINE FEATURES ────────────────────────────────────
# Exclude identifiers and target
exclude = [
    "game_pk", "game_date", "home_team", "away_team", "home_win",
    "home_gl_goalie_name", "away_gl_goalie_name",
    "home_gl_goalie_id", "away_gl_goalie_id",
]
feature_cols = [c for c in df.columns if c not in exclude]

# Only keep numeric columns
feature_cols = [c for c in feature_cols if df[c].dtype in ["float64", "float32", "int64", "int32"]]
print(f"  Features: {len(feature_cols)}")

X = df[feature_cols]
y = df["home_win"]

# ── 2. TIME-BASED TRAIN/TEST SPLIT ────────────────────────
# Train: 2022-23 + 2023-24, Test: 2024-25
df["game_date"] = pd.to_datetime(df["game_date"])
train_mask = df["game_date"] < "2024-10-01"
test_mask = df["game_date"] >= "2024-10-01"

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"\n  Train: {len(X_train):,} games ({y_train.mean():.3f} home win rate)")
print(f"  Test:  {len(X_test):,} games ({y_test.mean():.3f} home win rate)")

# ── 3. TRAIN XGBOOST ──────────────────────────────────────
print("\n=== Training XGBoost ===")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.3,
    min_child_weight=10,
    reg_alpha=1.0,
    reg_lambda=5.0,
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

# ── 4. EVALUATE ───────────────────────────────────────────
print("\n=== Evaluation on TEST set (2024-25 season) ===")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

print(f"\n  Accuracy:  {acc:.4f} ({acc:.1%})")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"  Log Loss:  {logloss:.4f}")
print(f"  Baseline:  {max(y_test.mean(), 1-y_test.mean()):.4f} (always pick majority)")

print(f"\n{classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win'])}")

# ── 5. FEATURE IMPORTANCE ─────────────────────────────────
print("\n=== Top 30 Features ===")
importance = pd.Series(
    model.feature_importances_, index=feature_cols
).sort_values(ascending=False)

for i, (feat, imp) in enumerate(importance.head(30).items()):
    print(f"  {i+1:2d}. {feat:50s} {imp:.4f}")

# ── 6. CONFIDENCE ANALYSIS ────────────────────────────────
print("\n=== Confidence Buckets ===")
test_results = pd.DataFrame({
    "prob_home_win": y_pred_proba,
    "actual": y_test.values,
})

bins = [0, 0.35, 0.45, 0.55, 0.65, 1.0]
labels = ["Strong Away", "Lean Away", "Toss-up", "Lean Home", "Strong Home"]
test_results["bucket"] = pd.cut(test_results["prob_home_win"], bins=bins, labels=labels)

for label in labels:
    subset = test_results[test_results["bucket"] == label]
    if len(subset) > 0:
        actual_home_rate = subset["actual"].mean()
        print(f"  {label:12s}: {len(subset):4d} games, "
              f"predicted ~{subset['prob_home_win'].mean():.1%} home win, "
              f"actual {actual_home_rate:.1%}")

# ── 7. SAVE MODEL ─────────────────────────────────────────
model_path = BASE / "xgb_game_model.json"
model.save_model(str(model_path))
print(f"\nModel saved: {model_path}")

# Save feature list
feat_path = BASE / "model_features.txt"
with open(feat_path, "w") as f:
    for c in feature_cols:
        f.write(c + "\n")
print(f"Feature list saved: {feat_path}")

print("\n=== DONE ===")