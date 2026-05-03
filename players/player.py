import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════
SKATER_PATH = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores\skater_features.parquet"
GOALIE_PATH = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores\goalie_features.parquet"

# ═══════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
skaters = pd.read_parquet(SKATER_PATH)
goalies = pd.read_parquet(GOALIE_PATH)
print(f"  Skaters : {skaters.shape[0]:,} rows x {skaters.shape[1]} cols")
print(f"  Goalies : {goalies.shape[0]:,} rows x {goalies.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════
#  TARGET  –  did the skater record 1+ points?
# ═══════════════════════════════════════════════════════════════
skaters["scored_point"] = (skaters["points"] >= 1).astype(int)

# ═══════════════════════════════════════════════════════════════
#  OPPOSING GOALIE FEATURES
# ═══════════════════════════════════════════════════════════════
print("Joining opposing-goalie features...")

goalie_stat_cols = [
    "g_goalsAgainst_avg3", "g_goalsAgainst_avg5", "g_goalsAgainst_avg10",
    "g_goalsAgainst_season_avg",
    "g_savePct_avg3", "g_savePct_avg5", "g_savePct_avg10", "g_savePct_season_avg",
    "g_shotsAgainst_avg3", "g_shotsAgainst_avg5", "g_shotsAgainst_avg10",
    "g_shotsAgainst_season_avg",
    "g_win_rate_3", "g_win_rate_5", "g_win_rate_10", "g_win_rate_season",
]

goalie_starters = (
    goalies[goalies["is_starter"] == 1][["game_pk", "team"] + goalie_stat_cols]
    .drop_duplicates(subset=["game_pk", "team"], keep="first")
    .copy()
)

# Goalie's team = skater's opponent
goalie_rename = {"team": "opponent"}
for c in goalie_stat_cols:
    goalie_rename[c] = "opp_" + c
goalie_starters = goalie_starters.rename(columns=goalie_rename)

n_before = len(skaters)
skaters = skaters.merge(goalie_starters, on=["game_pk", "opponent"], how="left")
assert len(skaters) == n_before, "Row duplication after goalie merge!"
print(f"  Done. Goalie-feature NaN rate: {skaters['opp_g_savePct_avg5'].isna().mean():.1%}")

# ═══════════════════════════════════════════════════════════════
#  OPPONENT TEAM DEFENSE FEATURES
# ═══════════════════════════════════════════════════════════════
print("Building opponent team-defense features...")

game_info = (
    skaters[["game_pk", "game_date", "home_team", "away_team",
             "home_score", "away_score"]]
    .drop_duplicates(subset=["game_pk"])
)

home = game_info.rename(columns={
    "home_team": "t", "home_score": "gf", "away_score": "ga"
})[["game_pk", "game_date", "t", "gf", "ga"]]

away = game_info.rename(columns={
    "away_team": "t", "away_score": "gf", "home_score": "ga"
})[["game_pk", "game_date", "t", "gf", "ga"]]

team_games = pd.concat([home, away], ignore_index=True)
team_games = team_games.sort_values(["t", "game_date"]).reset_index(drop=True)

for col in ["gf", "ga"]:
    shifted = team_games.groupby("t")[col].shift(1)
    for w in [5, 10, 20]:
        team_games[f"{col}_avg{w}"] = shifted.rolling(w, min_periods=3).mean()
    team_games[f"{col}_season_avg"] = shifted.expanding(min_periods=3).mean()

opp_stat_cols = [c for c in team_games.columns if "_avg" in c]
opp_team = team_games[["game_pk", "t"] + opp_stat_cols].copy()
opp_team = opp_team.rename(columns={"t": "opponent"})
opp_team = opp_team.rename(columns={c: f"opp_{c}" for c in opp_stat_cols})

n_before = len(skaters)
skaters = skaters.merge(opp_team, on=["game_pk", "opponent"], how="left")
assert len(skaters) == n_before, "Row duplication after team-defense merge!"
print(f"  Done.")

# ═══════════════════════════════════════════════════════════════
#  FILTER TO USABLE ROWS
# ═══════════════════════════════════════════════════════════════
print("Filtering to modellable rows...")
df = skaters[(skaters["has_history"] == 1) & (skaters["game_num"] >= 5)].copy()
# Derive back-to-back flags (no parquet change needed — derived from rest_days)
df["back_to_back"] = (df["rest_days"] <= 1).astype(int)
df["b2b_road"]     = ((df["rest_days"] <= 1) & (df["is_home"] == 0)).astype(int)
print(f"  Rows: {len(df):,}   Point rate: {df['scored_point'].mean():.3f}")

# ═══════════════════════════════════════════════════════════════
#  FEATURE COLUMNS
# ═══════════════════════════════════════════════════════════════
player_features = [
    "goals_avg3", "goals_avg5", "goals_avg10", "goals_avg20", "goals_season_avg",
    "assists_avg3", "assists_avg5", "assists_avg10", "assists_avg20", "assists_season_avg",
    "points_avg3", "points_avg5", "points_avg10", "points_avg20", "points_season_avg",
    "shots_avg3", "shots_avg5", "shots_avg10", "shots_avg20", "shots_season_avg",
    "pp_goals_avg3", "pp_goals_avg5", "pp_goals_avg10", "pp_goals_avg20", "pp_goals_season_avg",
    "toi_min_avg3", "toi_min_avg5", "toi_min_avg10", "toi_min_avg20", "toi_min_season_avg",
    "shifts_avg3", "shifts_avg5", "shifts_avg10", "shifts_avg20", "shifts_season_avg",
    "hits_avg3", "hits_avg5", "hits_avg10",
    "blocked_shots_avg3", "blocked_shots_avg5", "blocked_shots_avg10",
    "takeaways_avg3", "takeaways_avg5",
    "giveaways_avg3", "giveaways_avg5",
    "plus_minus_avg3", "plus_minus_avg5", "plus_minus_avg10", "plus_minus_avg20",
    "sh_pct_avg3", "sh_pct_avg5", "sh_pct_avg10", "sh_pct_avg20",
    "pts_per60_avg3", "pts_per60_avg5", "pts_per60_avg10", "pts_per60_avg20",
]

context_features = [
    "is_home", "is_center", "is_wing", "is_defense",
    "rest_days", "back_to_back", "b2b_road", "days_into_season", "game_num",
]

opp_goalie_features = [
    "opp_g_goalsAgainst_avg3", "opp_g_goalsAgainst_avg5", "opp_g_goalsAgainst_avg10",
    "opp_g_goalsAgainst_season_avg",
    "opp_g_savePct_avg3", "opp_g_savePct_avg5", "opp_g_savePct_avg10",
    "opp_g_savePct_season_avg",
    "opp_g_shotsAgainst_avg3", "opp_g_shotsAgainst_avg5", "opp_g_shotsAgainst_avg10",
    "opp_g_shotsAgainst_season_avg",
    "opp_g_win_rate_5", "opp_g_win_rate_10", "opp_g_win_rate_season",
]

opp_team_features = [
    "opp_ga_avg5", "opp_ga_avg10", "opp_ga_avg20", "opp_ga_season_avg",
    "opp_gf_avg5", "opp_gf_avg10", "opp_gf_avg20", "opp_gf_season_avg",
]

feature_cols = player_features + context_features + opp_goalie_features + opp_team_features

# Verify all columns exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print(f"  WARNING – dropping {len(missing)} missing cols: {missing}")
    feature_cols = [c for c in feature_cols if c in df.columns]

print(f"  Total features: {len(feature_cols)}")

# Allow up to 5 NaN features per row (XGBoost handles NaN natively)
df = df.dropna(subset=feature_cols, thresh=len(feature_cols) - 5)
df = df.sort_values("game_date").reset_index(drop=True)

X = df[feature_cols]
y = df["scored_point"]
dates = df["game_date"]

print(f"  Final dataset: {len(X):,} rows, {len(feature_cols)} features")
print(f"  Target: {y.mean():.3f} point rate  ({y.sum():,} positives / {len(y):,} total)")

# ═══════════════════════════════════════════════════════════════
#  TIME-SERIES CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TRAINING  –  5-FOLD TIME-SERIES CV")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=5)
oof_preds = np.full(len(y), np.nan)
models = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print(f"\n{'─' * 55}")
    print(f"  Fold {fold + 1}/5")
    print(f"  Train: {len(train_idx):>7,}  ({dates.iloc[train_idx[0]].date()} → {dates.iloc[train_idx[-1]].date()})")
    print(f"  Val:   {len(val_idx):>7,}  ({dates.iloc[val_idx[0]].date()} → {dates.iloc[val_idx[-1]].date()})")
    print(f"  Train pts rate: {y_tr.mean():.3f}  |  Val pts rate: {y_val.mean():.3f}")

    model = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=50,
        reg_alpha=1.0,
        reg_lambda=3.0,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = preds

    auc = roc_auc_score(y_val, preds)
    brier = brier_score_loss(y_val, preds)
    ll = log_loss(y_val, preds)
    print(f"  ➜ AUC={auc:.4f}   Brier={brier:.4f}   LogLoss={ll:.4f}")
    models.append(model)

# ═══════════════════════════════════════════════════════════════
#  OVERALL RESULTS
# ═══════════════════════════════════════════════════════════════
mask = ~np.isnan(oof_preds)

print("\n" + "=" * 60)
print("  OVERALL OUT-OF-FOLD RESULTS")
print("=" * 60)
print(f"  AUC      : {roc_auc_score(y[mask], oof_preds[mask]):.4f}")
print(f"  LogLoss  : {log_loss(y[mask], oof_preds[mask]):.4f}")
print(f"  Brier    : {brier_score_loss(y[mask], oof_preds[mask]):.4f}")

# ── Calibration ──────────────────────────────────────────────
print(f"\n  {'Predicted':>10s}  {'Actual':>10s}  {'Gap':>8s}")
print(f"  {'─'*10}  {'─'*10}  {'─'*8}")
frac_pos, mean_pred = calibration_curve(y[mask], oof_preds[mask], n_bins=10)
for mp, fp in zip(mean_pred, frac_pos):
    gap = fp - mp
    print(f"  {mp:10.4f}  {fp:10.4f}  {gap:+8.4f}")

# ═══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE  (last fold, gain-based)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TOP 25 FEATURES")
print("=" * 60)
importance = pd.Series(
    models[-1].feature_importances_, index=feature_cols
).sort_values(ascending=False)

for i, (feat, imp) in enumerate(importance.head(25).items()):
    bar = "█" * int(imp / importance.max() * 30)
    print(f"  {i+1:2d}. {feat:<45s} {imp:.4f}  {bar}")

# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE TIERS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CONFIDENCE TIER ANALYSIS")
print("=" * 60)

res = pd.DataFrame({"pred": oof_preds[mask], "actual": y[mask].values})

print("\n  OVERS  (player records 1+ points):")
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    sub = res[res["pred"] >= t]
    if len(sub) >= 20:
        print(f"    pred >= {t:.2f} :  {len(sub):>6,} picks   hit rate {sub['actual'].mean():.1%}")

print("\n  UNDERS  (player records 0 points):")
for t in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
    sub = res[res["pred"] <= t]
    if len(sub) >= 20:
        print(f"    pred <= {t:.2f} :  {len(sub):>6,} picks   miss rate {1 - sub['actual'].mean():.1%}")

# ═══════════════════════════════════════════════════════════════
#  BY-POSITION BREAKDOWN
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  RESULTS BY POSITION")
print("=" * 60)

df_eval = df.loc[mask].copy()
df_eval["pred"] = oof_preds[mask]

for col, name in [("is_center", "Centers"), ("is_wing", "Wings"), ("is_defense", "Defensemen")]:
    sub = df_eval[df_eval[col] == 1]
    if len(sub) > 100:
        auc = roc_auc_score(sub["scored_point"], sub["pred"])
        brier = brier_score_loss(sub["scored_point"], sub["pred"])
        rate = sub["scored_point"].mean()
        print(f"  {name:<15s}  n={len(sub):>6,}   base_rate={rate:.3f}   AUC={auc:.4f}   Brier={brier:.4f}")

print("\n✓ Done!")