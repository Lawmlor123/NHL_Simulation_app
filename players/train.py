import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import warnings
import pickle
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════
SKATER_PATH = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores\skater_features.parquet"
GOALIE_PATH = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores\goalie_features.parquet"
MODEL_DIR = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  TARGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════
TARGETS = {
    "points_1plus":  {"col": "points",   "threshold": 1, "desc": "1+ Points"},
    "points_2plus":  {"col": "points",   "threshold": 2, "desc": "2+ Points"},
    "goals_1plus":   {"col": "goals",    "threshold": 1, "desc": "1+ Goals"},
    "assists_1plus": {"col": "assists",  "threshold": 1, "desc": "1+ Assists"},
    "shots_3plus":   {"col": "shots",    "threshold": 3, "desc": "3+ SOG"},
    "shots_4plus":   {"col": "shots",    "threshold": 4, "desc": "4+ SOG"},
    "shots_5plus":   {"col": "shots",    "threshold": 5, "desc": "5+ SOG"},
}

# ── Per-target XGBoost param overrides ──────────────────────────────────────
# Low base-rate / high-threshold props need heavier regularization to prevent
# overconfidence at mid-high probability tiers (observed in calibration checks).
TARGET_PARAMS = {
    "points_2plus": {
        "max_depth": 4,
        "min_child_weight": 80,
        "reg_alpha": 1.5,
        "reg_lambda": 4.0,
    },
    "shots_4plus": {
        "max_depth": 4,
        "min_child_weight": 100,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
    },
    "shots_5plus": {
        "max_depth": 3,
        "min_child_weight": 150,
        "reg_alpha": 3.0,
        "reg_lambda": 7.0,
    },
}

# ═══════════════════════════════════════════════════════════════
#  LOAD + MERGE DATA  (same as player.py)
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
skaters = pd.read_parquet(SKATER_PATH)
goalies = pd.read_parquet(GOALIE_PATH)
print(f"  Skaters : {skaters.shape[0]:,} rows x {skaters.shape[1]} cols")
print(f"  Goalies : {goalies.shape[0]:,} rows x {goalies.shape[1]} cols")

# ── Opposing goalie features ─────────────────────────────────
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

goalie_rename = {"team": "opponent"}
for c in goalie_stat_cols:
    goalie_rename[c] = "opp_" + c
goalie_starters = goalie_starters.rename(columns=goalie_rename)

n_before = len(skaters)
skaters = skaters.merge(goalie_starters, on=["game_pk", "opponent"], how="left")
assert len(skaters) == n_before

# ── Opponent team defense features ───────────────────────────
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
assert len(skaters) == n_before

# ── Derive back-to-back flags (no parquet change needed) ─────────────────────
# rest_days <= 1 = played yesterday (classic B2B); b2b_road = B2B + away game,
# which is a known extra suppressor of player performance.
skaters["back_to_back"] = (skaters["rest_days"] <= 1).astype(int)
skaters["b2b_road"]     = ((skaters["rest_days"] <= 1) & (skaters["is_home"] == 0)).astype(int)

print("  Data merged.")

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
feature_cols = [c for c in feature_cols if c in skaters.columns]

# ═══════════════════════════════════════════════════════════════
#  FILTER TO MODELLABLE ROWS
# ═══════════════════════════════════════════════════════════════
df = skaters[(skaters["has_history"] == 1) & (skaters["game_num"] >= 5)].copy()
df = df.dropna(subset=feature_cols, thresh=len(feature_cols) - 5)
df = df.sort_values("game_date").reset_index(drop=True)

print(f"  Modellable rows: {len(df):,}")
print(f"  Features: {len(feature_cols)}")

# ═══════════════════════════════════════════════════════════════
#  TRAIN ALL TARGETS
# ═══════════════════════════════════════════════════════════════
trained_models = {}

for target_name, target_cfg in TARGETS.items():
    col = target_cfg["col"]
    threshold = target_cfg["threshold"]
    desc = target_cfg["desc"]

    print(f"\n{'═' * 60}")
    print(f"  TARGET: {desc}  ({col} >= {threshold})")
    print(f"{'═' * 60}")

    y = (df[col] >= threshold).astype(int)
    pos_rate = y.mean()
    print(f"  Positive rate: {pos_rate:.3f}  ({y.sum():,} / {len(y):,})")

    if pos_rate < 0.02 or pos_rate > 0.98:
        print(f"  ⚠ Skipping — rate too extreme for useful model")
        continue

    X = df[feature_cols]
    dates = df["game_date"]

    # ── Build per-target XGBoost params ──────────────────────
    base_params = {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    params = {**base_params, **TARGET_PARAMS.get(target_name, {})}
    overrides = TARGET_PARAMS.get(target_name, {})
    if overrides:
        override_str = ", ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"  ⚙ Param overrides: {override_str}")

    # ── Quick 3-fold TSCV to evaluate ────────────────────────
    tscv = TimeSeriesSplit(n_splits=3)
    oof_preds = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = XGBClassifier(**params)
        fold_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        oof_preds[val_idx] = fold_model.predict_proba(X_val)[:, 1]

    mask = ~np.isnan(oof_preds)
    auc = roc_auc_score(y[mask], oof_preds[mask])
    brier = brier_score_loss(y[mask], oof_preds[mask])
    ll = log_loss(y[mask], oof_preds[mask])
    print(f"  3-fold TSCV:  AUC={auc:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}")

    # ── Calibration: dedicated 75/25 chronological hold-out ──────────────────
    # FIX: Previously fitted isotonic on OOF preds from partial fold models.
    # Those models hadn't seen their val sets → raw probs were systematically
    # lower than the final model (trained on 100% of data). The calibrator
    # learned a mapping that was too aggressive at the high end → overconfidence.
    #
    # Fix: train a cal_model on the first 75% of data, predict on the last 25%
    # (true held-out, chronologically), fit isotonic on those. The cal_model
    # architecture matches the final model, so the raw probability distribution
    # is much closer → calibrator generalizes properly at inference time.
    cal_split = int(len(X) * 0.75)
    cal_model = XGBClassifier(**params)
    cal_model.fit(
        X.iloc[:cal_split], y.iloc[:cal_split],
        eval_set=[(X.iloc[cal_split:], y.iloc[cal_split:])],
        verbose=0,
    )
    cal_raw = cal_model.predict_proba(X.iloc[cal_split:])[:, 1]
    cal_y   = y.values[cal_split:]

    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(cal_raw, cal_y)
    brier_cal = brier_score_loss(cal_y, iso.predict(cal_raw))
    print(f"  Calibrated Brier (hold-out 25%): {brier_cal:.4f}")

    # ── Train final model on ALL data ────────────────────────
    print(f"  Training final model on all {len(X):,} rows...")
    final_model = XGBClassifier(**params)

    # Use last 20% as eval set for early stopping
    split_pt = int(len(X) * 0.8)
    final_model.fit(
        X.iloc[:split_pt], y.iloc[:split_pt],
        eval_set=[(X.iloc[split_pt:], y.iloc[split_pt:])],
        verbose=0,
    )

    trained_models[target_name] = {
        "model": final_model,
        "calibrator": iso,
        "auc": auc,
        "brier": brier,
        "pos_rate": pos_rate,
        "desc": desc,
    }

    print(f"  ✓ Saved: {desc}")

    # ── Confidence tiers ─────────────────────────────────────
    cal_preds = iso.predict(oof_preds[mask])
    res = pd.DataFrame({"pred": cal_preds, "actual": y[mask].values})

    print(f"\n  OVERS ({desc}):")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sub = res[res["pred"] >= t]
        if len(sub) >= 20:
            print(f"    pred >= {t:.2f} :  {len(sub):>6,} picks   hit rate {sub['actual'].mean():.1%}")

    print(f"  UNDERS ({desc}):")
    for t in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
        sub = res[res["pred"] <= t]
        if len(sub) >= 20:
            print(f"    pred <= {t:.2f} :  {len(sub):>6,} picks   miss rate {1 - sub['actual'].mean():.1%}")

# ═══════════════════════════════════════════════════════════════
#  SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════
save_path = os.path.join(MODEL_DIR, "nhl_player_models.pkl")
save_obj = {
    "models": trained_models,
    "feature_cols": feature_cols,
    "trained_date": datetime.now().isoformat(),
}

with open(save_path, "wb") as f:
    pickle.dump(save_obj, f)

print(f"\n{'═' * 60}")
print(f"  ALL MODELS SAVED")
print(f"{'═' * 60}")
print(f"  Path: {save_path}")
print(f"  Targets trained:")
for name, info in trained_models.items():
    print(f"    {info['desc']:<20s}  AUC={info['auc']:.4f}  base_rate={info['pos_rate']:.3f}")

print(f"\n✓ Done! Run predict.py next to generate tonight's picks.")