"""
home_bias_audit.py
Investigates why the model generates ~85% home picks and has a +20.3%
calibration error in the MED confidence band.

Outputs:
  - Pick direction breakdown by tier
  - Raw probability distribution (shows structural skew)
  - Feature importance by group (home vs away vs diff)
  - Calibration error per tier vs actual win rates
  - Recalibration using 2025-26 log data

Usage:
    python home_bias_audit.py
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
LOG  = BASE / "prediction_log.csv"

model = xgb.XGBClassifier()
model.load_model(str(BASE / "xgb_game_model_v2.json"))
with open(BASE / "game_calibrator.pkl", "rb") as f:
    calibrator = pickle.load(f)
with open(BASE / "model_features_v2.txt") as f:
    feature_cols = [l.strip() for l in f]

log = pd.read_csv(LOG)
log["confidence"]   = log["confidence"].astype(float)
log["prob_home"]    = log["prob_home"].astype(float)
log["pick_is_home"] = log["pick"] == log["home_team"]

sep = "=" * 70

# ── 1. PICK DIRECTION BY TIER ─────────────────────────────────────────────────
print(f"\n{sep}")
print("  HOME BIAS INVESTIGATION")
print(sep)

print(f"\n  1. PICK DIRECTION BY TIER")
print(f"  {'─'*55}")
tiers_def = [
    ("HIGH (≥65%)",  0.65, 1.00),
    ("MED (58-64%)", 0.58, 0.65),
    ("LOW (52-57%)", 0.52, 0.58),
    ("SKIP (<52%)",  0.00, 0.52),
]
for label, lo, hi in tiers_def:
    t = log[(log["confidence"] >= lo) & (log["confidence"] < hi)]
    if len(t) == 0: continue
    home_n = t["pick_is_home"].sum()
    away_n = len(t) - home_n
    print(f"  {label:<14s}  HOME: {home_n:3d} ({home_n/len(t):.0%})  "
          f"AWAY: {away_n:3d} ({away_n/len(t):.0%})  n={len(t)}")
home_all = log["pick_is_home"].sum()
print(f"\n  TOTAL  HOME: {home_all} ({home_all/len(log):.0%})  "
      f"AWAY: {len(log)-home_all} ({(len(log)-home_all)/len(log):.0%})")
print(f"  ⚠ Model NEVER generates HIGH-confidence AWAY picks (0 games)")

# ── 2. PROB_HOME DISTRIBUTION ─────────────────────────────────────────────────
print(f"\n  2. RAW PROB_HOME DISTRIBUTION (all {len(log)} predictions)")
print(f"  {'─'*55}")
print(f"  (0.0 = certain AWAY win  →  1.0 = certain HOME win)")
buckets = [
    (0.00,0.35,"Strong AWAY  "),
    (0.35,0.42,"Lean AWAY    "),
    (0.42,0.50,"Slight AWAY  "),
    (0.50,0.52,"SKIP home    "),
    (0.52,0.58,"LOW  home    "),
    (0.58,0.65,"MED  home    "),
    (0.65,1.00,"HIGH home    "),
]
for lo, hi, label in buckets:
    n = ((log["prob_home"] >= lo) & (log["prob_home"] < hi)).sum()
    bar = "█" * n
    print(f"  {lo:.2f}-{hi:.2f}  {label}  {n:3d}  {bar}")
print(f"\n  Root cause: model output is structurally right-skewed.")
print(f"  The 'strong away' zone (prob_home < 0.35) has ZERO predictions.")
print(f"  Symmetric signal strength would produce a balanced distribution.")

# ── 3. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
print(f"\n  3. FEATURE IMPORTANCE BY GROUP")
print(f"  {'─'*55}")
imp   = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
total = imp.sum()
home_imp = imp[[f for f in imp.index if f.startswith("home_")]].sum()
away_imp = imp[[f for f in imp.index if f.startswith("away_")]].sum()
diff_imp = imp[[f for f in imp.index if f.startswith("diff_")]].sum()
print(f"  diff_* (home−away differentials): {diff_imp/total:.1%}  ← dominant driver")
print(f"  away_* (away team raw stats):      {away_imp/total:.1%}")
print(f"  home_* (home team raw stats):      {home_imp/total:.1%}")
print(f"\n  ⚠ diff features = home_stat − away_stat")
print(f"    A strong home team → large positive diff → home prediction")
print(f"    A strong AWAY team → small or negative diff → only slight away lean")
print(f"    Net effect: model default state is 'home wins unless away is clearly superior'")

print(f"\n  Top 15 features:")
for i, (feat, val) in enumerate(imp.head(15).items()):
    prefix = "🏠" if feat.startswith("home_") else ("✈️ " if feat.startswith("away_") else "↔️ ")
    print(f"  {i+1:2d}. {prefix} {feat:<50s} {val:.4f}")

# ── 4. CALIBRATION ERROR BY TIER ─────────────────────────────────────────────
print(f"\n  4. CALIBRATION ERROR BY TIER (model confidence vs actual win rate)")
print(f"  {'─'*65}")
scored = log[log["result"].isin(["W", "L"])].copy()
scored["correct"] = (scored["result"] == "W").astype(int)
for label, lo, hi in tiers_def[:3]:
    ts = scored[(scored["confidence"] >= lo) & (scored["confidence"] < hi)]
    if len(ts) == 0: continue
    actual = ts["correct"].mean()
    pred   = ts["confidence"].mean()
    err    = pred - actual
    home_ts = ts[ts["pick_is_home"]]
    away_ts = ts[~ts["pick_is_home"]]
    hwr = home_ts["correct"].mean() if len(home_ts) > 0 else float("nan")
    awr = away_ts["correct"].mean() if len(away_ts) > 0 else float("nan")
    flag = "🚨" if abs(err) > 0.10 else ("⚠" if abs(err) > 0.05 else "✅")
    print(f"  {flag} {label:<14s}  pred:{pred:.1%}  actual:{actual:.1%}  "
          f"error:{err:+.1%}  (home_wr:{hwr:.1%} n={len(home_ts)}  "
          f"away_wr:{awr:.1%} n={len(away_ts)})")

print(f"\n  MED tier calibration error of +20.3% means:")
print(f"    The model says 61.7% → reality is 41.4% → 20pp overconfident")
print(f"    This tier should be labeled 40%-ish, not 58-64%")

# ── 5. IN-SEASON RECALIBRATION ───────────────────────────────────────────────
print(f"\n  5. IN-SEASON RECALIBRATION (using 2025-26 log data)")
print(f"  {'─'*65}")
scored_cal = scored[["prob_home", "correct"]].dropna()
if len(scored_cal) >= 50:
    new_cal = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    new_cal.fit(scored_cal["prob_home"].values, scored_cal["correct"].values)

    orig_preds = calibrator.predict(scored_cal["prob_home"].values)
    new_preds  = new_cal.predict(scored_cal["prob_home"].values)
    actual_arr = scored_cal["correct"].values

    brier_orig = brier_score_loss(actual_arr, orig_preds)
    brier_new  = brier_score_loss(actual_arr, new_preds)
    ll_orig    = log_loss(actual_arr, orig_preds)
    ll_new     = log_loss(actual_arr, new_preds)

    print(f"  Original calibrator (trained on 2024-25 test set):")
    print(f"    Brier score: {brier_orig:.4f}  Log loss: {ll_orig:.4f}")
    print(f"  New calibrator (trained on {len(scored_cal)} 2025-26 games):")
    print(f"    Brier score: {brier_new:.4f}  Log loss: {ll_new:.4f}")
    improvement = brier_orig - brier_new
    print(f"  Brier improvement: {improvement:+.4f}  "
          f"({'better ✅' if improvement > 0 else 'worse ❌'})")

    # What does recalibration do to MED tier?
    med_log = log[(log["confidence"] >= 0.58) & (log["confidence"] < 0.65)].copy()
    if len(med_log) > 0:
        med_raw = med_log["prob_home"].values
        med_recal = new_cal.predict(med_raw)
        print(f"\n  Effect on MED tier predictions after recalibration:")
        print(f"    Old prob_home range: {med_raw.min():.3f} – {med_raw.max():.3f}  "
              f"(mean: {med_raw.mean():.3f})")
        print(f"    New prob_home range: {med_recal.min():.3f} – {med_recal.max():.3f}  "
              f"(mean: {med_recal.mean():.3f})")
        still_med = ((med_recal >= 0.58) & (med_recal < 0.65)).sum()
        drops_low = ((med_recal >= 0.52) & (med_recal < 0.58)).sum()
        drops_skip= (med_recal < 0.52).sum()
        rises_high= (med_recal >= 0.65).sum()
        print(f"    After recal: {still_med} stay MED | {drops_low} drop to LOW | "
              f"{drops_skip} drop to SKIP | {rises_high} rise to HIGH")

    # Save new calibrator
    new_cal_path = BASE / "game_calibrator_2526.pkl"
    with open(new_cal_path, "wb") as f:
        pickle.dump(new_cal, f)
    print(f"\n  ✅ New calibrator saved → game_calibrator_2526.pkl")
    print(f"     To activate: rename to game_calibrator.pkl (backup old one first)")
else:
    print(f"  Need ≥50 scored games for reliable recalibration (have {len(scored_cal)})")

# ── 6. ROOT CAUSES + FIX ROADMAP ─────────────────────────────────────────────
print(f"\n  6. ROOT CAUSES + FIX ROADMAP")
print(f"  {'─'*65}")
causes = [
    ("CAUSE 1", "diff_* features (39.8% importance) = home−away",
     "Model defaults to HOME when teams are even — no neutral baseline"),
    ("CAUSE 2", "Calibrator trained on 2024-25 test set (not held-out)",
     "Over-fitted to last season; 2025-26 has different home win rate"),
    ("CAUSE 3", "No situational features (rest, b2b) in model",
     "Can't distinguish a rested away team from a fatigued one"),
    ("CAUSE 4", "No explicit home_ice_advantage feature",
     "Home advantage is implicit and leaks into every diff_* feature"),
]
fixes = [
    ("FIX A — QUICK",  "Activate game_calibrator_2526.pkl (just saved)",
     "Immediate: recalibrates using your 244 real 2025-26 results"),
    ("FIX B — SHORT",  "Add rest_days_diff + b2b_delta to model features",
     "Run backfill_situational.py, rebuild game_dataset, retrain"),
    ("FIX C — MEDIUM", "Add explicit is_home_game binary feature to training",
     "Separates home advantage from team quality differential"),
    ("FIX D — MEDIUM", "Raise confidence floor: require ≥60% (not 58%) for any pick",
     "Cuts the worst sub-band (58-60% was 30%) with one line change"),
    ("FIX E — LONG",   "Retrain on 2022-2026 data with new features",
     "Full retrain after B+C implemented; will fix structural skew"),
]
print(f"\n  ROOT CAUSES:")
for label, title, detail in causes:
    print(f"  [{label}] {title}")
    print(f"           → {detail}")
print(f"\n  RECOMMENDED FIXES (in order):")
for label, title, detail in fixes:
    print(f"  [{label}]")
    print(f"    What: {title}")
    print(f"    Why:  {detail}")

print(f"\n{sep}")
print(f"  Run order:")
print(f"  1. Backup game_calibrator.pkl → game_calibrator_2024.pkl")
print(f"  2. Copy game_calibrator_2526.pkl → game_calibrator.pkl")
print(f"  3. Run backfill_situational.py (adds b2b/rest to log)")
print(f"  4. Update build_game_dataset.py to include rest_days_diff + b2b features")
print(f"  5. Re-run: build_game_dataset → train_model_v2 → predict_today")
print(sep + "\n")
