"""
backfill_situational.py
Backfills day_of_week, is_weekend, home_b2b, away_b2b,
home_rest_days, away_rest_days into existing prediction_log.csv entries.

Uses the NHL score API (same endpoint as predict_today.py) to look back
up to 8 days before each prediction date and find each team's last game.

Safe to re-run — only fills rows where b2b columns are blank.
Usage:
    python backfill_situational.py
"""
import pandas as pd
import requests
from pathlib import Path
from datetime import date, datetime, timedelta

BASE     = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
LOG_FILE = BASE / "prediction_log.csv"
LOOKBACK = 8   # days to scan back per prediction date

# ── Load log ─────────────────────────────────────────────────────────────────
df = pd.read_csv(LOG_FILE)
df["pred_date"] = pd.to_datetime(df["pred_date"]).dt.date

# Ensure situational columns exist
for col in ["day_of_week", "is_weekend", "home_b2b", "away_b2b",
            "home_rest_days", "away_rest_days"]:
    if col not in df.columns:
        df[col] = ""

# Only process rows that still need backfill
needs = df["home_b2b"].isna() | (df["home_b2b"] == "")
rows_to_fill = df[needs]

if len(rows_to_fill) == 0:
    print("✅ All rows already have situational data. Nothing to do.")
    exit(0)

print(f"Backfilling {len(rows_to_fill)} rows across "
      f"{rows_to_fill['pred_date'].nunique()} dates ...\n")

# ── Cache: date → {team: last_game_date} so we only hit the API once per date ─
rest_cache = {}   # keyed by (pred_date, team)
score_cache = {}  # keyed by date string → API response

def fetch_score(d):
    """Fetch completed games for a date (cached)."""
    key = str(d)
    if key in score_cache:
        return score_cache[key]
    try:
        r = requests.get(f"https://api-web.nhle.com/v1/score/{d}", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        data = {}
    score_cache[key] = data
    return data

def get_rest(team, pred_date):
    """Return (rest_days, b2b) for a team as of pred_date."""
    cache_key = (pred_date, team)
    if cache_key in rest_cache:
        return rest_cache[cache_key]

    for offset in range(1, LOOKBACK + 1):
        check = pred_date - timedelta(days=offset)
        data = fetch_score(check)
        for g in data.get("games", []):
            if g.get("gameState") not in ("FINAL", "OFF"):
                continue
            home = g["homeTeam"]["abbrev"]
            away = g["awayTeam"]["abbrev"]
            if team in (home, away):
                rest = offset
                result = (rest, rest == 1)
                rest_cache[cache_key] = result
                return result

    result = (None, False)
    rest_cache[cache_key] = result
    return result

# ── Process each unique pred_date ─────────────────────────────────────────────
unique_dates = sorted(rows_to_fill["pred_date"].unique())
total_updated = 0

for pred_date in unique_dates:
    date_rows = df[df["pred_date"] == pred_date]
    dow = pred_date.strftime("%A")
    is_weekend = int(pred_date.weekday() >= 5)

    updated_this_date = 0
    for idx, row in date_rows.iterrows():
        if not (pd.isna(df.at[idx, "home_b2b"]) or df.at[idx, "home_b2b"] == ""):
            continue  # already filled

        home_rest, home_b2b = get_rest(row["home_team"], pred_date)
        away_rest, away_b2b = get_rest(row["away_team"], pred_date)

        df.at[idx, "day_of_week"]    = dow
        df.at[idx, "is_weekend"]     = is_weekend
        df.at[idx, "home_b2b"]       = int(home_b2b)
        df.at[idx, "away_b2b"]       = int(away_b2b)
        df.at[idx, "home_rest_days"] = home_rest if home_rest is not None else ""
        df.at[idx, "away_rest_days"] = away_rest if away_rest is not None else ""
        updated_this_date += 1

    total_updated += updated_this_date
    b2b_today = df[
        (df["pred_date"] == pred_date) &
        ((df["home_b2b"] == 1) | (df["away_b2b"] == 1))
    ]
    b2b_note = f"  ({len(b2b_today)} b2b game(s))" if len(b2b_today) > 0 else ""
    print(f"  {pred_date}  — {updated_this_date} rows filled{b2b_note}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(LOG_FILE, index=False)
print(f"\n✅ Done — {total_updated} rows updated and saved to {LOG_FILE}")

# ── Quick b2b summary ─────────────────────────────────────────────────────────
scored = df[df["result"].isin(["W", "L"])].copy()
scored["home_b2b"] = pd.to_numeric(scored["home_b2b"], errors="coerce").fillna(0)
scored["away_b2b"] = pd.to_numeric(scored["away_b2b"], errors="coerce").fillna(0)
scored["any_b2b"]  = (scored["home_b2b"] == 1) | (scored["away_b2b"] == 1)
scored["correct"]  = (scored["result"] == "W").astype(int)

print(f"\n{'='*60}")
print(f"  QUICK B2B SPLIT (all tiers, all scored games)")
print(f"{'='*60}")
for label, mask in [("Any B2B game", scored["any_b2b"]),
                    ("No B2B", ~scored["any_b2b"])]:
    grp = scored[mask]
    if len(grp) == 0:
        continue
    w = grp["correct"].sum()
    print(f"  {label:<18s}  {w}W-{len(grp)-w}L  ({w/len(grp):.1%})  n={len(grp)}")

# Pick-side b2b (team you picked is on b2b)
scored["pick_is_home"] = scored["pick"] == scored["home_team"]
scored["pick_b2b"] = (
    (scored["pick_is_home"] & (scored["home_b2b"] == 1)) |
    (~scored["pick_is_home"] & (scored["away_b2b"] == 1))
)
grp_pb2b = scored[scored["pick_b2b"]]
grp_npb2b = scored[~scored["pick_b2b"]]
if len(grp_pb2b) > 0:
    w = grp_pb2b["correct"].sum()
    print(f"  {'Pick on B2B':<18s}  {w}W-{len(grp_pb2b)-w}L  ({w/len(grp_pb2b):.1%})  n={len(grp_pb2b)}")
if len(grp_npb2b) > 0:
    w = grp_npb2b["correct"].sum()
    print(f"  {'Pick NOT on B2B':<18s}  {w}W-{len(grp_npb2b)-w}L  ({w/len(grp_npb2b):.1%})  n={len(grp_npb2b)}")
print(f"{'='*60}\n")
