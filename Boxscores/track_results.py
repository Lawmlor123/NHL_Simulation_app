"""
track_results.py
Backfill actual outcomes for past predictions and show model performance.
Usage:
    python track_results.py              # update results + show report
    python track_results.py --report     # just show report (no API calls)
    python track_results.py --days 30    # report for last 30 days only
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import date, datetime, timedelta
import sys

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
LOG_FILE = BASE / "prediction_log.csv"

# ── Parse arguments ────────────────────────────────────────
report_only = "--report" in sys.argv
days_filter = None
for i, arg in enumerate(sys.argv):
    if arg == "--days" and i + 1 < len(sys.argv):
        days_filter = int(sys.argv[i + 1])


def fetch_game_results(game_date):
    """Pull final scores from NHL API for a given date."""
    url = f"https://api-web.nhle.com/v1/score/{game_date}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠ Could not fetch results for {game_date}: {e}")
        return {}

    results = {}
    for g in data.get("games", []):
        state = g.get("gameState", "")
        if state not in ("FINAL", "OFF"):
            continue

        home_abbrev = g["homeTeam"]["abbrev"]
        away_abbrev = g["awayTeam"]["abbrev"]
        home_score = g["homeTeam"].get("score", 0)
        away_score = g["awayTeam"].get("score", 0)

        if home_score > away_score:
            winner = home_abbrev
        else:
            winner = away_abbrev

        key = (away_abbrev, home_abbrev)
        results[key] = {
            "actual_winner": winner,
            "home_goals": home_score,
            "away_goals": away_score,
        }

    return results


# ── 1. LOAD PREDICTION LOG ────────────────────────────────
if not LOG_FILE.exists():
    print("No prediction log found. Run predict_today.py first!")
    sys.exit(1)

df = pd.read_csv(LOG_FILE)
print(f"\n{'='*70}")
print(f"  NHL Model Performance Tracker")
print(f"{'='*70}")
print(f"\n  Loaded {len(df)} predictions from log")

# ── 2. BACKFILL RESULTS ───────────────────────────────────
if not report_only:
    needs_results = df[
        (df["actual_winner"].isna() | (df["actual_winner"] == ""))
        & (df["pred_date"] < str(date.today()))
    ]

    dates_to_check = needs_results["pred_date"].unique()

    if len(dates_to_check) == 0:
        print("  All past predictions already have results ✅\n")
    else:
        print(f"\n  Fetching results for {len(dates_to_check)} dates ...\n")

        updated_count = 0
        for d in sorted(dates_to_check):
            results = fetch_game_results(d)
            if not results:
                continue

            for idx, row in df[df["pred_date"] == d].iterrows():
                key = (row["away_team"], row["home_team"])
                if key in results:
                    res = results[key]
                    df.at[idx, "actual_winner"] = res["actual_winner"]
                    df.at[idx, "home_goals"] = res["home_goals"]
                    df.at[idx, "away_goals"] = res["away_goals"]
                    df.at[idx, "result"] = "W" if row["pick"] == res["actual_winner"] else "L"
                    updated_count += 1

            print(f"    {d}: updated {sum(1 for k in results if (df['pred_date'] == d).any())} games")

        df.to_csv(LOG_FILE, index=False)
        print(f"\n  💾 Updated {updated_count} predictions with actual results\n")

# ── 3. PERFORMANCE REPORT ─────────────────────────────────
# Only score HIGH (≥65%) and LOW (52-57%) — matches log_eligible in predict_today.py
# Excluded: MED (58-64%) chronic underperformer (41.4%), SKIP (<52%) poor signal (35.7%)
scored = df[
    df["result"].isin(["W", "L"]) &
    (
        (df["confidence"].astype(float) >= 0.65) |
        ((df["confidence"].astype(float) >= 0.52) & (df["confidence"].astype(float) < 0.58))
    )
].copy()

if days_filter:
    cutoff = str(date.today() - timedelta(days=days_filter))
    scored = scored[scored["pred_date"] >= cutoff]
    print(f"  📅 Filtering to last {days_filter} days (since {cutoff})")

if len(scored) == 0:
    print("\n  No scored predictions yet. Run this again after games finish!\n")
    sys.exit(0)

scored["correct"] = (scored["result"] == "W").astype(int)
scored["confidence"] = scored["confidence"].astype(float)

print(f"\n{'='*70}")
print(f"  MODEL PERFORMANCE REPORT")
date_range = f"{scored['pred_date'].min()} to {scored['pred_date'].max()}"
print(f"  Period: {date_range}")
print(f"{'='*70}")

# ── Overall record ────────────────────────────────────────
total = len(scored)
wins = scored["correct"].sum()
losses = total - wins
pct = wins / total if total > 0 else 0

print(f"\n  OVERALL RECORD")
print(f"  {'─'*50}")
print(f"  Record:     {wins}W - {losses}L  ({pct:.1%})")
print(f"  Total games: {total}")

# ── By confidence tier ────────────────────────────────────
print(f"\n  BY CONFIDENCE TIER")
print(f"  {'─'*60}")
print(f"  {'Tier':<12s} {'Record':>10s} {'Win%':>7s} {'Avg Conf':>9s} {'Games':>6s}")
print(f"  {'─'*60}")

tiers = [
    ("HIGH (≥65%)", scored[scored["confidence"] >= 0.65]),
    ("MED (58-64%)", scored[(scored["confidence"] >= 0.58) & (scored["confidence"] < 0.65)]),
    ("LOW (52-57%)", scored[(scored["confidence"] >= 0.52) & (scored["confidence"] < 0.58)]),
    ("SKIP (<52%)", scored[scored["confidence"] < 0.52]),
]

for label, tier_df in tiers:
    if len(tier_df) == 0:
        print(f"  {label:<12s} {'—':>10s}")
        continue
    tw = tier_df["correct"].sum()
    tl = len(tier_df) - tw
    tp = tw / len(tier_df)
    avg_c = tier_df["confidence"].mean()
    print(f"  {label:<12s} {tw}W-{tl}L{'':<4s} {tp:>6.1%}  {avg_c:>8.1%}  {len(tier_df):>5d}")

# ── By pick direction (home vs away) ─────────────────────
print(f"\n  BY PICK DIRECTION")
print(f"  {'─'*50}")

home_picks = scored[scored["pick"] == scored["home_team"]]
away_picks = scored[scored["pick"] == scored["away_team"]]

if len(home_picks) > 0:
    hw = home_picks["correct"].sum()
    hl = len(home_picks) - hw
    print(f"  HOME picks:  {hw}W-{hl}L  ({hw/len(home_picks):.1%})  n={len(home_picks)}")

if len(away_picks) > 0:
    aw = away_picks["correct"].sum()
    al = len(away_picks) - aw
    print(f"  AWAY picks:  {aw}W-{al}L  ({aw/len(away_picks):.1%})  n={len(away_picks)}")

# ── Value picks performance ───────────────────────────────
scored["edge"] = pd.to_numeric(scored["edge"], errors="coerce")
value = scored[scored["edge"] > 0.03]

if len(value) > 0:
    print(f"\n  💰 VALUE PICKS (edge > 3%)")
    print(f"  {'─'*50}")
    vw = value["correct"].sum()
    vl = len(value) - vw
    vp = vw / len(value)
    avg_edge = value["edge"].mean()
    print(f"  Record:     {vw}W-{vl}L  ({vp:.1%})")
    print(f"  Avg edge:   {avg_edge:+.1%}")
    print(f"  Games:      {len(value)}")

# ── Rolling performance (last 7 / 14 / 30 days) ──────────
print(f"\n  ROLLING PERFORMANCE")
print(f"  {'─'*50}")

for window_days in [7, 14, 30]:
    cutoff = str(date.today() - timedelta(days=window_days))
    window = scored[scored["pred_date"] >= cutoff]
    if len(window) == 0:
        continue
    ww = window["correct"].sum()
    wl = len(window) - ww
    wp = ww / len(window)
    print(f"  Last {window_days:>2d} days:  {ww}W-{wl}L  ({wp:.1%})  n={len(window)}")

# ── Daily breakdown (last 14 days) ────────────────────────
recent_cutoff = str(date.today() - timedelta(days=14))
recent = scored[scored["pred_date"] >= recent_cutoff]

if len(recent) > 0:
    print(f"\n  DAILY LOG (last 14 days)")
    print(f"  {'─'*60}")
    print(f"  {'Date':<12s} {'Record':>8s} {'Win%':>6s} {'High Conf':>10s}")
    print(f"  {'─'*60}")

    for d, day_df in recent.groupby("pred_date"):
        dw = day_df["correct"].sum()
        dl = len(day_df) - dw
        dp = dw / len(day_df)
        hc = day_df[day_df["confidence"] >= 0.65]
        if len(hc) > 0:
            hcw = hc["correct"].sum()
            hcl = len(hc) - hcw
            hc_str = f"{hcw}W-{hcl}L"
        else:
            hc_str = "—"
        print(f"  {d:<12s} {dw}W-{dl}L{'':<3s} {dp:>5.1%}  {hc_str:>10s}")

# ── Goalie status impact ──────────────────────────────────
if "home_goalie_status" in scored.columns:
    print(f"\n  GOALIE CONFIRMATION IMPACT")
    print(f"  {'─'*50}")

    both_confirmed = scored[
        (scored["home_goalie_status"].str.lower() == "confirmed") &
        (scored["away_goalie_status"].str.lower() == "confirmed")
    ]
    if len(both_confirmed) > 0:
        bcw = both_confirmed["correct"].sum()
        bcl = len(both_confirmed) - bcw
        print(f"  Both confirmed:  {bcw}W-{bcl}L  ({bcw/len(both_confirmed):.1%})  n={len(both_confirmed)}")

    any_fallback = scored[
        (scored["home_goalie_status"].str.lower() == "fallback") |
        (scored["away_goalie_status"].str.lower() == "fallback")
    ]
    if len(any_fallback) > 0:
        afw = any_fallback["correct"].sum()
        afl = len(any_fallback) - afw
        print(f"  Any fallback:    {afw}W-{afl}L  ({afw/len(any_fallback):.1%})  n={len(any_fallback)}")

# ── B2B situational split ────────────────────────────────────────────────────
if "home_b2b" in scored.columns and scored["home_b2b"].notna().any():
    scored["home_b2b_n"] = pd.to_numeric(scored["home_b2b"], errors="coerce").fillna(0)
    scored["away_b2b_n"] = pd.to_numeric(scored["away_b2b"], errors="coerce").fillna(0)
    scored["any_b2b"]    = (scored["home_b2b_n"] == 1) | (scored["away_b2b_n"] == 1)
    scored["pick_is_home"] = scored["pick"] == scored["home_team"]
    scored["opp_b2b"] = (
        (scored["pick_is_home"]  & (scored["away_b2b_n"] == 1)) |
        (~scored["pick_is_home"] & (scored["home_b2b_n"] == 1))
    )
    scored["pick_rest"] = scored.apply(
        lambda r: pd.to_numeric(r.get("home_rest_days"), errors="coerce")
        if r["pick_is_home"]
        else pd.to_numeric(r.get("away_rest_days"), errors="coerce"), axis=1
    )

    print(f"\n  🔁 BACK-TO-BACK SITUATIONAL SPLIT")
    print(f"  {'─'*60}")

    # Overall b2b split
    for label, mask in [("Any B2B game", scored["any_b2b"]),
                        ("No B2B",      ~scored["any_b2b"]),
                        ("Opp on B2B",   scored["opp_b2b"])]:
        g = scored[mask]
        if len(g) == 0: continue
        gw = g["correct"].sum()
        print(f"  {label:<18s}  {gw}W-{len(g)-gw}L  ({gw/len(g):.1%})  n={len(g)}")

    # HIGH vs LOW behave OPPOSITELY on b2b — show separately
    print(f"\n  {'Tier':<14s} {'B2B':>18s}   {'No B2B'}")
    print(f"  {'─'*55}")
    for tlabel, lo, hi in [("HIGH (≥65%)", 0.65, 1.0), ("LOW (52-57%)", 0.52, 0.58)]:
        t = scored[(scored["confidence"] >= lo) & (scored["confidence"] < hi)]
        b   = t[t["any_b2b"]];  nb  = t[~t["any_b2b"]]
        bw  = b["correct"].sum()  if len(b)  > 0 else 0
        nbw = nb["correct"].sum() if len(nb) > 0 else 0
        bs  = f"{bw}W-{len(b)-bw}L ({bw/len(b):.0%})"   if len(b)  > 0 else "—"
        nbs = f"{nbw}W-{len(nb)-nbw}L ({nbw/len(nb):.0%})" if len(nb) > 0 else "—"
        print(f"  {tlabel:<14s} {bs:>18s}   {nbs}")

    # Rest days warning: 3-5 days fresh = worst signal
    fresh = scored[(scored["pick_rest"] >= 3) & (scored["pick_rest"] <= 5)]
    tired = scored[(scored["pick_rest"] >= 1) & (scored["pick_rest"] <= 2)]
    if len(fresh) > 0 and len(tired) > 0:
        fw = fresh["correct"].sum(); tw = tired["correct"].sum()
        print(f"\n  Pick team rest:  1-2 days {tw}W-{len(tired)-tw}L ({tw/len(tired):.0%})"
              f"   3-5 days {fw}W-{len(fresh)-fw}L ({fw/len(fresh):.0%})"
              f"  ⚠" if fw/len(fresh) < tw/len(tired) else "")

# ── Worst misses (high confidence losses) ─────────────────
high_losses = scored[(scored["confidence"] >= 0.60) & (scored["result"] == "L")].sort_values(
    "confidence", ascending=False
)

if len(high_losses) > 0:
    print(f"\n  ⚠ ELEVATED CONFIDENCE LOSSES (≥60%)")
    print(f"  {'─'*70}")
    print(f"  {'Date':<12s} {'Matchup':<15s} {'Pick':>5s} {'Conf':>6s} {'Score':>8s}")
    print(f"  {'─'*70}")
    for _, row in high_losses.head(10).iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        score = f"{int(row['away_goals'])}-{int(row['home_goals'])}" if pd.notna(row.get("away_goals")) else "—"
        print(f"  {row['pred_date']:<12s} {matchup:<15s} {row['pick']:>5s} {row['confidence']:>5.1%} {score:>8s}")

print(f"\n{'='*70}")
print(f"  Log file: {LOG_FILE}")
print(f"  Run 'python track_results.py --days 7' for recent performance")
print(f"{'='*70}\n")