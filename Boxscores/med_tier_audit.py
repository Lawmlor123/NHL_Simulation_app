"""
med_tier_audit.py
Audit MED tier (58-64% confidence) losses to find clustering patterns.
Usage:
    python med_tier_audit.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
LOG_FILE = BASE / "prediction_log.csv"

df = pd.read_csv(LOG_FILE)
df["confidence"] = df["confidence"].astype(float)
df["pred_date"] = pd.to_datetime(df["pred_date"])

# ── Isolate MED tier with results ────────────────────────────────────────────
med = df[
    (df["confidence"] >= 0.58) &
    (df["confidence"] < 0.65) &
    (df["result"].isin(["W", "L"]))
].copy()

med_losses = med[med["result"] == "L"].copy()
med_wins   = med[med["result"] == "W"].copy()

sep = "=" * 70

print(f"\n{sep}")
print(f"  MED TIER AUDIT  (58–64% confidence)")
print(f"  Overall: {len(med_wins)}W - {len(med_losses)}L  ({len(med_wins)/len(med):.1%})")
print(f"  Auditing {len(med_losses)} losses across {med_losses['pred_date'].nunique()} dates")
print(sep)

# ── 1. PICK DIRECTION ────────────────────────────────────────────────────────
print(f"\n  1. PICK DIRECTION")
print(f"  {'─'*50}")
for label, subset in [("ALL MED", med), ("MED LOSSES", med_losses)]:
    home_n = (subset["pick"] == subset["home_team"]).sum()
    away_n = (subset["pick"] == subset["away_team"]).sum()
    total  = len(subset)
    hw = ((subset["pick"] == subset["home_team"]) & (subset["result"] == "W")).sum() if "result" in subset else "—"
    aw = ((subset["pick"] == subset["away_team"]) & (subset["result"] == "W")).sum() if "result" in subset else "—"
    print(f"  {label:<12s}  HOME picks: {home_n}  |  AWAY picks: {away_n}")

home_med = med[med["pick"] == med["home_team"]]
away_med = med[med["pick"] == med["away_team"]]
if len(home_med) > 0:
    hw = (home_med["result"] == "W").sum()
    print(f"             HOME record: {hw}W-{len(home_med)-hw}L  ({hw/len(home_med):.1%})")
if len(away_med) > 0:
    aw = (away_med["result"] == "W").sum()
    print(f"             AWAY record: {aw}W-{len(away_med)-aw}L  ({aw/len(away_med):.1%})")

# ── 2. GOALIE STATUS BREAKDOWN ───────────────────────────────────────────────
print(f"\n  2. GOALIE STATUS")
print(f"  {'─'*50}")
if "home_goalie_status" in med.columns and "away_goalie_status" in med.columns:
    med["goalie_situation"] = "Other"
    both_conf = (med["home_goalie_status"].str.lower() == "confirmed") & \
                (med["away_goalie_status"].str.lower() == "confirmed")
    any_fall  = (med["home_goalie_status"].str.lower() == "fallback") | \
                (med["away_goalie_status"].str.lower() == "fallback")
    med.loc[both_conf, "goalie_situation"] = "Both Confirmed"
    med.loc[any_fall,  "goalie_situation"] = "Any Fallback"

    for situation, grp in med.groupby("goalie_situation"):
        gw = (grp["result"] == "W").sum()
        gl = len(grp) - gw
        pct = gw / len(grp) if len(grp) > 0 else 0
        print(f"  {situation:<20s}  {gw}W-{gl}L  ({pct:.1%})  n={len(grp)}")
else:
    print("  No goalie status columns found in log.")

# ── 3. EDGE DISTRIBUTION ─────────────────────────────────────────────────────
print(f"\n  3. EDGE vs MARKET (did we have real edge or were we fighting it?)")
print(f"  {'─'*50}")
med["edge"] = pd.to_numeric(med["edge"], errors="coerce")
med_losses["edge"] = pd.to_numeric(med_losses["edge"], errors="coerce")

print(f"  MED ALL    avg edge: {med['edge'].mean():+.3f}  median: {med['edge'].median():+.3f}")
print(f"  MED LOSSES avg edge: {med_losses['edge'].mean():+.3f}  median: {med_losses['edge'].median():+.3f}")
print(f"  MED WINS   avg edge: {med_wins['edge'].mean():+.3f}  median: {med_wins['edge'].median():+.3f}")

neg_edge = med[med["edge"] < 0]
pos_edge = med[med["edge"] >= 0]
if len(neg_edge) > 0:
    ne_w = (neg_edge["result"] == "W").sum()
    print(f"\n  Negative edge picks:  {ne_w}W-{len(neg_edge)-ne_w}L  ({ne_w/len(neg_edge):.1%})  n={len(neg_edge)}")
if len(pos_edge) > 0:
    pe_w = (pos_edge["result"] == "W").sum()
    print(f"  Positive edge picks:  {pe_w}W-{len(pos_edge)-pe_w}L  ({pe_w/len(pos_edge):.1%})  n={len(pos_edge)}")

# ── 4. SCORE MARGIN ANALYSIS ─────────────────────────────────────────────────
print(f"\n  4. SCORE MARGINS ON LOSSES (blowout vs fluky?)")
print(f"  {'─'*50}")
med_losses["home_goals"] = pd.to_numeric(med_losses["home_goals"], errors="coerce")
med_losses["away_goals"] = pd.to_numeric(med_losses["away_goals"], errors="coerce")
med_losses["margin"] = (med_losses["home_goals"] - med_losses["away_goals"]).abs()

for bucket, label in [(1, "1-goal losses"), (2, "2-goal losses"), (3, "3+ goal losses")]:
    if bucket == 3:
        grp = med_losses[med_losses["margin"] >= 3]
    else:
        grp = med_losses[med_losses["margin"] == bucket]
    print(f"  {label:<20s}  n={len(grp)}  ({len(grp)/len(med_losses):.0%} of losses)")

avg_margin = med_losses["margin"].mean()
print(f"\n  Average losing margin: {avg_margin:.1f} goals")

# ── 5. DAY OF WEEK PATTERNS ──────────────────────────────────────────────────
print(f"\n  5. DAY OF WEEK PATTERNS")
print(f"  {'─'*50}")
med["dow"] = med["pred_date"].dt.day_name()
med_losses["dow"] = med_losses["pred_date"].dt.day_name()
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for day in day_order:
    day_all  = med[med["dow"] == day]
    day_loss = med_losses[med_losses["dow"] == day]
    if len(day_all) == 0:
        continue
    dw = (day_all["result"] == "W").sum()
    pct = dw / len(day_all)
    print(f"  {day:<12s}  {dw}W-{len(day_all)-dw}L  ({pct:.1%})  n={len(day_all)}")

# ── 6. TEAMS MOST OFTEN IN MED LOSSES ───────────────────────────────────────
print(f"\n  6. TEAMS MOST OFTEN IN MED LOSSES (pick side)")
print(f"  {'─'*50}")
team_losses = med_losses["pick"].value_counts().head(10)
for team, count in team_losses.items():
    team_all = med[med["pick"] == team]
    tw = (team_all["result"] == "W").sum()
    print(f"  {team:<6s}  losses: {count}  overall record as pick: {tw}W-{len(team_all)-tw}L")

# ── 7. CONFIDENCE WITHIN MED RANGE ──────────────────────────────────────────
print(f"\n  7. CONFIDENCE SUB-BANDS (where inside 58-64% are losses clustering?)")
print(f"  {'─'*50}")
bands = [(0.58, 0.60, "58-60%"), (0.60, 0.62, "60-62%"), (0.62, 0.65, "62-64%")]
for lo, hi, label in bands:
    grp = med[(med["confidence"] >= lo) & (med["confidence"] < hi)]
    if len(grp) == 0:
        continue
    gw = (grp["result"] == "W").sum()
    print(f"  {label}  {gw}W-{len(grp)-gw}L  ({gw/len(grp):.1%})  n={len(grp)}")

# ── 8. BACK-TO-BACK BREAKDOWN ────────────────────────────────────────────────
print(f"\n  8. BACK-TO-BACK ANALYSIS")
print(f"  {'─'*55}")

b2b_cols_present = "home_b2b" in med.columns and med["home_b2b"].notna().any() and (med["home_b2b"] != "").any()

if not b2b_cols_present:
    print("  ⚠ B2B data not yet backfilled — run backfill_situational.py first")
    print("    (python backfill_situational.py  from the Boxscores folder)")
else:
    med["home_b2b"] = pd.to_numeric(med["home_b2b"], errors="coerce").fillna(0)
    med["away_b2b"] = pd.to_numeric(med["away_b2b"], errors="coerce").fillna(0)
    med["any_b2b"]  = (med["home_b2b"] == 1) | (med["away_b2b"] == 1)
    med["pick_is_home"] = med["pick"] == med["home_team"]
    med["pick_b2b"] = (
        (med["pick_is_home"]  & (med["home_b2b"] == 1)) |
        (~med["pick_is_home"] & (med["away_b2b"] == 1))
    )
    med["opp_b2b"] = (
        (~med["pick_is_home"] & (med["home_b2b"] == 1)) |
        (med["pick_is_home"]  & (med["away_b2b"] == 1))
    )

    scenarios = [
        ("Any B2B game",        med["any_b2b"]),
        ("No B2B",             ~med["any_b2b"]),
        ("Pick team on B2B",    med["pick_b2b"]),
        ("Opponent on B2B",     med["opp_b2b"] & ~med["pick_b2b"]),
    ]
    for label, mask in scenarios:
        grp = med[mask]
        if len(grp) == 0:
            continue
        gw = (grp["result"] == "W").sum()
        print(f"  {label:<24s}  {gw}W-{len(grp)-gw}L  ({gw/len(grp):.1%})  n={len(grp)}")

    # Rest days split — pick team fresh (3+ days rest) vs tired (1-2 days)
    print(f"\n  Rest days for PICK team:")
    med["pick_rest"] = med.apply(
        lambda r: r["home_rest_days"] if r["pick_is_home"] else r["away_rest_days"], axis=1
    )
    med["pick_rest"] = pd.to_numeric(med["pick_rest"], errors="coerce")
    for lo, hi, label in [(1, 2, "1-2 days (tired)"), (3, 5, "3-5 days (fresh)"), (6, 99, "6+ days (rusty)")]:
        grp = med[(med["pick_rest"] >= lo) & (med["pick_rest"] <= hi)]
        if len(grp) == 0:
            continue
        gw = (grp["result"] == "W").sum()
        print(f"    {label:<22s}  {gw}W-{len(grp)-gw}L  ({gw/len(grp):.1%})  n={len(grp)}")

# ── 9. FULL LOSS LIST ─────────────────────────────────────────────────────────
print(f"\n  9. ALL {len(med_losses)} MED TIER LOSSES")
print(f"  {'─'*70}")
print(f"  {'Date':<12s} {'Matchup':<18s} {'Pick':>5s} {'Conf':>6s} {'Edge':>7s} {'Score':>7s} {'Goalie Sit.'}")
print(f"  {'─'*70}")
for _, row in med_losses.sort_values("pred_date").iterrows():
    matchup = f"{row['away_team']} @ {row['home_team']}"
    score = f"{int(row['away_goals'])}-{int(row['home_goals'])}" if pd.notna(row.get("away_goals")) else "—"
    gs = ""
    if "home_goalie_status" in row:
        h = str(row["home_goalie_status"]).lower()[:4]
        a = str(row["away_goalie_status"]).lower()[:4]
        gs = f"H:{h} A:{a}"
    edge_val = f"{row['edge']:+.3f}" if pd.notna(row.get("edge")) else "—"
    print(f"  {str(row['pred_date'])[:10]:<12s} {matchup:<18s} {row['pick']:>5s} {row['confidence']:>5.1%} {edge_val:>7s} {score:>7s}  {gs}")

print(f"\n{sep}")
print(f"  Tips:")
print(f"  - Run backfill_situational.py first to unlock section 8 (b2b analysis)")
print(f"  - Biggest levers found so far: goalie fallback (34.6%), home bias (69/70 picks)")
print(f"  - 58-60% sub-band is worst (30%) — consider raising LOW tier floor above 60%")
print(sep + "\n")
