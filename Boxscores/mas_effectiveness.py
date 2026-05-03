"""
mas_effectiveness.py — NHL MAS Effectiveness Dashboard
=======================================================
Comprehensive report on pick quality, ROI, calibration, and situational
splits across all confidence tiers.

Usage:
    python mas_effectiveness.py               # full report
    python mas_effectiveness.py --days 30     # last 30 days only
    python mas_effectiveness.py --tier HIGH   # filter to one tier
    python mas_effectiveness.py --update      # fetch latest results first

Tiers: HIGH (≥65%)  MED (58-64%)  LOW (52-58%)  SKIP (<52%)
"""

import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent
LOG_FILE = BASE / "prediction_log.csv"
VIG_STAKE = 110        # standard -110 vig (stake 110 to win 100)
WIN_PAYOUT = 100


# ── CLI args ──────────────────────────────────────────────────────────────────
args = sys.argv[1:]
do_update   = "--update" in args
days_filter = None
tier_filter = None
for i, a in enumerate(args):
    if a == "--days" and i + 1 < len(args):
        days_filter = int(args[i + 1])
    if a == "--tier" and i + 1 < len(args):
        tier_filter = args[i + 1].upper()


# ── Helpers ───────────────────────────────────────────────────────────────────
def roi(wins, losses):
    """ROI% at standard -110 vig."""
    total = wins + losses
    if total == 0:
        return None
    net = wins * WIN_PAYOUT - losses * VIG_STAKE
    return net / (total * VIG_STAKE) * 100


def fmt_roi(r):
    if r is None:
        return "  —   "
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.1f}%"


def fmt_rec(w, l):
    pct = w / (w + l) if (w + l) > 0 else 0
    return f"{w}W-{l}L ({pct:.1%})"


def tier_label(conf):
    if conf >= 0.65:  return "HIGH"
    if conf >= 0.58:  return "MED"
    if conf >= 0.52:  return "LOW"
    return "SKIP"


def goalie_trust(status):
    s = str(status).lower().strip()
    if s in ("confirmed", "manual", "roster-confirmed"):
        return "Confirmed"
    if s in ("likely", "expected", "probable"):
        return "Likely"
    if "conflict" in s:
        inner = s.replace("conflict-", "")
        if inner in ("confirmed", "likely", "expected", "probable"):
            return "Likely"
    return "Unconfirmed"


def section(title):
    print(f"\n  {'─'*62}")
    print(f"  {title}")
    print(f"  {'─'*62}")


def hdr(title):
    print(f"\n{'═'*66}")
    print(f"  {title}")
    print(f"{'═'*66}")


# ── Optional: fetch latest results ───────────────────────────────────────────
if do_update:
    hdr("UPDATING RESULTS FROM NHL API")
    df = pd.read_csv(LOG_FILE)
    pending_dates = df[df["result"].isna()]["pred_date"].unique()
    updated = 0
    for d in sorted(pending_dates):
        if d >= str(date.today()):
            continue
        url = f"https://api-web.nhle.com/v1/score/{d}"
        try:
            r = requests.get(url, timeout=12)
            data = r.json()
            results = {}
            for g in data.get("games", []):
                if g.get("gameState") not in ("FINAL", "OFF"):
                    continue
                ha = g["homeTeam"]["abbrev"]
                aa = g["awayTeam"]["abbrev"]
                hs = g["homeTeam"].get("score", 0)
                as_ = g["awayTeam"].get("score", 0)
                winner = ha if hs > as_ else aa
                results[(aa, ha)] = {"winner": winner, "home_goals": hs, "away_goals": as_}

            for idx, row in df[df["pred_date"] == d].iterrows():
                key = (row["away_team"], row["home_team"])
                if key in results and pd.isna(row.get("result")):
                    res = results[key]
                    df.at[idx, "actual_winner"] = res["winner"]
                    df.at[idx, "home_goals"]    = res["home_goals"]
                    df.at[idx, "away_goals"]    = res["away_goals"]
                    df.at[idx, "result"] = "W" if row["pick"] == res["winner"] else "L"
                    updated += 1
        except Exception as e:
            print(f"  ⚠  {d}: {e}")

    df.to_csv(LOG_FILE, index=False)
    print(f"  ✅ Updated {updated} predictions\n")


# ── Load data ─────────────────────────────────────────────────────────────────
if not LOG_FILE.exists():
    print("❌  prediction_log.csv not found. Run run_picks.py first.")
    sys.exit(1)

df = pd.read_csv(LOG_FILE)
df["confidence"] = df["confidence"].astype(float)
df["edge"]       = df["edge"].astype(float)
df["tier"]       = df["confidence"].apply(tier_label)

# Filter to resolved picks only
resolved = df[df["result"].isin(["W", "L"])].copy()
resolved["win"] = (resolved["result"] == "W").astype(int)

# Apply CLI filters
if days_filter:
    cutoff = str(date.today() - timedelta(days=days_filter))
    resolved = resolved[resolved["pred_date"] >= cutoff]
    print(f"\n  📅 Filtered: last {days_filter} days (since {cutoff})")

if tier_filter:
    resolved = resolved[resolved["tier"] == tier_filter]
    print(f"  🔍 Filtered: tier = {tier_filter}")

if len(resolved) == 0:
    print("\n  No resolved picks found for the selected filters.\n")
    sys.exit(0)

date_range = f"{resolved['pred_date'].min()} → {resolved['pred_date'].max()}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEADER SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
hdr(f"NHL MAS — EFFECTIVENESS REPORT  [{date_range}]")

total_w = resolved["win"].sum()
total_l = len(resolved) - total_w
total_roi = roi(total_w, total_l)
total_pct = total_w / len(resolved)
pending = df[~df["result"].isin(["W", "L"])]

print(f"\n  Picks resolved : {len(resolved)}")
print(f"  Picks pending  : {len(pending)}")
print(f"  Date range     : {date_range}")
print(f"\n  OVERALL RECORD : {fmt_rec(total_w, total_l)}")
print(f"  ROI @ -110 vig : {fmt_roi(total_roi)}")
print(f"  vs 50% baseline: {total_pct - 0.50:+.1%} ({'+' if total_pct>0.5 else ''}{'above' if total_pct>0.5 else 'below'} breakeven)")
breakeven = 0.5238  # wins needed to break even at -110
print(f"  vs break-even  : {total_pct - breakeven:+.1%} (need {breakeven:.1%} to profit)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BY CONFIDENCE TIER
# ══════════════════════════════════════════════════════════════════════════════
section("BY CONFIDENCE TIER")

tier_order = [("HIGH", "≥65%"), ("MED", "58-64%"), ("LOW", "52-58%"), ("SKIP", "<52%")]
print(f"  {'Tier':<13} {'Record':<18} {'Win%':>6}  {'ROI':>7}  {'AvgEdge':>8}  {'n':>4}")
print(f"  {'─'*62}")

for tier, pct_label in tier_order:
    t = resolved[resolved["tier"] == tier]
    if len(t) == 0:
        print(f"  {tier+' '+pct_label:<13} —")
        continue
    tw, tl = t["win"].sum(), len(t) - t["win"].sum()
    t_roi   = roi(tw, tl)
    avg_e   = t["edge"].mean()
    wp      = tw / len(t)
    roi_str = fmt_roi(t_roi)
    flag    = "✅" if wp >= breakeven else ("⚠️ " if wp >= 0.50 else "❌")
    print(f"  {flag} {tier+' ('+pct_label+')':<16} {fmt_rec(tw,tl):<16}  {wp:>6.1%}  "
          f"{roi_str:>7}  {avg_e:>+8.3f}  {len(t):>4}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. EDGE% CALIBRATION  (does higher edge = more wins?)
# ══════════════════════════════════════════════════════════════════════════════
section("EDGE% CALIBRATION  (is the edge signal predictive?)")

playable = resolved[resolved["tier"] != "SKIP"].copy()
edge_bins = [(-0.30, 0.00), (0.00, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.30)]
bin_labels = ["Negative", "0–3%", "3–6%", "6–10%", "10%+"]

print(f"  {'Edge bucket':<13} {'Record':<18} {'Win%':>6}  {'ROI':>7}  {'n':>4}")
print(f"  {'─'*52}")

for (lo, hi), label in zip(edge_bins, bin_labels):
    t = playable[(playable["edge"] >= lo) & (playable["edge"] < hi)]
    if len(t) == 0:
        continue
    tw, tl = t["win"].sum(), len(t) - t["win"].sum()
    wp = tw / len(t)
    flag = "✅" if wp >= breakeven else ("~" if wp >= 0.50 else "❌")
    print(f"  {flag} {label:<13} {fmt_rec(tw,tl):<16}  {wp:>6.1%}  "
          f"{fmt_roi(roi(tw,tl)):>7}  {len(t):>4}")

# Pearson correlation: edge vs win
corr = playable[["edge", "win"]].corr().iloc[0, 1]
print(f"\n  Edge↔Win correlation (Pearson r): {corr:+.3f}  "
      f"({'positive signal ✅' if corr > 0.05 else 'weak/noise ⚠️' if corr > 0 else 'negative ❌'})")


# ══════════════════════════════════════════════════════════════════════════════
# 4. GOALIE CONFIRMATION IMPACT  (core to today's fix)
# ══════════════════════════════════════════════════════════════════════════════
section("GOALIE CONFIRMATION IMPACT")

def pick_goalie_status(row):
    """Return the goalie status for the team we picked."""
    if row["pick"] == row["home_team"]:
        return goalie_trust(row["home_goalie_status"])
    else:
        return goalie_trust(row["away_goalie_status"])

def opp_goalie_status(row):
    """Return the goalie status for the opposing team."""
    if row["pick"] == row["home_team"]:
        return goalie_trust(row["away_goalie_status"])
    else:
        return goalie_trust(row["home_goalie_status"])

resolved["pick_goalie"]  = resolved.apply(pick_goalie_status, axis=1)
resolved["opp_goalie"]   = resolved.apply(opp_goalie_status, axis=1)
resolved["both_conf"]    = (
    (resolved["pick_goalie"].isin(["Confirmed", "Likely"])) &
    (resolved["opp_goalie"].isin(["Confirmed", "Likely"]))
)

print(f"  {'Goalie situation':<28} {'Record':<18} {'Win%':>6}  {'ROI':>7}  {'n':>4}")
print(f"  {'─'*62}")

splits = [
    ("Both confirmed/likely",    resolved[resolved["both_conf"]]),
    ("Pick goalie confirmed",    resolved[resolved["pick_goalie"].isin(["Confirmed", "Likely"])]),
    ("Opp goalie confirmed",     resolved[resolved["opp_goalie"].isin(["Confirmed", "Likely"])]),
    ("Pick goalie unconfirmed",  resolved[resolved["pick_goalie"] == "Unconfirmed"]),
    ("Both unconfirmed",         resolved[(resolved["pick_goalie"]=="Unconfirmed") & (resolved["opp_goalie"]=="Unconfirmed")]),
]

for label, t in splits:
    if len(t) == 0:
        continue
    tw, tl = t["win"].sum(), len(t) - t["win"].sum()
    wp = tw / len(t)
    flag = "✅" if wp >= breakeven else ("~" if wp >= 0.50 else "❌")
    print(f"  {flag} {label:<28} {fmt_rec(tw,tl):<16}  {wp:>6.1%}  "
          f"{fmt_roi(roi(tw,tl)):>7}  {len(t):>4}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PICK DIRECTION
# ══════════════════════════════════════════════════════════════════════════════
section("PICK DIRECTION  (home bias check)")

resolved["pick_direction"] = resolved.apply(
    lambda r: "Home" if r["pick"] == r["home_team"] else "Away", axis=1
)

print(f"  {'Direction':<10} {'Record':<18} {'Win%':>6}  {'ROI':>7}  {'AvgML':>7}  {'n':>4}")
print(f"  {'─'*58}")

for dir_label in ["Home", "Away"]:
    t = resolved[resolved["pick_direction"] == dir_label]
    if len(t) == 0:
        continue
    tw, tl = t["win"].sum(), len(t) - t["win"].sum()
    wp = tw / len(t)
    # avg moneyline for the picked team
    ml_col = "home_ml" if dir_label == "Home" else "away_ml"
    avg_ml = t[ml_col].mean() if ml_col in t else float("nan")
    flag = "✅" if wp >= breakeven else ("~" if wp >= 0.50 else "❌")
    print(f"  {flag} {dir_label:<10} {fmt_rec(tw,tl):<16}  {wp:>6.1%}  "
          f"{fmt_roi(roi(tw,tl)):>7}  {avg_ml:>+7.0f}  {len(t):>4}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SITUATIONAL SPLITS
# ══════════════════════════════════════════════════════════════════════════════
section("SITUATIONAL SPLITS")

def rest_adv(row):
    """Positive = pick team has more rest than opponent."""
    if row["pick"] == row["home_team"]:
        return row["home_rest_days"] - row["away_rest_days"]
    else:
        return row["away_rest_days"] - row["home_rest_days"]

def pick_b2b(row):
    if row["pick"] == row["home_team"]:
        return bool(row["home_b2b"])
    return bool(row["away_b2b"])

resolved["rest_adv"]   = resolved.apply(rest_adv, axis=1)
resolved["pick_is_b2b"] = resolved.apply(pick_b2b, axis=1)

situations = [
    ("Pick team B2B",        resolved[resolved["pick_is_b2b"]]),
    ("Pick team NOT B2B",    resolved[~resolved["pick_is_b2b"]]),
    ("Pick has rest edge",   resolved[resolved["rest_adv"] > 0]),
    ("Rest even",            resolved[resolved["rest_adv"] == 0]),
    ("Pick on short rest",   resolved[resolved["rest_adv"] < 0]),
    ("Weekend game",         resolved[resolved["is_weekend"] == 1.0]),
    ("Weekday game",         resolved[resolved["is_weekend"] != 1.0]),
]

print(f"  {'Situation':<26} {'Record':<18} {'Win%':>6}  {'ROI':>7}  {'n':>4}")
print(f"  {'─'*60}")

for label, t in situations:
    if len(t) == 0:
        continue
    tw, tl = t["win"].sum(), len(t) - t["win"].sum()
    wp = tw / len(t)
    flag = "✅" if wp >= breakeven else ("~" if wp >= 0.50 else "❌")
    print(f"  {flag} {label:<26} {fmt_rec(tw,tl):<16}  {wp:>6.1%}  "
          f"{fmt_roi(roi(tw,tl)):>7}  {len(t):>4}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. ROLLING PERFORMANCE TREND  (is it getting better?)
# ══════════════════════════════════════════════════════════════════════════════
section("ROLLING PERFORMANCE TREND  (recent vs historical)")

resolved_sorted = resolved.sort_values("pred_date")
today = date.today()

windows = [7, 14, 30]
print(f"  {'Window':<14} {'Record':<18} {'Win%':>6}  {'ROI':>7}  {'n':>4}")
print(f"  {'─'*52}")

for w in windows:
    cutoff = str(today - timedelta(days=w))
    t = resolved_sorted[resolved_sorted["pred_date"] >= cutoff]
    if len(t) == 0:
        print(f"  Last {w:>2} days   —  no data")
        continue
    tw, tl = t["win"].sum(), len(t) - t["win"].sum()
    wp = tw / len(t)
    flag = "✅" if wp >= breakeven else ("~" if wp >= 0.50 else "❌")
    print(f"  {flag} Last {w:>2} days  {fmt_rec(tw,tl):<16}  {wp:>6.1%}  "
          f"{fmt_roi(roi(tw,tl)):>7}  {len(t):>4}")

# All-time
t = resolved_sorted
tw, tl = t["win"].sum(), len(t) - t["win"].sum()
wp = tw / len(t)
flag = "✅" if wp >= breakeven else ("~" if wp >= 0.50 else "❌")
print(f"  {flag} {'All-time':<13} {fmt_rec(tw,tl):<16}  {wp:>6.1%}  "
      f"{fmt_roi(roi(tw,tl)):>7}  {len(t):>4}")

# Week-by-week breakdown
print(f"\n  Week-by-week:")
resolved_sorted["week"] = pd.to_datetime(resolved_sorted["pred_date"]).dt.to_period("W")
weeks = resolved_sorted.groupby("week")
print(f"  {'Week':<20} {'Record':<16} {'Win%':>6}  {'ROI':>7}")
for wk, grp in weeks:
    tw, tl = grp["win"].sum(), len(grp) - grp["win"].sum()
    wp = tw / len(grp) if len(grp) > 0 else 0
    flag = "✅" if wp >= breakeven else ("~" if wp >= 0.50 else "❌")
    print(f"  {flag} {str(wk):<20} {fmt_rec(tw,tl):<16} {wp:>6.1%}  {fmt_roi(roi(tw,tl)):>7}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. HIGH-CONFIDENCE MISSES & BEST BETS
# ══════════════════════════════════════════════════════════════════════════════
section("HIGH-CONFIDENCE MISSES  (≥63% confidence, lost)")

high_losses = resolved[(resolved["confidence"] >= 0.63) & (resolved["result"] == "L")].sort_values(
    "confidence", ascending=False
)

if len(high_losses) == 0:
    print("  None! ✅")
else:
    print(f"  {'Date':<12} {'Matchup':<16} {'Pick':<4} {'Conf':>6}  {'Edge':>7}  {'Score'}")
    print(f"  {'─'*58}")
    for _, r in high_losses.head(10).iterrows():
        score = f"{r['away_team']} {int(r['away_goals'] or 0)}-{int(r['home_goals'] or 0)} {r['home_team']}"
        print(f"  {r['pred_date']:<12} {r['away_team']+'@'+r['home_team']:<16} "
              f"{r['pick']:<4} {r['confidence']:>6.1%}  {r['edge']:>+7.3f}  {score}")

section("TOP PERFORMING SPOTS  (tier + situation combos, ≥5 picks)")

# Cross-tab: tier × goalie situation
playable2 = resolved[resolved["tier"] != "SKIP"].copy()
combos = []
for tier, _ in tier_order[:-1]:
    for g_label in ["Confirmed", "Likely", "Unconfirmed"]:
        t = playable2[(playable2["tier"] == tier) & (playable2["pick_goalie"] == g_label)]
        if len(t) < 5:
            continue
        tw = t["win"].sum()
        tl = len(t) - tw
        wp = tw / len(t)
        combos.append({
            "label":  f"{tier} + goalie {g_label}",
            "wins":   tw, "losses": tl, "n": len(t),
            "win_pct": wp, "roi": roi(tw, tl)
        })

combos.sort(key=lambda x: x["win_pct"], reverse=True)
print(f"  {'Combo':<30} {'Record':<18} {'Win%':>6}  {'ROI':>7}  {'n':>4}")
print(f"  {'─'*64}")
for c in combos:
    flag = "✅" if c["win_pct"] >= breakeven else ("~" if c["win_pct"] >= 0.50 else "❌")
    print(f"  {flag} {c['label']:<30} {fmt_rec(c['wins'],c['losses']):<16}  "
          f"{c['win_pct']:>6.1%}  {fmt_roi(c['roi']):>7}  {c['n']:>4}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CALIBRATION CHECK  (predicted confidence vs actual win rate)
# ══════════════════════════════════════════════════════════════════════════════
section("CONFIDENCE CALIBRATION  (model says X%, actually wins Y%?)")

bins = [(0.52, 0.55), (0.55, 0.58), (0.58, 0.61), (0.61, 0.64),
        (0.64, 0.67), (0.67, 0.70), (0.70, 1.00)]

print(f"  {'Conf band':<14} {'Predicted':>9}  {'Actual':>8}  {'Delta':>7}  {'n':>4}")
print(f"  {'─'*50}")

for lo, hi in bins:
    t = playable2[(playable2["confidence"] >= lo) & (playable2["confidence"] < hi)]
    if len(t) < 3:
        continue
    pred_avg = t["confidence"].mean()
    actual   = t["win"].mean()
    delta    = actual - pred_avg
    flag     = "✅" if abs(delta) < 0.05 else ("⚠️ " if abs(delta) < 0.10 else "❌")
    print(f"  {flag} {lo:.0%}–{hi:.0%}        {pred_avg:>9.1%}  {actual:>8.1%}  "
          f"{delta:>+7.1%}  {len(t):>4}")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*66}")
print(f"  Break-even win rate @ -110 vig: {breakeven:.2%}")
print(f"  To run with latest results:     python mas_effectiveness.py --update")
print(f"  To filter:  --days 14  |  --tier HIGH  |  --tier MED")
print(f"{'═'*66}\n")
