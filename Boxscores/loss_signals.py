"""
loss_signals.py  —  Team-Specific Signal Explorer
===================================================
Mines prediction_log.csv to find systematic patterns in the 118 losses.
Answers questions like:
  - Which teams are we systematically wrong about?
  - Which opponents do our picks always lose to?
  - Does home/away, B2B, or goalie status flip results for specific teams?
  - Are we picking teams at HIGH confidence that keep losing?

Usage:
    python loss_signals.py              # full report
    python loss_signals.py --team NJD   # single-team deep dive
    python loss_signals.py --min 4      # only show combos with ≥4 picks
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE     = Path(__file__).resolve().parent
LOG_FILE = BASE / "prediction_log.csv"
MIN_N    = 4       # minimum picks to show a pattern
BREAK_EVEN = 0.5238

# ── CLI ───────────────────────────────────────────────────────────────────────
args       = sys.argv[1:]
team_focus = None
for i, a in enumerate(args):
    if a == "--team" and i + 1 < len(args):
        team_focus = args[i + 1].upper()
    if a == "--min" and i + 1 < len(args):
        MIN_N = int(args[i + 1])

# ── Load & enrich ─────────────────────────────────────────────────────────────
df = pd.read_csv(LOG_FILE)
df["confidence"] = df["confidence"].astype(float)
r  = df[df["result"].isin(["W", "L"])].copy()
r["win"]      = (r["result"] == "W").astype(int)
r["pick_dir"] = r.apply(lambda x: "home" if x["pick"] == x["home_team"] else "away", axis=1)
r["opp"]      = r.apply(lambda x: x["away_team"] if x["pick"] == x["home_team"] else x["home_team"], axis=1)
r["month"]    = pd.to_datetime(r["pred_date"]).dt.strftime("%b")
r["tier"]     = r["confidence"].apply(
    lambda c: "HIGH" if c >= 0.65 else ("MED" if c >= 0.58 else ("LOW" if c >= 0.52 else "SKIP"))
)
r["pick_goalie_st"] = r.apply(
    lambda x: x["home_goalie_status"] if x["pick"] == x["home_team"] else x["away_goalie_status"], axis=1
)
r["pick_b2b"]  = r.apply(lambda x: x["home_b2b"]      if x["pick"] == x["home_team"] else x["away_b2b"],       axis=1)
r["opp_b2b"]   = r.apply(lambda x: x["away_b2b"]      if x["pick"] == x["home_team"] else x["home_b2b"],       axis=1)
r["pick_rest"] = r.apply(lambda x: x["home_rest_days"] if x["pick"] == x["home_team"] else x["away_rest_days"], axis=1)
r["opp_rest"]  = r.apply(lambda x: x["away_rest_days"] if x["pick"] == x["home_team"] else x["home_rest_days"], axis=1)
r["rest_adv"]  = r["pick_rest"] - r["opp_rest"]
r["goalie_conf"] = r["pick_goalie_st"].apply(
    lambda s: "confirmed" if str(s).lower() in ("confirmed", "manual", "likely", "probable")
    else "unconfirmed"
)

def rec(sub):
    w = sub["win"].sum(); l = len(sub) - w
    pct = w / (w + l) if (w + l) > 0 else 0
    roi = (w * 100 - l * 110) / ((w + l) * 110) * 100 if (w + l) > 0 else 0
    return w, l, pct, roi

def flag(pct):
    if pct >= BREAK_EVEN: return "✅"
    if pct >= 0.50:       return "~"
    if pct >= 0.40:       return "⚠️ "
    return "❌"

def hdr(t):
    print(f"\n{'═'*68}")
    print(f"  {t}")
    print(f"{'═'*68}")

def section(t):
    print(f"\n  {'─'*62}")
    print(f"  {t}")
    print(f"  {'─'*62}")


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-TEAM DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
if team_focus:
    t = r[r["pick"] == team_focus]
    hdr(f"DEEP DIVE: {team_focus}  ({len(t)} picks, {t['win'].sum()}W-{len(t)-t['win'].sum()}L)")

    if len(t) == 0:
        print(f"  No picks found for {team_focus}")
        sys.exit(0)

    section("By Direction")
    for d in ["home", "away"]:
        sub = t[t["pick_dir"] == d]
        if len(sub) == 0: continue
        w, l, pct, roi = rec(sub)
        print(f"  {flag(pct)} {d.upper():<6} {w}W-{l}L ({pct:.0%})  ROI {roi:+.1f}%")

    section("By Confidence Tier")
    for tier in ["HIGH", "MED", "LOW"]:
        sub = t[t["tier"] == tier]
        if len(sub) == 0: continue
        w, l, pct, roi = rec(sub)
        print(f"  {flag(pct)} {tier:<5} {w}W-{l}L ({pct:.0%})  ROI {roi:+.1f}%  (n={len(sub)})")

    section("By Goalie Status")
    for gs in ["confirmed", "unconfirmed"]:
        sub = t[t["goalie_conf"] == gs]
        if len(sub) == 0: continue
        w, l, pct, roi = rec(sub)
        print(f"  {flag(pct)} {gs:<14} {w}W-{l}L ({pct:.0%})  ROI {roi:+.1f}%  (n={len(sub)})")

    section("B2B Situations")
    for label, sub in [("Pick B2B", t[t["pick_b2b"]==1.0]),
                        ("Opp B2B",  t[t["opp_b2b"]==1.0]),
                        ("Both rested", t[(t["pick_b2b"]!=1.0)&(t["opp_b2b"]!=1.0)])]:
        if len(sub) < 2: continue
        w, l, pct, roi = rec(sub)
        print(f"  {flag(pct)} {label:<16} {w}W-{l}L ({pct:.0%})  ROI {roi:+.1f}%  (n={len(sub)})")

    section("By Opponent (≥2 matchups)")
    opp_rows = []
    for opp, sub in t.groupby("opp"):
        if len(sub) < 2: continue
        w, l, pct, roi = rec(sub)
        opp_rows.append((pct, opp, w, l, roi, len(sub)))
    for pct, opp, w, l, roi, n in sorted(opp_rows):
        print(f"  {flag(pct)} vs {opp:<5} {w}W-{l}L ({pct:.0%})  ROI {roi:+.1f}%  (n={n})")

    section("Game-by-game losses")
    losses = t[t["result"] == "L"].sort_values("pred_date")
    print(f"  {'Date':<12} {'vs':<4} {'Dir':<5} {'Tier':<5} {'Conf':>6}  "
          f"{'Edge':>7}  {'Goalie St':<14} {'Score'}")
    print(f"  {'─'*70}")
    for _, row in losses.iterrows():
        opp_team = row["opp"]
        score = f"{row['away_team']} {int(row['away_goals'] or 0)}-{int(row['home_goals'] or 0)} {row['home_team']}"
        print(f"  {row['pred_date']:<12} {opp_team:<4} {row['pick_dir']:<5} "
              f"{row['tier']:<5} {row['confidence']:>6.1%}  {row['edge']:>+7.3f}  "
              f"{row['pick_goalie_st']:<14} {score}")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════
losses = r[r["result"] == "L"]
hdr(f"LOSS SIGNAL EXPLORER  [{r['pred_date'].min()} to {r['pred_date'].max()}]")
print(f"\n  Total resolved: {len(r)}  |  W: {r['win'].sum()}  L: {len(losses)}  |  Win%: {r['win'].mean():.1%}")
print(f"  Drilling into {len(losses)} losses for hidden patterns...\n")


# ── 1. TEAM WIN RATES — sorted worst to best ──────────────────────────────────
section("TEAM WIN RATES  (when we pick them)  sorted worst→best")
print(f"  {'Team':<6} {'Record':<14} {'Win%':>5}  {'ROI':>7}  {'home':>9}  {'away':>9}  "
      f"{'HIGH':>7}  {'MED':>7}  {'LOW':>7}")
print(f"  {'─'*80}")

rows = []
for team, sub in r.groupby("pick"):
    if len(sub) < MIN_N: continue
    w, l, pct, roi = rec(sub)
    h = sub[sub["pick_dir"] == "home"];  hw, hl, hp, _ = rec(h) if len(h) else (0,0,0,0)
    a = sub[sub["pick_dir"] == "away"];  aw, al, ap, _ = rec(a) if len(a) else (0,0,0,0)
    hi = sub[sub["tier"] == "HIGH"];     hiw, hil, hip, _ = rec(hi) if len(hi) else (0,0,0,0)
    me = sub[sub["tier"] == "MED"];      mew, mel, mep, _ = rec(me) if len(me) else (0,0,0,0)
    lo = sub[sub["tier"] == "LOW"];      low_, lol, lop, _ = rec(lo) if len(lo) else (0,0,0,0)
    rows.append((pct, team, w, l, roi, hw, hl, hp, aw, al, ap, hiw, hil, hip, mew, mel, mep, low_, lol, lop))

for pct, team, w, l, roi, hw, hl, hp, aw, al, ap, hiw, hil, hip, mew, mel, mep, low_, lol, lop in sorted(rows):
    h_str  = f"{hw}-{hl} ({hp:.0%})" if hw+hl else "—"
    a_str  = f"{aw}-{al} ({ap:.0%})" if aw+al else "—"
    hi_str = f"{hiw}-{hil} ({hip:.0%})" if hiw+hil else "—"
    me_str = f"{mew}-{mel} ({mep:.0%})" if mew+mel else "—"
    lo_str = f"{low_}-{lol} ({lop:.0%})" if low_+lol else "—"
    f_ = flag(pct)
    print(f"  {f_} {team:<5} {w}W-{l}L ({pct:.0%})  {roi:>+6.0f}%  "
          f"{h_str:>9}  {a_str:>9}  {hi_str:>9}  {me_str:>9}  {lo_str:>9}")


# ── 2. OPPONENT EFFECT ────────────────────────────────────────────────────────
section("OPPONENT DIFFICULTY  (when we pick *against* them, how do WE do?)")
print(f"  {'Opp':<6} {'Our Record vs them':<20} {'Win%':>5}  note")
print(f"  {'─'*55}")

opp_rows = []
for opp, sub in r.groupby("opp"):
    if len(sub) < MIN_N: continue
    w, l, pct, roi = rec(sub)
    opp_rows.append((pct, opp, w, l, roi, len(sub)))

for pct, opp, w, l, roi, n in sorted(opp_rows):
    note = ""
    if pct <= 0.30: note = " ← AVOID picking against this team"
    if pct >= 0.70: note = " ← Great value picking against this team"
    print(f"  {flag(pct)} {opp:<5} {w}W-{l}L ({pct:.0%})  n={n}{note}")


# ── 3. KILLER COMBOS — team + situation pairs with notable patterns ────────────
section("KILLER COMBOS  (team × situation, ≥3 picks)")
print(f"  Pattern                          Record       Win%    ROI    note")
print(f"  {'─'*70}")

combos = []

for team, sub in r.groupby("pick"):
    if len(sub) < MIN_N: continue

    # home vs away
    for d in ["home", "away"]:
        s = sub[sub["pick_dir"] == d]
        if len(s) < 3: continue
        w, l, pct, roi = rec(s)
        combos.append((pct, f"{team} picked {d.upper()}", w, l, pct, roi, len(s)))

    # tier
    for tier in ["HIGH", "MED", "LOW"]:
        s = sub[sub["tier"] == tier]
        if len(s) < 3: continue
        w, l, pct, roi = rec(s)
        combos.append((pct, f"{team} @ {tier} conf", w, l, pct, roi, len(s)))

    # pick team on B2B
    s = sub[sub["pick_b2b"] == 1.0]
    if len(s) >= 2:
        w, l, pct, roi = rec(s)
        combos.append((pct, f"{team} on B2B", w, l, pct, roi, len(s)))

    # opponent on B2B
    s = sub[sub["opp_b2b"] == 1.0]
    if len(s) >= 2:
        w, l, pct, roi = rec(s)
        combos.append((pct, f"{team} vs opp B2B", w, l, pct, roi, len(s)))

    # unconfirmed goalie
    s = sub[sub["goalie_conf"] == "unconfirmed"]
    if len(s) >= 3:
        w, l, pct, roi = rec(s)
        combos.append((pct, f"{team} unconf goalie", w, l, pct, roi, len(s)))

    # rest advantage
    s = sub[sub["rest_adv"] > 0]
    if len(s) >= 3:
        w, l, pct, roi = rec(s)
        combos.append((pct, f"{team} rest advantage", w, l, pct, roi, len(s)))

# Sort: worst first (most losses = most signal), then best
combos.sort(key=lambda x: x[0])

print("\n  --- WORST PATTERNS (avoid) ---")
for _, label, w, l, pct, roi, n in [c for c in combos if c[0] <= 0.38]:
    print(f"  {flag(pct)} {label:<33} {w}W-{l}L ({pct:.0%})  {roi:>+6.1f}%  n={n}")

print("\n  --- BEST PATTERNS (exploit) ---")
for _, label, w, l, pct, roi, n in reversed([c for c in combos if c[0] >= 0.70]):
    print(f"  {flag(pct)} {label:<33} {w}W-{l}L ({pct:.0%})  {roi:>+6.1f}%  n={n}")


# ── 4. HIGH-CONFIDENCE TRAPS ──────────────────────────────────────────────────
section("HIGH-CONFIDENCE TRAPS  (teams where HIGH/MED picks underperform)")
print(f"  If we only played these teams at HIGH conf, how did we do?")
print(f"  {'Team':<6} {'HIGH record':<16} {'HIGH win%':>9}  {'Expected (≥65%)':>15}  verdict")
print(f"  {'─'*65}")

for team, sub in sorted(r.groupby("pick"), key=lambda x: x[0]):
    hi = sub[sub["tier"] == "HIGH"]
    if len(hi) < 3: continue
    w, l, pct, roi = rec(hi)
    verdict = "🚨 TRAP" if pct < 0.50 else ("⚠️  below expectation" if pct < 0.60 else "✅ solid")
    print(f"  {flag(pct)} {team:<5} {w}W-{l}L ({pct:.0%})         {pct:>9.0%}  vs 60%+ expected   {verdict}")


# ── 5. OPPONENT TRAPS — teams that beat our picks despite being the dog ────────
section("UPSET FACTORIES  (opponents that beat us when favoured)")
print("  Teams that kept winning as underdogs against our picks:")
print(f"  {'Opp':<6} {'Record vs our picks':<22} {'Win%':>5}  {'Avg their ML':>13}")
print(f"  {'─'*55}")

opp_upset = []
for opp, sub in r.groupby("opp"):
    if len(sub) < MIN_N: continue
    # opponent wins = our losses
    opp_wins = len(sub) - sub["win"].sum()
    opp_pct  = opp_wins / len(sub)
    # their avg moneyline (positive = underdog)
    opp_ml = sub.apply(
        lambda row: row["away_ml"] if row["pick"] == row["home_team"] else row["home_ml"], axis=1
    ).mean()
    opp_upset.append((opp_pct, opp, opp_wins, len(sub)-opp_wins, opp_ml))

for opp_pct, opp, opp_w, opp_l, opp_ml in sorted(opp_upset, reverse=True)[:12]:
    ml_str = f"{opp_ml:+.0f}" if not pd.isna(opp_ml) else "—"
    note = " ← frequent spoiler" if opp_pct >= 0.60 else ""
    print(f"  {'❌' if opp_pct>=0.55 else '~'} {opp:<5}  beat us {opp_w}/{opp_w+opp_l} times ({opp_pct:.0%})  avg ML: {ml_str}{note}")


# ── 6. MONTHLY PATTERNS ───────────────────────────────────────────────────────
section("MONTHLY PATTERNS")
month_order = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr"]
available   = [m for m in month_order if m in r["month"].values]
print(f"  {'Month':<6} {'Record':<14} {'Win%':>5}  {'ROI':>7}  note")
print(f"  {'─'*50}")
for m in available:
    sub = r[r["month"] == m]
    if len(sub) < 2: continue
    w, l, pct, roi = rec(sub)
    note = " ← best month" if pct == max(rec(r[r["month"]==mo])[2] for mo in available if len(r[r["month"]==mo])>=2) else ""
    print(f"  {flag(pct)} {m:<6} {w}W-{l}L ({pct:.0%})   {roi:>+6.1f}%{note}")


# ── 7. ACTIONABLE RULES — ranked by confidence ────────────────────────────────
section("ACTIONABLE RULES  (signals strong enough to act on)")
print()

rules = [
    # (win_pct, n, rule_text)
]

# Build rules from data
for team, sub in r.groupby("pick"):
    # Never pick team X
    if len(sub) >= MIN_N:
        w, l, pct, roi = rec(sub)
        if pct <= 0.25:
            rules.append((pct, len(sub), f"AVOID picking {team} — {w}W-{l}L ({pct:.0%}) all season"))

    # Never pick team X at home
    h = sub[sub["pick_dir"]=="home"]
    if len(h) >= MIN_N:
        w, l, pct, roi = rec(h)
        if pct <= 0.30:
            rules.append((pct, len(h), f"AVOID {team} at HOME — {w}W-{l}L ({pct:.0%})"))

    # Never pick team X on B2B
    b = sub[sub["pick_b2b"]==1.0]
    if len(b) >= 3:
        w, l, pct, roi = rec(b)
        if pct == 0.0:
            rules.append((pct, len(b), f"NEVER pick {team} on B2B — 0W-{l}L (0%)"))

    # Team X is gold at HIGH conf
    hi = sub[sub["tier"]=="HIGH"]
    if len(hi) >= 3:
        w, l, pct, roi = rec(hi)
        if pct >= 0.75:
            rules.append((pct, len(hi), f"TRUST {team} @ HIGH conf — {w}W-{l}L ({pct:.0%}) ROI {roi:+.0f}%"))

    # Team X is a trap at HIGH conf
    if len(hi) >= 3:
        w, l, pct, roi = rec(hi)
        if pct <= 0.40:
            rules.append((pct, len(hi), f"DISTRUST {team} @ HIGH conf — {w}W-{l}L ({pct:.0%}) despite high confidence"))

    # Pick team X away only
    a = sub[sub["pick_dir"]=="away"]
    if len(a) >= 3 and len(h) >= 3:
        wa, la, pa, _ = rec(a)
        wh, lh, ph, _ = rec(h)
        if pa >= 0.70 and ph <= 0.40:
            rules.append((pa, len(a), f"PICK {team} AWAY ONLY — away {wa}W-{la}L ({pa:.0%}) vs home {wh}W-{lh}L ({ph:.0%})"))

# Opponent rules
for opp, sub in r.groupby("opp"):
    if len(sub) < MIN_N: continue
    w, l, pct, roi = rec(sub)
    if pct >= 0.70:
        rules.append((pct, len(sub), f"PICK AGAINST {opp} freely — we go {w}W-{l}L ({pct:.0%}) vs them"))
    if pct <= 0.30:
        rules.append((1-pct, len(sub), f"NEVER pick against {opp} — they beat our pick {len(sub)-w}/{len(sub)} times"))

rules.sort(key=lambda x: x[0])

print("  AVOID rules:")
for pct, n, rule in [x for x in rules if "AVOID" in x[2] or "NEVER" in x[2] or "DISTRUST" in x[2]]:
    print(f"    ❌ {rule}  (n={n})")

print()
print("  EXPLOIT rules:")
for pct, n, rule in reversed([x for x in rules if "TRUST" in x[2] or "PICK" in x[2]]):
    print(f"    ✅ {rule}  (n={n})")

print(f"\n{'═'*68}")
print(f"  Run deep dive: python loss_signals.py --team <ABBREV>")
print(f"  e.g.:  python loss_signals.py --team COL")
print(f"         python loss_signals.py --team VGK")
print(f"         python loss_signals.py --team UTA")
print(f"{'═'*68}\n")
