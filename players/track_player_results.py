"""
track_player_results.py
Pull actual NHL boxscores, grade player prop predictions from predict.py.
Usage:
  python track_player_results.py                    # Grade yesterday
  python track_player_results.py --date 2026-03-01  # Specific date
  python track_player_results.py --all              # All unresolved
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import date, datetime, timedelta
import sys
import time

PREDICTIONS_DIR = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\predictions"
HISTORY_PATH = os.path.join(PREDICTIONS_DIR, "player_picks_history.csv")

# ── Parse arguments ─────────────────────────────────────────
target_date = date.today() - timedelta(days=1)
check_all = False
for i, arg in enumerate(sys.argv):
    if arg == "--date" and i + 1 < len(sys.argv):
        target_date = datetime.strptime(sys.argv[i + 1], "%Y-%m-%d").date()
    if arg == "--all":
        check_all = True

print(f"{'='*70}")
print(f"  NHL PLAYER PREDICTION TRACKER")
print(f"  Run: {date.today()}  |  Checking: {'ALL unresolved' if check_all else target_date}")
print(f"{'='*70}\n")


# ── Helpers ─────────────────────────────────────────────────
def clean_player_id(pid):
    """Normalize player_id to string without .0 suffix."""
    pid = str(pid).strip()
    if pid.endswith('.0'):
        pid = pid[:-2]
    return pid


def get_game_ids_for_date(check_date):
    """Get list of (game_id, home_team, away_team) from NHL API."""
    games = []
    try:
        url = f"https://api-web.nhle.com/v1/score/{check_date}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        for g in data.get('games', []):
            if g.get('gameState') in ('FINAL', 'OFF'):
                games.append({
                    'game_id': g['id'],
                    'home_team': g['homeTeam']['abbrev'],
                    'away_team': g['awayTeam']['abbrev'],
                })
    except Exception:
        pass

    if not games:
        try:
            url2 = f"https://api-web.nhle.com/v1/schedule/{check_date}"
            r2 = requests.get(url2, timeout=15)
            r2.raise_for_status()
            data2 = r2.json()
            for gw in data2.get('gameWeek', []):
                if gw['date'] == str(check_date):
                    for g in gw.get('games', []):
                        if g.get('gameState') in ('FINAL', 'OFF'):
                            games.append({
                                'game_id': g['id'],
                                'home_team': g['homeTeam']['abbrev'],
                                'away_team': g['awayTeam']['abbrev'],
                            })
        except Exception as e:
            print(f"    ⚠ Schedule error: {e}")

    return games


def get_boxscore_stats(game_id):
    """
    Pull individual player stats from a game boxscore.
    Returns dict keyed by player_id (str) and also by 'name:Full Name'.
    Each value: {player_name, goals, assists, points, sog}
    """
    players = {}
    try:
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        pstats = data.get('playerByGameStats', {})
        if not pstats:
            pstats = data.get('boxscore', {}).get('playerByGameStats', {})

        for side in ['homeTeam', 'awayTeam']:
            team_data = pstats.get(side, {})
            for pos_group in ['forwards', 'defense']:
                for p in team_data.get(pos_group, []):
                    pid = clean_player_id(p.get('playerId', ''))
                    name_obj = p.get('name', {})
                    if isinstance(name_obj, dict):
                        pname = name_obj.get('default', '')
                    else:
                        pname = str(name_obj)

                    goals = int(p.get('goals', 0))
                    assists = int(p.get('assists', 0))
                    points = goals + assists
                    sog = int(p.get('shots', p.get('sog', p.get('shotsOnGoal', 0))))

                    entry = {
                        'player_name': pname,
                        'goals': goals,
                        'assists': assists,
                        'points': points,
                        'sog': sog,
                    }

                    if pid:
                        players[pid] = entry
                    if pname:
                        players[f"name:{pname}"] = entry

    except Exception as e:
        print(f"    ⚠ Boxscore error for game {game_id}: {e}")

    return players


# ── Load history ────────────────────────────────────────────
if not os.path.exists(HISTORY_PATH):
    print(f"  ❌ No player_picks_history.csv found!")
    print(f"     Expected at: {HISTORY_PATH}")
    print(f"     Run predict.py first to generate predictions.")
    sys.exit(1)

df = pd.read_csv(HISTORY_PATH, dtype=str)
print(f"  Loaded {len(df)} player predictions from history\n")

# ── Determine dates to check ───────────────────────────────
if check_all:
    unresolved = df[df['actual_goals'].isna() | (df['actual_goals'] == '')]
    dates_to_check = sorted(unresolved['prediction_date'].unique())
    if len(dates_to_check) == 0:
        print("  ✅ All player predictions already graded!\n")
    else:
        print(f"  Found {len(dates_to_check)} date(s) with unresolved predictions\n")
else:
    dates_to_check = [str(target_date)]


# ═══════════════════════════════════════════════════════════════
#  GRADE PREDICTIONS
# ═══════════════════════════════════════════════════════════════
total_updates = 0

for check_date in dates_to_check:
    day_mask = df['prediction_date'] == check_date
    needs = df[day_mask & (df['actual_goals'].isna() | (df['actual_goals'] == ''))]

    if len(needs) == 0:
        continue

    print(f"  📅 {check_date}: {len(needs)} players to grade ...")

    game_infos = get_game_ids_for_date(check_date)
    if not game_infos:
        print(f"    ⚠ No final games found for {check_date}\n")
        continue

    print(f"    Found {len(game_infos)} final game(s)")

    # Pull all boxscores
    all_stats = {}
    for gi in game_infos:
        stats = get_boxscore_stats(gi['game_id'])
        all_stats.update(stats)
        time.sleep(0.3)

    id_count = len([k for k in all_stats if not k.startswith('name:')])
    print(f"    Retrieved stats for {id_count} skaters")

    if not all_stats:
        print(f"    ⚠ No boxscore data retrieved\n")
        continue

    # Grade each prediction
    day_graded = 0
    day_dnp = 0

    for idx in needs.index:
        row = df.loc[idx]
        pid = clean_player_id(row.get('player_id', ''))
        pname = str(row.get('player_name', ''))

        # Match by player_id
        stats = all_stats.get(pid)

        # Fallback: match by full name
        if stats is None and pname:
            stats = all_stats.get(f"name:{pname}")

        # Fallback: match by last name
        if stats is None and pname:
            last_name = pname.strip().split()[-1].lower() if pname.strip() else ''
            if last_name and len(last_name) > 2:
                for key, val in all_stats.items():
                    if key.startswith('name:') and last_name in key.lower():
                        stats = val
                        break

        if stats is None:
            df.at[idx, 'actual_goals'] = 'DNP'
            df.at[idx, 'actual_assists'] = 'DNP'
            df.at[idx, 'actual_points'] = 'DNP'
            df.at[idx, 'actual_shots'] = 'DNP'
            for hc in ['hit_goals_1plus', 'hit_points_1plus', 'hit_points_2plus',
                        'hit_assists_1plus', 'hit_shots_3plus', 'hit_shots_4plus',
                        'hit_shots_5plus']:
                df.at[idx, hc] = 'DNP'
            day_dnp += 1
            total_updates += 1
            continue

        goals = stats['goals']
        assists = stats['assists']
        points = stats['points']
        sog = stats['sog']

        df.at[idx, 'actual_goals'] = str(goals)
        df.at[idx, 'actual_assists'] = str(assists)
        df.at[idx, 'actual_points'] = str(points)
        df.at[idx, 'actual_shots'] = str(sog)

        df.at[idx, 'hit_goals_1plus'] = 'HIT' if goals >= 1 else 'MISS'
        df.at[idx, 'hit_points_1plus'] = 'HIT' if points >= 1 else 'MISS'
        df.at[idx, 'hit_points_2plus'] = 'HIT' if points >= 2 else 'MISS'
        df.at[idx, 'hit_assists_1plus'] = 'HIT' if assists >= 1 else 'MISS'
        df.at[idx, 'hit_shots_3plus'] = 'HIT' if sog >= 3 else 'MISS'
        df.at[idx, 'hit_shots_4plus'] = 'HIT' if sog >= 4 else 'MISS'
        df.at[idx, 'hit_shots_5plus'] = 'HIT' if sog >= 5 else 'MISS'

        day_graded += 1
        total_updates += 1

    print(f"    ✅ Graded {day_graded} players, {day_dnp} DNP\n")

# Save updated history
df.to_csv(HISTORY_PATH, index=False)
print(f"  💾 Saved {total_updates} updates → {HISTORY_PATH}\n")

# Also update daily files
for check_date in dates_to_check:
    daily_path = os.path.join(PREDICTIONS_DIR, f"daily_picks_{check_date}.csv")
    if os.path.exists(daily_path):
        day_data = df[df['prediction_date'] == check_date]
        day_data.to_csv(daily_path, index=False)


# ═══════════════════════════════════════════════════════════════
#  DISPLAY RESULTS FOR TARGET DATE
# ═══════════════════════════════════════════════════════════════
print(f"{'='*70}")
print(f"  📊 RESULTS FOR {target_date}")
print(f"{'='*70}\n")

day = df[df['prediction_date'] == str(target_date)].copy()

if len(day) == 0:
    print(f"  No predictions found for {target_date}\n")
else:
    for col in ['actual_goals', 'actual_assists', 'actual_points', 'actual_shots']:
        day[col] = pd.to_numeric(day[col], errors='coerce')

    played = day[day['actual_goals'].notna()].copy()

    if len(played) > 0:
        props_to_check = [
            ('pick_goals_1plus', 'hit_goals_1plus', 'prob_goals_1plus', '1+ Goals', 'actual_goals', 'G'),
            ('pick_points_1plus', 'hit_points_1plus', 'prob_points_1plus', '1+ Points', 'actual_points', 'P'),
            ('pick_points_2plus', 'hit_points_2plus', 'prob_points_2plus', '2+ Points', 'actual_points', 'P'),
            ('pick_assists_1plus', 'hit_assists_1plus', 'prob_assists_1plus', '1+ Assists', 'actual_assists', 'A'),
            ('pick_shots_3plus', 'hit_shots_3plus', 'prob_shots_3plus', '3+ SOG', 'actual_shots', 'SOG'),
            ('pick_shots_4plus', 'hit_shots_4plus', 'prob_shots_4plus', '4+ SOG', 'actual_shots', 'SOG'),
            ('pick_shots_5plus', 'hit_shots_5plus', 'prob_shots_5plus', '5+ SOG', 'actual_shots', 'SOG'),
        ]

        daily_summary = {}

        for pick_col, hit_col, prob_col, label, actual_col, suffix in props_to_check:
            if pick_col not in played.columns or hit_col not in played.columns:
                continue
            picks = played[played[pick_col] == 'YES'].copy()
            if len(picks) == 0:
                continue

            picks[prob_col] = pd.to_numeric(picks[prob_col], errors='coerce')
            picks = picks.sort_values(prob_col, ascending=False)

            hits = len(picks[picks[hit_col] == 'HIT'])
            graded = len(picks[picks[hit_col].isin(['HIT', 'MISS'])])
            pct = hits / graded * 100 if graded > 0 else 0
            daily_summary[label] = (hits, graded, pct)

            print(f"  {label} PICKS ({graded} graded):")
            print(f"  {'─' * 72}")
            print(f"  {'Player':<24s} {'Tm':>3s} {'Opp':>4s} {'Prob':>7s}  "
                  f"{'':>2s}  {'Actual':>8s}")
            print(f"  {'─' * 72}")

            for _, r in picks.iterrows():
                prob = float(r[prob_col]) if pd.notna(r.get(prob_col)) else 0
                hit_val = str(r.get(hit_col, ''))
                if hit_val == 'HIT':
                    icon = "✅"
                elif hit_val == 'MISS':
                    icon = "❌"
                else:
                    icon = "⏳"

                actual_val = r.get(actual_col)
                if pd.notna(actual_val):
                    actual_str = f"{int(actual_val)}{suffix}"
                else:
                    actual_str = "—"

                print(f"  {r['player_name']:<24s} {r['team']:>3s} "
                      f"{r['opponent']:>4s} {prob:>6.1%}  {icon}  {actual_str:>8s}")

            print(f"  {'─' * 72}")
            print(f"  Record: {hits}/{graded} ({pct:.1f}%)\n")

        if daily_summary:
            print(f"  {'═' * 50}")
            print(f"  DAILY SUMMARY — {target_date}")
            print(f"  {'═' * 50}")
            print(f"  {'Prop':<15s} {'Hits':>5s} {'Total':>6s} {'Hit%':>7s}")
            print(f"  {'─' * 35}")
            for label, (hits, total, pct) in daily_summary.items():
                print(f"  {label:<15s} {hits:>5d} {total:>6d} {pct:>6.1f}%")

        # Top performers
        top_scorers = played.nlargest(10, 'actual_points')
        if len(top_scorers) > 0:
            print(f"\n  🌟 TOP PERFORMERS OF THE NIGHT:")
            print(f"  {'─' * 60}")
            for _, r in top_scorers.iterrows():
                g = int(r['actual_goals'])
                a = int(r['actual_assists'])
                p = int(r['actual_points'])
                s = int(r['actual_shots']) if pd.notna(r.get('actual_shots')) else 0
                gp = f"{g}G {a}A {p}P {s}SOG"
                print(f"  {r['player_name']:<24s} {r['team']:>3s}  {gp}")

    else:
        pending = len(day[day['actual_goals'].isna() | (day['actual_goals'] == '')])
        dnp = len(day[day['actual_goals'] == 'DNP']) if 'actual_goals' in day.columns else 0
        print(f"  ⏳ {pending} predictions awaiting results, {dnp} DNP")


# ═══════════════════════════════════════════════════════════════
#  ALL-TIME RUNNING RECORD
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  📈 ALL-TIME PLAYER PROP RECORD")
print(f"{'='*70}\n")

graded = df[df['hit_points_1plus'].isin(['HIT', 'MISS'])].copy()

if len(graded) == 0:
    print("  No graded predictions yet.\n")
else:
    props_to_report = [
        ('pick_goals_1plus', 'hit_goals_1plus', 'prob_goals_1plus', '1+ Goals'),
        ('pick_points_1plus', 'hit_points_1plus', 'prob_points_1plus', '1+ Points'),
        ('pick_points_2plus', 'hit_points_2plus', 'prob_points_2plus', '2+ Points'),
        ('pick_assists_1plus', 'hit_assists_1plus', 'prob_assists_1plus', '1+ Assists'),
        ('pick_shots_3plus', 'hit_shots_3plus', 'prob_shots_3plus', '3+ SOG'),
        ('pick_shots_4plus', 'hit_shots_4plus', 'prob_shots_4plus', '4+ SOG'),
        ('pick_shots_5plus', 'hit_shots_5plus', 'prob_shots_5plus', '5+ SOG'),
    ]

    # ── Overall pick record ──────────────────────────────────
    print(f"  PICKS ABOVE THRESHOLD (flagged 'YES'):")
    print(f"  {'Prop':<15s} {'Hits':>6s} {'Total':>6s} {'Hit%':>7s} {'Base%':>7s} {'Edge':>7s}")
    print(f"  {'─' * 55}")

    for pick_col, hit_col, prob_col, label in props_to_report:
        if pick_col not in graded.columns or hit_col not in graded.columns:
            continue
        picks = graded[graded[pick_col] == 'YES']
        if len(picks) == 0:
            continue

        pick_graded = picks[picks[hit_col].isin(['HIT', 'MISS'])]
        hits = len(pick_graded[pick_graded[hit_col] == 'HIT'])
        total = len(pick_graded)
        if total == 0:
            continue
        hit_pct = hits / total * 100

        all_for_prop = graded[graded[hit_col].isin(['HIT', 'MISS'])]
        base_hits = len(all_for_prop[all_for_prop[hit_col] == 'HIT'])
        base_total = len(all_for_prop)
        base_pct = base_hits / base_total * 100 if base_total > 0 else 0

        edge = hit_pct - base_pct
        print(f"  {label:<15s} {hits:>6d} {total:>6d} {hit_pct:>6.1f}% {base_pct:>6.1f}% {edge:>+6.1f}%")

    # ── Calibration check ────────────────────────────────────
    print(f"\n  CALIBRATION CHECK — ACTUAL HIT RATE BY MODEL PROBABILITY TIER:")

    tiers = [
        ("60%+",   0.60, 1.01),
        ("50-60%", 0.50, 0.60),
        ("40-50%", 0.40, 0.50),
        ("30-40%", 0.30, 0.40),
        ("20-30%", 0.20, 0.30),
        ("<20%",   0.00, 0.20),
    ]

    for pick_col, hit_col, prob_col, label in props_to_report:
        if prob_col not in graded.columns or hit_col not in graded.columns:
            continue

        graded[prob_col] = pd.to_numeric(graded[prob_col], errors='coerce')
        prop_data = graded[graded[hit_col].isin(['HIT', 'MISS']) & graded[prob_col].notna()]
        if len(prop_data) == 0:
            continue

        has_data = False
        for _, lo, hi in tiers:
            tier = prop_data[(prop_data[prob_col] >= lo) & (prop_data[prob_col] < hi)]
            if len(tier) > 0:
                has_data = True
                break

        if not has_data:
            continue

        print(f"\n  {label}:")
        print(f"  {'Tier':<10s} {'Hits':>5s} {'Total':>6s} {'Actual%':>8s} {'AvgProb':>8s} {'Diff':>7s}")
        print(f"  {'─' * 48}")

        for tier_label, lo, hi in tiers:
            tier = prop_data[(prop_data[prob_col] >= lo) & (prop_data[prob_col] < hi)]
            if len(tier) == 0:
                continue
            th = len(tier[tier[hit_col] == 'HIT'])
            tt = len(tier)
            actual = th / tt * 100 if tt > 0 else 0
            avg_prob = tier[prob_col].mean() * 100
            diff = actual - avg_prob
            calib = "✅" if abs(diff) <= 5 else "⚠" if abs(diff) <= 10 else "❌"
            print(f"  {tier_label:<10s} {th:>5d} {tt:>6d} {actual:>7.1f}% {avg_prob:>7.1f}% {diff:>+6.1f}% {calib}")

    # ── Daily hit rates ──────────────────────────────────────
    print(f"\n  DAILY HIT RATES (last 10 days with picks):")
    print(f"  {'Date':<12s} {'1+Pts':>10s} {'1+Goals':>10s} {'3+SOG':>10s}")
    print(f"  {'─' * 48}")

    dates_sorted = sorted(graded['prediction_date'].unique(), reverse=True)[:10]

    for d in dates_sorted:
        day_df = graded[graded['prediction_date'] == d]

        def rate(subset, pick_col, hit_col):
            if pick_col not in subset.columns:
                return "—"
            picks = subset[subset[pick_col] == 'YES']
            gr = picks[picks[hit_col].isin(['HIT', 'MISS'])]
            if len(gr) == 0:
                return "—"
            h = len(gr[gr[hit_col] == 'HIT'])
            return f"{h}/{len(gr)}"

        pts_r = rate(day_df, 'pick_points_1plus', 'hit_points_1plus')
        gls_r = rate(day_df, 'pick_goals_1plus', 'hit_goals_1plus')
        sog_r = rate(day_df, 'pick_shots_3plus', 'hit_shots_3plus')
        print(f"  {d:<12s} {pts_r:>10s} {gls_r:>10s} {sog_r:>10s}")

    total_dates = len(graded['prediction_date'].unique())
    total_graded = len(graded)
    total_dnp = len(df[df['actual_goals'] == 'DNP'])
    print(f"\n  📅 Days tracked: {total_dates}")
    print(f"  👤 Players graded: {total_graded}  |  DNP: {total_dnp}")

print(f"\n{'='*70}")