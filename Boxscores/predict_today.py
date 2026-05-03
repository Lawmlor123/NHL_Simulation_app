"""
predict_today.py
Pull today's NHL schedule, build features, predict winners, SAVE to CSV log.
Usage: python predict_today.py [--date YYYY-MM-DD]
"""
import pandas as pd
import numpy as np
import pickle
import requests
import xgboost as xgb
from pathlib import Path
from datetime import date, datetime, timedelta
import sys

BASE = Path(__file__).parent  # Use current script directory, works on any platform

# Import goalie scraper
try:
    sys.path.insert(0, str(BASE))
    from scrape_goalies_v2 import scrape_starting_goalies, get_goalie_dict
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False
    print("  ⚠ scrape_goalies_v2.py not found — will use historical goalie data")

# Import manual goalie overrides (managed by Alexis daily)
try:
    from manual_overrides import GOALIE_OVERRIDES
except ImportError:
    GOALIE_OVERRIDES = {}

# ── Parse optional date argument ───────────────────────────
target_date = date.today()
for i, arg in enumerate(sys.argv):
    if arg == "--date" and i + 1 < len(sys.argv):
        target_date = datetime.strptime(sys.argv[i + 1], "%Y-%m-%d").date()

print(f"{'='*60}")
print(f"  NHL Game Predictions for {target_date}")
print(f"{'='*60}\n")

# ── Helper: American odds → implied probability ───────────
def american_to_implied(odds):
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return np.nan

# ── 1. LOAD MODEL AND DATA ────────────────────────────────
print("Loading model and data ...")
model = xgb.XGBClassifier()
model.load_model(str(BASE / "xgb_game_model_v2.json"))

# Load probability calibrator (trained on 2024-25 test set)
cal_path = BASE / "game_calibrator.pkl"
if cal_path.exists():
    with open(cal_path, "rb") as f:
        game_calibrator = pickle.load(f)
    print("  ✅ Calibrator loaded")
else:
    game_calibrator = None
    print("  ⚠ game_calibrator.pkl not found — using raw probabilities (run train_model_v2.py first)")

with open(BASE / "model_features_v2.txt") as f:
    feature_cols = [line.strip() for line in f.readlines()]

sk = pd.read_parquet(BASE / "skater_features.parquet")
gl = pd.read_parquet(BASE / "goalie_features.parquet")

# ── REST DAYS HELPER ──────────────────────────────────────
def get_team_rest_days(teams, as_of_date, lookback_days=8):
    """
    For each team, scan the past `lookback_days` days and find the most
    recent completed game. Returns a dict:
        { "BOS": {"last_game": date(2026,4,1), "rest_days": 1, "b2b": True}, ... }
    Uses the same NHL schedule API already in this script.
    """
    result = {t: {"last_game": None, "rest_days": None, "b2b": False} for t in teams}
    today = as_of_date

    # Walk backwards day by day
    for offset in range(1, lookback_days + 1):
        check_date = today - timedelta(days=offset)
        url = f"https://api-web.nhle.com/v1/score/{check_date}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        for g in data.get("games", []):
            if g.get("gameState") not in ("FINAL", "OFF"):
                continue
            home = g["homeTeam"]["abbrev"]
            away = g["awayTeam"]["abbrev"]
            for team in [home, away]:
                if team in result and result[team]["last_game"] is None:
                    rest = (today - check_date).days
                    result[team]["last_game"] = check_date
                    result[team]["rest_days"] = rest
                    result[team]["b2b"] = (rest == 1)

        # Stop early if all teams resolved
        if all(v["last_game"] is not None for v in result.values()):
            break

    return result


# ── 2. PULL TODAY'S SCHEDULE ───────────────────────────────
print(f"Pulling schedule for {target_date} ...\n")
url = f"https://api-web.nhle.com/v1/schedule/{target_date}"
r = requests.get(url, timeout=15)
r.raise_for_status()
schedule = r.json()

games = []
for game_week in schedule.get("gameWeek", []):
    if game_week["date"] == str(target_date):
        for g in game_week.get("games", []):
            # gameType: 1=preseason, 2=regular season, 3=playoffs.
            # Predict regular season AND playoffs (skip preseason).
            if g["gameType"] in (2, 3):
                games.append({
                    "game_id": g["id"],
                    "home_team": g["homeTeam"]["abbrev"],
                    "away_team": g["awayTeam"]["abbrev"],
                    "home_name": g["homeTeam"].get("placeName", {}).get("default", g["homeTeam"]["abbrev"]),
                    "away_name": g["awayTeam"].get("placeName", {}).get("default", g["awayTeam"]["abbrev"]),
                    "start_time": g.get("startTimeUTC", "TBD"),
                    "game_type": g["gameType"],  # 2=regular, 3=playoff (for downstream tagging)
                })

if not games:
    print("No regular-season or playoff games found for this date.")
    print("(Could be off-season, all-star break, or preseason)")
    sys.exit(0)

# Surface playoff vs regular split so it's obvious in the log
n_reg  = sum(1 for g in games if g["game_type"] == 2)
n_pof  = sum(1 for g in games if g["game_type"] == 3)
season_label = []
if n_reg: season_label.append(f"{n_reg} regular")
if n_pof: season_label.append(f"{n_pof} playoff")
print(f"Found {len(games)} games  ({', '.join(season_label)}):\n")
for g in games:
    tag = "[PLAYOFF]" if g["game_type"] == 3 else "         "
    print(f"  {tag}  {g['away_team']:>3s}  @  {g['home_team']:<3s}")
print()

# ── 2a. REST DAYS LOOKUP ───────────────────────────────────
all_teams = list({t for g in games for t in [g["home_team"], g["away_team"]]})
print("Calculating rest days for each team ...")
rest_data = get_team_rest_days(all_teams, target_date)

b2b_teams = [t for t, v in rest_data.items() if v["b2b"]]
if b2b_teams:
    print(f"  ⚠ Back-to-back teams today: {', '.join(sorted(b2b_teams))}")
else:
    print("  ✅ No back-to-back teams today")
print()

# ── 2b. FETCH CONFIRMED STARTING GOALIES FROM DAILYFACEOFF ──
print("Fetching confirmed starting goalies from DailyFaceoff ...")
scraped_goalies = {}
scraped_odds = {}

if SCRAPER_AVAILABLE:
    scraped_games = scrape_starting_goalies()
    if scraped_games:
        scraped_goalies = get_goalie_dict(scraped_games)

        # Build odds lookup keyed by (away_team, home_team)
        for sg in scraped_games:
            away = sg.get('away_team')
            home = sg.get('home_team')
            if away and home:
                scraped_odds[(away, home)] = {
                    'home_ml': sg.get('home_moneyline'),
                    'away_ml': sg.get('away_moneyline'),
                    'spread': sg.get('point_spread'),
                }

        confirmed = sum(1 for g in scraped_goalies.values()
                        if str(g.get('status', '')).lower() == 'confirmed')
        likely = sum(1 for g in scraped_goalies.values()
                     if str(g.get('status', '')).lower() in ('likely', 'expected'))
        print(f"\n  ✅ {confirmed} confirmed  🟡 {likely} likely  "
              f"❓ {len(scraped_goalies) - confirmed - likely} unconfirmed\n")
    else:
        print("  ⚠ Scraper returned no data — falling back to historical goalie data\n")
else:
    print("  ⚠ Scraper not available — falling back to historical goalie data\n")

# ── 2c. MANUAL GOALIE OVERRIDES (from manual_overrides.py — managed by Alexis) ─
if GOALIE_OVERRIDES:
    print(f"  ✏️  MANUAL GOALIE OVERRIDES ({len(GOALIE_OVERRIDES)} team(s))")
    print(f"  {'─' * 55}")
    for team, goalie_name in GOALIE_OVERRIDES.items():
        old = scraped_goalies.get(team, {})
        old_name   = old.get('name',   'None')
        old_status = old.get('status', 'None')
        scraped_goalies[team] = {
            'name':   goalie_name,
            'status': 'Manual',
            'gaa':    old.get('gaa'),    # preserve live stats if already scraped
            'svpct':  old.get('svpct'),
        }
        print(f"    {team}: {old_name} ({old_status}) → {goalie_name} (Manual)")
    print()
# ─────────────────────────────────────────────────────────────────────────────

# ── 3. BUILD FEATURES FOR EACH GAME ───────────────────────
print("Building features ...\n")

# Get rolling columns
rolling_cols = [c for c in sk.columns if any(
    c.endswith(suf) for suf in ["_avg3", "_avg5", "_avg10", "_avg20", "_season_avg"]
)]

sum_cols = [c for c in rolling_cols if any(
    stat in c for stat in ["goals", "points", "shots", "assists", "pp_goals"]
)]

goalie_rolling = [c for c in gl.columns if c.startswith("g_")]

# Get latest features for each player (most recent game)
sk_latest = sk.sort_values(["player_id", "game_date"]).groupby("player_id").last().reset_index()
gl_latest = gl.sort_values(["playerId", "game_date"]).groupby("playerId").last().reset_index()

# Pre-sort goalie starters for matching
gl_starters = gl[gl["is_starter"] == 1].sort_values("game_date").copy()

predictions = []

for game in games:
    home_team = game["home_team"]
    away_team = game["away_team"]

    # Get latest skater features for each team
    home_skaters = sk_latest[sk_latest["team"] == home_team]
    away_skaters = sk_latest[sk_latest["team"] == away_team]

    if len(home_skaters) == 0 or len(away_skaters) == 0:
        print(f"  WARNING: Missing skater data for {away_team} @ {home_team}")
        continue

    # Aggregate skater features
    row = {}

    # Home skater mean
    for col in rolling_cols:
        if col in home_skaters.columns:
            row[f"home_sk_{col}_mean"] = home_skaters[col].mean()
    # Away skater mean
    for col in rolling_cols:
        if col in away_skaters.columns:
            row[f"away_sk_{col}_mean"] = away_skaters[col].mean()

    # Home skater sum (offensive cols only)
    for col in sum_cols:
        if col in home_skaters.columns:
            row[f"home_sk_{col}_sum"] = home_skaters[col].sum()
    # Away skater sum
    for col in sum_cols:
        if col in away_skaters.columns:
            row[f"away_sk_{col}_sum"] = away_skaters[col].sum()

    # ── Goalie features: match scraped starter to parquet data ──
    home_goalie_name = "Unknown"
    away_goalie_name = "Unknown"
    home_goalie_status = "Unknown"
    away_goalie_status = "Unknown"
    home_goalie_gaa_live = None
    away_goalie_gaa_live = None
    home_goalie_svpct_live = None
    away_goalie_svpct_live = None

    # --- Home goalie ---
    home_matched = None
    home_scraped = scraped_goalies.get(home_team, {})
    if home_scraped.get('name'):
        home_goalie_name = home_scraped['name']
        home_goalie_status = str(home_scraped.get('status', 'Unknown'))
        home_goalie_gaa_live = home_scraped.get('gaa')
        home_goalie_svpct_live = home_scraped.get('svpct')

        # Match by last name in parquet
        last_name = home_goalie_name.strip().split()[-1] if home_goalie_name.strip() else ''
        if last_name:
            team_gl = gl_starters[gl_starters["team"] == home_team]
            name_match = team_gl[
                team_gl["goalie_name"].str.contains(last_name, case=False, na=False)
            ] if "goalie_name" in team_gl.columns else pd.DataFrame()

            if "player_name" in team_gl.columns and len(name_match) == 0:
                name_match = team_gl[
                    team_gl["player_name"].str.contains(last_name, case=False, na=False)
                ]

            if len(name_match) > 0:
                home_matched = name_match.iloc[-1]

    # Fallback: highest TOI goalie for team
    if home_matched is None:
        home_goalies_fb = gl_latest[gl_latest["team"] == home_team].sort_values("toi_min", ascending=False)
        if len(home_goalies_fb) > 0:
            home_matched = home_goalies_fb.iloc[0]
            if not home_scraped.get('name'):
                name_col = "goalie_name" if "goalie_name" in home_matched.index else "player_name"
                home_goalie_name = home_matched.get(name_col, "Unknown")
            home_goalie_status = "Fallback"

    if home_matched is not None:
        for col in goalie_rolling:
            if col in home_matched.index:
                row[f"home_gl_{col}"] = home_matched[col]

    # --- Away goalie ---
    away_matched = None
    away_scraped = scraped_goalies.get(away_team, {})
    if away_scraped.get('name'):
        away_goalie_name = away_scraped['name']
        away_goalie_status = str(away_scraped.get('status', 'Unknown'))
        away_goalie_gaa_live = away_scraped.get('gaa')
        away_goalie_svpct_live = away_scraped.get('svpct')

        last_name = away_goalie_name.strip().split()[-1] if away_goalie_name.strip() else ''
        if last_name:
            team_gl = gl_starters[gl_starters["team"] == away_team]
            name_match = team_gl[
                team_gl["goalie_name"].str.contains(last_name, case=False, na=False)
            ] if "goalie_name" in team_gl.columns else pd.DataFrame()

            if "player_name" in team_gl.columns and len(name_match) == 0:
                name_match = team_gl[
                    team_gl["player_name"].str.contains(last_name, case=False, na=False)
                ]

            if len(name_match) > 0:
                away_matched = name_match.iloc[-1]

    if away_matched is None:
        away_goalies_fb = gl_latest[gl_latest["team"] == away_team].sort_values("toi_min", ascending=False)
        if len(away_goalies_fb) > 0:
            away_matched = away_goalies_fb.iloc[0]
            if not away_scraped.get('name'):
                name_col = "goalie_name" if "goalie_name" in away_matched.index else "player_name"
                away_goalie_name = away_matched.get(name_col, "Unknown")
            away_goalie_status = "Fallback"

    if away_matched is not None:
        for col in goalie_rolling:
            if col in away_matched.index:
                row[f"away_gl_{col}"] = away_matched[col]

    # Differential features
    for col in rolling_cols:
        for agg in ["mean", "sum"]:
            hc = f"home_sk_{col}_{agg}"
            ac = f"away_sk_{col}_{agg}"
            dc = f"diff_sk_{col}_{agg}"
            if hc in row and ac in row:
                row[dc] = row[hc] - row[ac]

    for col in goalie_rolling:
        hc = f"home_gl_{col}"
        ac = f"away_gl_{col}"
        dc = f"diff_{col}"
        if hc in row and ac in row:
            row[dc] = row[hc] - row[ac]

    # Build feature vector
    feat_vector = pd.DataFrame([row])

    # Ensure all model features exist
    for col in feature_cols:
        if col not in feat_vector.columns:
            feat_vector[col] = np.nan

    feat_vector = feat_vector[feature_cols]

    # Predict — apply isotonic calibration if available
    raw_prob = model.predict_proba(feat_vector)[:, 1][0]
    if game_calibrator is not None:
        prob = float(game_calibrator.predict([raw_prob])[0])
    else:
        prob = raw_prob
    # raw_prob is stored in the log so the calibrator can be retrained
    # directly on XGB outputs (not on already-calibrated values)

    # Determine pick
    if prob > 0.5:
        pick = home_team
        confidence = prob
    else:
        pick = away_team
        confidence = 1 - prob

    # Confidence label
    if confidence >= 0.65:
        conf_label = "★★★ HIGH"
    elif confidence >= 0.58:
        conf_label = "★★  MED"
    elif confidence >= 0.52:
        conf_label = "★   LOW"
    else:
        conf_label = "    SKIP"

    # ── Goalie filter: flag games where either goalie is unconfirmed ──────────
    # Statuses we trust enough to log: Confirmed, Likely, Expected, Manual
    TRUSTED_STATUSES = {"confirmed", "likely", "expected", "manual"}
    goalie_ok = (
        str(home_goalie_status).lower() in TRUSTED_STATUSES and
        str(away_goalie_status).lower() in TRUSTED_STATUSES
    )

    # ── MED tier filter: 58-64% confidence is chronically underperforming ─────
    is_med_tier = 0.58 <= confidence < 0.65

    # ── Confidence floor: only log HIGH (≥65%) or LOW (52-57%) with good goalies ─
    # Blocks: MED (58-64%), SKIP (<52%), and unconfirmed goalie games
    # 58-60% sub-band was worst offender (30% win rate) — floor raised to 65% or <58%
    log_eligible = goalie_ok and (confidence >= 0.65 or (0.52 <= confidence < 0.58))

    # Get moneyline odds from scraper
    odds_key = (away_team, home_team)
    odds_info = scraped_odds.get(odds_key, {})
    home_ml = odds_info.get('home_ml')
    away_ml = odds_info.get('away_ml')
    home_implied = american_to_implied(home_ml)
    away_implied = american_to_implied(away_ml)

    # Calculate edge vs market
    if pick == home_team and home_implied is not None and not np.isnan(home_implied):
        market_implied = home_implied
        edge = confidence - market_implied
    elif pick == away_team and away_implied is not None and not np.isnan(away_implied):
        market_implied = away_implied
        edge = confidence - market_implied
    else:
        market_implied = None
        edge = None

    # ── Rest / back-to-back context ───────────────────────────
    home_rest = rest_data.get(home_team, {})
    away_rest = rest_data.get(away_team, {})

    predictions.append({
        "away": away_team,
        "home": home_team,
        "prob_home": prob,           # calibrated probability
        "raw_prob_home": raw_prob,   # raw XGB output — needed for future recalibration

        "pick": pick,
        "confidence": confidence,
        "conf_label": conf_label,
        "log_eligible": log_eligible,   # False = MED tier or unconfirmed goalie
        "skip_reason": (
            "MED tier" if is_med_tier else
            "unconfirmed goalie" if not goalie_ok else
            ""
        ),
        "home_b2b": int(home_rest.get("b2b", False)),
        "away_b2b": int(away_rest.get("b2b", False)),
        "home_rest_days": home_rest.get("rest_days"),
        "away_rest_days": away_rest.get("rest_days"),
        "home_goalie": home_goalie_name,
        "away_goalie": away_goalie_name,
        "home_goalie_status": home_goalie_status,
        "away_goalie_status": away_goalie_status,
        "home_goalie_gaa": home_goalie_gaa_live,
        "away_goalie_gaa": away_goalie_gaa_live,
        "home_goalie_svpct": home_goalie_svpct_live,
        "away_goalie_svpct": away_goalie_svpct_live,
        "home_ml": home_ml,
        "away_ml": away_ml,
        "home_implied": home_implied,
        "away_implied": away_implied,
        "market_implied": market_implied,
        "edge": edge,
    })

# ── 4. DISPLAY PREDICTIONS ────────────────────────────────
print(f"\n{'='*80}")
print(f"  PREDICTIONS FOR {target_date}")
print(f"{'='*80}\n")

# Goalie summary
print(f"  🏒 STARTING GOALIES")
print(f"  {'─'*75}")
for p in predictions:
    def goalie_icon(status):
        s = str(status).lower()
        if s == 'confirmed':
            return '✅'
        elif s in ('likely', 'expected'):
            return '🟡'
        elif s == 'fallback':
            return '🔄'
        else:
            return '❓'

    a_icon = goalie_icon(p['away_goalie_status'])
    h_icon = goalie_icon(p['home_goalie_status'])

    a_gaa = f" GAA:{float(p['away_goalie_gaa']):.2f}" if p.get('away_goalie_gaa') else ""
    h_gaa = f" GAA:{float(p['home_goalie_gaa']):.2f}" if p.get('home_goalie_gaa') else ""
    a_svp = f" SV%:{float(p['away_goalie_svpct']):.3f}" if p.get('away_goalie_svpct') else ""
    h_svp = f" SV%:{float(p['home_goalie_svpct']):.3f}" if p.get('home_goalie_svpct') else ""

    print(f"  {p['away']:>3s} @ {p['home']:<3s}")
    print(f"    {a_icon} {p['away']:>3s}  {p['away_goalie']:<25s} {p['away_goalie_status']}{a_gaa}{a_svp}")
    print(f"    {h_icon} {p['home']:>3s}  {p['home_goalie']:<25s} {p['home_goalie_status']}{h_gaa}{h_svp}")

# Main predictions table
print(f"\n  {'─'*75}")
print(f"  {'Matchup':<15s} {'Pick':>5s}  {'Conf':>5s}  {'Rating':<10s} "
      f"{'ML':>6s}  {'Impl':>5s}  {'Edge':>6s}  {'Goalies'}")
print(f"  {'─'*95}")

for p in sorted(predictions, key=lambda x: x["confidence"], reverse=True):
    matchup = f"{p['away']} @ {p['home']}"
    goalies = f"{p['away_goalie'][:12]} vs {p['home_goalie'][:12]}"

    # Show the moneyline for the picked team
    if p['pick'] == p['home']:
        pick_ml = p.get('home_ml', '')
    else:
        pick_ml = p.get('away_ml', '')
    ml_str = str(pick_ml) if pick_ml else "—"

    impl_str = f"{p['market_implied']:.0%}" if p.get('market_implied') is not None else "—"
    edge_str = f"{p['edge']:+.1%}" if p.get('edge') is not None else "—"

    # Flag good edges
    if p.get('edge') is not None and p['edge'] > 0.03:
        edge_flag = " 💰"
    elif p.get('edge') is not None and p['edge'] > 0:
        edge_flag = " ✓"
    else:
        edge_flag = ""

    log_flag = "" if p["log_eligible"] else f"  ⛔ SKIP ({p['skip_reason']})"
    b2b_flag = ""
    if p.get("home_b2b"):
        b2b_flag += f" 🔁{p['home']}(b2b)"
    if p.get("away_b2b"):
        b2b_flag += f" 🔁{p['away']}(b2b)"
    rest_note = ""
    hr = p.get("home_rest_days")
    ar = p.get("away_rest_days")
    if hr is not None and ar is not None:
        rest_note = f"  rest:{ar}d/{hr}d"
    print(f"  {matchup:<15s} {p['pick']:>5s}  {p['confidence']:>5.1%}  {p['conf_label']:<10s} "
          f"{ml_str:>6s}  {impl_str:>5s}  {edge_str:>6s}{edge_flag}{rest_note}{b2b_flag}{log_flag}")

# ── 5. VALUE PICKS (model edge vs market) ─────────────────
value_picks = [p for p in predictions if p.get('edge') is not None and p['edge'] > 0.03]
value_picks.sort(key=lambda x: x['edge'], reverse=True)

if value_picks:
    print(f"\n{'='*80}")
    print(f"  💰 VALUE PICKS — Model Edge > 3% vs Market")
    print(f"{'='*80}\n")
    print(f"  {'Matchup':<15s} {'Pick':>5s}  {'Model':>6s}  {'Market':>6s}  {'Edge':>6s}  {'ML':>6s}  {'Rating':<10s}")
    print(f"  {'─'*70}")

    for p in value_picks:
        matchup = f"{p['away']} @ {p['home']}"
        if p['pick'] == p['home']:
            pick_ml = p.get('home_ml', '')
        else:
            pick_ml = p.get('away_ml', '')
        ml_str = str(pick_ml) if pick_ml else "—"

        print(f"  {matchup:<15s} {p['pick']:>5s}  {p['confidence']:>5.1%}  "
              f"{p['market_implied']:>5.1%}  {p['edge']:>+5.1%}  {ml_str:>6s}  {p['conf_label']}")

# ── 6. SUMMARY ────────────────────────────────────────────
high_conf = [p for p in predictions if p["confidence"] >= 0.65]
print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
print(f"  Total games: {len(predictions)}")
print(f"  High confidence picks (≥65%): {len(high_conf)}")
print(f"  Value picks (edge > 3%): {len(value_picks)}")

# Goalie confidence breakdown
confirmed_g = sum(1 for p in predictions
                  for s in [p['home_goalie_status'], p['away_goalie_status']]
                  if str(s).lower() == 'confirmed')
likely_g = sum(1 for p in predictions
               for s in [p['home_goalie_status'], p['away_goalie_status']]
               if str(s).lower() in ('likely', 'expected'))
total_g = len(predictions) * 2
print(f"  Goalie status: ✅ {confirmed_g}/{total_g} confirmed  "
      f"🟡 {likely_g}/{total_g} likely  "
      f"❓ {total_g - confirmed_g - likely_g}/{total_g} other")

if high_conf:
    print(f"\n  🏒 TOP PICKS:")
    for p in high_conf:
        direction = "HOME" if p["prob_home"] > 0.5 else "AWAY"
        edge_note = ""
        if p.get('edge') is not None:
            edge_note = f" | edge: {p['edge']:+.1%}"
        print(f"     {p['pick']} ({direction}) — {p['confidence']:.1%} confidence{edge_note}")

if value_picks:
    print(f"\n  💰 BEST VALUE:")
    for p in value_picks[:3]:
        direction = "HOME" if p["prob_home"] > 0.5 else "AWAY"
        if p['pick'] == p['home']:
            pick_ml = p.get('home_ml', '')
        else:
            pick_ml = p.get('away_ml', '')
        print(f"     {p['pick']} ({direction}) — {p['confidence']:.1%} model vs "
              f"{p['market_implied']:.1%} market = {p['edge']:+.1%} edge  [ML: {pick_ml}]")

print(f"\n{'='*80}")

# ── 7. SAVE PREDICTIONS TO LOG ─────────────────────────────
log_file = BASE / "prediction_log.csv"

# Situational context for future audits
import calendar
dow = target_date.strftime("%A")           # e.g. "Wednesday"
dow_num = target_date.weekday()            # 0=Mon, 6=Sun
is_weekend = dow_num >= 5                  # Sat/Sun

log_rows = []
skipped_log = []
for p in predictions:
    if not p["log_eligible"]:
        skipped_log.append(f"{p['away']} @ {p['home']} ({p['skip_reason']})")
        continue
    log_rows.append({
        "pred_date":           str(target_date),
        "away_team":           p["away"],
        "home_team":           p["home"],
        "pick":                p["pick"],
        "prob_home":           round(p["prob_home"], 4),
        "raw_prob_home":       round(p.get("raw_prob_home", p["prob_home"]), 4),
        "confidence":          round(p["confidence"], 4),
        "conf_label":          p["conf_label"].strip(),
        "home_goalie":         p["home_goalie"],
        "away_goalie":         p["away_goalie"],
        "home_goalie_status":  p.get("home_goalie_status", ""),
        "away_goalie_status":  p.get("away_goalie_status", ""),
        "home_ml":             p.get("home_ml", ""),
        "away_ml":             p.get("away_ml", ""),
        "market_implied":      round(p["market_implied"], 4) if p.get("market_implied") is not None else "",
        "edge":                round(p["edge"], 4) if p.get("edge") is not None else "",
        # ── Situational flags (for future audit analysis) ──────────────────
        "day_of_week":         dow,
        "is_weekend":          int(is_weekend),
        "home_b2b":            p.get("home_b2b", ""),
        "away_b2b":            p.get("away_b2b", ""),
        "home_rest_days":      p.get("home_rest_days", ""),
        "away_rest_days":      p.get("away_rest_days", ""),
        # ── Result fields (filled in by track_results.py) ─────────────────
        "actual_winner":       "",
        "home_goals":          "",
        "away_goals":          "",
        "result":              "",
    })

log_df = pd.DataFrame(log_rows)

if log_file.exists():
    existing = pd.read_csv(log_file)
    existing = existing[existing["pred_date"] != str(target_date)]
    combined = pd.concat([existing, log_df], ignore_index=True)
else:
    combined = log_df

combined.to_csv(log_file, index=False)
print(f"\n  💾 Predictions saved to {log_file}")
print(f"     ({len(log_rows)} games logged for {target_date})")
if skipped_log:
    print(f"     ({len(skipped_log)} games NOT logged — filtered out)")
    for s in skipped_log:
        print(f"       ⛔ {s}")
print(f"     (Total predictions in log: {len(combined)})")
print(f"\n{'='*80}")