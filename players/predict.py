"""
predict.py  –  Generate NHL player prop predictions for tonight's games
─────────────────────────────────────────────────────────────────────────
Usage:
    python predict.py                       # auto-detect today's games
    python predict.py --date 2025-03-03     # specific date
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import requests
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

# Import goalie scraper
try:
    from scrape_goalies_v2 import scrape_starting_goalies, get_goalie_dict # type: ignore
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False
    print("  ⚠ scrape_goalies_v2.py not found — will use historical goalie data")

# ── NEW: Import manual goalie overrides ──────────────────────────────────
try:
    from manual_overrides import GOALIE_OVERRIDES
except ImportError:
    GOALIE_OVERRIDES = {}
# ─────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════
SKATER_PATH = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores\skater_features.parquet"
GOALIE_PATH = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores\goalie_features.parquet"
MODEL_PATH  = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\models\nhl_player_models.pkl"
OUTPUT_DIR  = r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  TEAM ABBREVIATION NORMALIZER
# ═══════════════════════════════════════════════════════════════
TEAM_NORM = {
    "LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL",
    "WAS": "WSH", "LAS": "VGK", "WIN": "WPG", "MON": "MTL",
    "VGS": "VGK", "CLB": "CBJ",
}

def norm_team(abbrev):
    abbrev = abbrev.strip().upper()
    return TEAM_NORM.get(abbrev, abbrev)

# ═══════════════════════════════════════════════════════════════
#  FETCH TONIGHT'S SCHEDULE FROM NHL API
# ═══════════════════════════════════════════════════════════════
def fetch_nhl_schedule(game_date):
    """Returns list of dicts: [{"away": "BOS", "home": "TOR"}, ...]"""
    url = f"https://api-web.nhle.com/v1/schedule/{game_date}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        games = []
        for week in data.get("gameWeek", []):
            if week.get("date") == game_date:
                for g in week.get("games", []):
                    away = norm_team(g["awayTeam"]["abbrev"])
                    home = norm_team(g["homeTeam"]["abbrev"])
                    state = g.get("gameState", "FUT")
                    games.append({"away": away, "home": home, "state": state})
        return games
    except Exception as e:
        print(f"  ⚠ Could not fetch schedule: {e}")
        return None


def manual_schedule():
    """Prompt user for tonight's games."""
    print("\n  Enter games manually.")
    print("  Format: AWAY@HOME  (comma-separated)")
    print("  Example: BOS@TOR,NYR@MTL,EDM@CGY")
    raw = input("  Games: ").strip()
    games = []
    for matchup in raw.split(","):
        parts = matchup.strip().split("@")
        if len(parts) == 2:
            games.append({
                "away": norm_team(parts[0]),
                "home": norm_team(parts[1]),
                "state": "FUT",
            })
    return games


def american_to_implied(odds):
    """Convert American odds (-110, +150 etc.) to implied probability."""
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return np.nan


# ═══════════════════════════════════════════════════════════════
#  FETCH CONFIRMED STARTING GOALIES
# ═══════════════════════════════════════════════════════════════
def fetch_confirmed_goalies():
    """
    Use DailyFaceoff scraper to get tonight's confirmed/likely starters.
    Returns:
        goalie_dict: {team_abbrev: {name, status, gaa, svpct, ...}}
        game_odds:   {(away, home): {home_ml, away_ml}}
    """
    if not SCRAPER_AVAILABLE:
        return {}, {}

    scraped_games = scrape_starting_goalies()
    if not scraped_games:
        return {}, {}

    goalie_dict = get_goalie_dict(scraped_games)

    # Build odds lookup keyed by (away_team, home_team)
    game_odds = {}
    for sg in scraped_games:
        away = sg.get('away_team')
        home = sg.get('home_team')
        if away and home:
            game_odds[(away, home)] = {
                'home_ml': sg.get('home_moneyline'),
                'away_ml': sg.get('away_moneyline'),
                'spread': sg.get('point_spread'),
            }

    return goalie_dict, game_odds


# ═══════════════════════════════════════════════════════════════
#  NEW: APPLY MANUAL GOALIE OVERRIDES
# ═══════════════════════════════════════════════════════════════
def apply_goalie_overrides(scraped_goalies, overrides):
    """
    Inject manual goalie overrides into the scraped_goalies dict.
    This runs BEFORE goalie_lookup is built, so the entire downstream
    pipeline (parquet matching, feature extraction, display) uses the
    overridden goalie automatically.

    Parameters
    ----------
    scraped_goalies : dict
        {team_abbrev: {name, status, gaa, svpct, ...}} from DailyFaceoff
    overrides : dict
        {team_abbrev: "Goalie Full Name"} from manual_overrides.py

    Returns
    -------
    scraped_goalies : dict (modified in place and returned)
    """
    applied = []

    for team, goalie_name in overrides.items():
        team = norm_team(team)
        old = scraped_goalies.get(team, {})
        old_name = old.get('name', 'None')
        old_status = old.get('status', 'None')

        # Preserve any existing scraped stats (GAA, SV%) if same goalie
        # Otherwise clear them so parquet lookup takes over
        if old_name and old_name.split()[-1].lower() == goalie_name.split()[-1].lower():
            # Same goalie — just upgrade status to Manual
            scraped_goalies[team] = {**old, 'status': 'Manual'}
            applied.append(f"    ✏️  {team}: {old_name} ({old_status}) → status upgraded to Manual")
        else:
            # Different goalie — replace entry, clear live stats
            scraped_goalies[team] = {
                'name': goalie_name,
                'status': 'Manual',
                'gaa': None,
                'svpct': None,
            }
            applied.append(f"    ✏️  {team}: {old_name} ({old_status}) → {goalie_name} (Manual)")

    return scraped_goalies, applied


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Game date YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    game_date = args.date or date.today().isoformat()

    print(f"\n{'═' * 70}")
    print(f"  NHL PLAYER PROP PREDICTIONS — {game_date}")
    print(f"{'═' * 70}")

    # ── 1. Load models ───────────────────────────────────────
    print("\nLoading models...")
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    models       = saved["models"]
    feature_cols = saved["feature_cols"]

    print(f"  {len(models)} model(s) loaded  (trained {saved['trained_date'][:10]})")
    for name, info in models.items():
        print(f"    {info['desc']:<20s}  AUC={info['auc']:.4f}  base_rate={info['pos_rate']:.3f}")

    # ── 2. Load feature data ─────────────────────────────────
    print("\nLoading feature data...")
    skaters = pd.read_parquet(SKATER_PATH)
    goalies = pd.read_parquet(GOALIE_PATH)
    skaters["game_date"] = pd.to_datetime(skaters["game_date"])
    goalies["game_date"] = pd.to_datetime(goalies["game_date"])
    print(f"  Skaters: {skaters.shape[0]:,} rows   Goalies: {goalies.shape[0]:,} rows")

    # ── 3. Get tonight's schedule ────────────────────────────
    print(f"\nFetching schedule for {game_date} ...")
    games = fetch_nhl_schedule(game_date)

    if not games:
        games = manual_schedule()

    if not games:
        print("  No games found. Exiting.")
        return

    # ── 3b. Fetch confirmed starting goalies from DailyFaceoff ──
    print(f"\nFetching confirmed starting goalies from DailyFaceoff...")
    scraped_goalies, game_odds = fetch_confirmed_goalies()

    if scraped_goalies:
        confirmed = sum(1 for g in scraped_goalies.values()
                        if str(g.get('status', '')).lower() == 'confirmed')
        likely = sum(1 for g in scraped_goalies.values()
                     if str(g.get('status', '')).lower() in ('likely', 'expected'))
        print(f"  ✅ {confirmed} confirmed  🟡 {likely} likely  "
              f"❓ {len(scraped_goalies) - confirmed - likely} unconfirmed")
    else:
        print(f"  ⚠ Could not scrape goalies — falling back to historical data")

    # ── 3c. NEW: Apply manual goalie overrides ───────────────
    if GOALIE_OVERRIDES:
        print(f"\n  ✏️  MANUAL GOALIE OVERRIDES ({len(GOALIE_OVERRIDES)} team(s))")
        print(f"  {'─' * 60}")
        scraped_goalies, override_log = apply_goalie_overrides(
            scraped_goalies, GOALIE_OVERRIDES
        )
        for msg in override_log:
            print(msg)
        print()
    # ─────────────────────────────────────────────────────────

    # Attach moneylines and implied probabilities to games
    for g in games:
        key = (g["away"], g["home"])
        odds = game_odds.get(key, {})
        g["home_ml"] = odds.get("home_ml")
        g["away_ml"] = odds.get("away_ml")
        g["home_implied"] = american_to_implied(odds.get("home_ml"))
        g["away_implied"] = american_to_implied(odds.get("away_ml"))

    print(f"\n  {'Game':<20s}  {'Status':>7s}  {'Away ML':>8s}  {'Home ML':>8s}  {'Away Impl':>9s}  {'Home Impl':>9s}")
    print(f"  {'─' * 70}")
    for g in games:
        aml = g.get('away_ml', '')
        hml = g.get('home_ml', '')
        aimp = f"{g['away_implied']:.1%}" if pd.notna(g.get('away_implied')) else "—"
        himp = f"{g['home_implied']:.1%}" if pd.notna(g.get('home_implied')) else "—"
        print(f"  {g['away']:>4s}  @  {g['home']:<4s}    {g.get('state', ''):>7s}  "
              f"{str(aml):>8s}  {str(hml):>8s}  {aimp:>9s}  {himp:>9s}")

    tonight_teams = set()
    for g in games:
        tonight_teams.add(g["away"])
        tonight_teams.add(g["home"])

    # ── 4. Latest features per player ────────────────────────
    recent = (
        skaters.sort_values("game_date")
        .groupby("player_id")
        .tail(1)
        .copy()
    )

    active = recent[recent["team"].isin(tonight_teams)].copy()
    print(f"\n  {len(active):,} skaters found on tonight's teams")

    if len(active) == 0:
        print("  ⚠ No matching players. Check that team abbreviations in your")
        print("    parquet match the NHL API codes. Exiting.")
        return

    # ── 5. Opposing-goalie features (UPGRADED with live scraper) ──
    goalie_stat_cols = [
        "g_goalsAgainst_avg3", "g_goalsAgainst_avg5", "g_goalsAgainst_avg10",
        "g_goalsAgainst_season_avg",
        "g_savePct_avg3", "g_savePct_avg5", "g_savePct_avg10", "g_savePct_season_avg",
        "g_shotsAgainst_avg3", "g_shotsAgainst_avg5", "g_shotsAgainst_avg10",
        "g_shotsAgainst_season_avg",
        "g_win_rate_3", "g_win_rate_5", "g_win_rate_10", "g_win_rate_season",
    ]

    # Get all starter rows from parquet, sorted by date
    goalie_all_starters = (
        goalies[goalies["is_starter"] == 1]
        .sort_values("game_date")
        .copy()
    )

    goalie_lookup = {}
    for team in tonight_teams:
        scraped = scraped_goalies.get(team, {})
        scraped_name = scraped.get('name', '')
        scraped_status = str(scraped.get('status', 'Unknown'))
        matched_row = None

        if scraped_name:
            # Try to find this specific goalie in our parquet data
            team_goalies = goalie_all_starters[goalie_all_starters["team"] == team]

            # Match by last name (most reliable)
            last_name = scraped_name.strip().split()[-1] if scraped_name.strip() else ''
            if last_name:
                name_match = team_goalies[
                    team_goalies["goalie_name"].str.contains(last_name, case=False, na=False)
                ]
                if len(name_match) > 0:
                    matched_row = name_match.iloc[-1]  # most recent game for THIS goalie

        # Fallback: most recent starter for team (old behavior)
        if matched_row is None:
            team_goalies = goalie_all_starters[goalie_all_starters["team"] == team]
            if len(team_goalies) > 0:
                matched_row = team_goalies.iloc[-1]
                scraped_status = "Fallback"

        if matched_row is not None:
            goalie_lookup[team] = {
                "goalie_name": scraped_name if scraped_name else matched_row.get("goalie_name", "Unknown"),
                "goalie_status": scraped_status,
                "goalie_gaa_live": scraped.get('gaa'),
                "goalie_svpct_live": scraped.get('svpct'),
            }
            for c in goalie_stat_cols:
                goalie_lookup[team][f"opp_{c}"] = matched_row.get(c, np.nan)

    print(f"\n  🏒 Starting Goalies:")
    print(f"  {'─' * 65}")
    for g in games:
        for side, team in [("Away", g["away"]), ("Home", g["home"])]:
            gl = goalie_lookup.get(team, {})
            name = gl.get("goalie_name", "?")
            status = gl.get("goalie_status", "?")
            gaa = gl.get("goalie_gaa_live")
            svpct = gl.get("goalie_svpct_live")

            if str(status).lower() == 'confirmed':
                icon = '✅'
            elif str(status).lower() in ('likely', 'expected'):
                icon = '🟡'
            elif str(status).lower() == 'manual':
                icon = '✏️'
            elif str(status).lower() == 'fallback':
                icon = '🔄'
            else:
                icon = '❓'

            gaa_str = f"  GAA:{float(gaa):.2f}" if gaa else ""
            svp_str = f"  SV%:{float(svpct):.3f}" if svpct else ""
            print(f"    {icon} {team:>3s}  {name:<25s} {status}{gaa_str}{svp_str}")

    # ── 6. Opponent team-defense features ────────────────────
    game_info = (
        skaters[["game_pk", "game_date", "home_team", "away_team",
                 "home_score", "away_score"]]
        .drop_duplicates(subset=["game_pk"])
    )

    home_g = game_info.rename(columns={
        "home_team": "t", "home_score": "gf", "away_score": "ga"
    })[["game_pk", "game_date", "t", "gf", "ga"]]

    away_g = game_info.rename(columns={
        "away_team": "t", "away_score": "gf", "home_score": "ga"
    })[["game_pk", "game_date", "t", "gf", "ga"]]

    team_games = (
        pd.concat([home_g, away_g], ignore_index=True)
        .sort_values(["t", "game_date"])
        .reset_index(drop=True)
    )

    team_defense = {}
    for team in tonight_teams:
        tg = team_games[team_games["t"] == team].sort_values("game_date")
        if len(tg) < 5:
            continue
        team_defense[team] = {
            "opp_ga_avg5":       tg["ga"].tail(5).mean(),
            "opp_ga_avg10":      tg["ga"].tail(10).mean(),
            "opp_ga_avg20":      tg["ga"].tail(20).mean(),
            "opp_ga_season_avg": tg["ga"].mean(),
            "opp_gf_avg5":       tg["gf"].tail(5).mean(),
            "opp_gf_avg10":      tg["gf"].tail(10).mean(),
            "opp_gf_avg20":      tg["gf"].tail(20).mean(),
            "opp_gf_season_avg": tg["gf"].mean(),
        }

    # ── 7. Build prediction rows ─────────────────────────────
    print("\nBuilding prediction rows...")

    pred_rows = []
    pred_date = pd.to_datetime(game_date)

    for g in games:
        away_team, home_team = g["away"], g["home"]

        for team, opponent, is_home in [
            (home_team, away_team, 1),
            (away_team, home_team, 0),
        ]:
            team_players = active[active["team"] == team]

            for _, player in team_players.iterrows():
                row = {}

                # Copy the player's latest rolling features
                for col in feature_cols:
                    if col in player.index:
                        row[col] = player[col]
                    else:
                        row[col] = np.nan

                # Override context features for tonight
                row["is_home"] = is_home

                last_game = pd.to_datetime(player["game_date"])
                row["rest_days"]    = max((pred_date - last_game).days, 0)
                row["back_to_back"] = int(row["rest_days"] <= 1)
                row["b2b_road"]     = int(row["rest_days"] <= 1 and is_home == 0)

                season_start = pd.to_datetime(
                    f"{pred_date.year if pred_date.month >= 10 else pred_date.year - 1}-10-10"
                )
                row["days_into_season"] = (pred_date - season_start).days
                row["game_num"] = int(player.get("game_num", 40)) + 1

                # Opposing goalie features
                opp_g = goalie_lookup.get(opponent, {})
                for k, v in opp_g.items():
                    if k not in ("goalie_name", "goalie_status",
                                 "goalie_gaa_live", "goalie_svpct_live") and k in feature_cols:
                        row[k] = v

                # Opponent team defense
                opp_d = team_defense.get(opponent, {})
                for k, v in opp_d.items():
                    if k in feature_cols:
                        row[k] = v

                # Metadata (prefixed with _ so we can split later)
                row["_player_name"] = player.get("player_name", "Unknown")
                row["_player_id"]   = player.get("player_id", 0)
                row["_team"]        = team
                row["_opponent"]    = opponent
                row["_is_home"]     = is_home
                row["_position"]    = player.get("position", "?")
                row["_opp_goalie"]  = opp_g.get("goalie_name", "?")
                row["_opp_goalie_status"] = opp_g.get("goalie_status", "?")
                row["_rest_days"]   = row["rest_days"]
                row["_home_ml"]     = g.get("home_ml")
                row["_away_ml"]     = g.get("away_ml")
                row["_home_implied"] = g.get("home_implied")
                row["_away_implied"] = g.get("away_implied")

                pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    print(f"  {len(pred_df):,} prediction rows built")

    if len(pred_df) == 0:
        print("  No prediction rows. Exiting.")
        return

    # ── 8. Run predictions ───────────────────────────────────
    print("\nScoring all models...")

    X_pred = pred_df[feature_cols].copy()

    for target_name, info in models.items():
        raw  = info["model"].predict_proba(X_pred)[:, 1]
        cal  = info["calibrator"].predict(raw)
        pred_df[f"prob_{target_name}"] = cal

    # ── 9. Build output table ────────────────────────────────
    meta_cols = [c for c in pred_df.columns if c.startswith("_")]
    prob_cols = sorted([c for c in pred_df.columns if c.startswith("prob_")])

    output = pred_df[meta_cols + prob_cols].copy()
    output.columns = [c.lstrip("_") for c in output.columns]
    output = output.sort_values("prob_points_1plus", ascending=False).reset_index(drop=True)

    # ── 10. Print results ────────────────────────────────────

    # --- Top Over picks (1+ points) ---
    overs = output[output["prob_points_1plus"] >= 0.50]
    print(f"\n{'═' * 90}")
    print(f"  OVER 0.5 POINTS — {len(overs)} picks  (model prob >= 50%)")
    print(f"{'═' * 90}")
    print(f"  {'#':>3s}  {'Player':<24s} {'Tm':>3s} {'Opp':>4s} {'Pos':>3s} "
          f"{'H/A':>3s} {'Rest':>4s} {'vs Goalie':<18s} {'Conf':>6s} {'Prob':>6s}")
    print(f"  {'─' * 88}")

    for i, (_, r) in enumerate(overs.head(50).iterrows(), 1):
        ha = "H" if r["is_home"] == 1 else "A"
        gs = str(r.get('opp_goalie_status', '?'))[:6]
        print(f"  {i:>3d}  {r['player_name']:<24s} {r['team']:>3s} {r['opponent']:>4s} "
              f"{r.get('position', '?'):>3s} {ha:>3s} {int(r.get('rest_days', 0)):>4d} "
              f"{r.get('opp_goalie', '?'):<18s} {gs:>6s} {r['prob_points_1plus']:>6.1%}")

    # --- Top Under picks (0 points) ---
    unders = output[output["prob_points_1plus"] <= 0.25].sort_values("prob_points_1plus")
    print(f"\n{'═' * 90}")
    print(f"  UNDER 0.5 POINTS — {len(unders)} picks  (model prob <= 25%)")
    print(f"{'═' * 90}")
    print(f"  {'#':>3s}  {'Player':<24s} {'Tm':>3s} {'Opp':>4s} {'Pos':>3s} "
          f"{'H/A':>3s} {'Prob':>6s} {'MissRate':>8s}")
    print(f"  {'─' * 60}")

    for i, (_, r) in enumerate(unders.head(50).iterrows(), 1):
        ha = "H" if r["is_home"] == 1 else "A"
        miss = 1 - r["prob_points_1plus"]
        print(f"  {i:>3d}  {r['player_name']:<24s} {r['team']:>3s} {r['opponent']:>4s} "
              f"{r.get('position', '?'):>3s} {ha:>3s} {r['prob_points_1plus']:>6.1%} {miss:>8.1%}")

    # --- Multi-target dashboard ---
    print(f"\n{'═' * 120}")
    print(f"  MULTI-TARGET DASHBOARD — TOP 40 PLAYERS BY 1+ POINTS PROB")
    print(f"{'═' * 120}")
    header = (f"  {'Player':<24s} {'Tm':>3s} {'Opp':>4s} {'Pos':>3s} "
              f"{'1+Pts':>7s} {'2+Pts':>7s} {'1+G':>7s} {'1+A':>7s} "
              f"{'3+SOG':>7s} {'4+SOG':>7s} {'5+SOG':>7s}")
    print(header)
    print(f"  {'─' * 114}")

    target_order = ["points_1plus", "points_2plus", "goals_1plus", "assists_1plus",
                    "shots_3plus", "shots_4plus", "shots_5plus"]

    for _, r in output.head(40).iterrows():
        line = f"  {r['player_name']:<24s} {r['team']:>3s} {r['opponent']:>4s} {r.get('position', '?'):>3s} "
        for t in target_order:
            col = f"prob_{t}"
            if col in r and pd.notna(r[col]):
                line += f" {r[col]:>6.1%} "
            else:
                line += f" {'N/A':>6s} "
        print(line)

    # --- Per-game breakdown ---
    for g in games:
        away_team, home_team = g["away"], g["home"]

        # Game header with odds
        aml = g.get('away_ml', '')
        hml = g.get('home_ml', '')
        odds_str = ""
        if aml or hml:
            odds_str = f"  [{away_team} ML:{aml} | {home_team} ML:{hml}]"

        # Goalie info
        ag = goalie_lookup.get(away_team, {})
        hg = goalie_lookup.get(home_team, {})

        print(f"\n{'─' * 90}")
        print(f"  {away_team} @ {home_team}{odds_str}")
        ag_status = ag.get('goalie_status', '?')
        hg_status = hg.get('goalie_status', '?')
        print(f"  Goalies: {ag.get('goalie_name', '?')} ({ag_status}) vs {hg.get('goalie_name', '?')} ({hg_status})")
        print(f"{'─' * 90}")

        game_players = output[
            (output["team"].isin([away_team, home_team]))
        ].sort_values("prob_points_1plus", ascending=False)

        print(f"  {'Player':<24s} {'Tm':>3s} {'Pos':>3s} {'1+Pts':>7s} {'1+G':>7s} "
              f"{'1+A':>7s} {'3+SOG':>7s}")
        print(f"  {'─' * 60}")

        for _, r in game_players.head(25).iterrows():
            pts  = r.get("prob_points_1plus", np.nan)
            gls  = r.get("prob_goals_1plus", np.nan)
            ast  = r.get("prob_assists_1plus", np.nan)
            sog  = r.get("prob_shots_3plus", np.nan)
            line = f"  {r['player_name']:<24s} {r['team']:>3s} {r.get('position', '?'):>3s} "
            line += f" {pts:>6.1%} " if pd.notna(pts) else f" {'':>6s} "
            line += f" {gls:>6.1%} " if pd.notna(gls) else f" {'':>6s} "
            line += f" {ast:>6.1%} " if pd.notna(ast) else f" {'':>6s} "
            line += f" {sog:>6.1%} " if pd.notna(sog) else f" {'':>6s} "
            print(line)

    # ── 11. Save CSV ─────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, f"predictions_{game_date}.csv")
    output.to_csv(csv_path, index=False)

    print(f"\n{'═' * 70}")
    print(f"  ✓ PREDICTIONS SAVED")
    print(f"{'═' * 70}")
    print(f"  File : {csv_path}")
    print(f"  Rows : {len(output):,} players")
    print(f"  Games: {len(games)}")
    print(f"\n  To find +EV bets: edge = model_prob - implied_prob.")
    print(f"  A +EV bet needs edge > ~2-3% to overcome vig.")
    print(f"  Moneylines from DailyFaceoff are included in the CSV.\n")
    # ── 12. TOP 10 DAILY PICKS ───────────────────────────────
    print(f"\n{'═' * 90}")
    print(f"  🔥 TOP 10 DAILY PICKS — {game_date}")
    print(f"{'═' * 90}")

    categories = [
        ("🥅 GOAL SCORERS (1+ Goals)", "prob_goals_1plus", 0.25, 0.40),
        ("📊 POINT GETTERS (1+ Points)", "prob_points_1plus", 0.45, 0.65),
        ("🏒 SHOT VOLUME (3+ SOG)", "prob_shots_3plus", 0.40, 0.60),
    ]

    for cat_name, cat_col, _, hot_thresh in categories:
        if cat_col not in output.columns:
            continue
        top = output.nlargest(10, cat_col)
        if len(top) == 0:
            continue

        print(f"\n  {cat_name}")
        print(f"  {'─' * 82}")
        print(f"  {'#':>2s}  {'Player':<24s} {'Tm':>3s} {'Opp':>4s} {'Pos':>3s} "
              f"{'Prob':>7s}  {'vs Goalie':<20s} {'Status':>8s}")
        print(f"  {'─' * 82}")

        for i, (_, r) in enumerate(top.iterrows(), 1):
            prob = r[cat_col]
            fire = "🔥🔥" if prob >= hot_thresh else "🔥" if prob >= hot_thresh * 0.75 else ""
            gs = str(r.get('opp_goalie_status', '?'))[:8]
            print(f"  {i:>2d}. {r['player_name']:<24s} {r['team']:>3s} "
                  f"{r['opponent']:>4s} {r.get('position', '?'):>3s} "
                  f"{prob:>6.1%}  {r.get('opp_goalie', '?'):<20s} {gs:>8s}  {fire}")

    # ── 13. SAVE DAILY PICKS SPREADSHEET ─────────────────────
    print(f"\n{'═' * 70}")
    print(f"  💾 SAVING DAILY PICKS SPREADSHEET")
    print(f"{'═' * 70}\n")

    all_prob_cols = [c for c in output.columns if c.startswith("prob_")]

    # Position-aware thresholds: defensemen have lower base rates for goals/SOG
    # than forwards — the same probability threshold produces different edge.
    # F = all non-D positions (C, LW, RW), D = defensemen.
    pick_thresholds = {
        "F": {
            'prob_points_1plus': 0.55,
            'prob_points_2plus': 0.30,
            'prob_goals_1plus': 0.28,
            'prob_assists_1plus': 0.45,
            'prob_shots_3plus': 0.52,
            'prob_shots_4plus': 0.38,
            'prob_shots_5plus': 0.27,
        },
        "D": {
            'prob_points_1plus': 0.50,
            'prob_points_2plus': 0.22,
            'prob_goals_1plus': 0.20,
            'prob_assists_1plus': 0.42,
            'prob_shots_3plus': 0.45,
            'prob_shots_4plus': 0.30,
            'prob_shots_5plus': 0.18,
        },
    }

    picks_rows = []
    for _, r in output.iterrows():
        pick_row = {
            'prediction_date': game_date,
            'player_name': r.get('player_name', ''),
            'player_id': r.get('player_id', ''),
            'team': r.get('team', ''),
            'opponent': r.get('opponent', ''),
            'position': r.get('position', ''),
            'is_home': int(r.get('is_home', 0)),
            'opp_goalie': r.get('opp_goalie', ''),
            'opp_goalie_status': r.get('opp_goalie_status', ''),
            'rest_days': int(r.get('rest_days', 0)),
        }

        for col in all_prob_cols:
            val = r.get(col, np.nan)
            pick_row[col] = round(float(val), 4) if pd.notna(val) else ''

        pos_key = "D" if str(r.get('position', 'F')) == "D" else "F"
        for col, thresh in pick_thresholds[pos_key].items():
            prop_name = col.replace('prob_', '')
            if col in r.index and pd.notna(r.get(col)) and float(r[col]) >= thresh:
                pick_row[f"pick_{prop_name}"] = 'YES'
            else:
                pick_row[f"pick_{prop_name}"] = ''

        if r.get('is_home') == 1:
            pick_row['team_ml'] = r.get('home_ml', '')
        else:
            pick_row['team_ml'] = r.get('away_ml', '')

        for res_col in ['actual_goals', 'actual_assists', 'actual_points',
                        'actual_shots', 'hit_goals_1plus', 'hit_points_1plus',
                        'hit_points_2plus', 'hit_assists_1plus', 'hit_shots_3plus',
                        'hit_shots_4plus', 'hit_shots_5plus']:
            pick_row[res_col] = ''

        picks_rows.append(pick_row)

    df_picks = pd.DataFrame(picks_rows)

    daily_path = os.path.join(OUTPUT_DIR, f"daily_picks_{game_date}.csv")
    df_picks.to_csv(daily_path, index=False)
    print(f"  📁 Daily picks:    {daily_path}  ({len(df_picks)} players)")

    history_path = os.path.join(OUTPUT_DIR, "player_picks_history.csv")
    if os.path.exists(history_path):
        df_hist = pd.read_csv(history_path, dtype=str)
        df_hist = df_hist[df_hist['prediction_date'] != game_date]
        df_combined = pd.concat([df_hist, df_picks.astype(str)], ignore_index=True)
    else:
        df_combined = df_picks.astype(str)
    df_combined.to_csv(history_path, index=False)
    print(f"  📁 Master history: {history_path}  ({len(df_combined)} total rows)")

    pick_counts = {}
    for col in pick_thresholds["F"].keys():   # prop names identical across positions
        prop_name = col.replace('prob_', '')
        flag_col = f"pick_{prop_name}"
        if flag_col in df_picks.columns:
            count = len(df_picks[df_picks[flag_col] == 'YES'])
            if count > 0:
                pick_counts[prop_name] = count

    if pick_counts:
        print(f"\n  Today's picks above threshold:")
        for prop, count in pick_counts.items():
            print(f"    {prop:<20s}: {count} players")

    print(f"\n  ✅ Run 'python track_player_results.py' tomorrow to grade results.")
    print(f"{'═' * 70}\n")

if __name__ == "__main__":
    main()