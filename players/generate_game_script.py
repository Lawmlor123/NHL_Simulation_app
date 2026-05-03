"""
generate_game_script.py
Reads today's game predictions and top player picks, then writes:
  1. A ~3500-character narration script for the Smart Thinking AI (Alexis Rivers)
  2. A ready-to-paste YouTube upload description with title and hashtags

Outputs (saved to NHL_Player/script_briefs/):
  game_script_{date}.txt    ← paste into AI voice tool
  youtube_desc_{date}.txt   ← paste into YouTube upload page
"""

import pandas as pd
import os
from datetime import date, datetime
from pathlib import Path

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player")
GAME_LOG    = BASE / "Boxscores" / "prediction_log.csv"
PICKS_DIR   = BASE / "predictions"
OUTPUT_DIR  = BASE / "script_briefs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEAM_NAMES = {
    "ANA":"Anaheim Ducks","ARI":"Arizona Coyotes","BOS":"Boston Bruins",
    "BUF":"Buffalo Sabres","CAR":"Carolina Hurricanes","CBJ":"Columbus Blue Jackets",
    "CGY":"Calgary Flames","CHI":"Chicago Blackhawks","COL":"Colorado Avalanche",
    "DAL":"Dallas Stars","DET":"Detroit Red Wings","EDM":"Edmonton Oilers",
    "FLA":"Florida Panthers","LAK":"LA Kings","MIN":"Minnesota Wild",
    "MTL":"Montreal Canadiens","NJD":"New Jersey Devils","NSH":"Nashville Predators",
    "NYI":"NY Islanders","NYR":"NY Rangers","OTT":"Ottawa Senators",
    "PHI":"Philadelphia Flyers","PIT":"Pittsburgh Penguins","SEA":"Seattle Kraken",
    "SJS":"San Jose Sharks","STL":"St. Louis Blues","TBL":"Tampa Bay Lightning",
    "TOR":"Toronto Maple Leafs","UTA":"Utah Hockey Club","VAN":"Vancouver Canucks",
    "VGK":"Vegas Golden Knights","WPG":"Winnipeg Jets","WSH":"Washington Capitals",
}

TEAM_SHORT = {
    "ANA":"Anaheim","ARI":"Arizona","BOS":"Boston","BUF":"Buffalo",
    "CAR":"Carolina","CBJ":"Columbus","CGY":"Calgary","CHI":"Chicago",
    "COL":"Colorado","DAL":"Dallas","DET":"Detroit","EDM":"Edmonton",
    "FLA":"Florida","LAK":"LA Kings","MIN":"Minnesota","MTL":"Montreal",
    "NJD":"New Jersey","NSH":"Nashville","NYI":"NY Islanders","NYR":"NY Rangers",
    "OTT":"Ottawa","PHI":"Philadelphia","PIT":"Pittsburgh","SEA":"Seattle",
    "SJS":"San Jose","STL":"St. Louis","TBL":"Tampa Bay","TOR":"Toronto",
    "UTA":"Utah","VAN":"Vancouver","VGK":"Vegas","WPG":"Winnipeg","WSH":"Washington",
}

def full(abbr):  return TEAM_NAMES.get(str(abbr).upper(), abbr)
def short(abbr): return TEAM_SHORT.get(str(abbr).upper(), abbr)


def stars(label):
    """Return a clean star tier string."""
    if "HIGH" in str(label).upper():  return "HIGH confidence ★★★"
    if "MED"  in str(label).upper():  return "MEDIUM confidence ★★"
    if "LOW"  in str(label).upper():  return "LOW confidence ★"
    return "SKIP"


def build_game_script(game_date_str, today_games, picks_df):
    """Build the ~3500-character Alexis Rivers narration script."""

    date_obj     = datetime.strptime(game_date_str, "%Y-%m-%d")
    date_display = date_obj.strftime("%B %d, %Y").replace(" 0", " ").replace("  ", " ")
    num_games    = len(today_games)

    # ── Sort: HIGH first, then by confidence desc ──────────────────────────────
    today_games = today_games.copy()
    today_games["conf_num"] = pd.to_numeric(today_games["confidence"], errors="coerce").fillna(0)
    today_games["sort_key"] = today_games["conf_label"].str.upper().apply(
        lambda x: 0 if "HIGH" in x else (1 if "MED" in x else (2 if "LOW" in x else 3))
    )
    today_games = today_games.sort_values(["sort_key", "conf_num"], ascending=[True, False])

    # ── Top player props ───────────────────────────────────────────────────────
    top_points, top_goals, top_shots = [], [], []
    if picks_df is not None:
        for col in ["prob_points_1plus", "prob_goals_1plus", "prob_shots_3plus"]:
            picks_df[col] = pd.to_numeric(picks_df.get(col, 0), errors="coerce").fillna(0)
        picks_df["prob_points_1plus"] = pd.to_numeric(picks_df["prob_points_1plus"], errors="coerce")
        top_points = picks_df.nlargest(5, "prob_points_1plus")[
            ["player_name", "team", "prob_points_1plus"]
        ].values.tolist()
        top_goals  = picks_df.nlargest(3, "prob_goals_1plus")[
            ["player_name", "team", "prob_goals_1plus"]
        ].values.tolist()
        top_shots  = picks_df.nlargest(3, "prob_shots_3plus")[
            ["player_name", "team", "prob_shots_3plus"]
        ].values.tolist()

    lines = []
    sep   = "=" * 80

    # ── Header (not spoken) ────────────────────────────────────────────────────
    lines += [
        f"NHL GAME SCRIPT — {date_display}",
        "For: Smart Thinking AI Model (Alexis Rivers)",
        "Runtime: ~1.5 minutes",
        sep, "",
    ]

    # ── OPENING ───────────────────────────────────────────────────────────────
    lines += [
        "Hi, I am Alexis Rivers of Smart Thinking, and of course this is for",
        "entertainment purposes only.",
        "",
    ]

    # ── SLATE INTRO ───────────────────────────────────────────────────────────
    game_word = "game" if num_games == 1 else "games"
    high_count = sum(1 for _, g in today_games.iterrows()
                     if "HIGH" in str(g["conf_label"]).upper())
    value_count = sum(1 for _, g in today_games.iterrows()
                      if pd.to_numeric(g.get("edge", None), errors="coerce") > 0)

    lines.append(
        f"Tonight we have {num_games} {game_word} on the slate, and the model has "
        f"been through all of them. "
        + (f"There {'is' if high_count == 1 else 'are'} {high_count} high-confidence "
           f"{'pick' if high_count == 1 else 'picks'} tonight"
           + (f" with positive edge against the market." if value_count > 0 else ".")
           if high_count > 0 else "Let's see what the data is telling us.")
    )
    lines.append(
        "The model does not guess — it runs the numbers and gives us a probability. "
        "Here is exactly what it is pointing to tonight."
    )
    lines += [""]

    # ── GAME-BY-GAME ──────────────────────────────────────────────────────────
    for _, g in today_games.iterrows():
        away      = str(g["away_team"])
        home      = str(g["home_team"])
        pick      = str(g["pick"])
        conf_pct  = pd.to_numeric(g["confidence"], errors="coerce")
        conf_str  = f"{conf_pct * 100:.1f}%" if pd.notna(conf_pct) else "—"
        label     = stars(g["conf_label"])
        edge_val  = pd.to_numeric(g.get("edge", None), errors="coerce")
        edge_str  = (f"+{edge_val*100:.1f}%" if edge_val > 0 else f"{edge_val*100:.1f}%") if pd.notna(edge_val) else "—"
        has_value = pd.notna(edge_val) and edge_val > 0
        is_skip   = "SKIP" in str(g["conf_label"]).upper()
        market_pct = pd.to_numeric(g.get("market_implied", None), errors="coerce")
        home_ml   = str(g.get("home_ml", ""))
        away_ml   = str(g.get("away_ml", ""))

        is_home_pick = (pick == home)
        opponent     = away if is_home_pick else home
        location     = "at home" if is_home_pick else "on the road"
        pick_ml      = home_ml if is_home_pick else away_ml

        hg      = str(g.get("home_goalie", "TBD"))
        ag      = str(g.get("away_goalie", "TBD"))
        hg_st   = str(g.get("home_goalie_status", ""))
        ag_st   = str(g.get("away_goalie_status", ""))
        pick_goalie = hg if is_home_pick else ag
        opp_goalie  = ag if is_home_pick else hg
        pick_g_st   = hg_st if is_home_pick else ag_st
        opp_g_st    = ag_st if is_home_pick else hg_st

        if is_skip:
            lines.append(
                f"{short(away)} at {short(home)} — the model says to sit this one out. "
                f"Too close to call, not enough edge either way. No play here tonight."
            )
            lines += [""]
            continue

        # Opening sentence
        lines.append(
            f"{short(away)} at {short(home)}. The model is on the "
            f"{short(pick)} {location} — {conf_str} confidence. {label}."
        )

        # Edge vs market
        if has_value and pd.notna(market_pct):
            lines.append(
                f"The market has {short(pick)} implied at {market_pct*100:.1f}% — "
                f"the model is {conf_pct*100:.1f}%. That gap of {edge_str} is real value. "
                f"This is the kind of edge the model is built to find."
            )
        elif has_value:
            lines.append(
                f"There is a positive edge of {edge_str} against the market line. "
                f"The model sees this {short(pick)} win as more likely than the books do — real value."
            )
        else:
            lines.append(
                f"The model likes {short(pick)}, but the market already has them "
                f"implied at {market_pct*100:.1f}% — our model says {conf_pct*100:.1f}%. "
                f"Negative edge of {edge_str}. Lean only — no moneyline value tonight."
            )

        # Goalie context
        if "Manual" in [hg_st, ag_st]:
            lines.append(
                f"Goalie confirmed: {pick_goalie} starts for {short(pick)}, "
                f"{opp_goalie} gets the start for {short(opponent)} — both manually verified tonight."
            )
        else:
            lines.append(
                f"Expected goalies: {pick_goalie} for {short(pick)}, "
                f"{opp_goalie} for {short(opponent)}."
            )

        # Value summary line
        if has_value:
            ml_note = f" at {pick_ml}" if pick_ml and pick_ml not in ("nan", "", "None") else ""
            lines.append(
                f"This is the model's top-rated game on the slate tonight. "
                f"{short(pick)}{ml_note} is the play. "
                f"When the model is at {conf_str} confidence and the edge is positive, "
                f"this is exactly the kind of situation you want to be involved in."
            )
        else:
            lines.append(
                f"The lean is {short(pick)}, but the market is already ahead of the model here. "
                f"Use this game for player props — there are good individual spots on both sides. "
                f"Avoid the moneyline."
            )

        lines += [""]

    # ── PLAYER PROPS ──────────────────────────────────────────────────────────
    if top_points:
        p1_name, p1_team, p1_prob = top_points[0]
        p1_pct = f"{float(p1_prob)*100:.0f}%"
        lines.append(
            f"On the player side — {p1_name} is the number one name on the entire "
            f"board tonight. The model has him at {p1_pct} probability to record "
            f"at least one point. That is an elite individual prop number — "
            f"anything above 75% is a strong spot and {p1_pct} is exceptional."
        )
        if len(top_points) > 1:
            runners = [
                f"{r[0]} at {float(r[2])*100:.0f}%"
                for r in top_points[1:4]
            ]
            lines.append(
                "Right behind on the points board: " + ", ".join(runners) + ". "
                "The model is loaded on the points market tonight — shop for the "
                "best price across your books before locking anything in."
            )
        lines += [""]

    if top_goals:
        goal_names = [
            f"{r[0]} ({float(r[2])*100:.0f}%)"
            for r in top_goals
        ]
        lines.append(
            "Top anytime goal scorers per the model tonight: "
            + ", ".join(goal_names) + ". "
            "Check your books for anytime goal prices on these names — "
            "any plus-odds price on a 35%+ probability is positive expected value."
        )
        lines += [""]

    if top_shots:
        shot_names = [
            f"{r[0]} ({float(r[2])*100:.0f}%)"
            for r in top_shots
        ]
        lines.append(
            "On the shots market — "
            + ", ".join(shot_names)
            + " are the model's top names for three or more shots tonight. "
            "Shots props are often the most underpriced market on the board. "
            "If you can find a 2.5 shots line on any of these players, the model "
            "is pointing you firmly to the over."
        )
        lines += [""]

    # ── PARLAY CALLOUT ────────────────────────────────────────────────────────
    high_games_list = [g for _, g in today_games.iterrows()
                       if "HIGH" in str(g["conf_label"]).upper()]
    if high_games_list and top_points and len(top_points) >= 2:
        hg   = high_games_list[0]
        pick = short(str(hg["pick"]))
        p1   = top_points[0][0]
        p2   = top_points[1][0]
        lines.append(
            f"If you are building a parlay tonight — the {pick} ML "
            f"paired with {p1} on points and {p2} on points is a three-leg "
            f"stack where every single leg has a strong model probability behind it. "
            f"Size it appropriately and do not chase the juice."
        )
        lines += [""]

    # ── CLOSING ───────────────────────────────────────────────────────────────
    lines += [
        f"That is your full {date_display} breakdown. {num_games} {game_word} "
        "tonight — the model has spoken. If you found this breakdown useful, "
        "drop a like, leave your picks in the comments, and subscribe for daily "
        "NHL model output throughout the rest of the season. "
        "Good luck tonight, enjoy every shift, "
        "and as always — this is for entertainment purposes only!",
        "",
        sep,
    ]

    return "\n".join(lines)


def build_youtube_description(game_date_str, today_games, picks_df):
    """Build the YouTube upload description: title + body + hashtags."""

    date_obj     = datetime.strptime(game_date_str, "%Y-%m-%d")
    date_display = date_obj.strftime("%B %d, %Y").replace(" 0", " ").replace("  ", " ")
    month_year   = date_obj.strftime("%B %Y")

    today_games  = today_games.copy()
    today_games["conf_num"] = pd.to_numeric(today_games["confidence"], errors="coerce").fillna(0)

    # ── YouTube Title ──────────────────────────────────────────────────────────
    # Format: "NHL Picks March 27 | Rangers vs Blackhawks + Sabres vs Wings | Smart Thinking"
    game_parts = []
    for _, g in today_games.iterrows():
        away = short(str(g["away_team"]))
        home = short(str(g["home_team"]))
        game_parts.append(f"{away} vs {home}")
    games_str = " + ".join(game_parts) if game_parts else "Tonight's Games"

    short_date = date_obj.strftime("%B %-d").replace(" 0", " ") if os.name != "nt" else \
                 date_obj.strftime("%B %d").replace(" 0", " ").replace("  ", " ")

    yt_title = f"NHL Picks {short_date} | {games_str} | Smart Thinking"

    # ── High confidence callouts for description ───────────────────────────────
    high_games  = today_games[today_games["conf_label"].str.upper().str.contains("HIGH", na=False)]
    value_games = today_games[
        pd.to_numeric(today_games["edge"], errors="coerce") > 0
    ]

    pick_bullets = []
    for _, g in today_games.iterrows():
        if "SKIP" in str(g["conf_label"]).upper():
            continue
        pick  = str(g["pick"])
        away  = str(g["away_team"])
        home  = str(g["home_team"])
        conf  = pd.to_numeric(g["confidence"], errors="coerce")
        edge  = pd.to_numeric(g.get("edge", None), errors="coerce")
        label = stars(g["conf_label"])
        edge_s = (f"+{edge*100:.1f}%" if edge > 0 else f"{edge*100:.1f}%") if pd.notna(edge) else ""
        value_note = " ✅ VALUE" if pd.notna(edge) and edge > 0 else " ⚠️ NO EDGE"
        pick_bullets.append(
            f"• {short(pick)} ({short(away)}@{short(home)}) — {conf*100:.1f}% | {label} | edge {edge_s}{value_note}"
        )

    # ── Top prop callout ───────────────────────────────────────────────────────
    prop_line = ""
    if picks_df is not None and "prob_points_1plus" in picks_df.columns:
        picks_df["prob_points_1plus"] = pd.to_numeric(picks_df["prob_points_1plus"], errors="coerce")
        top = picks_df.nlargest(3, "prob_points_1plus")[["player_name", "prob_points_1plus"]].values
        prop_parts = [f"{r[0]} ({float(r[1])*100:.0f}%)" for r in top]
        prop_line = "🏒 Top point props: " + ", ".join(prop_parts)

    # ── Hashtags ───────────────────────────────────────────────────────────────
    team_tags = []
    for _, g in today_games.iterrows():
        for abbr in [g["away_team"], g["home_team"]]:
            name = short(str(abbr)).replace(" ", "").replace(".", "")
            team_tags.append(f"#{name}")

    hashtags = (
        "#NHLPicks #NHLPredictions #NHLBetting #HockeyPicks "
        "#SportsBetting #NHLProps #SmartThinking #AlexisRivers "
        + " ".join(team_tags[:6])
        + f" #NHL{date_obj.strftime('%Y')}"
    )

    # ── Assemble ───────────────────────────────────────────────────────────────
    picks_block = "\n".join(pick_bullets) if pick_bullets else "— no games today —"

    body = f"""📅 {date_display} | NHL Daily Picks & Props | Smart Thinking

Tonight's model output — game picks, confidence ratings, edge vs the market, and the top player prop spots on the board.

🎯 TONIGHT'S GAME PICKS
{picks_block}

{prop_line}

🤖 About the Model
Smart Thinking uses a machine learning model trained on NHL historical data to generate probability estimates for game outcomes and individual player stats. Picks are based on model output, not gut feeling.

⚠️ For entertainment purposes only. Please gamble responsibly.

👍 Like and subscribe for daily NHL picks throughout the season!
💬 Drop your picks in the comments — let's see who calls it.

{hashtags}"""

    sep = "=" * 80
    output = "\n".join([
        f"NHL YOUTUBE DESCRIPTION — {date_display}",
        sep,
        f"TITLE (copy this line exactly):",
        yt_title,
        sep,
        "DESCRIPTION (paste into YouTube description box):",
        "",
        body,
        "",
        sep,
    ])

    return output


def main():
    game_date_str = date.today().isoformat()
    print(f"\n  Generating game script + YouTube description for {game_date_str} ...")

    # ── Load game predictions ──────────────────────────────────────────────────
    if not GAME_LOG.exists():
        print(f"  ⚠ Game prediction log not found: {GAME_LOG}")
        return 1
    games_df    = pd.read_csv(GAME_LOG, dtype=str)
    today_games = games_df[games_df["pred_date"] == game_date_str].copy()

    if len(today_games) == 0:
        print(f"  ⚠ No games found for {game_date_str} in prediction log.")
        return 1

    # ── Load player picks ──────────────────────────────────────────────────────
    picks_file = PICKS_DIR / f"daily_picks_{game_date_str}.csv"
    picks_df   = None
    if picks_file.exists():
        picks_df = pd.read_csv(picks_file, dtype=str)
    else:
        print(f"  ⚠ Player picks not found: {picks_file} — props section will be skipped.")

    # ── Build and save game script ─────────────────────────────────────────────
    script_text = build_game_script(game_date_str, today_games, picks_df)
    script_path = OUTPUT_DIR / f"game_script_{game_date_str}.txt"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)
    char_count = len(script_text)
    print(f"  ✅ Game script saved ({char_count:,} chars): {script_path}")

    # ── Build and save YouTube description ────────────────────────────────────
    yt_text  = build_youtube_description(game_date_str, today_games, picks_df)
    yt_path  = OUTPUT_DIR / f"youtube_desc_{game_date_str}.txt"
    with open(yt_path, "w", encoding="utf-8") as f:
        f.write(yt_text)
    print(f"  ✅ YouTube description saved: {yt_path}")

    return 0


if __name__ == "__main__":
    exit(main())
