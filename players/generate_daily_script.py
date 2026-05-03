"""
generate_daily_script.py
Reads today's game predictions and player picks, then writes a ready-to-use
~1.5 minute narration script for the Smart Thinking AI model (Alexis Rivers).

Output: NHL_Player/script_briefs/ai_script_{date}.txt
"""

import pandas as pd
import os
from datetime import date, datetime
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player")
GAME_LOG    = BASE / "Boxscores" / "prediction_log.csv"
PICKS_DIR   = BASE / "predictions"
OUTPUT_DIR  = BASE / "script_briefs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEAM_NAMES = {
    "ANA":"Anaheim","ARI":"Arizona","BOS":"Boston","BUF":"Buffalo",
    "CAR":"Carolina","CBJ":"Columbus","CGY":"Calgary","CHI":"Chicago",
    "COL":"Colorado","DAL":"Dallas","DET":"Detroit","EDM":"Edmonton",
    "FLA":"Florida","LAK":"LA Kings","MIN":"Minnesota","MTL":"Montreal",
    "NJD":"New Jersey","NSH":"Nashville","NYI":"NY Islanders","NYR":"NY Rangers",
    "OTT":"Ottawa","PHI":"Philadelphia","PIT":"Pittsburgh","SEA":"Seattle",
    "SJS":"San Jose","STL":"St. Louis","TBL":"Tampa Bay","TOR":"Toronto",
    "UTA":"Utah","VAN":"Vancouver","VGK":"Vegas","WPG":"Winnipeg","WSH":"Washington",
}

def full(abbr):
    return TEAM_NAMES.get(str(abbr).upper(), abbr)


def build_script(game_date_str, games_df, picks_df):
    """Build the narration script text."""

    # ── Parse game picks ───────────────────────────────────────────────────────
    today_games = games_df[games_df["pred_date"] == game_date_str].copy()

    high_conf   = today_games[today_games["conf_label"].str.contains("HIGH", na=False)]
    med_conf    = today_games[today_games["conf_label"].str.contains("MED", na=False)]
    value_games = today_games[pd.to_numeric(today_games["edge"], errors="coerce") > 0.08]
    underdog_value = today_games[
        (pd.to_numeric(today_games["edge"], errors="coerce") > 0.06) &
        (pd.to_numeric(today_games["home_ml"], errors="coerce") > 100) |
        (pd.to_numeric(today_games["away_ml"], errors="coerce") > 100)
    ] if len(today_games) > 0 else pd.DataFrame()

    skip_games = today_games[today_games["conf_label"].str.upper().str.contains("SKIP", na=False)]
    num_games  = len(today_games)

    # ── Parse player picks ─────────────────────────────────────────────────────
    top_points = []
    top_goals  = []
    top_shots  = []
    if picks_df is not None and "prob_points_1plus" in picks_df.columns:
        picks_df["prob_points_1plus"] = pd.to_numeric(picks_df["prob_points_1plus"], errors="coerce")
        picks_df["prob_goals_1plus"]  = pd.to_numeric(picks_df.get("prob_goals_1plus", 0), errors="coerce")
        picks_df["prob_shots_3plus"]  = pd.to_numeric(picks_df.get("prob_shots_3plus", 0), errors="coerce")

        top_points = picks_df.nlargest(5, "prob_points_1plus")["player_name"].tolist()
        top_goals  = picks_df.nlargest(3, "prob_goals_1plus")["player_name"].tolist()
        top_shots  = picks_df.nlargest(3, "prob_shots_3plus")["player_name"].tolist()

    # ── Compose script ─────────────────────────────────────────────────────────
    lines = []
    date_obj = datetime.strptime(game_date_str, "%Y-%m-%d")
    date_display = date_obj.strftime("%B %d, %Y").replace(" 0", " ").replace("  ", " ")

    # Header (not spoken — metadata for the AI tool)
    lines.append(f"NHL DAILY SCRIPT — {date_display}")
    lines.append("For: Smart Thinking AI Model (Alexis Rivers)")
    lines.append("Runtime: ~1.5 minutes")
    lines.append("=" * 80)
    lines.append("")

    # ── OPENING ───────────────────────────────────────────────────────────────
    lines.append("Hi, I am Alexis Rivers of Smart Thinking, and of course this is for")
    lines.append("entertainment purposes only.")
    lines.append("")

    # ── GAME COUNT ────────────────────────────────────────────────────────────
    if num_games == 1:
        game_word = "game"
    else:
        game_word = "games"
    lines.append(f"{'Tonight' if num_games > 0 else 'Today'} we have {num_games} {game_word} on the schedule, "
                 f"and, um, the model has done all the heavy lifting.")
    lines.append("Here's what it's telling us.")
    lines.append("")

    # ── HIGH CONFIDENCE GAMES ─────────────────────────────────────────────────
    if len(high_conf) > 0:
        hc_list = []
        for _, g in high_conf.iterrows():
            pick  = str(g["pick"])
            away  = str(g["away_team"])
            home  = str(g["home_team"])
            is_home_pick = (pick == home)
            location = "at home" if is_home_pick else "on the road"
            hc_list.append(f"{full(pick)} {location} against {full(away if is_home_pick else home)}")

        if len(hc_list) == 1:
            lines.append(f"The game the model feels most confident about tonight is {hc_list[0]}.")
            lines.append("That's a strong lean — the data lines up clearly in one direction.")
        else:
            joined = " and ".join(hc_list)
            lines.append(f"The games the model feels strongest about are {joined}.")
            lines.append("Both of those are high-confidence leans — everything points the same way.")
        lines.append("")

    # ── VALUE / UNDERDOG ANGLE ────────────────────────────────────────────────
    best_value = today_games.copy()
    best_value["edge_num"] = pd.to_numeric(best_value["edge"], errors="coerce")
    best_value = best_value[best_value["edge_num"] > 0.08].sort_values("edge_num", ascending=False)

    if len(best_value) > 0:
        g = best_value.iloc[0]
        pick  = str(g["pick"])
        away  = str(g["away_team"])
        home  = str(g["home_team"])
        is_home_pick = (pick == home)
        opp   = full(away if is_home_pick else home)
        pick_ml_col = "home_ml" if is_home_pick else "away_ml"
        ml_val = g.get(pick_ml_col, None)
        is_underdog = ml_val and pd.to_numeric(ml_val, errors="coerce") > 100

        if is_underdog:
            lines.append(f"The spot that really jumps out is {full(pick)} hosting {opp}.")
            lines.append(f"The books have {full(pick)} as underdogs tonight, which is surprising,")
            lines.append(f"because our model sees them winning this game more often than not.")
            lines.append("That gap between what the market thinks and what the model thinks... ")
            lines.append("that's the biggest value edge on the entire slate tonight.")
        else:
            lines.append(f"The best value spot tonight is {full(pick)} against {opp}.")
            lines.append("The model sees a meaningful edge here... the market just hasn't fully priced it in.")
        lines.append("")

    # ── SECONDARY VALUE GAMES ─────────────────────────────────────────────────
    secondary = best_value.iloc[1:3] if len(best_value) > 1 else pd.DataFrame()
    if len(secondary) > 0:
        sec_names = []
        for _, g in secondary.iterrows():
            pick  = str(g["pick"])
            away  = str(g["away_team"])
            home  = str(g["home_team"])
            is_home_pick = (pick == home)
            opp   = full(away if is_home_pick else home)
            location = "at home" if is_home_pick else "on the road"
            sec_names.append(f"{full(pick)} {location}")
        if len(sec_names) == 1:
            lines.append(f"{sec_names[0]} is another game where the model sees value against the market.")
        else:
            lines.append(f"{' and '.join(sec_names)} are also spots where the model sees value.")
        lines.append("")

    # ── SKIP GAMES ────────────────────────────────────────────────────────────
    if len(skip_games) > 0:
        s = skip_games.iloc[0]
        lines.append(f"The {full(s['away_team'])} at {full(s['home_team'])} game is one the model says to sit out —")
        lines.append("too close to call, not enough edge either way.")
        lines.append("")

    # ── PLAYER PROPS ──────────────────────────────────────────────────────────
    if top_points:
        p1 = top_points[0]
        rest = top_points[1:4]

        # First player gets the spotlight
        lines.append(f"On the player side, um, {p1} is in a class of his own tonight —")
        lines.append("the model has him as close to a lock as you'll find on the scoresheet.")

        if rest:
            if len(rest) == 1:
                lines.append(f"{rest[0]} is right behind him as a strong point play as well.")
            elif len(rest) == 2:
                lines.append(f"{rest[0]} and {rest[1]} are both flagged as strong plays tonight too.")
            else:
                lines.append(f"{', '.join(rest[:-1])}, and {rest[-1]} are all flagged as strong plays.")
        lines.append("")

    # ── CLOSING ───────────────────────────────────────────────────────────────
    lines.append(f"That's your {date_display} breakdown.")
    if num_games > 1:
        lines.append(f"{num_games} games tonight... it's going to be a great night of hockey.")
    lines.append("Enjoy every shift, and as always, this is for entertainment purposes only!")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    game_date_str = date.today().isoformat()
    print(f"\n  Generating AI narration script for {game_date_str} ...")

    # Load game predictions
    if not GAME_LOG.exists():
        print(f"  ⚠ Game prediction log not found: {GAME_LOG}")
        return 1
    games_df = pd.read_csv(GAME_LOG, dtype=str)

    # Load player picks
    picks_file = PICKS_DIR / f"daily_picks_{game_date_str}.csv"
    picks_df = None
    if picks_file.exists():
        picks_df = pd.read_csv(picks_file, dtype=str)
    else:
        print(f"  ⚠ Player picks file not found: {picks_file}")
        print(f"    Script will be generated without player prop section.")

    # Build and save script
    script_text = build_script(game_date_str, games_df, picks_df)
    output_path = OUTPUT_DIR / f"ai_script_{game_date_str}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_text)

    print(f"  ✅ AI script saved to:")
    print(f"     {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
