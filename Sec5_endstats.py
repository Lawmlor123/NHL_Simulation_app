# ============================================================
# SECTION 5: DATA + PRINTERS
# ============================================================

# --- Injury profiles ---
injury_profiles = {
    "David Pastrnak":{"role":"forward","impact":0.2},
    "Charlie McAvoy":{"role":"defense","impact":0.2},
    "Connor McDavid":{"role":"forward","impact":0.35},
    "Leon Draisaitl":{"role":"forward","impact":0.3},
    "Auston Matthews":{"role":"forward","impact":0.25},
    "Morgan Rielly":{"role":"defense","impact":0.15},
    "Matthew Tkachuk":{"role":"forward","impact":0.25},
    "Aleksander Barkov":{"role":"forward","impact":0.2},
    "Sidney Crosby":{"role":"forward","impact":0.25},
    "Alex Ovechkin":{"role":"forward","impact":0.25},
    "Nathan MacKinnon":{"role":"forward","impact":0.3},
    "Cale Makar":{"role":"defense","impact":0.25},
    "Kirill Kaprizov":{"role":"forward","impact":0.25},
    "Jack Hughes":{"role":"forward","impact":0.25},
    "Artemi Panarin":{"role":"forward","impact":0.25},
    "Adam Fox":{"role":"defense","impact":0.2},
    "Nikita Kucherov":{"role":"forward","impact":0.3},
    "Victor Hedman":{"role":"defense","impact":0.2}
}

# --- Team rosters (with factors optional) ---
team_rosters = {
    "Boston Bruins":[{"name":"David Pastrnak","factor":1.25},{"name":"Charlie McAvoy","factor":0.95}],
    "Toronto Maple Leafs":[{"name":"Auston Matthews","factor":1.25},{"name":"Morgan Rielly","factor":0.95}],
    "Tampa Bay Lightning":[{"name":"Nikita Kucherov","factor":1.25},{"name":"Victor Hedman","factor":0.95}],
    "Colorado Avalanche":[{"name":"Nathan MacKinnon","factor":1.25},{"name":"Cale Makar","factor":0.95}],
    "Edmonton Oilers":[{"name":"Connor McDavid","factor":1.3},{"name":"Leon Draisaitl","factor":1.2}],
    "New York Rangers":[{"name":"Artemi Panarin","factor":1.2},{"name":"Adam Fox","factor":0.95}],
    "New Jersey Devils":["Jack Hughes","Dougie Hamilton"],
    "Ottawa Senators":["Brady Tkachuk","Thomas Chabot"],
    "Pittsburgh Penguins":["Sidney Crosby","Kris Letang"],
    "Washington Capitals":["Alex Ovechkin","John Carlson"],
    "Vancouver Canucks":["Elias Pettersson","Quinn Hughes"],
    "Florida Panthers":["Matthew Tkachuk","Aleksander Barkov"],
    "Minnesota Wild":["Kirill Kaprizov"],
    "Nashville Predators":["Filip Forsberg","Roman Josi"],
    "Vegas Golden Knights":["Jack Eichel","Mark Stone"],
    "Los Angeles Kings":["Anze Kopitar","Drew Doughty"],
    "Columbus Blue Jackets":["Johnny Gaudreau","Zach Werenski"],
    "Arizona Coyotes":["Clayton Keller"],
    "Buffalo Sabres":["Rasmus Dahlin"],
    "Winnipeg Jets":["Kyle Connor","Josh Morrissey"],
    "San Jose Sharks":["Logan Couture","Tomas Hertl"],
    "St. Louis Blues":["Robert Thomas"],
    "Calgary Flames":["Jonathan Huberdeau"],
    "Chicago Blackhawks":["Connor Bedard"],
    "Detroit Red Wings":["Dylan Larkin","Moritz Seider"],
    "Dallas Stars":["Jason Robertson","Miro Heiskanen"],
    "Seattle Kraken":["Matty Beniers"],
    "Philadelphia Flyers":["Travis Konecny"],
    "Montreal Canadiens":["Cole Caufield","Nick Suzuki"],
    "Anaheim Ducks":["Trevor Zegras","Mason McTavish"],
    "Carolina Hurricanes":["Sebastian Aho","Andrei Svechnikov"],
    "New York Islanders":["Mathew Barzal"],
    "Utah Mammoth":[]
}

# --- Team stats ---
team_stats = {
    "Boston Bruins":{"GF":3.24,"GA":2.64,"PP":23.3,"PK":82.6},
    "Toronto Maple Leafs":{"GF":3.63,"GA":3.01,"PP":26.8,"PK":76.9},
    "Tampa Bay Lightning":{"GF":3.5,"GA":3.3,"PP":28.6,"PK":79.0},
    "Colorado Avalanche":{"GF":3.35,"GA":2.99,"PP":25.2,"PK":79.6},
    "Edmonton Oilers":{"GF":3.96,"GA":3.2,"PP":29.5,"PK":79.5},
    "New York Rangers":{"GF":3.3,"GA":2.7,"PP":25.2,"PK":82.0},
    "New Jersey Devils":{"GF":3.4,"GA":3.1,"PP":21.9,"PK":81.5},
    "Ottawa Senators":{"GF":3.2,"GA":3.4,"PP":20.4,"PK":79.0},
    "Pittsburgh Penguins":{"GF":3.2,"GA":3.1,"PP":21.7,"PK":80.2},
    "Washington Capitals":{"GF":2.9,"GA":3.2,"PP":21.3,"PK":80.1},
    "Vancouver Canucks":{"GF":3.4,"GA":2.9,"PP":23.7,"PK":79.5},
    "Florida Panthers":{"GF":3.5,"GA":3.0,"PP":25.5,"PK":78.2},
    "Minnesota Wild":{"GF":3.2,"GA":2.9,"PP":22.0,"PK":81.1},
    "Nashville Predators":{"GF":3.0,"GA":3.0,"PP":19.3,"PK":80.0},
    "Vegas Golden Knights":{"GF":3.4,"GA":2.8,"PP":21.4,"PK":79.2},
    "Los Angeles Kings":{"GF":3.2,"GA":3.1,"PP":20.0,"PK":78.1},
    "Columbus Blue Jackets":{"GF":3.0,"GA":3.6,"PP":19.8,"PK":76.3},
    "Arizona Coyotes":{"GF":2.8,"GA":3.2,"PP":18.9,"PK":77.1},
    "Buffalo Sabres":{"GF":3.1,"GA":3.3,"PP":23.4,"PK":78.9},
    "Winnipeg Jets":{"GF":3.1,"GA":2.6,"PP":21.2,"PK":80.9},
    "San Jose Sharks":{"GF":2.7,"GA":3.6,"PP":18.7,"PK":75.0},
    "St. Louis Blues":{"GF":3.0,"GA":3.3,"PP":19.2,"PK":76.5},
    "Calgary Flames":{"GF":3.0,"GA":3.2,"PP":20.9,"PK":78.8},
    "Chicago Blackhawks":{"GF":2.7,"GA":3.7,"PP":17.2,"PK":76.0},
    "Detroit Red Wings":{"GF":3.0,"GA":3.3,"PP":21.1,"PK":78.6},
    "Dallas Stars":{"GF":3.4,"GA":2.7,"PP":27.0,"PK":82.2},
    "Seattle Kraken":{"GF":3.1,"GA":2.9,"PP":20.5,"PK":79.3},
    "Philadelphia Flyers":{"GF":2.8,"GA":3.0,"PP":17.1,"PK":78.2},
    "Montreal Canadiens":{"GF":2.9,"GA":3.6,"PP":16.9,"PK":72.9},
    "Anaheim Ducks":{"GF":2.7,"GA":3.5,"PP":17.3,"PK":75.6},
    "Carolina Hurricanes":{"GF":3.3,"GA":2.5,"PP":23.0,"PK":84.4},
    "New York Islanders":{"GF":2.9,"GA":2.8,"PP":18.5,"PK":81.2},
    "Utah Mammoth":{"GF":3.0,"GA":3.0,"PP":20.0,"PK":80.0}
}

# --- Goalies ---
goalies = {
    "Boston Bruins":{"starter":{"SV":0.918,"name":"Jeremy Swayman"},"backup":{"SV":0.902,"name":"Brandon Bussi"}},
    "Toronto Maple Leafs":{"starter":{"SV":0.912,"name":"Ilya Samsonov"},"backup":{"SV":0.898,"name":"Joseph Woll"}},
    "Tampa Bay Lightning":{"starter":{"SV":0.915,"name":"Andrei Vasilevskiy"},"backup":{"SV":0.899,"name":"Jonas Johansson"}},
    "Colorado Avalanche":{"starter":{"SV":0.910,"name":"Alexandar Georgiev"},"backup":{"SV":0.899,"name":"Justus Annunen"}},
    "Edmonton Oilers":{"starter":{"SV":0.913,"name":"Stuart Skinner"},"backup":{"SV":0.892,"name":"Calvin Pickard"}},
    "New York Rangers":{"starter":{"SV":0.919,"name":"Igor Shesterkin"},"backup":{"SV":0.900,"name":"Jonathan Quick"}},
    "New Jersey Devils":{"starter":{"SV":0.907,"name":"Vitek Vanecek"},"backup":{"SV":0.897,"name":"Akira Schmid"}},
    "Ottawa Senators":{"starter":{"SV":0.905,"name":"Joonas Korpisalo"},"backup":{"SV":0.893,"name":"Anton Forsberg"}},
    "Pittsburgh Penguins":{"starter":{"SV":0.913,"name":"Tristan Jarry"},"backup":{"SV":0.899,"name":"Alex Nedeljkovic"}},
    "Washington Capitals":{"starter":{"SV":0.909,"name":"Darcy Kuemper"},"backup":{"SV":0.895,"name":"Charlie Lindgren"}},
    "Vancouver Canucks":{"starter":{"SV":0.920,"name":"Thatcher Demko"},"backup":{"SV":0.901,"name":"Casey DeSmith"}},
    "Florida Panthers":{"starter":{"SV":0.913,"name":"Sergei Bobrovsky"},"backup":{"SV":0.898,"name":"Anthony Stolarz"}},
    "Minnesota Wild":{"starter":{"SV":0.912,"name":"Filip Gustavsson"},"backup":{"SV":0.901,"name":"Marc-Andre Fleury"}},
    "Nashville Predators":{"starter":{"SV":0.916,"name":"Juuse Saros"},"backup":{"SV":0.900,"name":"Kevin Lankinen"}},
    "Vegas Golden Knights":{"starter":{"SV":0.909,"name":"Logan Thompson"},"backup":{"SV":0.902,"name":"Adin Hill"}},
    "Los Angeles Kings":{"starter":{"SV":0.907,"name":"Cam Talbot"},"backup":{"SV":0.898,"name":"Pheonix Copley"}},
    "Columbus Blue Jackets":{"starter":{"SV":0.902,"name":"Elvis Merzlikins"},"backup":{"SV":0.892,"name":"Spencer Martin"}},
    "Arizona Coyotes":{"starter":{"SV":0.908,"name":"Karel Vejmelka"},"backup":{"SV":0.895,"name":"Connor Ingram"}},
    "Buffalo Sabres":{"starter":{"SV":0.906,"name":"Ukko-Pekka Luukkonen"},"backup":{"SV":0.894,"name":"Devon Levi"}},
    "Winnipeg Jets":{"starter":{"SV":0.920,"name":"Connor Hellebuyck"},"backup":{"SV":0.899,"name":"Laurent Brossoit"}},
    "San Jose Sharks":{"starter":{"SV":0.898,"name":"Kaapo Kahkonen"},"backup":{"SV":0.890,"name":"Mackenzie Blackwood"}},
    "St. Louis Blues":{"starter":{"SV":0.909,"name":"Jordan Binnington"},"backup":{"SV":0.896,"name":"Joel Hofer"}},
    "Calgary Flames":{"starter":{"SV":0.910,"name":"Jacob Markstrom"},"backup":{"SV":0.898,"name":"Dan Vladar"}},
    "Chicago Blackhawks":{"starter":{"SV":0.907,"name":"Petr Mrazek"},"backup":{"SV":0.892,"name":"Arvid Soderblom"}},
    "Detroit Red Wings":{"starter":{"SV":0.905,"name":"Ville Husso"},"backup":{"SV":0.894,"name":"James Reimer"}},
    "Dallas Stars":{"starter":{"SV":0.917,"name":"Jake Oettinger"},"backup":{"SV":0.902,"name":"Scott Wedgewood"}},
    "Seattle Kraken":{"starter":{"SV":0.907,"name":"Philipp Grubauer"},"backup":{"SV":0.897,"name":"Joey Daccord"}},
    "Philadelphia Flyers":{"starter":{"SV":0.910,"name":"Carter Hart"},"backup":{"SV":0.896,"name":"Samuel Ersson"}},
    "Montreal Canadiens":{"starter":{"SV":0.903,"name":"Jake Allen"},"backup":{"SV":0.893,"name":"Sam Montembeault"}},
    "Anaheim Ducks":{"starter":{"SV":0.906,"name":"John Gibson"},"backup":{"SV":0.895,"name":"Lukas Dostal"}},
    "Carolina Hurricanes":{"starter":{"SV":0.916,"name":"Frederik Andersen"},"backup":{"SV":0.905,"name":"Antti Raanta"}},
    "New York Islanders":{"starter":{"SV":0.918,"name":"Ilya Sorokin"},"backup":{"SV":0.907,"name":"Semyon Varlamov"}},
    "Utah Mammoth":{"starter":{"SV":0.905,"name":"Generic Starter"},"backup":{"SV":0.895,"name":"Generic Backup"}}
}

# =========================
# IMPORTS FOR HELPERS
# =========================
import numpy as np
from Sec1_Core_Inj import season_injury_impact

# --- Key mapping for JSON-safe exports ---
_json_key_map = {
    "Sh%": "sh_pct",
    "Sv%": "sv_pct",
    "OT%": "ot_pct",
    "Diff/G": "diff_pg",
    "Pace/G": "pace_pg",
    "ST%": "st_pct",
    "MOV": "mov",
    "InjAdj": "inj_adj",
    "CloseG%": "closeg_pct",
    "FatigueG%": "fatigueg_pct",
    "1G%": "onegoal_pct",
    "H2HΔ(top8)": "h2h_delta_top8"
}

def sanitize_keys(record: dict) -> dict:
    """Return JSON-safe snake_case keys for API/export use."""
    return {_json_key_map.get(k, k.replace("/", "_").lower()): v for k, v in record.items()}
# --- Printers ---
def print_team_averages(season_stats, standings=None, season_notes=None):
    print("\n=== Per-Team Season Shot Averages & Gambler Values ===")
    ot_rates = []
    for team, stats in season_stats.items():
        games = (standings[team]["W"] + standings[team]["L"] + standings[team]["OT"]) if standings else 0
        if games == 0:
            continue
        sf = stats["SF"]/games
        sa = stats["SA"]/games
        sh_pct = (stats["GF"]/stats["SF"]*100 if stats["SF"]>0 else 0)
        sv_pct = (1 - stats["GA"]/stats["SA"] if stats["SA"]>0 else 0)
        ot_pct = standings[team]["OT"] / games * 100 if standings else 0
        diff_pg = (stats["GF"] - stats["GA"]) / games
        pace = sf + sa
        st_strength = team_stats[team]["PP"] + team_stats[team]["PK"]
        mov = (stats["GF"] - stats["GA"]) / games
        inj_factor = (season_injury_impact.get(team, 0.0) / games) if games > 0 else 0.0

        # close/1‑goal/fatigue rates
        close_games = fatigue_flags = one_goal_games = 0
        if season_notes:
            for (date,h,a), note in season_notes.items():
                if team in (h,a):
                    gf = season_stats[team]["GF"]; ga = season_stats[team]["GA"]
                    if abs(gf-ga)==1:
                        close_games+=1; one_goal_games+=1
                    if "Fatigue" in note:
                        fatigue_flags+=1
        close_game_pct = close_games/games*100 if games else 0
        fatigue_pct = fatigue_flags/games*100 if games else 0
        one_goal_pct = one_goal_games/games*100 if games else 0

        # H2H vs top tier
        top_tier = {"Boston Bruins","Colorado Avalanche","Edmonton Oilers","Toronto Maple Leafs",
                    "New York Rangers","Dallas Stars","Florida Panthers","Carolina Hurricanes"}
        h2h_diff = []
        if season_notes:
            for (date,h,a), note in season_notes.items():
                if team in (h,a):
                    opp = a if team==h else h
                    if opp in top_tier:
                        gf = season_stats[team]["GF"]; ga = season_stats[team]["GA"]
                        h2h_diff.append(gf-ga)
        h2h_delta = np.mean(h2h_diff) if h2h_diff else 0

        ot_rates.append((standings[team]["OT"], games))
        print(
            f"{team:20s}  SF/G:{sf:.1f} SA/G:{sa:.1f} Sh%:{sh_pct:.1f}% Sv%:{sv_pct:.3f} "
            f"OT%:{ot_pct:.1f}% Diff/G:{diff_pg:+.2f} Pace/G:{pace:.1f} ST%:{st_strength:.1f} "
            f"MOV:{mov:+.2f} InjAdj:{inj_factor:+.2f} CloseG%:{close_game_pct:.1f}% "
            f"FatigueG%:{fatigue_pct:.1f}% 1G%:{one_goal_pct:.1f}% H2HΔ(top8):{h2h_delta:+.2f}"
        )
    if ot_rates:
        total_ot = sum(ot for ot,g in ot_rates)
        total_games = sum(g for ot,g in ot_rates)
        league_avg_ot = total_ot/total_games*100 if total_games else 0
        print(f"\nNHL Average OT%: {league_avg_ot:.1f}%")

def print_streaks(season_streaks):
    print("\n=== Win/Loss/OT Streaks (per team) ===")
    for team, streaks in season_streaks.items():
        print(f"{team:20s}  MaxW:{streaks['maxW']}  MaxL:{streaks['maxL']}  MaxOT:{streaks['maxOT']}")
        # --- NEW Monte Carlo streak printer ---
def print_monte_carlo_streaks(mc_results):
    """
    Print aggregated streak probabilities from Monte Carlo results.
    Expects results as returned by monte_carlo_league (Section 4).
    Keys use API-stable format: win_3+, loss_5+, ot_2+, etc.
    """
    print("\n=== Monte Carlo Streak Likelihoods (per team) ===")
    for team, stats in mc_results.items():
        sp = stats.get("streak_probs", {})
        if not sp:
            continue
        w5 = sp.get("win_5+", 0.0)
        l3 = sp.get("loss_3+", 0.0)
        o2 = sp.get("ot_2+", 0.0)
        print(f"{team:20s}  Win≥5:{w5}%  Loss≥3:{l3}%  OT≥2:{o2}%")

# --- NEW printer for playoff qualification odds ---
def print_monte_carlo_playoff_odds(mc_results):
    """
    Print playoff qualification odds from Monte Carlo results.
    Each team's dict is expected to contain 'playoff_pct' (Section 4).
    """
    print("\n=== Monte Carlo Playoff Odds (per team) ===")
    for team, stats in mc_results.items():
        pct = stats.get("playoff_pct", None)
        if pct is None:
            continue
        print(f"{team:20s}  Playoff%: {pct:.1f}%")

def print_sample_playbyplay(season_logs):
    if season_logs:
        first_game = next(iter(season_logs.keys()))
        print("\n=== Sample Play-by-Play (first 10 events) ===")
        print(f"Game: {first_game}")
        for event in season_logs[first_game][:10]:
            print(event)

def print_sample_doctors_note(season_notes):
    if season_notes:
        key = next(iter(season_notes.keys()))
        print("\n=== Sample Doctor’s Note ===")
        print(season_notes[key])

# --- Structured Doctor’s Note export ---
def get_doctors_note_json(season_notes, key, mc_results=None):
    """
    Return a structured JSON-ready doctor's note with summary, edge verdict, and factors list.
    If Monte Carlo results are provided, will also add streak probability language.
    """
    raw = season_notes.get(key, "")
    if not raw:
        return {}

    factors = []
    if "Goalie" in raw:
        factors.append("goalie adjustment noted")
    if "Fatigue" in raw:
        factors.append("fatigue flagged")
    if "Special" in raw or "PP" in raw or "PK" in raw:
        factors.append("special teams impact")

    edge_team = None
    if "Bruins" in raw:
        edge_team = "Boston Bruins"
    elif "Maple Leafs" in raw:
        edge_team = "Toronto Maple Leafs"
    elif "Oilers" in raw:
        edge_team = "Edmonton Oilers"

    summary = f"{edge_team} projected edge" if edge_team else "No clear edge"

    # --- UX embellishment: streak reference ---
    if mc_results and edge_team and edge_team in mc_results:
        sp = mc_results[edge_team].get("streak_probs", {})
        w3 = sp.get("win_3+")
        w7 = sp.get("win_7+")
        if w3 is not None and w7 is not None:
            streak_summary = f"{edge_team} have a {w3}% chance of ≥3 wins, but only {w7}% for ≥7."
            factors.append(streak_summary)

    return {
        "summary": summary,
        "edge": edge_team,
        "factors": factors
    }

# --- Explicit exports ---
__all__ = [
    "injury_profiles", "team_rosters", "team_stats", "goalies",
    "print_team_averages", "print_streaks", "print_monte_carlo_streaks",
    "print_monte_carlo_playoff_odds",  # newly added export
    "print_sample_playbyplay", "print_sample_doctors_note",
    "sanitize_keys", "get_doctors_note_json"
]

# ========== END OF SECTION 5 =================================