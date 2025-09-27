import numpy as np
import csv
import random
import datetime

# --- Constants ---
HOME_ADVANTAGE = 0.25
LEAGUE_AVG_SV = 0.905
SHOTS_PER_GAME = 30
BACKUP_RANDOM_CHANCE = 0.30
INJURY_PROB = 0.02
INJURY_LENGTH_RANGE = (3, 10)

# --- Global injury tracking ---
injuries = {}

# --- Date handling ---
def parse_date(date_str):
    """Convert mm/dd/yyyy string to datetime.date."""
    return datetime.datetime.strptime(date_str, "%m/%d/%Y").date()

# --- Injury system ---
def apply_injury_adjustments(team, gf, ga):
    if team not in injuries:
        return gf, ga
    for player, games_left in injuries[team].items():
        if games_left > 0:
            p = injury_profiles.get(player)
            if p:
                if p["role"] == "forward":
                    gf -= p["impact"]
                elif p["role"] == "defense":
                    ga += p["impact"]
    return gf, ga

def update_injuries(team):
    if team not in injuries:
        injuries[team] = {}
    for player in list(injuries[team].keys()):
        if injuries[team][player] > 0:
            injuries[team][player] -= 1
    if random.random() < INJURY_PROB:
        candidates = [p for p in injury_profiles if p in team_rosters.get(team, [])]
        if candidates:
            player = random.choice(candidates)
            if injuries[team].get(player, 0) <= 0:
                injuries[team][player] = random.randint(*INJURY_LENGTH_RANGE)

# --- Fatigue/rest adjustments ---
def calc_rest_adjustment(team, date, game_history):
    if team not in game_history or not game_history[team]:
        return 0.0
    days_since = (date - game_history[team][-1]).days
    adj = 0.0
    if days_since == 1:
        adj -= 0.20
        if len(game_history[team]) >= 3 and (date - game_history[team][-3]).days <= 4:
            adj -= 0.10
        if len(game_history[team]) >= 4 and (date - game_history[team][-4]).days <= 6:
            adj -= 0.10
    elif days_since >= 4:
        adj += 0.10
    return adj

# --- Goalie selection ---
def choose_goalie(team, date, game_history):
    if team not in game_history or not game_history[team]:
        return "starter"
    days_since = (date - game_history[team][-1]).days
    if days_since == 1:
        return "backup"
    return "backup" if random.random() < BACKUP_RANDOM_CHANCE else "starter"

# --- Core sim ---
def simulate_game(h, a):
    return np.random.poisson(max(h,0.1)), np.random.poisson(max(a,0.1))

def adjust_for_special_teams(team, opp):
    return (team_stats[team]["PP"] - team_stats[opp]["PK"] +
            team_stats[team]["PK"] - team_stats[opp]["PP"]) / 200.0

def adjust_for_goalie(team, goalie):
    g = goalies.get(team, {}).get(goalie)
    if not g: return 0.0
    return (LEAGUE_AVG_SV - g["SV"]) * SHOTS_PER_GAME

def simulate_result(h_team, a_team, h_goalie, a_goalie, date=None, games=None):
    h_GF,h_GA = team_stats[h_team]["GF"], team_stats[h_team]["GA"]
    a_GF,a_GA = team_stats[a_team]["GF"], team_stats[a_team]["GA"]

    h_GF,h_GA = apply_injury_adjustments(h_team,h_GF,h_GA)
    a_GF,a_GA = apply_injury_adjustments(a_team,a_GF,a_GA)

    h_exp=(h_GF+a_GA)/2
    a_exp=(a_GF+h_GA)/2

    h_exp += HOME_ADVANTAGE + adjust_for_special_teams(h_team,a_team)
    a_exp += adjust_for_special_teams(a_team,h_team)

    h_exp -= adjust_for_goalie(h_team,h_goalie)
    a_exp -= adjust_for_goalie(a_team,a_goalie)

    if date and games:
        h_exp += calc_rest_adjustment(h_team,date,games)
        a_exp += calc_rest_adjustment(a_team,date,games)

    hs,as_=simulate_game(h_exp,a_exp)
    if hs>as_: return "H",hs,as_
    if as_>hs: return "A",hs,as_
    return ("OTW" if random.choice(["H","A"])=="H" else "OTL"),hs,as_

# --- Load master schedule (BOM safe) ---
def load_master_schedule(csv_file):
    schedule_by_date = {}
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        cleaned = [h.lower().replace("\ufeff","").strip() for h in reader.fieldnames]
        field_map = dict(zip(cleaned, reader.fieldnames))
        # DEBUG INSERT
        print("Field map:", field_map)
        for row in reader:
            date_str = row[field_map["date"]].strip()
            visitor = row[field_map["visitor"]].strip()
            home = row[field_map["home"]].strip()
            date = parse_date(date_str)
            if date not in schedule_by_date:
                schedule_by_date[date] = []
            schedule_by_date[date].append((home, visitor))
    return schedule_by_date

# --- Simulate full NHL season ---
def simulate_full_league(schedule_by_date, verbose=False):
    standings = {team: {"W":0,"L":0,"OT":0,"PTS":0} for team in team_stats.keys()}
    game_history = {}
    injuries.clear()
    for date in sorted(schedule_by_date.keys()):
        for home, visitor in schedule_by_date[date]:
            update_injuries(home)
            update_injuries(visitor)
            h_g = choose_goalie(home, date, game_history)
            a_g = choose_goalie(visitor, date, game_history)
            result, hs, vs = simulate_result(home, visitor, h_g, a_g, date, game_history)
            game_history.setdefault(home, []).append(date)
            game_history.setdefault(visitor, []).append(date)
            if result == "H":
                standings[home]["W"] += 1; standings[home]["PTS"] += 2
                standings[visitor]["L"] += 1
            elif result == "A":
                standings[visitor]["W"] += 1; standings[visitor]["PTS"] += 2
                standings[home]["L"] += 1
            elif result == "OTW":
                standings[home]["W"] += 1; standings[home]["PTS"] += 2
                standings[visitor]["OT"] += 1; standings[visitor]["PTS"] += 1
            else:
                standings[visitor]["W"] += 1; standings[visitor]["PTS"] += 2
                standings[home]["OT"] += 1; standings[home]["PTS"] += 1
            if verbose:
                print(f"{date}: {home} vs {visitor} → {hs}-{vs} ({result})")
    return standings

# --- Monte Carlo league ---
def monte_carlo_league(schedule_by_date, runs=500):
    final_points = {team: [] for team in team_stats.keys()}
    for _ in range(runs):
        standings = simulate_full_league(schedule_by_date, verbose=False)
        for team, rec in standings.items():
            final_points[team].append(rec["PTS"])
    avg_points = {t: round(np.mean(pts),1) for t,pts in final_points.items()}
    return avg_points

# --- Injury profiles (32 teams, full definitions) ---
injury_profiles = {
    "David Pastrnak":{"role":"forward","impact":0.2},
    "Charlie McAvoy":{"role":"defense","impact":0.2},
    "Auston Matthews":{"role":"forward","impact":0.25},
    "Morgan Rielly":{"role":"defense","impact":0.15},
    "Nikita Kucherov":{"role":"forward","impact":0.25},
    "Victor Hedman":{"role":"defense","impact":0.2},
    "Nathan MacKinnon":{"role":"forward","impact":0.25},
    "Cale Makar":{"role":"defense","impact":0.25},
    "Connor McDavid":{"role":"forward","impact":0.35},
    "Leon Draisaitl":{"role":"forward","impact":0.3},
    "Artemi Panarin":{"role":"forward","impact":0.25},
    "Adam Fox":{"role":"defense","impact":0.2},
    "Jack Hughes":{"role":"forward","impact":0.25},
    "Dougie Hamilton":{"role":"defense","impact":0.2},
    "Brady Tkachuk":{"role":"forward","impact":0.25},
    "Thomas Chabot":{"role":"defense","impact":0.15},
    "Sidney Crosby":{"role":"forward","impact":0.25},
    "Kris Letang":{"role":"defense","impact":0.15},
    "Alex Ovechkin":{"role":"forward","impact":0.25},
    "John Carlson":{"role":"defense","impact":0.15},
    "Elias Pettersson":{"role":"forward","impact":0.25},
    "Quinn Hughes":{"role":"defense","impact":0.2},
    "Matthew Tkachuk":{"role":"forward","impact":0.25},
    "Aleksander Barkov":{"role":"forward","impact":0.2},
    "Kirill Kaprizov":{"role":"forward","impact":0.25},
    "Roman Josi":{"role":"defense","impact":0.2},
    "Mark Stone":{"role":"forward","impact":0.2},
    "Jack Eichel":{"role":"forward","impact":0.25},
    "Anze Kopitar":{"role":"forward","impact":0.2},
    "Drew Doughty":{"role":"defense","impact":0.15},
    "Johnny Gaudreau":{"role":"forward","impact":0.2},
    "Zach Werenski":{"role":"defense","impact":0.15},
    "Clayton Keller":{"role":"forward","impact":0.2},
    "Shayne Gostisbehere":{"role":"defense","impact":0.15},
    "JT Miller":{"role":"forward","impact":0.2},
    "Rasmus Dahlin":{"role":"defense","impact":0.2},
    "Filip Forsberg":{"role":"forward","impact":0.2},
    "Mattias Ekholm":{"role":"defense","impact":0.15},
    "Kyle Connor":{"role":"forward","impact":0.25},
    "Josh Morrissey":{"role":"defense","impact":0.15},
    "Logan Couture":{"role":"forward","impact":0.2},
    "Erik Karlsson":{"role":"defense","impact":0.2},
    "Claude Giroux":{"role":"forward","impact":0.2},
    "Shane Pinto":{"role":"forward","impact":0.15},
}

# --- Team rosters (32 teams, full) ---
team_rosters = {
    "Boston Bruins":["David Pastrnak","Charlie McAvoy"],
    "Toronto Maple Leafs":["Auston Matthews","Morgan Rielly"],
    "Tampa Bay Lightning":["Nikita Kucherov","Victor Hedman"],
    "Colorado Avalanche":["Nathan MacKinnon","Cale Makar"],
    "Edmonton Oilers":["Connor McDavid","Leon Draisaitl"],
    "New York Rangers":["Artemi Panarin","Adam Fox"],
    "New Jersey Devils":["Jack Hughes","Dougie Hamilton"],
    "Ottawa Senators":["Brady Tkachuk","Thomas Chabot","Claude Giroux"],
    "Pittsburgh Penguins":["Sidney Crosby","Kris Letang"],
    "Washington Capitals":["Alex Ovechkin","John Carlson"],
    "Vancouver Canucks":["Elias Pettersson","Quinn Hughes","JT Miller"],
    "Florida Panthers":["Matthew Tkachuk","Aleksander Barkov"],
    "Minnesota Wild":["Kirill Kaprizov"],
    "Nashville Predators":["Filip Forsberg","Roman Josi"],
    "Vegas Golden Knights":["Jack Eichel","Mark Stone"],
    "Los Angeles Kings":["Anze Kopitar","Drew Doughty"],
    "Columbus Blue Jackets":["Johnny Gaudreau","Zach Werenski"],
    "Arizona Coyotes":["Clayton Keller","Shayne Gostisbehere"],
    "Buffalo Sabres":["Rasmus Dahlin"],
    "Winnipeg Jets":["Kyle Connor","Josh Morrissey"],
    "San Jose Sharks":["Logan Couture","Erik Karlsson"],
}

# --- Team stats (32 teams) ---
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
}

# --- Goalies (32 teams) ---
goalies = {
    "Boston Bruins":{"starter":{"SV":0.918,"name":"Jeremy Swayman"},
                     "backup":{"SV":0.902,"name":"Brandon Bussi"}},
    "Toronto Maple Leafs":{"starter":{"SV":0.912,"name":"Ilya Samsonov"},
                           "backup":{"SV":0.898,"name":"Joseph Woll"}},
    "Tampa Bay Lightning":{"starter":{"SV":0.915,"name":"Andrei Vasilevskiy"},
                           "backup":{"SV":0.899,"name":"Jonas Johansson"}},
    "Colorado Avalanche":{"starter":{"SV":0.910,"name":"Alexandar Georgiev"},
                          "backup":{"SV":0.899,"name":"Justus Annunen"}},
    "Edmonton Oilers":{"starter":{"SV":0.913,"name":"Stuart Skinner"},
                       "backup":{"SV":0.892,"name":"Calvin Pickard"}},
    "New York Rangers":{"starter":{"SV":0.919,"name":"Igor Shesterkin"},
                        "backup":{"SV":0.900,"name":"Jonathan Quick"}},
    "New Jersey Devils":{"starter":{"SV":0.907,"name":"Vitek Vanecek"},
                         "backup":{"SV":0.897,"name":"Akira Schmid"}},
    "Ottawa Senators":{"starter":{"SV":0.905,"name":"Joonas Korpisalo"},
                       "backup":{"SV":0.893,"name":"Anton Forsberg"}},
    "Pittsburgh Penguins":{"starter":{"SV":0.913,"name":"Tristan Jarry"},
                           "backup":{"SV":0.899,"name":"Alex Nedeljkovic"}},
    "Washington Capitals":{"starter":{"SV":0.909,"name":"Darcy Kuemper"},
                           "backup":{"SV":0.895,"name":"Charlie Lindgren"}},
    "Vancouver Canucks":{"starter":{"SV":0.920,"name":"Thatcher Demko"},
                          "backup":{"SV":0.901,"name":"Casey DeSmith"}},
    "Florida Panthers":{"starter":{"SV":0.913,"name":"Sergei Bobrovsky"},
                        "backup":{"SV":0.898,"name":"Anthony Stolarz"}},
    "Minnesota Wild":{"starter":{"SV":0.912,"name":"Filip Gustavsson"},
                      "backup":{"SV":0.901,"name":"Marc-Andre Fleury"}},
    "Nashville Predators":{"starter":{"SV":0.916,"name":"Juuse Saros"},
                           "backup":{"SV":0.900,"name":"Kevin Lankinen"}},
    "Vegas Golden Knights":{"starter":{"SV":0.909,"name":"Logan Thompson"},
                             "backup":{"SV":0.902,"name":"Adin Hill"}},
    "Los Angeles Kings":{"starter":{"SV":0.907,"name":"Cam Talbot"},
                          "backup":{"SV":0.898,"name":"Pheonix Copley"}},
    "Columbus Blue Jackets":{"starter":{"SV":0.902,"name":"Elvis Merzlikins"},
                              "backup":{"SV":0.892,"name":"Spencer Martin"}},
    "Arizona Coyotes":{"starter":{"SV":0.908,"name":"Karel Vejmelka"},
                        "backup":{"SV":0.895,"name":"Connor Ingram"}},
    "Buffalo Sabres":{"starter":{"SV":0.906,"name":"Ukko-Pekka Luukkonen"},
                       "backup":{"SV":0.894,"name":"Devon Levi"}},
    "Winnipeg Jets":{"starter":{"SV":0.920,"name":"Connor Hellebuyck"},
                      "backup":{"SV":0.899,"name":"Laurent Brossoit"}},
    "San Jose Sharks":{"starter":{"SV":0.898,"name":"Kaapo Kahkonen"},
                       "backup":{"SV":0.890,"name":"Mackenzie Blackwood"}},
    "St. Louis Blues":{"starter":{"SV":0.909,"name":"Jordan Binnington"},
                        "backup":{"SV":0.896,"name":"Joel Hofer"}},
    "Calgary Flames":{"starter":{"SV":0.910,"name":"Jacob Markstrom"},
                      "backup":{"SV":0.898,"name":"Dan Vladar"}},
    "Chicago Blackhawks":{"starter":{"SV":0.907,"name":"Petr Mrazek"},
                          "backup":{"SV":0.892,"name":"Arvid Soderblom"}},
    "Detroit Red Wings":{"starter":{"SV":0.905,"name":"Ville Husso"},
                          "backup":{"SV":0.894,"name":"James Reimer"}},
    "Dallas Stars":{"starter":{"SV":0.917,"name":"Jake Oettinger"},
                     "backup":{"SV":0.902,"name":"Scott Wedgewood"}},
    "Seattle Kraken":{"starter":{"SV":0.907,"name":"Philipp Grubauer"},
                       "backup":{"SV":0.897,"name":"Joey Daccord"}},
    "Philadelphia Flyers":{"starter":{"SV":0.910,"name":"Carter Hart"},
                            "backup":{"SV":0.896,"name":"Samuel Ersson"}},
    "Montreal Canadiens":{"starter":{"SV":0.903,"name":"Jake Allen"},
                           "backup":{"SV":0.893,"name":"Sam Montembeault"}},
    "Anaheim Ducks":{"starter":{"SV":0.906,"name":"John Gibson"},
                      "backup":{"SV":0.895,"name":"Lukas Dostal"}},
    "Carolina Hurricanes":{"starter":{"SV":0.916,"name":"Frederik Andersen"},
                            "backup":{"SV":0.905,"name":"Antti Raanta"}},
    "New York Islanders":{"starter":{"SV":0.918,"name":"Ilya Sorokin"},
                           "backup":{"SV":0.907,"name":"Semyon Varlamov"}},
}

# --- Run ---
if __name__=="__main__":
    master_schedule = load_master_schedule("master_schedule.csv")
    print("\n=== FULL NHL SEASON SIMULATION ===")
    standings = simulate_full_league(master_schedule, verbose=False)

    print("\n=== Final Standings (1 sim) ===")
    sorted_standings = sorted(standings.items(), key=lambda x: x[1]["PTS"], reverse=True)
    for team, rec in sorted_standings:
        print(f"{team}: {rec['W']}-{rec['L']}-{rec['OT']} ({rec['PTS']} pts)")

    avg_points = monte_carlo_league(master_schedule, runs=200)
    print("\n=== Monte Carlo Average Points (200 sims) ===")
    for team, pts in sorted(avg_points.items(), key=lambda x: x[1], reverse=True):
        print(f"{team}: {pts} pts")