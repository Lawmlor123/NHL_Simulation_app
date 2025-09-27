import requests
import datetime
import json
import csv
from pathlib import Path

BASE = "https://api-web.nhle.com/v1"

TEAM_NAMES = {
    "ANA": "Anaheim Ducks",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CBJ": "Columbus Blue Jackets",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "NYI": "New York Islanders",
    "PHI": "Philadelphia Flyers",
    "STL": "St. Louis Blues",
    "UTA": "Utah Hockey Club",
    "WSH": "Washington Capitals",
    "MTL": "Montreal Canadiens",
    "PIT": "Pittsburgh Penguins",
    "TBL": "Tampa Bay Lightning",
    "CAR": "Carolina Hurricanes",
    "NJD": "New Jersey Devils",
    "NYR": "New York Rangers",
    "VGK": "Vegas Golden Knights",
    "LAK": "Los Angeles Kings",
}

# ------------------------------
# CSV loader with BOM fix
# ------------------------------

def load_csv_data(file_path):
    data = {}
    try:
        with open(file_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Normalize headers
            fixed_fieldnames = []
            for name in reader.fieldnames:
                clean = name.strip().lower()
                if "team" in clean:  # catches "ï»¿team"
                    fixed_fieldnames.append("Team")
                else:
                    fixed_fieldnames.append(name.strip())
            reader.fieldnames = fixed_fieldnames

            for row in reader:
                team = row.get("Team")
                if team:
                    data[team.strip()] = row
        print(f"✅ Loaded {len(data)} rows from {file_path}")
    except FileNotFoundError:
        print(f"⚠️ CSV not found: {file_path}")
    return data

CSV_DATA = {
    "2022-2023": load_csv_data("nhl_2023_team_stats.csv"),
    "2023-2024": load_csv_data("nhl_2024_team_stats.csv"),
    "2024-2025": load_csv_data("nhl_2025_team_stats.csv"),
}

# ------------------------------
# NHL API helpers
# ------------------------------

def fetch_schedule(date: str):
    url = f"{BASE}/schedule/{date}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_boxscore(game_id: str):
    url = f"{BASE}/gamecenter/{game_id}/boxscore"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_team_schedule(team: str, season_id: str):
    url = f"{BASE}/club-schedule-season/{team}/{season_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

# ------------------------------
# Advanced Stat: Scoring First
# ------------------------------

def fetch_scoring_first_stats(team_abbrev: str, season_id: str):
    """Return record when scoring first and when opponent scores first."""
    data = fetch_team_schedule(team_abbrev, season_id)
    games = data.get("games", [])
    wins_first = losses_first = 0
    wins_against = losses_against = 0

    for g in games:
        if g.get("gameState") != "FINAL":
            continue
        gid = g["id"]

        try:
            pbp = requests.get(f"{BASE}/gamecenter/{gid}/play-by-play", timeout=15).json()
        except Exception:
            continue

        plays = pbp.get("plays", [])
        first_goal_team = None
        for play in plays:
            if play.get("typeDescKey") == "goal":
                first_goal_team = play.get("details", {}).get("eventOwnerTeamAbbrev")
                break
        if not first_goal_team:
            continue

        winner = None
        if g.get("homeTeam", {}).get("winner"):
            winner = g["homeTeam"].get("abbrev")
        elif g.get("awayTeam", {}).get("winner"):
            winner = g["awayTeam"].get("abbrev")

        if first_goal_team == team_abbrev:
            if winner == team_abbrev:
                wins_first += 1
            else:
                losses_first += 1
        else:
            if winner == team_abbrev:
                wins_against += 1
            else:
                losses_against += 1

    gp_first = wins_first + losses_first
    gp_against = wins_against + losses_against
    pct_first = (wins_first / gp_first * 100) if gp_first else 0.0
    pct_against = (wins_against / gp_against * 100) if gp_against else 0.0

    return {
        "ForRecord": f"{wins_first}-{losses_first}",
        "ForPct": round(pct_first, 1),
        "AgainstRecord": f"{wins_against}-{losses_against}",
        "AgainstPct": round(pct_against, 1)
    }

# ------------------------------
# Season Stats
# ------------------------------

def fetch_season_stats(team_abbrev: str, season_id: str):
    season_key = f"{season_id[:4]}-{season_id[4:]}"
    team_full = TEAM_NAMES.get(team_abbrev, team_abbrev)

    # --- CSV first ---
    csv_row = CSV_DATA.get(season_key, {}).get(team_full)
    if csv_row:
        try:
            W = int(csv_row.get("W", 0))
            L = int(csv_row.get("L", 0))
            GF = int(csv_row.get("GF", 0))
            GA = int(csv_row.get("GA", 0))
            win_pct = float(csv_row.get("PTS%", 0))
            gd_pg = (GF - GA) / (W + L) if (W + L) else 0
            pp_pct = float(csv_row.get("PP%", 0))
            pk_pct = float(csv_row.get("PK%", 0))
        except Exception:
            return None
        return {
            "season": season_key,
            "W": W,
            "L": L,
            "GF": GF,
            "GA": GA,
            "Win%": f"{win_pct:.2f}",
            "GD/G": f"{gd_pg:.2f}",
            "PP%": f"{pp_pct:.1f}",
            "PK%": f"{pk_pct:.1f}"
        }

    # --- Fallback to API ---
    data = fetch_team_schedule(team_abbrev, season_id)
    games = data.get("games", [])
    if not games:
        return None

    W = L = GF = GA = PPG = PPA = PKGA = PKSA = 0
    for g in games:
        if g.get("gameState") != "FINAL":
            continue
        gid = g["id"]
        box = fetch_boxscore(str(gid))
        home = box["homeTeam"]["abbrev"]
        away = box["awayTeam"]["abbrev"]

        if team_abbrev == home:
            team_stats = box["homeTeam"]
            opp_stats = box["awayTeam"]
        else:
            team_stats = box["awayTeam"]
            opp_stats = box["homeTeam"]

        if team_stats["score"] > opp_stats["score"]:
            W += 1
        else:
            L += 1

        GF += team_stats["score"]
        GA += opp_stats["score"]

        ts = team_stats.get("teamStats", {})
        os = opp_stats.get("teamStats", {})

        team_ppg = ts.get("powerPlayGoals") or ts.get("powerPlay", {}).get("goals", 0)
        team_ppa = ts.get("powerPlayOpportunities") or ts.get("powerPlay", {}).get("opportunities", 0)

        opp_ppg = os.get("powerPlayGoals") or os.get("powerPlay", {}).get("goals", 0)
        opp_ppa = os.get("powerPlayOpportunities") or os.get("powerPlay", {}).get("opportunities", 0)

        PPG += team_ppg
        PPA += team_ppa
        PKGA += opp_ppg
        PKSA += opp_ppa

    GP = W + L
    win_pct = (W / GP) if GP else 0
    gd_pg = ((GF - GA) / GP) if GP else 0
    pp_pct = (100 * PPG / PPA) if PPA else 0
    pk_pct = (100 * (1 - (PKGA / PKSA))) if PKSA else 0

    return {
        "season": season_key,
        "W": W,
        "L": L,
        "GF": GF,
        "GA": GA,
        "Win%": f"{win_pct:.2f}",
        "GD/G": f"{gd_pg:.2f}",
        "PP%": f"{pp_pct:.1f}",
        "PK%": f"{pk_pct:.1f}"
    }

# ------------------------------
# Build team report
# ------------------------------

def build_team_report(team_abbrev: str):
    seasons = ["20232024", "20242025", "20252026"]
    rows = []
    for s in seasons:
        stats = fetch_season_stats(team_abbrev, s)
        if stats:
            stats["season"] = f"{s[:4]}-{s[4:]}"  # YYYY-YYYY style
            if s == "20242025":
                try:
                    sf = fetch_scoring_first_stats(team_abbrev, s)
                except Exception:
                    sf = {"ForRecord": "0-0", "ForPct": 0.0,
                          "AgainstRecord": "0-0", "AgainstPct": 0.0}
                stats["ScoringFirstRecord"] = sf["ForRecord"]
                stats["ScoringFirstPct"] = sf["ForPct"]
                stats["OpponentFirstRecord"] = sf["AgainstRecord"]
                stats["OpponentFirstPct"] = sf["AgainstPct"]
            rows.append(stats)
    return rows

# ------------------------------
# Write Dashboard (full HTML/JS included)
# ------------------------------

def write_dashboard(home_abbrev, away_abbrev, home_report, away_report, game_date: str):
    home_full = TEAM_NAMES.get(home_abbrev, home_abbrev)
    away_full = TEAM_NAMES.get(away_abbrev, away_abbrev)
    filename = f"dashboard_{game_date}_{away_abbrev}_vs_{home_abbrev}.html"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{away_full} vs {home_full} - WTP Three Season Comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; padding: 1rem; background: #f4f7fb; }}
h1 {{ text-align: center; font-size: 24px; margin: 0; }}
h2 {{ text-align: center; font-size: 18px; margin: 6px 0 0; }}
.years {{ font-size: 12px; text-align: center; margin-bottom: 12px; color: #555; }}
.container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; justify-content: center; }}
.panel {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); padding: 0.5rem; }}
.panel h3 {{ font-size: 14px; margin: 0 0 6px; text-align: center; }}
.panel h3 img {{ vertical-align: middle; margin-right: 6px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 10px; table-layout: fixed; }}
th, td {{ padding: 2px 3px; text-align: right; border: 1px solid #ddd; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; width: 60px; }}
th {{ background: #f0f0f0; }}
.comparison {{ background: #fafafa; font-size: 12px; }}
.wtp-edge {{ margin: 1rem auto; background: #fff; border-radius: 10px;
             box-shadow: 0 3px 8px rgba(0,0,0,0.15); padding: 1rem; text-align: center; }}
.wtp-edge h2 {{ margin-top: 0; font-size: 24px; color: #003366; }}
.wtp-edge p {{ font-size: 22px; font-weight: bold; margin: 0.5rem 0; }}
</style>
</head>
<body>
<h1>{away_full} vs {home_full}</h1>
<h2>WTP Team Stats Comparison</h2>
<div class="years">(2023-24 / 2024-25 / 2025-26)</div>

<div class="container">
  <div class="panel">
    <h3><img src="https://assets.nhle.com/logos/nhl/svg/{home_abbrev}_light.svg" height="44"> {home_full}</h3>
    <table><thead>
      <tr><th>Season</th><th>W</th><th>L</th><th>GF</th><th>GA</th><th>Win%</th><th>GD/G</th><th>PP%</th><th>PK%</th></tr>
    </thead><tbody id="home-data"></tbody></table>
  </div>

  <div class="panel comparison">
    <h3>WTP Key Comparison Notes</h3>
    <div id="comparison-notes"><p>Loading...</p></div>
  </div>

  <div class="panel">
    <h3><img src="https://assets.nhle.com/logos/nhl/svg/{away_abbrev}_light.svg" height="44"> {away_full}</h3>
    <table><thead>
      <tr><th>Season</th><th>W</th><th>L</th><th>GF</th><th>GA</th><th>Win%</th><th>GD/G</th><th>PP%</th><th>PK%</th></tr>
    </thead><tbody id="away-data"></tbody></table>
  </div>
</div>

<div class="wtp-edge">
  <h2>🏒 WTP Edge</h2>
  <p id="wtp-edge-text">Loading...</p>
</div>

<script>
const homeStats = {json.dumps(home_report)};
const awayStats = {json.dumps(away_report)};
const homeFull = "{home_full}";
const awayFull = "{away_full}";

function populateTable(teamStats, elementId) {{
  const tbody = document.getElementById(elementId);
  tbody.innerHTML = "";
  teamStats.forEach(s => {{
    tbody.innerHTML += `
      <tr>
        <td>${{s.season}}</td>
        <td>${{s.W}}</td>
        <td>${{s.L}}</td>
        <td>${{s.GF}}</td>
        <td>${{s.GA}}</td>
        <td>${{s["Win%"]}}</td>
        <td>${{s["GD/G"]}}</td>
        <td>${{s["PP%"]}}</td>
        <td>${{s["PK%"]}}</td>
      </tr>`;
  }});
}}

function comparisonNotes(home, away) {{
  const homeCurrent = home.find(s => s.season === "2024-2025") || home[home.length-1];
  const awayCurrent = away.find(s => s.season === "2024-2025") || away[away.length-1];

  let html = "";
  html += `<p><strong>Win % (Current):</strong> ${{homeFull}} ${{homeCurrent["Win%"]}} vs ${{awayFull}} ${{awayCurrent["Win%"]}}</p>`;
  html += `<p><strong>Goal Diff/Game:</strong> ${{homeFull}} ${{homeCurrent["GD/G"]}} vs ${{awayFull}} ${{awayCurrent["GD/G"]}}</p>`;
  html += `<p><strong>Special Teams:</strong> ${{homeFull}} PP ${{homeCurrent["PP%"]}}%, PK ${{homeCurrent["PK%"]}}% |
               ${{awayFull}} PP ${{awayCurrent["PP%"]}}%, PK ${{awayCurrent["PK%"]}}%</p>`;

  const stiHome = parseFloat(homeCurrent["PP%"]) + parseFloat(homeCurrent["PK%"]);
  const stiAway = parseFloat(awayCurrent["PP%"]) + parseFloat(awayCurrent["PK%"]);
  html += `<p><strong>Special Teams Index (PP% + PK%):</strong> ${{homeFull}} ${{stiHome.toFixed(1)}} | ${{awayFull}} ${{stiAway.toFixed(1)}}</p>`;

  if ("ScoringFirstRecord" in homeCurrent && "ScoringFirstRecord" in awayCurrent) {{
    const homeVal = (homeCurrent["ScoringFirstRecord"] === "0-0") ? "Not available (preseason/no data yet)" :
      `${{homeCurrent["ScoringFirstRecord"]}} (${{homeCurrent["ScoringFirstPct"]}}%)`;
    const awayVal = (awayCurrent["ScoringFirstRecord"] === "0-0") ? "Not available (preseason/no data yet)" :
      `${{awayCurrent["ScoringFirstRecord"]}} (${{awayCurrent["ScoringFirstPct"]}}%)`;
    html += `<p><strong>Scoring First Record:</strong> ${{homeFull}} ${{homeVal}} | ${{awayFull}} ${{awayVal}}</p>`;
  }}

  if ("OpponentFirstRecord" in homeCurrent && "OpponentFirstRecord" in awayCurrent) {{
    const homeVal = (homeCurrent["OpponentFirstRecord"] === "0-0") ? "Not available (preseason/no data yet)" :
      `${{homeCurrent["OpponentFirstRecord"]}} (${{homeCurrent["OpponentFirstPct"]}}%)`;
    const awayVal = (awayCurrent["OpponentFirstRecord"] === "0-0") ? "Not available (preseason/no data yet)" :
      `${{awayCurrent["OpponentFirstRecord"]}} (${{awayCurrent["OpponentFirstPct"]}}%)`;
    html += `<p><strong>Opponent Scoring First Record:</strong> ${{homeFull}} ${{homeVal}} | ${{awayFull}} ${{awayVal}}</p>`;
  }}

  document.getElementById("comparison-notes").innerHTML = html;

  // WTP Edge calculation
  function calcEdge(team) {{
    const win = parseFloat(team["Win%"]);
    const gd = Math.max(-2, Math.min(2, parseFloat(team["GD/G"]))); 
    const gdNorm = (gd + 2) / 4; 
    const sti = (parseFloat(team["PP%"]) + parseFloat(team["PK%"])) / 200;
    let sfVal = 0;
    if ("ScoringFirstPct" in team && "OpponentFirstPct" in team) {{
      sfVal = (parseFloat(team["ScoringFirstPct"]) + parseFloat(team["OpponentFirstPct"])) / 200;
    }}
    return (win * 0.40) + (gdNorm * 0.20) + (sti * 0.20) + (sfVal * 0.20);
  }}

  const scoreHome = calcEdge(homeCurrent);
  const scoreAway = calcEdge(awayCurrent);

  let probHome = 50, probAway = 50;
  if (scoreHome + scoreAway > 0) {{
    probHome = (scoreHome / (scoreHome + scoreAway)) * 100;
    probAway = (scoreAway / (scoreHome + scoreAway)) * 100;
  }}

  let favored = "";
  if (probHome > probAway) {{
    favored = `<strong>${{homeFull}} favored</strong><br>`;
  }} else if (probAway > probHome) {{
    favored = `<strong>${{awayFull}} favored</strong><br>`;
  }} else {{
    favored = `<strong>Even matchup</strong><br>`;
  }}

  let edgeText = favored + `${{homeFull}}: ${{probHome.toFixed(0)}}% | ${{awayFull}}: ${{probAway.toFixed(0)}}%`;

  document.getElementById("wtp-edge-text").innerHTML = edgeText;
}}

populateTable(homeStats, "home-data");
populateTable(awayStats, "away-data");
comparisonNotes(homeStats, awayStats);
</script>
</body>
</html>"""
    Path(filename).write_text(html, encoding="utf-8")
    print(f"✅ Wrote {filename}")

# ------------------------------
# Main driver
# ------------------------------

def process_games_for_date(date: str, label: str):
    print(f"\nFetching schedule for {date} ({label})...\n")
    schedule = fetch_schedule(date)
    games = schedule.get("gameWeek", [])[0].get("games", [])
    if not games:
        print(f"No games found for {date}.")
        return
    for g in games:
        away = g.get("awayTeam", {}).get("abbrev")
        home = g.get("homeTeam", {}).get("abbrev")
        print(f"\n=== {away} at {home} ===")
        home_report = build_team_report(home)
        away_report = build_team_report(away)
        write_dashboard(home, away, home_report, away_report, date)

def main():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    process_games_for_date(yesterday.strftime("%Y-%m-%d"), "yesterday")
    process_games_for_date(today.strftime("%Y-%m-%d"), "today")

if __name__ == "__main__":
    main()