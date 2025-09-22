import requests
import datetime
import json
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
    # add more as needed
}

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
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_season_stats(team_abbrev: str, season_id: str):
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

        PPG += team_stats.get("powerPlayGoals", 0)
        PPA += team_stats.get("powerPlayOpportunities", 0)
        PKGA += opp_stats.get("powerPlayGoals", 0)
        PKSA += opp_stats.get("powerPlayOpportunities", 0)

    GP = W + L
    win_pct = (W / GP) if GP else 0
    gd_per_game = ((GF - GA) / GP) if GP else 0
    pp_pct = (100 * PPG / PPA) if PPA else 0
    pk_pct = (100 * (1 - (PKGA / PKSA))) if PKSA else 0

    return {
        "season": season_id,
        "W": W,
        "L": L,
        "GF": GF,
        "GA": GA,
        "Win%": f"{win_pct:.2f}",
        "GD/G": f"{gd_per_game:.2f}",
        "PP%": f"{pp_pct:.1f}",
        "PK%": f"{pk_pct:.1f}"
    }

def build_team_report(team_abbrev: str):
    seasons = ["20232024", "20242025", "20252026"]
    rows = []
    for s in seasons:
        stats = fetch_season_stats(team_abbrev, s)
        if stats:
            stats["season"] = f"{s[:4]}–{s[4:]}"
            rows.append(stats)
    return rows

def write_dashboard(home_abbrev, away_abbrev, home_report, away_report, game_date: str):
    home_full = TEAM_NAMES.get(home_abbrev, home_abbrev)
    away_full = TEAM_NAMES.get(away_abbrev, away_abbrev)

    filename = f"dashboard_{game_date}_{away_abbrev}_vs_{home_abbrev}.html"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{away_full} vs {home_full} - Three Season Comparison</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 1rem;
      background: #f4f7fb;
    }}
    h1 {{ text-align: center; font-size: 24px; margin: 0; }}
    h2 {{ text-align: center; font-size: 18px; margin: 6px 0 0; }}
    .years {{
      font-size: 12px; text-align: center; margin-bottom: 12px; color: #555;
    }}
    .container {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 10px;
      justify-content: center;
    }}
    .panel {{
      background: #fff; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      padding: 0.5rem;
    }}
    .panel h3 {{
      font-size: 14px; margin: 0 0 6px; text-align: center;
    }}
    .panel h3 img {{
      vertical-align: middle; margin-right: 6px;
    }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 10px; table-layout: fixed;
    }}
    th, td {{
      padding: 2px 3px; text-align: right; border: 1px solid #ddd;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; width: 60px; }}
    th {{ background: #f0f0f0; }}
    .comparison {{ background: #fafafa; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{away_full} vs {home_full}</h1>
  <h2>Team Stats Comparison</h2>
  <div class="years">(2023–24 / 2024–25 / 2025–26)</div>

  <div class="container">
    <div class="panel">
      <h3><img src="https://assets.nhle.com/logos/nhl/svg/{home_abbrev}.svg" alt="" height="22"> {home_full}</h3>
      <table>
        <thead>
          <tr><th>Season</th><th>W</th><th>L</th><th>GF</th><th>GA</th>
              <th>Win%</th><th>GD/G</th><th>PP%</th><th>PK%</th></tr>
        </thead>
        <tbody id="home-data"></tbody>
      </table>
    </div>

    <div class="panel comparison">
      <h3>Key Comparison Notes</h3>
      <div id="comparison-notes"><p>Loading...</p></div>
    </div>

    <div class="panel">
      <h3><img src="https://assets.nhle.com/logos/nhl/svg/{away_abbrev}.svg" alt="" height="22"> {away_full}</h3>
      <table>
        <thead>
          <tr><th>Season</th><th>W</th><th>L</th><th>GF</th><th>GA</th>
              <th>Win%</th><th>GD/G</th><th>PP%</th><th>PK%</th></tr>
        </thead>
        <tbody id="away-data"></tbody>
      </table>
    </div>
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
      const lastHome = home[home.length - 1];
      const lastAway = away[away.length - 1];

      let html = "";
      html += `<p><strong>Win % (Current):</strong> ${{homeFull}} ${{lastHome["Win%"]}} vs ${{awayFull}} ${{lastAway["Win%"]}}</p>`;
      html += `<p><strong>Goal Diff/Game:</strong> ${{homeFull}} ${{lastHome["GD/G"]}} vs ${{awayFull}} ${{lastAway["GD/G"]}}</p>`;
      html += `<p><strong>Special Teams:</strong> ${{homeFull}} PP ${{lastHome["PP%"]}}%, PK ${{lastHome["PK%"]}}% |
                   ${{awayFull}} PP ${{lastAway["PP%"]}}%, PK ${{lastAway["PK%"]}}%</p>`;

      // Weighted Edge calculation
      const wWin = 0.50, wGD = 0.25, wPP = 0.125, wPK = 0.125;

      function score(team) {{
        return (parseFloat(team["Win%"]) * wWin) +
               (parseFloat(team["GD/G"]) * wGD) +
               (parseFloat(team["PP%"]) / 100 * wPP) +
               (parseFloat(team["PK%"]) / 100 * wPK);
      }}

      const scoreHome = score(lastHome);
      const scoreAway = score(lastAway);

      if (scoreHome > scoreAway) {{
        html += `<p><strong>Edge:</strong> ${{homeFull}} (overall weighted edge score higher).</p>`;
      }} else if (scoreHome < scoreAway) {{
        html += `<p><strong>Edge:</strong> ${{awayFull}} (overall weighted edge score higher).</p>`;
      }} else {{
        html += `<p><strong>Edge:</strong> Even matchup (scores equal).</p>`;
      }}

      document.getElementById("comparison-notes").innerHTML = html;
    }}

    populateTable(homeStats, "home-data");
    populateTable(awayStats, "away-data");
    comparisonNotes(homeStats, awayStats);
  </script>
</body>
</html>"""

    Path(filename).write_text(html, encoding="utf-8")
    print(f"✅ Wrote {filename}")

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