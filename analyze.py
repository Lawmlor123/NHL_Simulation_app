import pandas as pd
import os
from datetime import datetime

# --- Step 1. Load data ---
reports = pd.read_csv("game_reports.csv")
schedule = pd.read_csv("nhl_schedule_preseason.csv")

# Debug: show column names
print("\n--- Column Names Check ---")
print("Reports columns:", reports.columns.tolist())
print("Schedule columns:", schedule.columns.tolist())
print("---------------------------\n")

# --- Step 2. Auto-detect today's date ---
# On Windows, use %#m/%#d/%Y (no leading zeros)
# On Mac/Linux, you can use %-m/%-d/%Y
today = datetime.now().strftime("%#m/%#d/%Y")
print(f"📅 Using today's date from system: {today}")

today_sched = schedule[schedule["game_date"] == today]

if today_sched.empty:
    print(f"No games found in schedule for {today}")
    exit()

# --- Step 3. Join schedule + reports (only to link teams) ---
today_games = reports.merge(
    today_sched,
    left_on=["Date", "HomeTeam", "AwayTeam"],
    right_on=["game_date", "home_team", "away_team"],
    how="inner"
)

if today_games.empty:
    print(f"No matching reports found between reports and schedule for {today}")
    exit()

# --- Step 4. Build multi-season dashboards ---
os.makedirs("dashboards", exist_ok=True)

for _, game in today_games.iterrows():
    home = str(game["HomeTeam"])
    away = str(game["AwayTeam"])
    date = str(game["game_date"])

    filename = f"dashboards/{home}_vs_{away}_dashboard.html".replace(" ", "_")

    # Pull ALL seasons for the two teams
    home_stats = reports[reports["HomeTeam"] == home][
        ["StatsSeason", "Home_W", "Home_L", "Home_GF", "Home_GA", "Home_WinPct", "Home_GD/Game", "Home_PP%", "Home_PK%"]
    ].drop_duplicates("StatsSeason")

    away_stats = reports[reports["AwayTeam"] == away][
        ["StatsSeason", "Away_W", "Away_L", "Away_GF", "Away_GA", "Away_WinPct", "Away_GD/Game", "Away_PP%", "Away_PK%"]
    ].drop_duplicates("StatsSeason")

    # Sort by season (latest last)
    home_stats = home_stats.sort_values("StatsSeason")
    away_stats = away_stats.sort_values("StatsSeason")

    # Build HTML tables for multi-season
    def make_table(df, side):
        rows = ""
        for _, row_data in df.iterrows():
            if side == "home":
                rows += f"<tr><td>{row_data['StatsSeason']}</td><td>{row_data['Home_W']}</td><td>{row_data['Home_L']}</td><td>{row_data['Home_GF']}</td><td>{row_data['Home_GA']}</td><td>{row_data['Home_WinPct']}</td><td>{row_data['Home_GD/Game']}</td><td>{row_data['Home_PP%']}</td><td>{row_data['Home_PK%']}</td></tr>"
            else:
                rows += f"<tr><td>{row_data['StatsSeason']}</td><td>{row_data['Away_W']}</td><td>{row_data['Away_L']}</td><td>{row_data['Away_GF']}</td><td>{row_data['Away_GA']}</td><td>{row_data['Away_WinPct']}</td><td>{row_data['Away_GD/Game']}</td><td>{row_data['Away_PP%']}</td><td>{row_data['Away_PK%']}</td></tr>"
        return rows

    home_table = make_table(home_stats, "home")
    away_table = make_table(away_stats, "away")

    # Use latest season for comparison notes
    latest_home = home_stats.iloc[-1]
    latest_away = away_stats.iloc[-1]

    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>{home} vs {away}</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 30px;
          background: #fafafa;
          color: #222;
        }}
        h1, h2 {{
          text-align: center;
          margin-bottom: 10px;
        }}
        h3 {{
          text-align: center;
          margin-top: 0;
        }}
        .container {{
          display: flex;
          justify-content: space-between;
          gap: 20px;
          margin-top: 30px;
        }}
        .card {{
          background: white;
          border: 1px solid #ccc;
          border-radius: 6px;
          padding: 15px;
          width: 32%;
          box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          margin-top: 10px;
          font-size: 14px;
        }}
        th {{
          background: #eee;
          padding: 6px;
        }}
        td {{
          padding: 6px;
        }}
        tr:nth-child(even) {{
          background: #f9f9f9;
        }}
        p {{
          margin: 8px 0;
          line-height: 1.4em;
        }}
        b {{
          color: #000;
        }}
      </style>
    </head>
    <body>
      <h1>{home} vs {away}</h1>
      <h2>Team Stats Comparison ({home_stats['StatsSeason'].min()} vs {home_stats['StatsSeason'].max()})</h2>
      <div class="container">
        <div class="card">
          <h3>{home}</h3>
          <table>
            <tr><th>Season</th><th>W</th><th>L</th><th>GF</th><th>GA</th><th>Win%</th><th>GD/G</th><th>PP%</th><th>PK%</th></tr>
            {home_table}
          </table>
        </div>
        <div class="card">
          <h3>Key Comparison Notes</h3>
          <p><b>Win % (Latest Season):</b> {home} {latest_home['Home_WinPct']} vs {away} {latest_away['Away_WinPct']}</p>
          <p><b>Goal Diff/Game (Latest Season):</b> {home} {latest_home['Home_GD/Game']} vs {away} {latest_away['Away_GD/Game']}</p>
          <p><b>Special Teams (Latest Season):</b> {home} PP {latest_home['Home_PP%']} / PK {latest_home['Home_PK%']} |
             {away} PP {latest_away['Away_PP%']} / PK {latest_away['Away_PK%']}</p>
          <p><b>Edge:</b> {"Home team has the higher recent win rate" if latest_home['Home_WinPct'] > latest_away['Away_WinPct'] else "Away team has the higher recent win rate"}</p>
        </div>
        <div class="card">
          <h3>{away}</h3>
          <table>
            <tr><th>Season</th><th>W</th><th>L</th><th>GF</th><th>GA</th><th>Win%</th><th>GD/G</th><th>PP%</th><th>PK%</th></tr>
            {away_table}
          </table>
        </div>
      </div>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Multi-season dashboard saved: {filename}")