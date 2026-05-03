import requests, sqlite3, pandas as pd

db_path = "nhl_data.db"

# Connect / create database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table with UNIQUE constraint
cursor.execute("""
CREATE TABLE IF NOT EXISTS games (
    game_date TEXT,
    home_team TEXT,
    away_team TEXT,
    start_time_utc TEXT,
    UNIQUE(game_date, home_team, away_team) ON CONFLICT IGNORE
)
""")

# Prompt user
date_str = input("Enter a date (YYYY-MM-DD): ")

# Fetch from API
url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
print(f"Fetching {url}")
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    games = data.get("gameWeek", [])[0].get("games", []) if data.get("gameWeek") else []

    if games:
        for game in games:
            home = game["homeTeam"]["placeName"]["default"]
            away = game["awayTeam"]["placeName"]["default"]
            time = game["startTimeUTC"]

            cursor.execute("""
                INSERT OR IGNORE INTO games (game_date, home_team, away_team, start_time_utc)
                VALUES (?,?,?,?)
            """, (date_str, home, away, time))
            print(f"✅ {away} @ {home} stored")
    else:
        print(f"-- No games found on {date_str}")

except Exception as e:
    print(f"❌ Error: {e}")

# Save and close
conn.commit()
conn.close()

print("🎉 Finished updating daily schedule")