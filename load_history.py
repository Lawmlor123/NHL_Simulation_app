import requests
import sqlite3
import sys
from datetime import datetime, timedelta
from tqdm import tqdm  # progress bar

def fetch_day(date_str):
    """Fetch schedule for a single day (YYYY-MM-DD)."""
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    resp = requests.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()

def load_season(season):
    """Load one season into SQLite (e.g. 20242025)."""
    start_year = int(season[:4])
    end_year = int(season[4:])

    start_date = datetime(start_year, 10, 1)
    end_date = datetime(end_year, 7, 1)

    conn = sqlite3.connect("nhl_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            start_time_utc TEXT,
            home_score INTEGER,
            away_score INTEGER
        )
    """)

    total_games = 0
    days = (end_date - start_date).days + 1

    # Wrap date loop with tqdm for progress bar
    for i in tqdm(range(days), desc=f"Loading {season}"):
        day = start_date + timedelta(days=i)
        date_str = day.strftime("%Y-%m-%d")
        data = fetch_day(date_str)

        if data and "gameWeek" in data:
            for day_info in data["gameWeek"]:
                for g in day_info.get("games", []):
                    game_date = date_str
                    home = g.get("homeTeam", {}).get("abbrev")
                    away = g.get("awayTeam", {}).get("abbrev")
                    time = g.get("startTimeUTC")
                    home_score = g.get("homeTeam", {}).get("score")
                    away_score = g.get("awayTeam", {}).get("score")

                    if home and away:
                        cursor.execute("""
                            INSERT OR REPLACE INTO games 
                            (game_date, home_team, away_team, start_time_utc, home_score, away_score)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (game_date, home, away, time, home_score, away_score))
                        total_games += 1

    conn.commit()
    conn.close()
    print(f"✅ Loaded season {season} with {total_games} games")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_history.py <SEASON>")
        print("Example: python load_history.py 20242025")
        sys.exit(1)

    season = sys.argv[1]
    load_season(season)