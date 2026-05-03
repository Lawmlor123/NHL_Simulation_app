import requests
import datetime

BASE = "https://api-web.nhle.com/v1"

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

def main():
    today = datetime.date.today().strftime("%Y-%m-%d")
    print(f"Fetching schedule for {today}...")
    schedule = fetch_schedule(today)

    games = schedule.get("gameWeek", [])[0].get("games", [])
    if not games:
        print("No games found today.")
        return

    for g in games:
        game_id = g.get("id")   # <-- THIS is the correct ID format
        away = g.get("awayTeam", {}).get("abbrev")
        home = g.get("homeTeam", {}).get("abbrev")
        print(f"\n=== {away} at {home} (gameId={game_id}) ===")

        box = fetch_boxscore(str(game_id))

        teams = box.get("boxscore", {}).get("teams", {})
        for side, info in teams.items():
            name = info.get("team", {}).get("name", "Unknown")
            goals = info.get("teamStats", {}).get("goals", 0)
            print(f"{name}: {goals}")

if __name__ == "__main__":
    main()