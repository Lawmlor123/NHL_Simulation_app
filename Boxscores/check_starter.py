"""
check_starters.py
Peek at the NHL API to see if projected starters are available yet.
"""
import requests
from datetime import date
import json

today = str(date.today())
url = f"https://api-web.nhle.com/v1/schedule/{today}"
r = requests.get(url, timeout=15)
schedule = r.json()

for game_week in schedule.get("gameWeek", []):
    if game_week["date"] == today:
        for g in game_week.get("games", []):
            game_id = g["id"]
            away = g["awayTeam"]["abbrev"]
            home = g["homeTeam"]["abbrev"]

            print(f"\n{'='*50}")
            print(f"  {away} @ {home} (ID: {game_id})")
            print(f"{'='*50}")

            landing = requests.get(
                f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing",
                timeout=15
            ).json()

            for key in ["homeTeam", "awayTeam"]:
                team = landing.get(key, {})
                goalie = team.get("probableGoalie", team.get("goalie", None))
                print(f"  {key} probable goalie: {goalie}")

            print(f"\n  Top-level keys: {list(landing.keys())}")

            for key in ["lineup", "lineups", "projectedLineup", "matchup"]:
                if key in landing:
                    print(f"\n  Found '{key}' field!")
                    print(f"  {json.dumps(landing[key], indent=2)[:500]}")