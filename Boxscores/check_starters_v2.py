"""
check_starters_v2.py
Dig deeper into the matchup field for goalie info.
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

            # Check probableGoalie in team objects
            for side in ["homeTeam", "awayTeam"]:
                team = landing.get(side, {})
                print(f"\n  {side} keys: {list(team.keys())}")
                for k in ["probableGoalie", "goalie", "starter"]:
                    if k in team:
                        print(f"    {k}: {json.dumps(team[k], indent=4)}")

            # Check matchup for goalie comparison
            matchup = landing.get("matchup", {})
            print(f"\n  matchup keys: {list(matchup.keys())}")

            if "goalieComparison" in matchup:
                print(f"\n  GOALIE COMPARISON FOUND:")
                print(json.dumps(matchup["goalieComparison"], indent=2)[:1000])

            if "goalie" in str(matchup.keys()).lower():
                for k in matchup:
                    if "goalie" in k.lower():
                        print(f"\n  {k}:")
                        print(json.dumps(matchup[k], indent=2)[:1000])

            # Only check first game to keep output short
            print("\n  (checking first game only)")
            break
        break