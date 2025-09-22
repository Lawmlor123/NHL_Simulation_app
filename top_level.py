import requests, json

game_id = "2025010010"
url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
data = requests.get(url).json()

print("Top-level keys:", list(data.keys()))

pbs = data.get("playerByGameStats")
if not pbs:
    print("\nplayerByGameStats is EMPTY or missing:", pbs)
else:
    print("\nKeys inside playerByGameStats:", list(pbs.keys()))
    if "forwards" in pbs and pbs["forwards"]:
        print("\nSample forward:", pbs["forwards"][0])