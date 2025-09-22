import requests

game_id = "2025010010"  # TOR @ OTT example
url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
data = requests.get(url).json()

# Show the keys inside playerByGameStats
pbs = data.get("playerByGameStats", {})
print("Top-level keys in playerByGameStats:", pbs.keys())

# Print one sample player record
for k, v in list(pbs.items())[:1]:
    print("Sample key:", k)
    print("Sample value keys:", v.keys())
    print(v)
    break