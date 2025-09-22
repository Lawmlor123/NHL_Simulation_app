import requests

game_id = "2025010010"
url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
data = requests.get(url).json()

print(data.keys())  # show the top-level keys