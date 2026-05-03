import json
import requests

# Pull one boxscore using a gamePk from your file
game_pk = 2022010056
url = f"https://api-web.nhle.com/v1/gamecenter/{game_pk}/boxscore"

resp = requests.get(url)
print("Status:", resp.status_code)

if resp.status_code == 200:
    box = resp.json()
    print("\nTop-level keys:", list(box.keys()))
    
    # Save it so we can inspect
    with open("sample_boxscore.json", "w") as f:
        json.dump(box, f, indent=2)
    print("\nSaved to sample_boxscore.json - send me the first 200 lines")