import requests

def get_shots_for_date(date="2024-10-10"):
    sb_url = f"https://api-web.nhle.com/v1/scoreboard/{date}"
    sb_resp = requests.get(sb_url).json()
    days = sb_resp.get("gamesByDate", [])
    if not days:
        print(f"No games found for {date}")
        return []

    for day in days:
        for g in day.get("games", []):
            game_id = g["id"]
            away = g["awayTeam"]["abbrev"]
            home = g["homeTeam"]["abbrev"]
            gtype = g["gameType"]
            print(f"{away} @ {home} | gameId={game_id} | type={gtype}")

            pbp_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
            pbp_events = requests.get(pbp_url).json().get("plays", [])
            shots = [e for e in pbp_events if e.get("typeDescKey") == "shot-on-goal"]
            print(f"  Shots: {len(shots)}")
            # safe printing
            for s in shots[:5]:
                d = s.get("details", {})
                period = s.get("period", "?")
                time = s.get("timeInPeriod", "?")
                shot_type = d.get("shotType", "?")
                shooter = d.get("shooterId", "?")
                print(f"    P{period} {time} | {shot_type} by {shooter}")

    return days

if __name__ == "__main__":
    get_shots_for_date("2024-10-10")  # known regular season start last year