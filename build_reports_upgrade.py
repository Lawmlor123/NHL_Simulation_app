import requests
import datetime
import csv
import argparse

BASE = "https://api-web.nhle.com/v1"

TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH",
    "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL",
    "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"
]

def fetch_boxscore(game_id: str):
    url = f"{BASE}/gamecenter/{game_id}/boxscore"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_club_schedule(team: str, season: str):
    url = f"{BASE}/club-schedule-season/{team}/{season}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def toi_to_seconds(toi_str: str) -> int:
    try:
        m, s = map(int, toi_str.split(":"))
        return m * 60 + s
    except Exception:
        return 0

def summarize_boxscore(data: dict, debug_once=False):
    home = data.get("homeTeam", {})
    away = data.get("awayTeam", {})
    pbs = data.get("playerByGameStats", {})

    game_id = data.get("id")
    game_date = data.get("gameDate", "")
    lines, rows = [], []

    lines.append("Final Score:")
    lines.append(f"{away.get('placeName', {}).get('default','Away')} "
                 f"({away.get('abbrev','')}) : {away.get('score',0)}")
    lines.append(f"{home.get('placeName', {}).get('default','Home')} "
                 f"({home.get('abbrev','')}) : {home.get('score',0)}")

    if debug_once:
        print("\n=== DEBUG: full playerByGameStats keys ===")
        print(pbs.keys())
        if "forwards" in pbs and pbs["forwards"]:
            print("Sample forward:", pbs["forwards"][0])
        print("=== END DEBUG ===\n")

    # Skaters
    skaters = []
    for group in ("forwards", "defense"):
        for pdata in pbs.get(group, []):
            skaters.append({
                "name": pdata.get("name", {}).get("default", "Unknown"),
                "team": pdata.get("team", {}).get("abbrev", ""),
                "goals": pdata.get("goals", 0),
                "assists": pdata.get("assists", 0),
                "shots": pdata.get("shots", 0),
                "toi": pdata.get("toi", "0:00"),
                "saves": "",
                "shots_against": ""
            })

    skaters.sort(key=lambda x: toi_to_seconds(x["toi"]), reverse=True)
    lines.append("\nTop 5 Skaters (by TOI):")
    for p in skaters[:5]:
        lines.append(f"{p['name']} ({p['team']}): "
                     f"G={p['goals']} A={p['assists']} Shots={p['shots']} TOI={p['toi']}")
    rows.extend([{"game_id": game_id, "date": game_date, **s} for s in skaters])

    # Goalies
    lines.append("\nGoalies:")
    for pdata in pbs.get("goalies", []):
        name = pdata.get("name", {}).get("default", "Unknown")
        team = pdata.get("team", {}).get("abbrev", "")
        saves = pdata.get("saves", 0)
        shots = pdata.get("shotsAgainst", 0)
        toi = pdata.get("toi", "0:00")
        lines.append(f"{name} ({team}): {saves}/{shots} SV, TOI {toi}")
        rows.append({
            "game_id": game_id, "date": game_date,
            "team": team, "name": name,
            "goals": "", "assists": "", "shots": "",
            "toi": toi, "saves": saves, "shots_against": shots
        })

    return "\n".join(lines), rows

def export_to_csv(all_game_stats, filename="yesterday_game_stats.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game_id","date","team","name",
            "goals","assists","shots","toi","saves","shots_against"
        ])
        writer.writeheader()
        writer.writerows(all_game_stats)
    print(f"Saved {len(all_game_stats)} rows to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", help="Optional specific gameId to fetch")
    args = parser.parse_args()

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    season = f"{today.year}{today.year+1}"

    seen_games, all_stats = set(), []
    debug_done = False

    if args.game_id:
        game_id = args.game_id
        print(f"Fetching single game {game_id}...")
        box = fetch_boxscore(game_id)
        report, rows = summarize_boxscore(box, debug_once=True)
        print(report)
        all_stats.extend(rows)
    else:
        print(f"Fetching all team games for {yesterday}...\n")
        for team in TEAMS:
            try:
                sched = fetch_club_schedule(team, season)
                for g in sched.get("games", []):
                    if g.get("gameDate") == yesterday.strftime("%Y-%m-%d"):
                        game_id = g.get("id")
                        if game_id in seen_games:
                            continue
                        seen_games.add(game_id)
                        print(f"\n=== {g['awayTeam']['abbrev']} at {g['homeTeam']['abbrev']} "
                              f"(gameId={game_id}) ===")
                        box = fetch_boxscore(str(game_id))
                        report, rows = summarize_boxscore(box, debug_once=not debug_done)
                        debug_done = True
                        print(report)
                        all_stats.extend(rows)
            except Exception as e:
                print(f"Error fetching {team}: {e}")

    export_to_csv(all_stats)

if __name__ == "__main__":
    main()