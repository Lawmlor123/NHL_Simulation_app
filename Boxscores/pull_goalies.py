"""
pull_goalies.py  –  Pull goalie game-by-game box scores (3 seasons)
"""
import requests, time, pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
SEASONS = [20222023, 20232024, 20242025, 20252026]
LIMIT = 100

def pull_goalie_season(season):
    rows, start = [], 0
    while True:
        url = (
            f"https://api.nhle.com/stats/rest/en/goalie/summary?"
            f"isAggregate=false&isGame=true"
            f"&cayenneExp=seasonId={season} and gameTypeId=2"
            f"&start={start}&limit={LIMIT}"
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()["data"]
        if not data:
            break
        rows.extend(data)
        print(f"  Season {season}: {len(rows)} rows ...", end="\r")
        start += LIMIT
        time.sleep(0.35)
    print(f"  Season {season}: {len(rows)} goalie rows      ")
    return pd.DataFrame(rows)

def main():
    print("Pulling goalie box scores ...\n")
    frames = []
    for s in SEASONS:
        df = pull_goalie_season(s)
        frames.append(df)

    goalies = pd.concat(frames, ignore_index=True)

    print(f"\nShape: {goalies.shape}")
    print(f"\nColumns:\n{goalies.columns.tolist()}")

    # Show one row so we can see the field names
    print(f"\nSample row:")
    for k, v in goalies.iloc[0].to_dict().items():
        print(f"  {k}: {v}")

    # Save raw
    out = BASE / "goalie_boxscores_raw.parquet"
    goalies.to_parquet(out, index=False)
    print(f"\nSaved: {out}")

    # Check: can we determine game winners?
    if "wins" in goalies.columns:
        winners = goalies[goalies["wins"] == 1]
        print(f"\nGames with a goalie W:  {winners['gameId'].nunique():,}")
        print(f"Total unique games:     {goalies['gameId'].nunique():,}")
        coverage = winners['gameId'].nunique() / goalies['gameId'].nunique()
        print(f"Coverage:               {coverage:.1%}")

if __name__ == "__main__":
    main()