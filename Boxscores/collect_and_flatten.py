"""
collect_and_flatten.py
Pulls boxscore data from NHL API for every completed game
in nhl_2022_2025.json, flattens player stats, saves parquet.

Resume-safe: if interrupted, just run again.
"""

import json, time, requests
import pandas as pd
from pathlib import Path
from collections import Counter

# ── CONFIG ──────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player")
SCHEDULE  = BASE / "Outputs" / "nhl_2022_2025.json"
OUT_DIR   = BASE / "Boxscores"
SKATER_F  = OUT_DIR / "skater_stats.parquet"
GOALIE_F  = OUT_DIR / "goalie_stats.parquet"
PROG_F    = OUT_DIR / "collect_progress.json"
DELAY     = 0.3
SAVE_EVERY = 500

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. BUILD GAME LIST ─────────────────────────────────────────────
print("Loading schedule ...")
with open(SCHEDULE) as f:
    sched = json.load(f)

games = []
for day in sched["days"]:
    for g in day["games"]:
        if g.get("gameState") in ("FINAL", "OFF"):
            games.append({
                "game_pk":    g["gamePk"],
                "game_date":  day["date"],
                "season":     g["season"],
                "game_type":  g["gameType"],
                "home_team":  g["homeTeam"]["abbrev"],
                "away_team":  g["awayTeam"]["abbrev"],
                "home_score": g["homeTeam"].get("score"),
                "away_score": g["awayTeam"].get("score"),
            })

print(f"Completed games in schedule: {len(games)}")
type_map = {1: "Pre", 2: "Reg", 3: "Post"}
for gt, n in sorted(Counter(g["game_type"] for g in games).items()):
    print(f"  {str(type_map.get(gt, gt)):5s}: {n}")

# ── 2. LOAD RESUME STATE ───────────────────────────────────────────
done = set()
skater_rows, goalie_rows = [], []

if PROG_F.exists():
    done = set(json.load(open(PROG_F)))
    if SKATER_F.exists():
        skater_rows = pd.read_parquet(SKATER_F).to_dict("records")
    if GOALIE_F.exists():
        goalie_rows = pd.read_parquet(GOALIE_F).to_dict("records")
    print(f"Resuming: {len(done)} games done, "
          f"{len(skater_rows)} skater / {len(goalie_rows)} goalie rows")

todo = [g for g in games if g["game_pk"] not in done]
print(f"Remaining: {len(todo)}")
if not todo:
    print("Nothing to do!")
    raise SystemExit

# ── 3. FLATTEN HELPER ──────────────────────────────────────────────
def flatten(box, meta):
    sk, gl = [], []
    pbs = box.get("playerByGameStats", {})

    for side in ("homeTeam", "awayTeam"):
        is_home = side == "homeTeam"
        team = box[side]["abbrev"]
        opp  = box["awayTeam" if is_home else "homeTeam"]["abbrev"]
        base = {**meta, "team": team, "opponent": opp, "is_home": is_home}

        for grp in ("forwards", "defense"):
            for p in pbs.get(side, {}).get(grp, []):
                sk.append({**base,
                    "player_id":     p["playerId"],
                    "player_name":   p["name"]["default"],
                    "position":      p["position"],
                    "sweater":       p.get("sweaterNumber"),
                    "goals":         p.get("goals", 0),
                    "assists":       p.get("assists", 0),
                    "points":        p.get("points", 0),
                    "plus_minus":    p.get("plusMinus", 0),
                    "pim":           p.get("pim", 0),
                    "hits":          p.get("hits", 0),
                    "shots":         p.get("sog", 0),
                    "blocked_shots": p.get("blockedShots", 0),
                    "pp_goals":      p.get("powerPlayGoals", 0),
                    "faceoff_pct":   p.get("faceoffWinningPctg"),
                    "toi":           p.get("toi", "0:00"),
                    "shifts":        p.get("shifts", 0),
                    "giveaways":     p.get("giveaways", 0),
                    "takeaways":     p.get("takeaways", 0),
                })

        for p in pbs.get(side, {}).get("goalies", []):
            gl.append({**base,
                "player_id":        p["playerId"],
                "player_name":      p["name"]["default"],
                "position":         "G",
                "sweater":          p.get("sweaterNumber"),
                "saves":            p.get("saves", 0),
                "shots_against":    p.get("shotsAgainst", 0),
                "goals_against":    p.get("goalsAgainst", 0),
                "save_pctg":        p.get("savePctg"),
                "es_goals_against": p.get("evenStrengthGoalsAgainst", 0),
                "pp_goals_against": p.get("powerPlayGoalsAgainst", 0),
                "sh_goals_against": p.get("shorthandedGoalsAgainst", 0),
                "toi":              p.get("toi", "0:00"),
                "decision":         p.get("decision", ""),
            })
    return sk, gl

# ── 4. COLLECT + FLATTEN ───────────────────────────────────────────
def checkpoint():
    if skater_rows:
        pd.DataFrame(skater_rows).to_parquet(SKATER_F, index=False)
    if goalie_rows:
        pd.DataFrame(goalie_rows).to_parquet(GOALIE_F, index=False)
    json.dump(list(done), open(PROG_F, "w"))

t0 = time.time()
errors = []

for i, meta in enumerate(todo):
    pk  = meta["game_pk"]
    url = f"https://api-web.nhle.com/v1/gamecenter/{pk}/boxscore"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            sk, gl = flatten(r.json(), meta)
            skater_rows.extend(sk)
            goalie_rows.extend(gl)
            done.add(pk)
        else:
            errors.append((pk, f"HTTP {r.status_code}"))
    except Exception as e:
        errors.append((pk, str(e)))

    if (i+1) % 100 == 0:
        el = time.time() - t0
        eta = (len(todo)-i-1) / ((i+1)/el) / 60
        print(f"  [{i+1}/{len(todo)}]  skaters={len(skater_rows):,}"
              f"  goalies={len(goalie_rows):,}  ETA {eta:.1f}m")

    if (i+1) % SAVE_EVERY == 0:
        checkpoint()
        print(f"  -- checkpoint saved --")

    time.sleep(DELAY)

# ── 5. FINAL SAVE ─────────────────────────────────────────────────
# Deduplicate in case of interrupted checkpoint resume
if skater_rows:
    df = pd.DataFrame(skater_rows).drop_duplicates(
        subset=["game_pk", "player_id", "team"])
    df.to_parquet(SKATER_F, index=False)
    print(f"\nSkaters: {len(df):,} rows  |  {df['player_id'].nunique()} players")

if goalie_rows:
    df = pd.DataFrame(goalie_rows).drop_duplicates(
        subset=["game_pk", "player_id", "team"])
    df.to_parquet(GOALIE_F, index=False)
    print(f"Goalies: {len(df):,} rows  |  {df['player_id'].nunique()} players")

json.dump(list(done), open(PROG_F, "w"))

elapsed = time.time() - t0
print(f"\n{'='*55}")
print(f"DONE  -  {elapsed/60:.1f} minutes")
print(f"Files: {SKATER_F.name}, {GOALIE_F.name}")
if errors:
    print(f"Errors: {len(errors)} games")
    for pk, err in errors[:5]:
        print(f"  {pk}: {err}")
print(f"{'='*55}")