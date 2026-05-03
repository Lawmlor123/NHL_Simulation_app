from pathlib import Path
import pandas as pd
from aggregate_player_strength import load_collector_games

collector_path = Path(r"C:\Users\shell\OneDrive\Documents\NHL_Player\Outputs\nhl_2022_2025.json")
games_df = load_collector_games(collector_path)

games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.strftime("%Y-%m-%d")
games_df["game_pk"] = games_df["game_pk"].astype("int64")

games_df.to_parquet(r"..\Features\game_history_2022_2025.parquet", index=False)