import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("nhl_data.db")

query = """
SELECT 
    CASE 
        WHEN game_date BETWEEN '2021-07-01' AND '2022-06-30' THEN '2021-22'
        WHEN game_date BETWEEN '2022-07-01' AND '2023-06-30' THEN '2022-23'
        WHEN game_date BETWEEN '2023-07-01' AND '2024-06-30' THEN '2023-24'
        WHEN game_date BETWEEN '2024-07-01' AND '2025-06-30' THEN '2024-25'
    END AS season,
    COUNT(*) AS games,
    AVG(home_score + away_score) AS avg_goals_per_game
FROM games
GROUP BY season
ORDER BY season;
"""

df = pd.read_sql_query(query, conn)
conn.close()

print(df)

plt.figure(figsize=(8,5))
plt.plot(df["season"], df["avg_goals_per_game"], marker="o")
plt.title("Average Goals per Game by Season")
plt.xlabel("Season")
plt.ylabel("Goals per Game")
plt.grid(True)
plt.show()