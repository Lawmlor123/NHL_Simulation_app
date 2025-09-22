import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load rosters
# ---------------------------
dal = pd.read_csv("Dallas_roster_2025.csv")
stl = pd.read_csv("StLouis_roster_2025.csv")

dal["Team"] = "Dallas"
stl["Team"] = "St. Louis"

# Combine
rosters = pd.concat([dal, stl])

# ---------------------------
# Position summary
# ---------------------------
pos_summary = rosters.groupby(["Team", "Pos"]).agg(
    Avg_Age=("Age", "mean"),
    Min_Age=("Age", "min"),
    Max_Age=("Age", "max"),
    Count=("Age", "count")
).reset_index()

print("\n--- Position Summary ---")
print(pos_summary)

# ---------------------------
# Age buckets
# ---------------------------
bins = [0, 21, 26, 30, 100]
labels = ["≤21", "22-26", "27-30", "31+"]

rosters["Age_Bucket"] = pd.cut(rosters["Age"], bins=bins, labels=labels, right=True)

bucket_summary = rosters.groupby(["Team", "Pos", "Age_Bucket"]).size().reset_index(name="Count")

print("\n--- Age Buckets ---")
print(bucket_summary)

# ---------------------------
# Export to Excel
# ---------------------------
with pd.ExcelWriter("roster_summary.xlsx") as writer:
    pos_summary.to_excel(writer, sheet_name="Position Summary", index=False)
    bucket_summary.to_excel(writer, sheet_name="Age Buckets", index=False)

print("\n✅ Results exported to roster_summary.xlsx")

# ---------------------------
# Charts
# ---------------------------

# 1. Average Age by Position (DAL vs STL)
plt.figure(figsize=(8,6))
for team in rosters["Team"].unique():
    team_data = pos_summary[pos_summary["Team"] == team]
    plt.bar(team_data["Pos"] + " (" + team[:3] + ")", 
            team_data["Avg_Age"], 
            label=team)
plt.title("Average Age by Position")
plt.ylabel("Average Age")
plt.legend()
plt.tight_layout()
plt.savefig("avg_age_by_position.png")
plt.show()

# 2. Age Buckets by Team/Position (stacked bars)
pivot_data = bucket_summary.pivot_table(index=["Team","Pos"], 
                                        columns="Age_Bucket", 
                                        values="Count", 
                                        fill_value=0)
pivot_data.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Roster Age Buckets by Position")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.savefig("age_buckets.png")
plt.show()