import csv, json

with open("NHL_Teams_Logos.csv", newline='', encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    # normalize the header
    fieldnames = [h.strip().lower() for h in reader.fieldnames]

    # figure out which columns are team name and URL
    team_key = None
    url_key = None
    for h in fieldnames:
        if "team" in h: team_key = h
        if "logo" in h: url_key = h

    logos = {}
    for row in reader:
        team = row[reader.fieldnames[0]].strip()
        url = row[reader.fieldnames[1]].strip()
        logos[team] = url

with open("NHL_Teams_Logos.json", "w", encoding="utf-8") as f:
    json.dump(logos, f, indent=2)

print("✅ Done! File saved as NHL_Teams_Logos.json")