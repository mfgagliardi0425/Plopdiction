import json
import pandas as pd
import re


df = pd.read_csv("ml_data/games_optimized.csv")
with open("odds_cache/espn_closing_spreads_detailed.json", "r", encoding="utf-8") as f:
    detailed = json.load(f)


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\s]", "", (name or "")).lower()).strip()

matches = 0
for _, row in df.iterrows():
    date = row["game_date"]
    home = normalize(row["home_team"])
    away = normalize(row["away_team"])
    found = False
    for entry in detailed.values():
        if entry.get("date") != date:
            continue
        eh = normalize(entry.get("home_team"))
        ea = normalize(entry.get("away_team"))
        if (eh in home or home in eh) and (ea in away or away in ea):
            found = True
            break
    if found:
        matches += 1

print("matches", matches)
