import json
from collections import Counter

with open("odds_cache/espn_closing_spreads_detailed.json", "r", encoding="utf-8") as f:
    detailed = json.load(f)

entries = [v for v in detailed.values() if v.get("date") == "2025-12-02"]
print("Entries for 2025-12-02:", len(entries))
print(entries)
