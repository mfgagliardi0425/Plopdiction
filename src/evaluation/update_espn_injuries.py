"""Fetch and cache ESPN injury reports for today's games."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_injuries import fetch_today_injuries


def main() -> None:
    teams = fetch_today_injuries()
    total = sum(len(v) for v in teams.values())
    print(f"Cached injuries for {len(teams)} teams. Total entries: {total}.")
    for team, entries in teams.items():
        if entries:
            print(f"  {team}: {len(entries)}")


if __name__ == "__main__":
    main()
