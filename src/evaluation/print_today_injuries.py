"""Print ESPN injuries for all teams playing today."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_injuries import fetch_today_injuries


def main() -> None:
    teams = fetch_today_injuries()
    if not teams:
        print("No injuries found for today.")
        return

    for team, entries in sorted(teams.items()):
        print(f"\n{team} Status")
        if not entries:
            print("  (none)")
            continue
        for e in entries:
            player = e.get("player") or ""
            status = (e.get("status") or "").upper()
            if not status:
                status = "UNKNOWN"
            print(f"  - {player} [{status}]")


if __name__ == "__main__":
    main()
