"""Build or refresh the ESPN player ID database."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_player_ids import build_player_id_db


def parse_args():
    parser = argparse.ArgumentParser(description="Update ESPN player ID database")
    parser.add_argument("--season", type=int, default=2026, help="Season year (e.g., 2026 for 2025-26)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_player_id_db(args.season)
    print(f"Saved player ID DB for season {payload.get('season_year')} with {len(payload.get('players', {}))} players.")


if __name__ == "__main__":
    main()
