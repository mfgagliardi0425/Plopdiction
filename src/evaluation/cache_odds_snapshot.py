"""Cache a timestamped Odds API snapshot for opening line tracking."""
import argparse
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.odds_api import cache_odds_snapshot, get_nba_odds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a timestamped odds snapshot")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    odds_data = get_nba_odds(markets="spreads")
    cache_path = cache_odds_snapshot(odds_data, target_date.isoformat())
    print(f"Saved odds snapshot: {cache_path}")


if __name__ == "__main__":
    main()
