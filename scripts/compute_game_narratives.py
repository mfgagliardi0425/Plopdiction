import argparse
from datetime import date

from evaluation.game_narratives import build_narratives, save_narratives


def parse_args():
    parser = argparse.ArgumentParser(description="Compute game narratives (blown leads, clutch scoring)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    narratives = build_narratives(start_date, end_date)
    if start_date == end_date:
        save_narratives(start_date, narratives)
    print(f"Computed narratives for {len(narratives)} games.")


if __name__ == "__main__":
    main()
