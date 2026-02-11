import argparse
from datetime import date, timedelta
from typing import Iterable

import sys

sys.path.append("src")

from evaluation.send_results_to_discord import send_results_for_date


def _iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ATS results and send to Discord")
    parser.add_argument("--date", help="Date (YYYY-MM-DD); defaults to yesterday")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date and (args.start or args.end):
        raise ValueError("Use --date or --start/--end, not both")

    if args.start or args.end:
        if not args.start or not args.end:
            raise ValueError("Both --start and --end are required")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        if end < start:
            raise ValueError("end date must be on or after start date")
        for day in _iter_dates(start, end):
            send_results_for_date(day)
        return

    target_date = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=1))
    send_results_for_date(target_date)


if __name__ == "__main__":
    main()
