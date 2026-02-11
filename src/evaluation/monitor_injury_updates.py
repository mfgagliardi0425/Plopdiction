"""Monitor injury updates and rerun tonight's model if affected teams play today."""
import argparse
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_api import get_espn_games_for_date
from data_fetching.espn_injuries import fetch_today_injuries
from data_fetching.injury_dataset import load_dataset, update_injury_dataset
from evaluation.tonight_spread_predictions_summary import format_summary, generate_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor ESPN injury updates")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date to monitor (YYYY-MM-DD)")
    parser.add_argument("--season", default="2025-26", help="Season for player stats")
    parser.add_argument("--model", default="ml_data/best_model_with_spreads.joblib")
    parser.add_argument("--force", action="store_true", help="Bypass ESPN injuries cache")
    parser.add_argument("--rerun", action="store_true", help="Rerun predictions if affected team plays today")
    parser.add_argument("--fast", action="store_true", help="Skip PPG refresh if injuries are unchanged")
    return parser.parse_args()


def build_injury_signature(teams: dict) -> dict:
    sig = {}
    for team, entries in teams.items():
        key = sorted((e.get("player"), e.get("status"), e.get("comment")) for e in entries)
        sig[team] = key
    return sig


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    if args.fast:
        current_injuries = fetch_today_injuries(target_date, force_refresh=args.force)
        existing = load_dataset()
        existing_teams = existing.get("teams", {}) if isinstance(existing, dict) else {}
        if build_injury_signature(current_injuries) == build_injury_signature(existing_teams):
            print("No injury changes detected (fast mode). Skipping PPG refresh.")
            return

    dataset, changes = update_injury_dataset(target_date, season=args.season, force_refresh=args.force)
    changed_teams = set(changes.get("teams", []))

    if not changed_teams:
        print("No injury changes detected.")
        return

    print("Injury updates detected:")
    for item in changes.get("players", []):
        print(f"  - {item}")

    if not args.rerun:
        return

    games = get_espn_games_for_date(target_date)
    todays_teams = {g["home_team"] for g in games} | {g["away_team"] for g in games}

    if not (changed_teams & todays_teams):
        print("No updated teams play today. Skipping rerun.")
        return

    rows, _ = generate_rows(target_date, args.model)
    text, _ = format_summary(target_date, rows, model_path=args.model)
    print("\n" + text)


if __name__ == "__main__":
    main()
