"""Send tonight's prediction summary to Discord immediately."""
import argparse
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.tonight_spread_predictions_summary import format_summary, generate_rows
from evaluation.discord_notifier import send_discord_message
from evaluation.live_ats_tracking import save_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Send tonight's prediction summary to Discord")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    rows, _ = generate_rows(target_date, "ml_data/best_model_with_spreads.joblib", use_cached_dataset=args.fast)
    if not rows:
        print("No games with odds and features found for tonight.")
        return

    save_predictions(target_date, rows)
    message, _ = format_summary(target_date, rows)
    sent = send_discord_message(message, username="NBA Predictions")
    if sent:
        print("Sent Discord message.")
    else:
        print("Discord message not sent.")


if __name__ == "__main__":
    main()
