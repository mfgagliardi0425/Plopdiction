"""Send tonight's prediction summary to Discord immediately."""
import argparse
from datetime import date
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.tonight_spread_predictions_summary import (
    format_prediction_narrative,
    format_summary,
    format_pick_rationale,
    generate_rows,
)
from evaluation.discord_notifier import send_discord_message
from evaluation.live_ats_tracking import save_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Send tonight's prediction summary to Discord")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--model", default="ml_data/best_model_with_spreads.joblib")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    parser.add_argument("--preview", action="store_true", help="Print summary and rationale before sending")
    parser.add_argument(
        "--min-start-est",
        default=None,
        help="Only include games at/after this EST time (HH:MM)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    rows, _ = generate_rows(
        target_date,
        args.model,
        use_cached_dataset=args.fast,
        min_start_est=args.min_start_est,
    )
    if not rows:
        print("No games with odds and features found for tonight.")
        return

    save_predictions(target_date, rows)
    narrative_text = format_prediction_narrative(target_date, rows)
    narrative_dir = Path("tracking/narratives")
    narrative_dir.mkdir(parents=True, exist_ok=True)
    narrative_path = narrative_dir / f"{target_date.isoformat()}_prediction_narrative.txt"
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write(narrative_text)
    message, _ = format_summary(target_date, rows, model_path=args.model)
    rationale = format_pick_rationale(rows)
    if args.preview:
        print(message)
        print("")
        print(rationale)
        print("")
        print(f"Saved narrative summary to {narrative_path}")

    webhook = os.getenv("DISCORD_WEBHOOK_URL_DAILY_PREDICTIONS")
    if not webhook:
        print("DISCORD_WEBHOOK_URL_DAILY_PREDICTIONS not set.")
        return

    sent = send_discord_message(message, username="NBA Predictions", webhook_url=webhook)
    if sent:
        print("Sent Discord message.")
    else:
        print("Discord message not sent.")


if __name__ == "__main__":
    main()
