"""
Daily pipeline:
- Ingest yesterday's results
- Scrape ESPN closing spreads for yesterday
- Rebuild dataset
- Retrain model
- Evaluate ATS thresholds
- Print tonight's predictions summary
"""
import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.daily_update import main as daily_update_main
from ml.edge_threshold_eval import main as edge_eval_main
from evaluation.tonight_spread_predictions_summary import format_summary, generate_rows
from evaluation.discord_notifier import send_discord_message
from evaluation.update_espn_injuries import main as update_espn_injuries_main
from evaluation.live_ats_tracking import save_predictions, evaluate_date


def parse_args():
    parser = argparse.ArgumentParser(description="Run daily end-to-end pipeline")
    parser.add_argument("--date", help="Date to ingest (YYYY-MM-DD), defaults to yesterday")
    return parser.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=1))

    # Run daily update (ingest + scrape + rebuild + retrain)
    sys.argv = ["daily_update", "--date", target_date.isoformat(), "--rebuild-dataset", "--retrain"]
    daily_update_main()

    # Evaluate ATS thresholds on current model
    sys.argv = ["edge_threshold_eval"]
    edge_eval_main()

    # Update ESPN injuries for today's games
    update_espn_injuries_main()

    # Print tonight's predictions summary and notify Discord
    target_today = date.today()
    rows, _ = generate_rows(target_today, "ml_data/best_model_with_spreads.joblib")
    if rows:
        save_predictions(target_today, rows)
        message, _ = format_summary(target_today, rows)
        print("\n" + message + "\n")
        send_discord_message(message)
    else:
        print("\nNo games with odds and features found for tonight.\n")

    # Evaluate yesterday's live ATS performance
    eval_summary = evaluate_date(target_date)
    print(f"Live ATS evaluation for {target_date.isoformat()}: {eval_summary}")


if __name__ == "__main__":
    main()
