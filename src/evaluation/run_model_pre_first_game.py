"""Run predictions one hour before the first game of the day."""
import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
import os
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_api import get_espn_games_for_date
from evaluation.discord_notifier import send_discord_message
from evaluation.live_ats_tracking import save_predictions
from evaluation.tonight_spread_predictions_summary import format_summary, generate_rows


TRACK_DIR = Path("tracking/automation")
TRACK_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model one hour before first game")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--model", default="ml_data/best_model_with_spreads.joblib")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    parser.add_argument("--send-discord", action="store_true", help="Send summary to Discord")
    return parser.parse_args()


def _first_game_time_et(target_date: date) -> datetime | None:
    games = get_espn_games_for_date(target_date)
    if not games:
        return None
    start_times = []
    for game in games:
        start_time = game.get("start_time_utc")
        if not start_time:
            continue
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        start_times.append(start_dt.astimezone(ZoneInfo("America/New_York")))
    return min(start_times) if start_times else None


def _load_marker(marker_path: Path) -> dict:
    if not marker_path.exists():
        return {}
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_marker(marker_path: Path, payload: dict) -> None:
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    marker_path = TRACK_DIR / f"model_pre_game_{target_date.isoformat()}.json"

    first_game = _first_game_time_et(target_date)
    if not first_game:
        print("No games found for today.")
        return

    run_time = first_game - timedelta(hours=1)
    now = datetime.now(ZoneInfo("America/New_York"))

    marker = _load_marker(marker_path)
    if marker.get("ran"):
        print("Model already ran for this date.")
        return

    if now < run_time:
        print(f"Too early. First game at {first_game.isoformat()}, run at {run_time.isoformat()} ET.")
        return

    rows, _ = generate_rows(target_date, args.model, use_cached_dataset=args.fast)
    if not rows:
        print("No games with odds and features found.")
        return

    save_predictions(target_date, rows)
    summary, _ = format_summary(target_date, rows, model_path=args.model)
    print(summary)

    if args.send_discord:
        webhook = os.getenv("DISCORD_WEBHOOK_URL_DAILY_PREDICTIONS")
        if not webhook:
            print("DISCORD_WEBHOOK_URL_DAILY_PREDICTIONS not set.")
        else:
            sent = send_discord_message(summary, username="NBA Predictions", webhook_url=webhook)
            if sent:
                print("Sent Discord message.")
            else:
                print("Discord message not sent.")

    _save_marker(marker_path, {
        "ran": True,
        "ran_at": now.isoformat(),
        "first_game_et": first_game.isoformat(),
        "run_time_et": run_time.isoformat(),
    })


if __name__ == "__main__":
    main()
