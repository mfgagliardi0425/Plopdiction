"""
Send a Discord alert 1 hour before the first NBA game of the day.
Run this script on a schedule (e.g., every 15 minutes).
"""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from evaluation.discord_notifier import send_discord_message

ALERT_STATE_DIR = Path("tracking/alerts")
ALERT_STATE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_espn_scoreboard(game_date: date) -> dict:
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        f"?dates={game_date.strftime('%Y%m%d')}"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def get_first_game_time_utc(scoreboard: dict) -> datetime | None:
    times = []
    for event in scoreboard.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        start_time = competitions[0].get("date")
        if not start_time:
            continue
        try:
            times.append(datetime.fromisoformat(start_time.replace("Z", "+00:00")))
        except Exception:
            continue
    return min(times) if times else None


def alert_already_sent(game_date: date) -> bool:
    return (ALERT_STATE_DIR / f"first_game_alert_{game_date.isoformat()}.json").exists()


def mark_alert_sent(game_date: date, payload: dict) -> None:
    path = ALERT_STATE_DIR / f"first_game_alert_{game_date.isoformat()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    today = date.today()
    if alert_already_sent(today):
        return

    scoreboard = fetch_espn_scoreboard(today)
    first_game_time = get_first_game_time_utc(scoreboard)
    if not first_game_time:
        return

    now_utc = datetime.now(timezone.utc)
    alert_time = first_game_time - timedelta(hours=1)

    if now_utc < alert_time:
        return

    message = (
        f"NBA first game alert: first tip is at {first_game_time.strftime('%H:%M UTC')}\n"
        f"This message was sent 1 hour before tip."
    )

    sent = send_discord_message(message, username="NBA Alerts")
    if sent:
        mark_alert_sent(today, {
            "sent_at": now_utc.isoformat(),
            "first_game_time": first_game_time.isoformat(),
        })


if __name__ == "__main__":
    main()
