"""Schedule a one-time task for 1 hour before the first game."""
import argparse
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_api import get_espn_games_for_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule 1-hour-before-first-game task")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--task-name", default="NBA Pregame Model 1hr")
    default_cmd = Path(__file__).resolve().parents[2] / "scripts" / "run_pregame_model.cmd"
    parser.add_argument("--cmd-path", default=str(default_cmd))
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


def _run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    cmd_path = args.cmd_path
    first_game = _first_game_time_et(target_date)
    if not first_game:
        print("No games found for today.")
        return

    run_time = first_game - timedelta(hours=1)
    now = datetime.now(ZoneInfo("America/New_York"))
    if run_time <= now:
        print("Run time already passed. Running model now.")
        result = _run_cmd([cmd_path])
        if result.returncode != 0:
            print(result.stderr.strip() or "Failed to run pregame model.")
        return

    task_name = args.task_name
    run_date = run_time.date()
    time_str = run_time.strftime("%H:%M")
    date_str = run_date.strftime("%m/%d/%Y")

    _run_cmd(["schtasks", "/Delete", "/TN", task_name, "/F"])
    result = _run_cmd([
        "schtasks",
        "/Create",
        "/F",
        "/TN",
        task_name,
        "/TR",
        f'"{cmd_path}"',
        "/SC",
        "ONCE",
        "/ST",
        time_str,
        "/SD",
        date_str,
    ])

    if result.returncode != 0:
        print(result.stderr.strip() or "Failed to create one-time task.")
        return

    print(f"Scheduled {task_name} at {date_str} {time_str}.")


if __name__ == "__main__":
    main()
