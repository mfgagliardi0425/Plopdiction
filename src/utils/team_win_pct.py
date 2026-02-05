import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.sportradar.com/nba"
DATA_DIR = Path("data")


@dataclass
class TeamRecord:
    team_id: str
    name: str
    wins: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses

    @property
    def win_pct(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games


def build_url(path: str, params: Optional[dict] = None) -> str:
    if params:
        return f"{BASE_URL}{path}?" + "&".join(f"{k}={v}" for k, v in params.items())
    return f"{BASE_URL}{path}"


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def iter_game_files(data_dir: Path) -> Iterable[Path]:
    if not data_dir.exists():
        return []
    return data_dir.glob("*/**/*.json")


def parse_team_display(team: dict) -> Tuple[str, str]:
    team_id = team.get("id") or "unknown"
    market = team.get("market") or ""
    name = team.get("name") or ""
    display = (market + " " + name).strip() or team.get("alias") or team_id
    return team_id, display


def extract_points(team: dict) -> Optional[int]:
    if "points" in team and isinstance(team["points"], int):
        return team["points"]
    scoring = team.get("scoring")
    if isinstance(scoring, dict) and "points" in scoring:
        return scoring.get("points")
    return None


def build_records(data_dir: Path) -> Dict[str, TeamRecord]:
    records: Dict[str, TeamRecord] = {}

    for file_path in iter_game_files(data_dir):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception:
            continue

        status = (game.get("status") or "").lower()
        if status not in {"closed", "complete", "completed", "final"}:
            continue

        home = game.get("home", {})
        away = game.get("away", {})

        home_id, home_name = parse_team_display(home)
        away_id, away_name = parse_team_display(away)

        home_points = extract_points(home)
        away_points = extract_points(away)

        if home_points is None or away_points is None:
            continue

        if home_id not in records:
            records[home_id] = TeamRecord(team_id=home_id, name=home_name)
        if away_id not in records:
            records[away_id] = TeamRecord(team_id=away_id, name=away_name)

        if home_points > away_points:
            records[home_id].wins += 1
            records[away_id].losses += 1
        elif away_points > home_points:
            records[away_id].wins += 1
            records[home_id].losses += 1

    return records


def get_schedule(day: date, retries: int = 3, backoff: float = 2.0) -> dict:
    access_level = get_env("SPORTRADAR_ACCESS_LEVEL", "trial")
    language = get_env("SPORTRADAR_LANGUAGE", "en")
    version = get_env("SPORTRADAR_VERSION", "v8")
    api_key = get_env("SPORTRADAR_API_KEY")

    path = f"/{access_level}/{version}/{language}/games/{day.year}/{day.month:02d}/{day.day:02d}/schedule.json"
    url = build_url(path, {"api_key": api_key})

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        response = requests.get(url, timeout=30)
        if response.status_code == 429:
            wait = backoff * (attempt + 1)
            print(f"Rate limited when fetching {day.isoformat()} schedule. Waiting {wait:.1f}s...")
            time.sleep(wait)
            last_error = requests.exceptions.HTTPError("429 Too Many Requests")
            continue
        response.raise_for_status()
        return response.json()

    if last_error:
        raise last_error
    raise RuntimeError("Failed to fetch schedule")


def get_upcoming_games(start_date: date, end_date: date) -> List[dict]:
    games: List[dict] = []
    current = start_date
    while current <= end_date:
        data = get_schedule(current)
        day_games = data.get("games", [])
        for game in day_games:
            status = (game.get("status") or "").lower()
            if status in {"scheduled", "created", "inprogress", "delayed"}:
                games.append(game)
        current += timedelta(days=1)
    return games


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_matchups(games: List[dict], records: Dict[str, TeamRecord]) -> None:
    if not games:
        print("No upcoming games found in the selected date range.")
        return

    print("\nUpcoming games with team win % (season to date):\n")
    for game in games:
        scheduled = game.get("scheduled")
        try:
            game_date = datetime.fromisoformat(scheduled.replace("Z", "+00:00")).date()
        except Exception:
            game_date = None

        home = game.get("home", {})
        away = game.get("away", {})
        home_id, home_name = parse_team_display(home)
        away_id, away_name = parse_team_display(away)

        home_record = records.get(home_id)
        away_record = records.get(away_id)

        home_pct = format_pct(home_record.win_pct) if home_record else "N/A"
        away_pct = format_pct(away_record.win_pct) if away_record else "N/A"

        date_label = game_date.isoformat() if game_date else "Unknown date"
        print(f"{date_label}: {away_name} ({away_pct}) @ {home_name} ({home_pct})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate team win % for upcoming NBA games")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_date = date.fromisoformat(args.start) if args.start else date.today()
    end_date = date.fromisoformat(args.end) if args.end else (start_date + timedelta(days=7))

    records = build_records(DATA_DIR)
    if not records:
        print("No completed games found in data/. Download games first.")
        return

    games = get_upcoming_games(start_date, end_date)
    print_matchups(games, records)


if __name__ == "__main__":
    main()
