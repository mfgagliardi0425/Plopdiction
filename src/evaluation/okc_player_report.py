"""Print Oklahoma City Thunder player availability for today's game."""
import os
from datetime import date
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

access_level = os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial")
language = os.getenv("SPORTRADAR_LANGUAGE", "en")
version = os.getenv("SPORTRADAR_VERSION", "v8")
api_key = os.getenv("SPORTRADAR_API_KEY")

base_url = f"https://api.sportradar.com/nba/{access_level}/{version}/{language}"


def get_schedule(day: date) -> dict:
    url = f"{base_url}/games/{day.year}/{day.month:02d}/{day.day:02d}/schedule.json"
    response = requests.get(url, params={"api_key": api_key}, timeout=30)
    response.raise_for_status()
    return response.json()


def get_team_profile(team_id: str) -> dict:
    url = f"{base_url}/teams/{team_id}/profile.json"
    response = requests.get(url, params={"api_key": api_key}, timeout=30)
    response.raise_for_status()
    return response.json()


def find_team_id(schedule: dict) -> Optional[str]:
    targets = {"oklahoma city", "thunder", "okc"}
    for game in schedule.get("games", []):
        for side in ("home", "away"):
            team = game.get(side, {})
            name = (team.get("name") or "").lower()
            market = (team.get("market") or "").lower()
            alias = (team.get("alias") or "").lower()
            if name in targets or market in targets or alias in targets:
                return team.get("id")
    return None


def main() -> None:
    today = date.today()
    schedule = get_schedule(today)

    team_id = find_team_id(schedule)
    if not team_id:
        print("Could not find OKC on today's schedule.")
        return

    profile = get_team_profile(team_id)
    players = profile.get("players", [])
    def is_active(status: str) -> bool:
        return (status or "").lower() in {"active", "act"}

    active = [p for p in players if is_active(p.get("status"))]
    inactive = [p for p in players if not is_active(p.get("status"))]

    print("Oklahoma City Thunder - Player Report (SportRadar)")
    print(f"Active ({len(active)}):")
    for p in active:
        print(f"  {p.get('full_name')} - {p.get('position')} - {p.get('status')}")

    print(f"\nInactive ({len(inactive)}):")
    for p in inactive:
        print(f"  {p.get('full_name')} - {p.get('position')} - {p.get('status')}")


if __name__ == "__main__":
    main()
