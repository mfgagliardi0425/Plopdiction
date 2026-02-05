import os
from datetime import date

import requests
from dotenv import load_dotenv

load_dotenv()

access_level = os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial")
language = os.getenv("SPORTRADAR_LANGUAGE", "en")
version = os.getenv("SPORTRADAR_VERSION", "v8")
api_key = os.getenv("SPORTRADAR_API_KEY")

base_url = f"https://api.sportradar.com/nba/{access_level}/{version}/{language}"


def get_schedule(day: date):
    url = f"{base_url}/games/{day.year}/{day.month:02d}/{day.day:02d}/schedule.json"
    response = requests.get(url, params={"api_key": api_key}, timeout=30)
    response.raise_for_status()
    return response.json()


def get_team_profile(team_id: str):
    url = f"{base_url}/teams/{team_id}/profile.json"
    response = requests.get(url, params={"api_key": api_key}, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    today = date.today()
    schedule = get_schedule(today)
    games = schedule.get("games", [])
    if not games:
        print("No games scheduled today")
        return

    game = games[0]
    home = game.get("home", {})
    away = game.get("away", {})
    home_id = home.get("id")
    away_id = away.get("id")

    print(f"Testing roster endpoints for {away.get('name')} @ {home.get('name')}")

    for team_id, label in [(home_id, "home"), (away_id, "away")]:
        if not team_id:
            continue
        profile = get_team_profile(team_id)
        players = profile.get("players", [])
        active = [p for p in players if p.get("status") == "active"]
        print(f"{label} roster players: {len(players)} | active: {len(active)}")
        if active:
            print("Sample active player:", active[0].get("full_name"))


if __name__ == "__main__":
    main()
