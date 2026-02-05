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


def probe_game(game_id: str):
    candidates = [
        f"{base_url}/games/{game_id}/summary.json",
        f"{base_url}/games/{game_id}/boxscore.json",
        f"{base_url}/games/{game_id}/pbp.json",
        f"{base_url}/games/{game_id}/roster.json",
        f"{base_url}/games/{game_id}/lineups.json",
        f"{base_url}/games/{game_id}/lineup.json",
        f"{base_url}/games/{game_id}/depth_chart.json",
    ]

    for url in candidates:
        try:
            response = requests.get(url, params={"api_key": api_key}, timeout=30)
            status = response.status_code
            print(f"{status} -> {url}")
            if status == 200:
                data = response.json()
                print(f"  Keys: {list(data.keys())[:10]}")
        except Exception as exc:
            print(f"ERROR -> {url}: {exc}")


def main():
    today = date.today()
    schedule = get_schedule(today)
    games = schedule.get("games", [])
    if not games:
        print("No games scheduled today")
        return

    game_id = games[0].get("id")
    print(f"Probing endpoints for game: {game_id}")
    probe_game(game_id)


if __name__ == "__main__":
    main()
