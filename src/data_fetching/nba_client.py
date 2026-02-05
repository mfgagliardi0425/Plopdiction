import os
from datetime import date
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

BASE_URL = "https://api.sportradar.com/nba"


def build_url(path: str, params: dict | None = None) -> str:
    if params:
        return f"{BASE_URL}{path}?{urlencode(params)}"
    return f"{BASE_URL}{path}"


def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def get_daily_schedule(day: date) -> dict:
    access_level = get_env("SPORTRADAR_ACCESS_LEVEL", "trial")
    language = get_env("SPORTRADAR_LANGUAGE", "en")
    version = get_env("SPORTRADAR_VERSION", "v8")
    api_key = get_env("SPORTRADAR_API_KEY")

    path = f"/{access_level}/{version}/{language}/games/{day.year}/{day.month:02d}/{day.day:02d}/schedule.json"
    url = build_url(path, {"api_key": api_key})

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    load_dotenv()
    today = date.today()
    data = get_daily_schedule(today)
    games = data.get("games", [])
    print(f"{len(games)} games found for {today.isoformat()}")
    if games:
        first = games[0]
        print("Example game id:", first.get("id"))
        print("Away:", first.get("away", {}).get("name"))
        print("Home:", first.get("home", {}).get("name"))
