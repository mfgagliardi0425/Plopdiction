"""Fetch team injury reports from ESPN injuries endpoint."""
import json
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import requests

CACHE_PATH = Path("odds_cache/espn_injuries.json")
CACHE_TTL_SECONDS = 60 * 60  # 1 hour

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.espn.com/",
}

INJURY_URL = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        if time.time() - CACHE_PATH.stat().st_mtime > CACHE_TTL_SECONDS:
            return {}
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(payload: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def fetch_today_injuries(target_date: date | None = None, force_refresh: bool = False) -> Dict[str, List[dict]]:
    """Fetch injuries for teams playing on target_date using ESPN injuries endpoint."""
    if target_date is None:
        target_date = date.today()

    if not force_refresh:
        cached = _load_cache()
        if cached and cached.get("date") == target_date.isoformat():
            teams_cached = cached.get("teams", {})
            total_cached = sum(len(v) for v in teams_cached.values())
            if total_cached > 0:
                return teams_cached

    scoreboard_url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        f"?dates={target_date.strftime('%Y%m%d')}"
    )
    response = requests.get(scoreboard_url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    scoreboard = response.json()

    playing_teams = set()
    for event in scoreboard.get("events", []):
        for comp in event.get("competitions", []):
            for competitor in comp.get("competitors", []):
                team = competitor.get("team", {})
                name = team.get("displayName") or team.get("name")
                if name:
                    playing_teams.add(name)

    injuries_resp = requests.get(INJURY_URL, headers=HEADERS, timeout=20)
    injuries_resp.raise_for_status()
    data = injuries_resp.json()

    teams: Dict[str, List[dict]] = {}
    for team in data.get("injuries", []):
        team_name = team.get("displayName")
        team_id = team.get("id")
        if not team_name or team_name not in playing_teams:
            continue
        rows = []
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete", {})
            athlete_id = None
            for link in athlete.get("links", []) or []:
                href = link.get("href") or ""
                if "/id/" in href:
                    try:
                        athlete_id = int(href.split("/id/")[-1].split("/")[0])
                        break
                    except ValueError:
                        continue
            rows.append({
                "player": athlete.get("displayName"),
                "athlete_id": athlete_id,
                "team_id": team_id,
                "position": athlete.get("position", {}).get("abbreviation"),
                "status": inj.get("status"),
                "comment": inj.get("longComment") or inj.get("shortComment"),
            })
        teams[team_name] = rows

    payload = {"date": target_date.isoformat(), "teams": teams}
    _save_cache(payload)
    return teams
