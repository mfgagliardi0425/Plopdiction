"""Build and maintain a local ESPN player ID database."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests

DATA_PATH = Path("data/players/espn_player_ids.json")
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize(name: str) -> str:
    return " ".join("".join(ch for ch in (name or "") if ch.isalnum() or ch.isspace()).lower().split())


def load_player_id_db() -> dict:
    if not DATA_PATH.exists():
        return {}
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_player_id_db(payload: dict) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _fetch_team_ids(season_year: int) -> List[int]:
    url = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/{season_year}/teams?lang=en&region=us"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    items = data.get("items", [])
    team_ids = []
    for item in items:
        ref = item.get("$ref") or ""
        if "/teams/" in ref:
            try:
                team_ids.append(int(ref.split("/teams/")[-1].split("?")[0].split("/")[0]))
            except ValueError:
                continue
    return team_ids


def _fetch_team_athletes(season_year: int, team_id: int) -> List[dict]:
    url = (
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/{season_year}/"
        f"teams/{team_id}/athletes?lang=en&region=us"
    )
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    items = data.get("items", [])
    players = []
    for item in items:
        ref = item.get("$ref") or ""
        if "/athletes/" in ref:
            try:
                athlete_id = int(ref.split("/athletes/")[-1].split("?")[0].split("/")[0])
            except ValueError:
                continue
            athlete = requests.get(ref, headers=HEADERS, timeout=30).json()
            name = athlete.get("displayName") or athlete.get("fullName")
            if not name:
                continue
            players.append({
                "athlete_id": athlete_id,
                "name": name,
                "name_norm": _normalize(name),
                "team_id": team_id,
                "team": athlete.get("team", {}).get("displayName"),
            })
    return players


def build_player_id_db(season_year: int = 2026) -> dict:
    team_ids = _fetch_team_ids(season_year)
    players: Dict[str, dict] = {}

    for team_id in team_ids:
        for p in _fetch_team_athletes(season_year, team_id):
            key = p["name_norm"]
            if key and key not in players:
                players[key] = p

    payload = {
        "season_year": season_year,
        "last_updated": _now_iso(),
        "players": players,
    }
    save_player_id_db(payload)
    return payload


def lookup_athlete_id(name: str, db: dict | None = None) -> int | None:
    if db is None:
        db = load_player_id_db()
    players = db.get("players", {}) if isinstance(db, dict) else {}
    return (players.get(_normalize(name), {}) or {}).get("athlete_id")
