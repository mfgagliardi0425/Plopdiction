"""Fetch current season NBA player stats (PPG) from stats.nba.com."""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

STATS_URL = "https://stats.nba.com/stats/leaguedashplayerstats"
CACHE_PATH = Path("odds_cache/nba_player_stats_ppg.json")
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _load_cache() -> Optional[dict]:
    if not CACHE_PATH.exists():
        return None
    try:
        if time.time() - CACHE_PATH.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_cache_any() -> Optional[dict]:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(payload: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fetch_player_ppg(season: str = "2025-26") -> List[dict]:
    cached = _load_cache()
    if cached and cached.get("season") == season:
        return cached.get("players", [])

    params = {
        "Season": season,
        "SeasonType": "Regular Season",
        "PerMode": "PerGame",
        "LeagueID": "00",
        "MeasureType": "Base",
        "PlusMinus": "N",
        "PaceAdjust": "N",
        "Rank": "N",
        "Outcome": "",
        "Location": "",
        "Month": "0",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "OpponentTeamID": "0",
        "VsConference": "",
        "VsDivision": "",
        "GameSegment": "",
        "Period": "0",
        "LastNGames": "0",
        "PORound": "0",
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(STATS_URL, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        fallback = _load_cache_any()
        if fallback and fallback.get("players"):
            return fallback.get("players", [])
        return []

    result_sets = data.get("resultSets") or []
    if not result_sets:
        fallback = _load_cache_any()
        if fallback and fallback.get("players"):
            return fallback.get("players", [])
        return []

    headers = result_sets[0].get("headers", [])
    rows = result_sets[0].get("rowSet", [])

    try:
        idx_name = headers.index("PLAYER_NAME")
        idx_team = headers.index("TEAM_NAME")
        idx_ppg = headers.index("PTS")
    except ValueError:
        return []

    players = []
    for row in rows:
        players.append({
            "player": row[idx_name],
            "team": row[idx_team],
            "ppg": float(row[idx_ppg]),
        })

    _save_cache({"season": season, "players": players})
    return players


def build_team_ppg_index(players: List[dict]) -> Dict[str, List[dict]]:
    team_map: Dict[str, List[dict]] = {}
    for p in players:
        team_map.setdefault(p["team"], []).append(p)

    for team, plist in team_map.items():
        plist.sort(key=lambda x: x["ppg"], reverse=True)
        team_map[team] = plist

    return team_map
