"""Fetch NBA official injury report data from stats.nba.com."""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

CACHE_PATH = Path("odds_cache/nba_injury_report.json")
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

URL_CANDIDATES = [
    "https://stats.nba.com/js/data/injury/leagueinjuryreport.json",
    "https://stats.nba.com/js/data/league/2025/leagueinjuryreport.json",
    "https://stats.nba.com/js/data/league/2026/leagueinjuryreport.json",
    "https://stats.nba.com/stats/injuryreport?LeagueID=00",
]


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


def _save_cache(payload: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_rows(data: dict) -> List[dict]:
    result_sets = data.get("resultSets") or []
    if result_sets:
        headers = result_sets[0].get("headers", [])
        rows = result_sets[0].get("rowSet", [])
    else:
        return []

    def idx(name: str) -> int:
        return headers.index(name)

    try:
        i_team = idx("TEAM_NAME")
        i_player = idx("PLAYER_NAME")
        i_status = idx("STATUS") if "STATUS" in headers else idx("INJURY_STATUS")
        i_reason = idx("COMMENT") if "COMMENT" in headers else idx("INJURY_COMMENT")
    except Exception:
        return []

    parsed = []
    for row in rows:
        parsed.append({
            "team": row[i_team],
            "player": row[i_player],
            "status": row[i_status],
            "comment": row[i_reason],
        })
    return parsed


def fetch_injury_report() -> List[dict]:
    cached = _load_cache()
    if cached and "rows" in cached:
        return cached["rows"]

    for url in URL_CANDIDATES:
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()
            rows = _parse_rows(data)
            if rows:
                _save_cache({"rows": rows})
                return rows
        except Exception:
            continue

    return []


def build_team_injury_index(rows: List[dict]) -> Dict[str, List[dict]]:
    team_map: Dict[str, List[dict]] = {}
    for r in rows:
        team_map.setdefault(r["team"], []).append(r)
    return team_map
