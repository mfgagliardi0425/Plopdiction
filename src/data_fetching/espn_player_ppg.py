"""Fetch player PPG from ESPN core API using athlete IDs."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

CACHE_PATH = Path("odds_cache/espn_player_ppg.json")
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(payload: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _get_cache_entry(cache: dict, season_year: int, athlete_id: int) -> Optional[float]:
    season_key = str(season_year)
    if season_key not in cache:
        return None
    entry = cache[season_key].get(str(athlete_id))
    if not entry:
        return None
    ts = entry.get("updated_at")
    if ts:
        try:
            updated = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if (datetime.utcnow() - updated.replace(tzinfo=None)).total_seconds() > CACHE_TTL_SECONDS:
                return None
        except Exception:
            return None
    return entry.get("ppg")


def _set_cache_entry(cache: dict, season_year: int, athlete_id: int, ppg: float) -> None:
    season_key = str(season_year)
    cache.setdefault(season_key, {})[str(athlete_id)] = {
        "ppg": ppg,
        "updated_at": _now_iso(),
    }


def fetch_player_ppg(athlete_id: int, season_year: int) -> Optional[float]:
    cache = _load_cache()
    cached = _get_cache_entry(cache, season_year, athlete_id)
    if cached is not None:
        return cached

    url = (
        "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
        f"seasons/{season_year}/types/2/athletes/{athlete_id}/statistics/0"
        "?lang=en&region=us"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    splits = data.get("splits", {})
    categories = splits.get("categories", [])
    ppg = None
    for cat in categories:
        for stat in cat.get("stats", []):
            if stat.get("name") == "avgPoints":
                ppg = stat.get("value")
                break
        if ppg is not None:
            break

    if ppg is None:
        return None

    _set_cache_entry(cache, season_year, athlete_id, float(ppg))
    _save_cache(cache)
    return float(ppg)


def fetch_player_ppg_bulk(athlete_ids: list[int], season_year: int, delay_seconds: float = 0.15) -> dict[int, float]:
    """Fetch PPG for multiple athletes with cache reuse."""
    cache = _load_cache()
    results: dict[int, float] = {}
    missing: list[int] = []

    for athlete_id in athlete_ids:
        cached = _get_cache_entry(cache, season_year, athlete_id)
        if cached is not None:
            results[athlete_id] = float(cached)
        else:
            missing.append(athlete_id)

    for athlete_id in missing:
        ppg = fetch_player_ppg(athlete_id, season_year)
        if ppg is not None:
            results[athlete_id] = float(ppg)
        if delay_seconds:
            time.sleep(delay_seconds)

    return results
