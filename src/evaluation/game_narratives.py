"""Compute game narrative metrics (blown leads, clutch scoring) from play-by-play logs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.matchup_model import DATA_DIR, parse_game_date

NARRATIVE_DIR = Path("tracking/narratives")
NARRATIVE_DIR.mkdir(parents=True, exist_ok=True)

CLUTCH_SECONDS = 5 * 60  # last 5 minutes of 4Q
BLOWN_LEAD_THRESHOLD = 10


@dataclass
class GameNarrative:
    game: str
    home_team: str
    away_team: str
    home_points: int
    away_points: int
    max_home_lead: int
    max_away_lead: int
    blown_lead_team: Optional[str]
    clutch_home_points: int
    clutch_away_points: int
    clutch_margin: int


def _parse_clock(clock: str) -> Optional[int]:
    if not clock or ":" not in clock:
        return None
    try:
        mins, secs = clock.split(":")
        return int(mins) * 60 + int(secs)
    except Exception:
        return None


def _iter_game_files(start_date: date, end_date: date) -> List[Path]:
    files: List[Path] = []
    current = start_date
    while current <= end_date:
        day_dir = Path(DATA_DIR) / current.isoformat()
        if day_dir.exists():
            files.extend(day_dir.glob("*.json"))
        current += timedelta(days=1)
    return files


def _extract_game_meta(game: dict) -> Tuple[str, str, str, int, int]:
    home = game.get("home", {})
    away = game.get("away", {})
    home_name = f"{home.get('market','')} {home.get('name','')}".strip()
    away_name = f"{away.get('market','')} {away.get('name','')}".strip()
    home_points = int(home.get("points") or 0)
    away_points = int(away.get("points") or 0)
    return f"{away_name} @ {home_name}", home_name, away_name, home_points, away_points


def compute_narrative_stats(game: dict) -> Optional[dict]:
    status = (game.get("status") or "").lower()
    if status not in {"closed", "complete", "completed", "final"}:
        return None

    game_date = parse_game_date(game)
    if not game_date:
        return None

    game_key, home_name, away_name, home_points, away_points = _extract_game_meta(game)
    periods = game.get("periods", [])
    if not periods:
        return None

    max_home_lead = 0
    max_away_lead = 0
    clutch_home_points = 0
    clutch_away_points = 0

    last_home = 0
    last_away = 0

    for period in periods:
        events = period.get("events", [])
        if not events:
            continue
        for evt in events:
            home_pts = evt.get("home_points")
            away_pts = evt.get("away_points")
            if home_pts is None or away_pts is None:
                continue
            try:
                home_pts = int(home_pts)
                away_pts = int(away_pts)
            except Exception:
                continue

            lead = home_pts - away_pts
            if lead > max_home_lead:
                max_home_lead = lead
            if -lead > max_away_lead:
                max_away_lead = -lead

            # clutch: last 5 minutes of 4th quarter
            if period.get("type") == "quarter" and period.get("number") == 4:
                seconds = _parse_clock(evt.get("clock"))
                if seconds is not None and seconds <= CLUTCH_SECONDS:
                    if home_pts > last_home:
                        clutch_home_points += home_pts - last_home
                    if away_pts > last_away:
                        clutch_away_points += away_pts - last_away

            last_home = home_pts
            last_away = away_pts

    final_margin = home_points - away_points
    winner = home_name if final_margin > 0 else away_name if final_margin < 0 else None

    blown_lead_team = None
    if max_home_lead >= BLOWN_LEAD_THRESHOLD and winner == away_name:
        blown_lead_team = home_name
    if max_away_lead >= BLOWN_LEAD_THRESHOLD and winner == home_name:
        blown_lead_team = away_name

    return {
        "game": game_key,
        "game_date": game_date,
        "home_team": home_name,
        "away_team": away_name,
        "home_points": home_points,
        "away_points": away_points,
        "max_home_lead": max_home_lead,
        "max_away_lead": max_away_lead,
        "blown_lead_team": blown_lead_team,
        "blown_lead_side": "home" if blown_lead_team == home_name else "away" if blown_lead_team == away_name else None,
        "clutch_home_points": clutch_home_points,
        "clutch_away_points": clutch_away_points,
        "clutch_margin": clutch_home_points - clutch_away_points,
    }


def build_narratives(start_date: date, end_date: date) -> Dict[str, dict]:
    narratives = {}
    for file_path in _iter_game_files(start_date, end_date):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception:
            continue

        narrative = compute_narrative_stats(game)
        if not narrative:
            continue
        narratives[narrative["game"]] = {
            "game": narrative["game"],
            "home_team": narrative["home_team"],
            "away_team": narrative["away_team"],
            "home_points": narrative["home_points"],
            "away_points": narrative["away_points"],
            "max_home_lead": narrative["max_home_lead"],
            "max_away_lead": narrative["max_away_lead"],
            "blown_lead_team": narrative["blown_lead_team"],
            "clutch_home_points": narrative["clutch_home_points"],
            "clutch_away_points": narrative["clutch_away_points"],
            "clutch_margin": narrative["clutch_margin"],
        }

    return narratives


def save_narratives(target_date: date, narratives: Dict[str, dict]) -> Path:
    path = NARRATIVE_DIR / f"{target_date.isoformat()}_narratives.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": target_date.isoformat(), "games": narratives}, f, indent=2)
    return path
