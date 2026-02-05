"""Maintain an in-house injury dataset enriched with player stats."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

from data_fetching.espn_injuries import fetch_today_injuries
from data_fetching.nba_player_stats import build_team_ppg_index, fetch_player_ppg
from data_fetching.espn_player_ppg import fetch_player_ppg as fetch_espn_player_ppg
from data_fetching.espn_player_ppg import fetch_player_ppg_bulk
from data_fetching.espn_player_ids import load_player_id_db, lookup_athlete_id

DATA_PATH = Path("data/injuries/injury_dataset.json")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_dataset() -> dict:
    if not DATA_PATH.exists():
        return {}
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_dataset(payload: dict) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _normalize(name: str) -> str:
    return " ".join("".join(ch for ch in (name or "") if ch.isalnum() or ch.isspace()).lower().split())


def build_injury_dataset(target_date: date, season: str, force_refresh: bool = False) -> dict:
    injuries = fetch_today_injuries(target_date, force_refresh=force_refresh)
    players = fetch_player_ppg(season)
    team_ppg = build_team_ppg_index(players) if players else {}

    espn_ppg_map: dict[int, float] = {}
    player_id_db = load_player_id_db()
    if not team_ppg:
        try:
            season_year = int(season.split("-")[0]) + 1
        except Exception:
            season_year = date.today().year
        needed_ids = []
        for _, entries in injuries.items():
            for e in entries:
                if (e.get("status") or "").lower() != "out":
                    continue
                athlete_id = e.get("athlete_id")
                if not athlete_id:
                    athlete_id = lookup_athlete_id(e.get("player"), player_id_db)
                if athlete_id:
                    needed_ids.append(int(athlete_id))
        if needed_ids:
            espn_ppg_map = fetch_player_ppg_bulk(sorted(set(needed_ids)), season_year)

    # Build rank map by team
    team_rank = {}
    for team, plist in team_ppg.items():
        team_rank[team] = { _normalize(p["player"]): i + 1 for i, p in enumerate(plist) }

    teams_out: Dict[str, List[dict]] = {}
    for team, entries in injuries.items():
        rank_map = team_rank.get(team, {})
        enriched = []
        for e in entries:
            player = e.get("player") or ""
            player_norm = _normalize(player)
            athlete_id = e.get("athlete_id")
            if not athlete_id:
                athlete_id = lookup_athlete_id(player, player_id_db)
            ppg = None
            for p in team_ppg.get(team, []):
                if _normalize(p.get("player")) == player_norm:
                    ppg = p.get("ppg")
                    break
            if ppg is None:
                if athlete_id:
                    if espn_ppg_map:
                        ppg = espn_ppg_map.get(int(athlete_id))
                    if ppg is None:
                        try:
                            season_year = int(season.split("-")[0]) + 1
                        except Exception:
                            season_year = date.today().year
                        ppg = fetch_espn_player_ppg(int(athlete_id), season_year)
            enriched.append({
                "player": player,
                "athlete_id": athlete_id,
                "position": e.get("position"),
                "status": e.get("status"),
                "comment": e.get("comment"),
                "ppg": ppg,
                "ppg_rank": rank_map.get(player_norm),
                "updated_at": _now_iso(),
            })
        teams_out[team] = enriched

    return {
        "date": target_date.isoformat(),
        "last_updated": _now_iso(),
        "season": season,
        "teams": teams_out,
    }


def diff_injury_dataset(old: dict, new: dict) -> Dict[str, List[str]]:
    """Return changed teams and players between two datasets."""
    changes = {"teams": [], "players": []}
    old_teams = old.get("teams", {}) if isinstance(old, dict) else {}
    new_teams = new.get("teams", {}) if isinstance(new, dict) else {}

    for team, new_entries in new_teams.items():
        old_entries = old_teams.get(team, [])
        old_key = {(e.get("player"), e.get("status"), e.get("comment")) for e in old_entries}
        new_key = {(e.get("player"), e.get("status"), e.get("comment")) for e in new_entries}
        if old_key != new_key:
            changes["teams"].append(team)
            changed_players = sorted({p for (p, _, _) in (old_key ^ new_key) if p})
            changes["players"].extend([f"{team}: {p}" for p in changed_players])

    return changes


def update_injury_dataset(
    target_date: date | None = None,
    season: str = "2025-26",
    force_refresh: bool = False,
) -> Tuple[dict, Dict[str, List[str]]]:
    if target_date is None:
        target_date = date.today()

    old = load_dataset()
    new = build_injury_dataset(target_date, season, force_refresh=force_refresh)
    changes = diff_injury_dataset(old, new)
    save_dataset(new)
    return new, changes
