"""Compute injury-based spread adjustments using official NBA injury report + PPG ranks."""
from typing import Dict, List

from datetime import date

from data_fetching.espn_injuries import fetch_today_injuries
from data_fetching.nba_player_stats import build_team_ppg_index, fetch_player_ppg
from data_fetching.espn_player_ppg import fetch_player_ppg as fetch_espn_player_ppg
from data_fetching.espn_player_ppg import fetch_player_ppg_bulk
from data_fetching.espn_player_ids import load_player_id_db, lookup_athlete_id
from data_fetching.injury_dataset import load_dataset


STATUS_WEIGHTS = {
    "out": 1.0,
}

# Scale from PPG to spread points
POINTS_PER_PPG = 0.15  # 20 PPG -> ~3.0 points

# Boost factor for top scorers (1-based rank by PPG)
RANK_MULTIPLIER = {
    1: 1.8,
    2: 1.5,
    3: 1.3,
    4: 1.15,
    5: 1.1,
}


def ppg_multiplier(ppg: float) -> float:
    if ppg >= 25:
        return 1.8
    if ppg >= 20:
        return 1.5
    if ppg >= 15:
        return 1.3
    if ppg >= 10:
        return 1.15
    return 1.0


def normalize(name: str) -> str:
    return " ".join("".join(ch for ch in (name or "") if ch.isalnum() or ch.isspace()).lower().split())


def build_injury_adjustments(
    target_date: date | None = None,
    season: str = "2025-26",
    use_cached_dataset: bool = False,
) -> Dict[str, float]:
    """Return per-team adjustment (positive values = points removed from that team)."""
    injuries_by_team = fetch_today_injuries(target_date)
    if not injuries_by_team:
        return {}

    if use_cached_dataset:
        cached = load_dataset()
        if cached and cached.get("date") == target_date.isoformat():
            adjustments: Dict[str, float] = {}
            for team, plist in cached.get("teams", {}).items():
                penalty = 0.0
                for inj in plist:
                    status = (inj.get("status") or "").lower()
                    if status != "out":
                        continue
                    ppg = inj.get("ppg")
                    if ppg is None:
                        continue
                    base = float(ppg) * POINTS_PER_PPG
                    rank = inj.get("ppg_rank")
                    mult = RANK_MULTIPLIER.get(rank, 1.0) if rank else ppg_multiplier(float(ppg))
                    penalty += base * mult
                if penalty > 0:
                    adjustments[team] = penalty
            if adjustments:
                return adjustments

    players = fetch_player_ppg(season)
    team_ppg = build_team_ppg_index(players) if players else {}

    # Prefetch ESPN PPG for OUT players if NBA stats are unavailable
    espn_ppg_map: dict[int, float] = {}
    player_id_db = load_player_id_db()
    if not team_ppg:
        try:
            season_year = int(season.split("-")[0]) + 1
        except Exception:
            season_year = date.today().year
        needed_ids = []
        for plist in injuries_by_team.values():
            for inj in plist:
                if (inj.get("status") or "").lower() != "out":
                    continue
                athlete_id = inj.get("athlete_id")
                if not athlete_id:
                    athlete_id = lookup_athlete_id(inj.get("player"), player_id_db)
                if athlete_id:
                    needed_ids.append(int(athlete_id))
        if needed_ids:
            espn_ppg_map = fetch_player_ppg_bulk(sorted(set(needed_ids)), season_year)

    if not team_ppg:
        missing_ids = any(
            any(not e.get("athlete_id") for e in entries)
            for entries in injuries_by_team.values()
        )
        if missing_ids:
            injuries_by_team = fetch_today_injuries(target_date, force_refresh=True)

    # Build player -> rank mapping per team
    team_rank = {}
    for team, plist in team_ppg.items():
        team_rank[team] = {normalize(p["player"]): i + 1 for i, p in enumerate(plist)}

    adjustments: Dict[str, float] = {}
    for team, plist in injuries_by_team.items():
        penalty = 0.0
        rank_map = team_rank.get(team, {})
        for inj in plist:
            status = (inj.get("status") or "").lower()
            weight = STATUS_WEIGHTS.get(status)
            if weight is None:
                continue
            player_norm = normalize(inj.get("player"))
            rank = rank_map.get(player_norm)
            # base penalty from PPG (NBA stats, fallback to ESPN by athlete_id)
            ppg = None
            for p in team_ppg.get(team, []):
                if normalize(p.get("player")) == player_norm:
                    ppg = p.get("ppg")
                    break
            if ppg is None:
                athlete_id = inj.get("athlete_id")
                if not athlete_id:
                    athlete_id = lookup_athlete_id(inj.get("player"), player_id_db)
                if athlete_id:
                    if espn_ppg_map:
                        ppg = espn_ppg_map.get(int(athlete_id))
                    if ppg is None:
                        try:
                            season_year = int(season.split("-")[0]) + 1
                        except Exception:
                            season_year = date.today().year
                        ppg = fetch_espn_player_ppg(int(athlete_id), season_year)
            if ppg is None:
                continue

            base = float(ppg) * POINTS_PER_PPG
            if rank:
                mult = RANK_MULTIPLIER.get(rank, 1.0)
            else:
                mult = ppg_multiplier(float(ppg))
            penalty += base * mult * weight
        if penalty > 0:
            adjustments[team] = penalty

    return adjustments


def apply_injury_adjustment(away_team: str, home_team: str, pred_away_margin: float, adjustments: Dict[str, float]) -> float:
    """Adjust away margin for injuries (away penalties reduce margin, home penalties increase)."""
    away_penalty = adjustments.get(away_team, 0.0)
    home_penalty = adjustments.get(home_team, 0.0)
    return pred_away_margin - away_penalty + home_penalty
