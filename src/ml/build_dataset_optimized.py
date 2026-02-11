"""
Rebuild dataset with ESPN closing spreads for training/testing split.

Training: December 1 - January 15, 2026
Testing: January 16-31, 2026
"""
import argparse
import csv
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.matchup_model import (
    DATA_DIR,
    build_team_history,
    compute_team_stats,
    extract_points,
    parse_game_date,
    parse_team_display,
)
from evaluation.game_narratives import compute_narrative_stats
from data_fetching.espn_api import get_espn_games_for_date, scrape_espn_gamecast_spreads
from data_fetching.odds_api import get_opening_spreads_for_date
from evaluation.spread_utils import normalize_team_name


OUTPUT_DIR = Path("ml_data")
OUTPUT_DIR.mkdir(exist_ok=True)

ESPN_SPREADS_CACHE = Path("odds_cache/espn_closing_spreads.json")
ESPN_DETAILED_SPREADS_CACHE = Path("odds_cache/espn_closing_spreads_detailed.json")


def load_espn_spreads_cache() -> dict:
    """Load cached ESPN spreads."""
    if ESPN_SPREADS_CACHE.exists():
        try:
            with open(ESPN_SPREADS_CACHE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def load_espn_spreads_detailed_cache() -> dict:
    """Load detailed ESPN spreads cache with matchup info."""
    if ESPN_DETAILED_SPREADS_CACHE.exists():
        try:
            with open(ESPN_DETAILED_SPREADS_CACHE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_espn_spreads_cache(spreads: dict):
    """Save ESPN spreads to cache."""
    ESPN_SPREADS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(ESPN_SPREADS_CACHE, 'w') as f:
        json.dump(spreads, f, indent=2)


def iter_game_files(start_date: date, end_date: date):
    """Iterate through game files in date range."""
    current = start_date
    files = []
    while current <= end_date:
        day_dir = DATA_DIR / current.isoformat()
        if day_dir.exists():
            files.extend(day_dir.glob("*.json"))
        current += timedelta(days=1)
    return files


def recent_avg(games: list, n: int, attr: str) -> float:
    if not games:
        return 0.0
    recent = games[-n:] if len(games) >= n else games
    values = [getattr(g, attr) for g in recent]
    return sum(values) / len(values) if values else 0.0


def recent_win_pct(games: list, n: int) -> float:
    if not games:
        return 0.0
    recent = games[-n:] if len(games) >= n else games
    wins = sum(1 for g in recent if g.margin > 0)
    return wins / len(recent) if recent else 0.0


def head_to_head_stats(home_games: list, away_id: str) -> tuple[float, float, int]:
    matchups = [g for g in home_games if getattr(g, "opponent_id", "") == away_id]
    if not matchups:
        return 0.0, 0.0, 0
    avg_margin = sum(g.margin for g in matchups) / len(matchups)
    win_pct = sum(1 for g in matchups if g.margin > 0) / len(matchups)
    return avg_margin, win_pct, len(matchups)


def build_team_narrative_history(data_dir: Path) -> dict:
    history = {}
    for file_path in data_dir.glob("*/**/*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception:
            continue

        narrative = compute_narrative_stats(game)
        if not narrative:
            continue

        home = game.get("home", {})
        away = game.get("away", {})
        home_id = home.get("id")
        away_id = away.get("id")
        if not home_id or not away_id:
            continue

        game_date = narrative["game_date"]
        clutch_margin = narrative["clutch_margin"]
        blown_side = narrative.get("blown_lead_side")

        history.setdefault(home_id, []).append({
            "game_date": game_date,
            "clutch_margin": clutch_margin,
            "max_lead": narrative["max_home_lead"],
            "blew_lead": blown_side == "home",
        })
        history.setdefault(away_id, []).append({
            "game_date": game_date,
            "clutch_margin": -clutch_margin,
            "max_lead": narrative["max_away_lead"],
            "blew_lead": blown_side == "away",
        })

    for team_id in history:
        history[team_id].sort(key=lambda x: x["game_date"])

    return history


def build_features_for_game(
    game: dict,
    history: dict,
    half_life: float,
    market_spread: float = 0.0,
    opening_spread: float | None = None,
    narrative_history: dict | None = None,
) -> dict:
    """Build feature vector for a game."""
    game_date = parse_game_date(game)
    if not game_date:
        return {}

    home = game.get("home", {})
    away = game.get("away", {})
    home_id, home_name = parse_team_display(home)
    away_id, away_name = parse_team_display(away)

    home_games = [g for g in history.get(home_id, []) if g.game_date < game_date]
    away_games = [g for g in history.get(away_id, []) if g.game_date < game_date]

    if not home_games or not away_games:
        return {}

    home_stats = compute_team_stats(home_id, home_games, home_name, half_life)
    away_stats = compute_team_stats(away_id, away_games, away_name, half_life)

    rest_diff = 0.0
    home_b2b = 0.0
    away_b2b = 0.0
    if home_stats.last_game_date and away_stats.last_game_date:
        home_rest = (game_date - home_stats.last_game_date).days
        away_rest = (game_date - away_stats.last_game_date).days
        rest_diff = home_rest - away_rest
        home_b2b = 1.0 if home_rest == 0 else 0.0
        away_b2b = 1.0 if away_rest == 0 else 0.0

    home_recent_margin_3 = recent_avg(home_games, 3, "margin")
    home_recent_margin_5 = recent_avg(home_games, 5, "margin")
    home_recent_margin_10 = recent_avg(home_games, 10, "margin")
    away_recent_margin_3 = recent_avg(away_games, 3, "margin")
    away_recent_margin_5 = recent_avg(away_games, 5, "margin")
    away_recent_margin_10 = recent_avg(away_games, 10, "margin")

    home_recent_win_pct_3 = recent_win_pct(home_games, 3)
    home_recent_win_pct_5 = recent_win_pct(home_games, 5)
    away_recent_win_pct_3 = recent_win_pct(away_games, 3)
    away_recent_win_pct_5 = recent_win_pct(away_games, 5)

    home_recent_points_for_5 = recent_avg(home_games, 5, "points_for")
    home_recent_points_against_5 = recent_avg(home_games, 5, "points_against")
    away_recent_points_for_5 = recent_avg(away_games, 5, "points_for")
    away_recent_points_against_5 = recent_avg(away_games, 5, "points_against")

    home_narr = []
    away_narr = []
    if narrative_history:
        home_narr = [n for n in narrative_history.get(home_id, []) if n["game_date"] < game_date]
        away_narr = [n for n in narrative_history.get(away_id, []) if n["game_date"] < game_date]

    def recent_narrative_avg(entries: list, n: int, field: str) -> float:
        if not entries:
            return 0.0
        recent = entries[-n:] if len(entries) >= n else entries
        values = [e.get(field, 0.0) for e in recent]
        return sum(values) / len(values) if values else 0.0

    def recent_blown_rate(entries: list, n: int) -> float:
        if not entries:
            return 0.0
        recent = entries[-n:] if len(entries) >= n else entries
        blew = sum(1 for e in recent if e.get("blew_lead"))
        return blew / len(recent) if recent else 0.0

    h2h_margin_avg, h2h_win_pct, h2h_games_played = head_to_head_stats(home_games, away_id)

    features = {
        "home_weighted_margin": home_stats.weighted_home_margin,
        "away_weighted_margin": away_stats.weighted_away_margin,
        "home_weighted_win_pct": home_stats.weighted_win_pct,
        "away_weighted_win_pct": away_stats.weighted_win_pct,
        "home_recent_10_win_pct": home_stats.recent_10_win_pct,
        "away_recent_10_win_pct": away_stats.recent_10_win_pct,
        "home_weighted_points_for": home_stats.weighted_points_for,
        "away_weighted_points_for": away_stats.weighted_points_for,
        "home_weighted_points_against": home_stats.weighted_points_against,
        "away_weighted_points_against": away_stats.weighted_points_against,
        "home_weighted_point_diff": home_stats.weighted_points_for - home_stats.weighted_points_against,
        "away_weighted_point_diff": away_stats.weighted_points_for - away_stats.weighted_points_against,
        "home_recent_margin_3": home_recent_margin_3,
        "home_recent_margin_5": home_recent_margin_5,
        "home_recent_margin_10": home_recent_margin_10,
        "away_recent_margin_3": away_recent_margin_3,
        "away_recent_margin_5": away_recent_margin_5,
        "away_recent_margin_10": away_recent_margin_10,
        "home_recent_win_pct_3": home_recent_win_pct_3,
        "home_recent_win_pct_5": home_recent_win_pct_5,
        "away_recent_win_pct_3": away_recent_win_pct_3,
        "away_recent_win_pct_5": away_recent_win_pct_5,
        "home_recent_points_for_5": home_recent_points_for_5,
        "home_recent_points_against_5": home_recent_points_against_5,
        "away_recent_points_for_5": away_recent_points_for_5,
        "away_recent_points_against_5": away_recent_points_against_5,
        "home_blown_rate_10": recent_blown_rate(home_narr, 10),
        "away_blown_rate_10": recent_blown_rate(away_narr, 10),
        "home_clutch_margin_10": recent_narrative_avg(home_narr, 10, "clutch_margin"),
        "away_clutch_margin_10": recent_narrative_avg(away_narr, 10, "clutch_margin"),
        "home_max_lead_10": recent_narrative_avg(home_narr, 10, "max_lead"),
        "away_max_lead_10": recent_narrative_avg(away_narr, 10, "max_lead"),
        "home_h2h_margin_avg": h2h_margin_avg,
        "home_h2h_win_pct": h2h_win_pct,
        "h2h_games_played": float(h2h_games_played),
        "rest_diff": rest_diff,
        "home_b2b": home_b2b,
        "away_b2b": away_b2b,
        "home_games_played": float(len(home_games)),
        "away_games_played": float(len(away_games)),
        "market_spread": market_spread,
        "line_move": float(market_spread - opening_spread) if opening_spread is not None else 0.0,
    }
    return features


def get_espn_spread_for_game(game: dict, espn_spreads: dict, espn_detailed: dict) -> float:
    """
    Try to find ESPN closing spread for a game.
    Uses game date and team names to match.
    
    ESPN spread cache format: "YYYY-MM-DD_HOME_TEAM_NAME": spread_value
    Example: "2026-01-06_Pacers": -6.5
    """
    game_date = parse_game_date(game)
    if not game_date:
        return 0.0
    
    home = game.get("home", {})
    away = game.get("away", {})
    home_name = (home.get("market", "") + " " + home.get("name", "")).strip()
    away_name = (away.get("market", "") + " " + away.get("name", "")).strip()

    def normalize(name: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\s]", "", name or "").lower()).strip()

    date_str = game_date.isoformat()
    date_candidates = [date_str]
    try:
        date_candidates.append((game_date - timedelta(days=1)).isoformat())
        date_candidates.append((game_date + timedelta(days=1)).isoformat())
    except Exception:
        pass
    home_norm = normalize(home_name)
    away_norm = normalize(away_name)

    # Prefer detailed cache: match by date + teams
    if espn_detailed:
        for key, entry in espn_detailed.items():
            if not any(key.startswith(d) for d in date_candidates):
                continue
            entry_home = normalize(entry.get("home_team", ""))
            entry_away = normalize(entry.get("away_team", ""))
            if len(entry_home) < 3 or len(entry_away) < 3:
                continue
            if (entry_home in home_norm or home_norm in entry_home) and (entry_away in away_norm or away_norm in entry_away):
                spread = entry.get("closing_spread_away")
                if spread is not None:
                    return float(spread)

    # Fallback to simple cache (date + home team key)
    home_token = home_name.split()[-1] if home_name else ""
    for d in date_candidates:
        key = f"{d}_{home_token}"
        if key in espn_spreads:
            spread = espn_spreads[key]
            if spread is not None:
                return float(spread)

    return 0.0


def build_dataset(start_date: date, end_date: date, half_life: float, output_path: Path) -> None:
    """Build dataset with ESPN closing spreads."""
    print(f"\nBuilding dataset from {start_date} to {end_date}")
    print(f"Output: {output_path}\n")
    
    history, _ = build_team_history(DATA_DIR)
    narrative_history = build_team_narrative_history(DATA_DIR)
    
    # Load ESPN spreads cache
    espn_spreads = load_espn_spreads_cache()
    espn_detailed = load_espn_spreads_detailed_cache()
    print(f"Loaded {len(espn_spreads)} cached ESPN spreads")
    print(f"Loaded {len(espn_detailed)} cached ESPN detailed spreads\n")
    
    fieldnames = [
        "game_date",
        "home_team",
        "away_team",
        "actual_margin",
        "home_weighted_margin",
        "away_weighted_margin",
        "home_weighted_win_pct",
        "away_weighted_win_pct",
        "home_recent_10_win_pct",
        "away_recent_10_win_pct",
        "home_weighted_points_for",
        "away_weighted_points_for",
        "home_weighted_points_against",
        "away_weighted_points_against",
        "home_weighted_point_diff",
        "away_weighted_point_diff",
        "home_recent_margin_3",
        "home_recent_margin_5",
        "home_recent_margin_10",
        "away_recent_margin_3",
        "away_recent_margin_5",
        "away_recent_margin_10",
        "home_recent_win_pct_3",
        "home_recent_win_pct_5",
        "away_recent_win_pct_3",
        "away_recent_win_pct_5",
        "home_recent_points_for_5",
        "home_recent_points_against_5",
        "away_recent_points_for_5",
        "away_recent_points_against_5",
        "home_blown_rate_10",
        "away_blown_rate_10",
        "home_clutch_margin_10",
        "away_clutch_margin_10",
        "home_max_lead_10",
        "away_max_lead_10",
        "home_h2h_margin_avg",
        "home_h2h_win_pct",
        "h2h_games_played",
        "rest_diff",
        "home_b2b",
        "away_b2b",
        "home_games_played",
        "away_games_played",
        "market_spread",
        "line_move",
    ]

    rows = []
    opening_spreads_by_date: dict[date, dict] = {}
    for file_path in iter_game_files(start_date, end_date):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception:
            continue

        status = (game.get("status") or "").lower()
        if status not in {"closed", "complete", "completed", "final"}:
            continue

        home = game.get("home", {})
        away = game.get("away", {})
        home_points = extract_points(home)
        away_points = extract_points(away)
        if home_points is None or away_points is None:
            continue

        game_date = parse_game_date(game)
        if not game_date:
            continue

        # Get ESPN closing spread if available
        market_spread = get_espn_spread_for_game(game, espn_spreads, espn_detailed)

        opening_spreads = opening_spreads_by_date.get(game_date)
        if opening_spreads is None:
            opening_spreads = get_opening_spreads_for_date(game_date.isoformat())
            opening_spreads_by_date[game_date] = opening_spreads
        
        opening_spread = None
        if opening_spreads:
            home_name = f"{home.get('market','')} {home.get('name','')}".strip()
            away_name = f"{away.get('market','')} {away.get('name','')}".strip()
            open_key = f"{normalize_team_name(away_name)}@{normalize_team_name(home_name)}"
            opening_spread = opening_spreads.get(open_key)

        features = build_features_for_game(
            game,
            history,
            half_life,
            market_spread=market_spread,
            opening_spread=opening_spread,
            narrative_history=narrative_history,
        )
        if not features:
            continue

        home_name = f"{home.get('market','')} {home.get('name','')}".strip()
        away_name = f"{away.get('market','')} {away.get('name','')}".strip()

        row = {
            "game_date": game_date.isoformat(),
            "home_team": home_name,
            "away_team": away_name,
            "actual_margin": home_points - away_points,
            **features,
        }
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved dataset: {output_path} ({len(rows)} rows)")
    print(f"\nDataset split:")
    print(f"  Training (Dec 1 - Jan 15): Use for model training")
    print(f"  Testing (Jan 16-31): Use for model evaluation")


def main():
    parser = argparse.ArgumentParser(description="Build optimized ML dataset for NBA predictions")
    parser.add_argument("--start", default="2025-12-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD); defaults to yesterday")
    args = parser.parse_args()

    train_start = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else (date.today() - timedelta(days=1))

    output_path = OUTPUT_DIR / "games_optimized.csv"
    build_dataset(train_start, end_date, half_life=10.0, output_path=output_path)

    print(f"\n{'='*70}")
    print(f"Dataset ready for training")
    print(f"Next steps:")
    print(f"1. Retrain model with a recent cutoff date")
    print(f"2. Evaluate on the most recent test window")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
