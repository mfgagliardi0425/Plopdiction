"""
Build a supervised ML dataset from historical games.
"""
import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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

OUTPUT_DIR = Path("ml_data")
OUTPUT_DIR.mkdir(exist_ok=True)

TRACKING_DIR = Path("tracking")


def load_market_spreads() -> Dict[str, Dict[str, float]]:
    """
    Load market spreads from tracking files.
    Returns: {game_date: {game_id: market_spread}}
    """
    spreads = {}
    if not TRACKING_DIR.exists():
        return spreads
    
    for tracking_file in sorted(TRACKING_DIR.glob("*.json")):
        try:
            with open(tracking_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            game_date = data.get("date")
            if not game_date:
                continue
            
            spreads[game_date] = {}
            for game in data.get("games", []):
                game_id = game.get("game_id")
                market = game.get("market", {})
                spread = market.get("spread")
                if game_id and spread is not None:
                    spreads[game_date][game_id] = spread
        except Exception:
            continue
    
    return spreads


def iter_game_files(start_date: date, end_date: date) -> List[Path]:
    files: List[Path] = []
    current = start_date
    while current <= end_date:
        day_dir = DATA_DIR / current.isoformat()
        if day_dir.exists():
            files.extend(day_dir.glob("*.json"))
        current += timedelta(days=1)
    return files


def recent_avg(games: List, n: int, attr: str) -> float:
    if not games:
        return 0.0
    recent = games[-n:] if len(games) >= n else games
    values = [getattr(g, attr) for g in recent]
    return sum(values) / len(values) if values else 0.0


def recent_win_pct(games: List, n: int) -> float:
    if not games:
        return 0.0
    recent = games[-n:] if len(games) >= n else games
    wins = sum(1 for g in recent if g.margin > 0)
    return wins / len(recent) if recent else 0.0

def build_team_narrative_history(data_dir: Path) -> Dict[str, list]:
    history: Dict[str, list] = {}
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


def head_to_head_stats(home_games: List, away_id: str) -> tuple[float, float, int]:
    matchups = [g for g in home_games if getattr(g, "opponent_id", "") == away_id]
    if not matchups:
        return 0.0, 0.0, 0
    avg_margin = sum(g.margin for g in matchups) / len(matchups)
    win_pct = sum(1 for g in matchups if g.margin > 0) / len(matchups)
    return avg_margin, win_pct, len(matchups)


def build_features_for_game(
    game: dict,
    history: Dict[str, list],
    half_life: float,
    market_spread: Optional[float] = None,
    opening_spread: Optional[float] = None,
    narrative_history: Optional[Dict[str, list]] = None,
) -> Dict[str, float]:
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

    h2h_margin_avg, h2h_win_pct, h2h_games_played = head_to_head_stats(home_games, away_id)

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
        "home_h2h_margin_avg": h2h_margin_avg,
        "home_h2h_win_pct": h2h_win_pct,
        "h2h_games_played": float(h2h_games_played),
        "rest_diff": rest_diff,
        "home_b2b": home_b2b,
        "away_b2b": away_b2b,
        "home_games_played": float(len(home_games)),
        "away_games_played": float(len(away_games)),
        "market_spread": market_spread if market_spread is not None else 0.0,
        "line_move": float(market_spread - opening_spread) if opening_spread is not None else 0.0,
        "home_blown_rate_10": recent_blown_rate(home_narr, 10),
        "away_blown_rate_10": recent_blown_rate(away_narr, 10),
        "home_clutch_margin_10": recent_narrative_avg(home_narr, 10, "clutch_margin"),
        "away_clutch_margin_10": recent_narrative_avg(away_narr, 10, "clutch_margin"),
        "home_max_lead_10": recent_narrative_avg(home_narr, 10, "max_lead"),
        "away_max_lead_10": recent_narrative_avg(away_narr, 10, "max_lead"),
    }
    return features


def build_dataset(start_date: date, end_date: date, half_life: float, output_path: Path) -> None:
    history, _ = build_team_history(DATA_DIR)
    narrative_history = build_team_narrative_history(DATA_DIR)
    market_spreads = load_market_spreads()

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
    for file_path in iter_game_files(start_date, end_date):
        try:
            import json
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

        features = build_features_for_game(game, history, half_life, market_spread=None, narrative_history=narrative_history)
        if not features:
            continue

        home_name = f"{home.get('market','')} {home.get('name','')}".strip()
        away_name = f"{away.get('market','')} {away.get('name','')}".strip()

        # Look up market spread if available
        game_id = game.get("id")
        if game_date.isoformat() in market_spreads and game_id in market_spreads[game_date.isoformat()]:
            market_spread = market_spreads[game_date.isoformat()][game_id]
            # Rebuild features with market spread
            features = build_features_for_game(game, history, half_life, market_spread=market_spread, narrative_history=narrative_history)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML dataset from historical games")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--half-life", type=float, default=10.0, help="Half-life in games")
    parser.add_argument("--output", default="ml_data/games.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_dataset(start_date, end_date, args.half_life, output_path)


if __name__ == "__main__":
    main()
