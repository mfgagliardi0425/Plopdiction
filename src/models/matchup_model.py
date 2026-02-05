import argparse
import json
import math
import os
import time
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.sportradar.com/nba"
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
SUMMARY_CACHE_DIR = CACHE_DIR / "summary"


@dataclass
class GameResult:
    game_id: str
    game_date: date
    is_home: bool
    points_for: int
    points_against: int
    opponent_id: str = ""

    @property
    def margin(self) -> int:
        return self.points_for - self.points_against


@dataclass
class TeamStats:
    team_id: str
    name: str
    weighted_win_pct: float
    weighted_margin: float
    weighted_home_margin: float
    weighted_away_margin: float
    weighted_points_for: float
    weighted_points_against: float
    recent_10_win_pct: float
    last_game_date: Optional[date]


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def build_url(path: str, params: Optional[dict] = None) -> str:
    if params:
        return f"{BASE_URL}{path}?" + "&".join(f"{k}={v}" for k, v in params.items())
    return f"{BASE_URL}{path}"


def iter_game_files(data_dir: Path) -> Iterable[Path]:
    if not data_dir.exists():
        return []
    return data_dir.glob("*/**/*.json")


def parse_team_display(team: dict) -> Tuple[str, str]:
    team_id = team.get("id") or "unknown"
    market = team.get("market") or ""
    name = team.get("name") or ""
    display = (market + " " + name).strip() or team.get("alias") or team_id
    return team_id, display


def extract_points(team: dict) -> Optional[int]:
    if "points" in team and isinstance(team["points"], int):
        return team["points"]
    scoring = team.get("scoring")
    if isinstance(scoring, dict) and "points" in scoring:
        return scoring.get("points")
    return None


def parse_game_date(game: dict) -> Optional[date]:
    scheduled = game.get("scheduled")
    if not scheduled:
        return None
    try:
        return datetime.fromisoformat(scheduled.replace("Z", "+00:00")).date()
    except Exception:
        return None


def build_team_history(data_dir: Path) -> Tuple[Dict[str, List[GameResult]], Dict[str, str]]:
    history: Dict[str, List[GameResult]] = {}
    names: Dict[str, str] = {}

    for file_path in iter_game_files(data_dir):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception:
            continue

        status = (game.get("status") or "").lower()
        if status not in {"closed", "complete", "completed", "final"}:
            continue

        game_date = parse_game_date(game)
        if not game_date:
            continue

        home = game.get("home", {})
        away = game.get("away", {})
        home_id, home_name = parse_team_display(home)
        away_id, away_name = parse_team_display(away)
        game_id = game.get("id")
        if not game_id:
            continue

        home_points = extract_points(home)
        away_points = extract_points(away)
        if home_points is None or away_points is None:
            continue

        names[home_id] = home_name
        names[away_id] = away_name

        history.setdefault(home_id, []).append(
            GameResult(
                game_id=game_id,
                game_date=game_date,
                is_home=True,
                points_for=home_points,
                points_against=away_points,
                opponent_id=away_id,
            )
        )
        history.setdefault(away_id, []).append(
            GameResult(
                game_id=game_id,
                game_date=game_date,
                is_home=False,
                points_for=away_points,
                points_against=home_points,
                opponent_id=home_id,
            )
        )

    for team_id in history:
        history[team_id].sort(key=lambda g: g.game_date)

    return history, names


def weighted_stats(games: List[GameResult], half_life_games: float) -> Tuple[float, float, float, float]:
    if not games:
        return 0.0, 0.0, 0.0, 0.0

    weights: List[float] = []
    total = 0.0
    weighted_wins = 0.0
    weighted_margin = 0.0
    weighted_for = 0.0
    weighted_against = 0.0

    most_recent_index = len(games) - 1
    for idx, game in enumerate(games):
        games_ago = most_recent_index - idx
        weight = 0.5 ** (games_ago / half_life_games)
        weights.append(weight)
        total += weight
        weighted_wins += weight * (1.0 if game.margin > 0 else 0.0)
        weighted_margin += weight * game.margin
        weighted_for += weight * game.points_for
        weighted_against += weight * game.points_against

    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        weighted_wins / total,
        weighted_margin / total,
        weighted_for / total,
        weighted_against / total,
    )


def compute_team_stats(
    team_id: str,
    games: List[GameResult],
    name: str,
    half_life_games: float,
) -> TeamStats:
    win_pct, margin, pts_for, pts_against = weighted_stats(games, half_life_games)

    home_games = [g for g in games if g.is_home]
    away_games = [g for g in games if not g.is_home]
    home_margin = weighted_stats(home_games, half_life_games)[1] if home_games else margin
    away_margin = weighted_stats(away_games, half_life_games)[1] if away_games else margin

    recent = games[-10:]
    recent_win_pct = 0.0
    if recent:
        recent_win_pct = sum(1 for g in recent if g.margin > 0) / len(recent)

    last_game_date = games[-1].game_date if games else None

    return TeamStats(
        team_id=team_id,
        name=name,
        weighted_win_pct=win_pct,
        weighted_margin=margin,
        weighted_home_margin=home_margin,
        weighted_away_margin=away_margin,
        weighted_points_for=pts_for,
        weighted_points_against=pts_against,
        recent_10_win_pct=recent_win_pct,
        last_game_date=last_game_date,
    )


def get_schedule(day: date, retries: int = 3, backoff: float = 2.0) -> dict:
    access_level = get_env("SPORTRADAR_ACCESS_LEVEL", "trial")
    language = get_env("SPORTRADAR_LANGUAGE", "en")
    version = get_env("SPORTRADAR_VERSION", "v8")
    api_key = get_env("SPORTRADAR_API_KEY")

    path = f"/{access_level}/{version}/{language}/games/{day.year}/{day.month:02d}/{day.day:02d}/schedule.json"
    url = build_url(path, {"api_key": api_key})

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        response = requests.get(url, timeout=30)
        if response.status_code == 429:
            wait = backoff * (attempt + 1)
            print(f"Rate limited when fetching {day.isoformat()} schedule. Waiting {wait:.1f}s...")
            time.sleep(wait)
            last_error = requests.exceptions.HTTPError("429 Too Many Requests")
            continue
        response.raise_for_status()
        return response.json()

    if last_error:
        raise last_error
    raise RuntimeError("Failed to fetch schedule")


def get_upcoming_games(start_date: date, end_date: date) -> List[dict]:
    games: List[dict] = []
    current = start_date
    while current <= end_date:
        data = get_schedule(current)
        day_games = data.get("games", [])
        for game in day_games:
            status = (game.get("status") or "").lower()
            if status in {"scheduled", "created", "inprogress", "delayed"}:
                games.append(game)
        current += timedelta(days=1)
    return games


def get_team_profile(team_id: str, delay: float = 2.0, retries: int = 3) -> dict:
    access_level = get_env("SPORTRADAR_ACCESS_LEVEL", "trial")
    language = get_env("SPORTRADAR_LANGUAGE", "en")
    version = get_env("SPORTRADAR_VERSION", "v8")
    api_key = get_env("SPORTRADAR_API_KEY")

    url = f"{BASE_URL}/{access_level}/{version}/{language}/teams/{team_id}/profile.json"

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        response = requests.get(url, params={"api_key": api_key}, timeout=30)
        if response.status_code == 429:
            wait = delay * (attempt + 1)
            print(f"Rate limited when fetching roster {team_id}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            last_error = requests.exceptions.HTTPError("429 Too Many Requests")
            continue
        response.raise_for_status()
        return response.json()

    if last_error:
        raise last_error
    raise RuntimeError("Failed to fetch roster")


def get_game_summary(game_id: str, delay: float = 2.0, retries: int = 3) -> dict:
    SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = SUMMARY_CACHE_DIR / f"{game_id}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    access_level = get_env("SPORTRADAR_ACCESS_LEVEL", "trial")
    language = get_env("SPORTRADAR_LANGUAGE", "en")
    version = get_env("SPORTRADAR_VERSION", "v8")
    api_key = get_env("SPORTRADAR_API_KEY")

    url = f"{BASE_URL}/{access_level}/{version}/{language}/games/{game_id}/summary.json"

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        response = requests.get(url, params={"api_key": api_key}, timeout=30)
        if response.status_code == 429:
            wait = delay * (attempt + 1)
            print(f"Rate limited when fetching summary {game_id}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            last_error = requests.exceptions.HTTPError("429 Too Many Requests")
            continue
        response.raise_for_status()
        data = response.json()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    if last_error:
        raise last_error
    raise RuntimeError("Failed to fetch game summary")


def parse_minutes(value: Optional[str]) -> float:
    if not value:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    parts = value.split(":")
    if len(parts) != 2:
        return 0.0
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes + seconds / 60.0
    except ValueError:
        return 0.0


def build_player_averages(
    team_id: str,
    games: List[GameResult],
    half_life_games: float,
    max_games: int = 10,
    delay: float = 2.0,
) -> Dict[str, dict]:
    if not games:
        return {}

    recent_games = games[-max_games:]
    player_totals: Dict[str, dict] = {}

    most_recent_index = len(recent_games) - 1
    for idx, game in enumerate(recent_games):
        games_ago = most_recent_index - idx
        weight = 0.5 ** (games_ago / half_life_games)

        summary = get_game_summary(game.game_id, delay=delay)
        side_key = "home" if summary.get("home", {}).get("id") == team_id else "away"
        players = summary.get(side_key, {}).get("players", [])

        for player in players:
            stats = player.get("statistics", {})
            minutes = parse_minutes(stats.get("minutes"))
            if minutes <= 0:
                continue

            player_id = player.get("id")
            if not player_id:
                continue

            points = float(stats.get("points", 0) or 0)
            plus = float(stats.get("plus", 0) or 0)
            minus = float(stats.get("minus", 0) or 0)
            plus_minus = plus - minus

            entry = player_totals.setdefault(
                player_id,
                {
                    "name": player.get("full_name") or "Unknown",
                    "weighted_points": 0.0,
                    "weighted_pm": 0.0,
                    "total_weight": 0.0,
                },
            )

            entry["weighted_points"] += weight * points
            entry["weighted_pm"] += weight * plus_minus
            entry["total_weight"] += weight

    averages: Dict[str, dict] = {}
    for player_id, entry in player_totals.items():
        if entry["total_weight"] <= 0:
            continue
        averages[player_id] = {
            "name": entry["name"],
            "avg_points": entry["weighted_points"] / entry["total_weight"],
            "avg_pm": entry["weighted_pm"] / entry["total_weight"],
        }

    return averages


def compute_missing_player_impact(
    team_profile: dict,
    player_averages: Dict[str, dict],
    points_weight: float,
    pm_weight: float,
    impact_scale: float,
) -> float:
    missing_impact = 0.0
    for player in team_profile.get("players", []):
        if player.get("status") == "ACT":
            continue
        player_id = player.get("id")
        if not player_id:
            continue
        averages = player_averages.get(player_id)
        if not averages:
            continue
        impact = (averages["avg_points"] * points_weight) + (averages["avg_pm"] * pm_weight)
        missing_impact += impact
    return missing_impact * impact_scale


def count_active_players(team_profile: dict) -> int:
    players = team_profile.get("players", [])
    active = [p for p in players if p.get("status") == "active"]
    return len(active)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def timezone_penalty(game: dict, preferred_hour: int) -> Tuple[float, float]:
    scheduled = game.get("scheduled")
    time_zones = game.get("time_zones")
    if not scheduled or not isinstance(time_zones, dict):
        return 0.0, 0.0

    home_tz = time_zones.get("home")
    away_tz = time_zones.get("away")
    if not home_tz or not away_tz:
        return 0.0, 0.0

    try:
        game_dt = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
        home_local = game_dt.astimezone(ZoneInfo(home_tz))
        away_local = game_dt.astimezone(ZoneInfo(away_tz))
        home_penalty = abs(home_local.hour - preferred_hour)
        away_penalty = abs(away_local.hour - preferred_hour)
        return float(home_penalty), float(away_penalty)
    except Exception:
        return 0.0, 0.0


def predict_game(
    game: dict,
    stats: Dict[str, TeamStats],
    player_averages: Dict[str, Dict[str, dict]],
    include_rosters: bool,
    include_player_impact: bool,
    home_advantage: float,
    margin_scale: float,
    rest_weight: float,
    time_weight: float,
    roster_weight: float,
    b2b_weight: float,
    time_zone_weight: float,
    preferred_local_hour: int,
    player_points_weight: float,
    player_pm_weight: float,
    player_impact_scale: float,
) -> Tuple[str, str, float, float, float]:
    home = game.get("home", {})
    away = game.get("away", {})

    home_id, home_name = parse_team_display(home)
    away_id, away_name = parse_team_display(away)

    home_stats = stats.get(home_id)
    away_stats = stats.get(away_id)

    if not home_stats or not away_stats:
        return home_name, away_name, 0.0, 0.5, 0.5

    base_margin = (
        home_stats.weighted_home_margin - away_stats.weighted_away_margin
    ) + home_advantage

    rest_diff = 0.0
    game_date = parse_game_date(game)
    if game_date and home_stats.last_game_date and away_stats.last_game_date:
        home_rest = (game_date - home_stats.last_game_date).days
        away_rest = (game_date - away_stats.last_game_date).days
        rest_diff = clamp(home_rest - away_rest, -3, 3)

    rest_adj = rest_weight * rest_diff

    b2b_adj = 0.0
    if game_date and home_stats.last_game_date and away_stats.last_game_date:
        home_rest = (game_date - home_stats.last_game_date).days
        away_rest = (game_date - away_stats.last_game_date).days
        home_b2b = 1.0 if home_rest == 0 else 0.0
        away_b2b = 1.0 if away_rest == 0 else 0.0
        b2b_adj = b2b_weight * (away_b2b - home_b2b)

    scheduled = game.get("scheduled")
    time_adj = 0.0
    if scheduled:
        try:
            game_dt = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
            if 12 <= game_dt.hour < 20:
                time_adj = time_weight
        except Exception:
            pass

    home_tz_penalty, away_tz_penalty = timezone_penalty(game, preferred_local_hour)
    tz_adj = time_zone_weight * (away_tz_penalty - home_tz_penalty)

    roster_adj = 0.0
    missing_adj = 0.0
    if include_rosters or include_player_impact:
        try:
            home_profile = get_team_profile(home_id)
            away_profile = get_team_profile(away_id)

            if include_rosters:
                roster_adj = roster_weight * (
                    count_active_players(home_profile) - count_active_players(away_profile)
                )

            if include_player_impact:
                home_missing = compute_missing_player_impact(
                    home_profile,
                    player_averages.get(home_id, {}),
                    player_points_weight,
                    player_pm_weight,
                    player_impact_scale,
                )
                away_missing = compute_missing_player_impact(
                    away_profile,
                    player_averages.get(away_id, {}),
                    player_points_weight,
                    player_pm_weight,
                    player_impact_scale,
                )
                missing_adj = -home_missing + away_missing
        except Exception:
            roster_adj = 0.0
            missing_adj = 0.0

    expected_margin = base_margin + rest_adj + b2b_adj + time_adj + tz_adj + roster_adj + missing_adj
    home_win_prob = sigmoid(expected_margin / margin_scale)
    away_win_prob = 1.0 - home_win_prob

    return home_name, away_name, expected_margin, home_win_prob, away_win_prob


def print_predictions(
    games: List[dict],
    stats: Dict[str, TeamStats],
    player_averages: Dict[str, Dict[str, dict]],
    include_rosters: bool,
    include_player_impact: bool,
    home_advantage: float,
    margin_scale: float,
    rest_weight: float,
    time_weight: float,
    roster_weight: float,
    b2b_weight: float,
    time_zone_weight: float,
    preferred_local_hour: int,
    player_points_weight: float,
    player_pm_weight: float,
    player_impact_scale: float,
) -> None:
    if not games:
        print("No upcoming games found in the selected date range.")
        return

    print("\nMatchup predictions (win prob split & expected margin):\n")
    for game in games:
        home_name, away_name, margin, home_prob, away_prob = predict_game(
            game,
            stats,
            player_averages,
            include_rosters,
            include_player_impact,
            home_advantage,
            margin_scale,
            rest_weight,
            time_weight,
            roster_weight,
            b2b_weight,
            time_zone_weight,
            preferred_local_hour,
            player_points_weight,
            player_pm_weight,
            player_impact_scale,
        )
        scheduled = game.get("scheduled")
        date_label = "Unknown date"
        if scheduled:
            try:
                date_label = datetime.fromisoformat(scheduled.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                pass
        print(
            f"{date_label}: {away_name} @ {home_name} | "
            f"Expected Margin (home): {margin:+.1f} | "
            f"Win Prob: home {home_prob*100:.1f}% / away {away_prob*100:.1f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matchup model with win probability + margin")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--half-life", type=float, default=10.0, help="Half-life in games for recency weighting")
    parser.add_argument("--include-rosters", action="store_true", help="Fetch team profiles to include roster size")
    parser.add_argument(
        "--include-player-impact",
        action="store_true",
        help="Use inactive players and recent per-game stats to adjust expected margin",
    )
    parser.add_argument("--home-adv", type=float, default=2.5, help="Home court advantage in points")
    parser.add_argument("--margin-scale", type=float, default=8.5, help="Scale for converting margin to win prob")
    parser.add_argument("--rest-weight", type=float, default=0.5, help="Points per rest-day advantage")
    parser.add_argument("--time-weight", type=float, default=0.2, help="Points added for earlier start times")
    parser.add_argument("--roster-weight", type=float, default=0.1, help="Points per active player difference")
    parser.add_argument("--b2b-weight", type=float, default=1.0, help="Points penalty for back-to-back games")
    parser.add_argument("--time-zone-weight", type=float, default=0.1, help="Points per hour away from preferred local time")
    parser.add_argument("--preferred-local-hour", type=int, default=19, help="Preferred local start time (hour 0-23)")
    parser.add_argument("--player-points-weight", type=float, default=1.0, help="Weight for player PPG in impact")
    parser.add_argument("--player-pm-weight", type=float, default=0.25, help="Weight for player +/- in impact")
    parser.add_argument("--player-impact-scale", type=float, default=0.1, help="Scale applied to missing player impact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DATA_DIR.exists():
        print("No data/ directory found. Download game data first.")
        return

    history, names = build_team_history(DATA_DIR)
    if not history:
        print("No completed games found in data/. Download games first.")
        return

    team_stats: Dict[str, TeamStats] = {}
    for team_id, games in history.items():
        team_stats[team_id] = compute_team_stats(team_id, games, names.get(team_id, team_id), args.half_life)

    start_date = date.fromisoformat(args.start) if args.start else date.today()
    end_date = date.fromisoformat(args.end) if args.end else (start_date + timedelta(days=7))

    games = get_upcoming_games(start_date, end_date)

    player_averages: Dict[str, Dict[str, dict]] = {}
    if args.include_player_impact:
        team_ids = set()
        for game in games:
            home_id = game.get("home", {}).get("id")
            away_id = game.get("away", {}).get("id")
            if home_id:
                team_ids.add(home_id)
            if away_id:
                team_ids.add(away_id)

        for team_id in sorted(team_ids):
            team_games = history.get(team_id, [])
            player_averages[team_id] = build_player_averages(
                team_id,
                team_games,
                args.half_life,
                max_games=10,
            )

    print_predictions(
        games,
        team_stats,
        player_averages,
        args.include_rosters,
        args.include_player_impact,
        args.home_adv,
        args.margin_scale,
        args.rest_weight,
        args.time_weight,
        args.roster_weight,
        args.b2b_weight,
        args.time_zone_weight,
        args.preferred_local_hour,
        args.player_points_weight,
        args.player_pm_weight,
        args.player_impact_scale,
    )


if __name__ == "__main__":
    main()
