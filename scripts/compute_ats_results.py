import argparse
import json
from datetime import date
from pathlib import Path
import sys

sys.path.append("src")

import joblib
import pandas as pd

from evaluation.live_ats_tracking import _load_game_results
from evaluation.tonight_spread_predictions_summary import FEATURE_COLS, build_fake_game, build_name_index, find_team_id
from ml.build_dataset_optimized import build_features_for_game, load_espn_spreads_cache, load_espn_spreads_detailed_cache
from models.matchup_model import build_team_history, DATA_DIR


def _normalize(name: str) -> str:
    return " ".join("".join(ch for ch in (name or "") if ch.isalnum() or ch.isspace()).lower().split())


TEAM_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def abbr(team_name: str) -> str:
    return TEAM_ABBR.get(team_name, team_name)


def _get_cached_spread(target_date: date, away_team: str, home_team: str, espn_detailed: dict, espn_simple: dict) -> float | None:
    date_str = target_date.isoformat()
    away_norm = _normalize(away_team)
    home_norm = _normalize(home_team)

    for key, entry in espn_detailed.items():
        if not key.startswith(date_str):
            continue
        entry_home = _normalize(entry.get("home_team", ""))
        entry_away = _normalize(entry.get("away_team", ""))
        if (entry_home in home_norm or home_norm in entry_home) and (entry_away in away_norm or away_norm in entry_away):
            spread = entry.get("closing_spread_away")
            if spread is not None:
                return float(spread)

    home_token = home_team.split()[-1] if home_team else ""
    key = f"{date_str}_{home_token}"
    if key in espn_simple:
        spread = espn_simple[key]
        if spread is not None:
            return float(spread)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ATS results for a date")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    results = _load_game_results(target_date)
    if not results:
        print("Missing results for that date.")
        return

    history, names = build_team_history(DATA_DIR)
    name_index = build_name_index(names)
    espn_simple = load_espn_spreads_cache()
    espn_detailed = load_espn_spreads_detailed_cache()

    model = joblib.load("ml_data/best_model_with_spreads.joblib")

    total = 0
    correct = 0
    lines = []

    for game_key in results.keys():
        try:
            away_team, home_team = game_key.split(" @ ")
        except ValueError:
            continue

        market_spread = _get_cached_spread(target_date, away_team, home_team, espn_detailed, espn_simple)
        if market_spread is None:
            continue

        home_id = find_team_id(home_team, name_index)
        away_id = find_team_id(away_team, name_index)
        if not home_id or not away_id:
            continue

        fake_game = build_fake_game(target_date, home_id, names[home_id], away_id, names[away_id])
        features = build_features_for_game(fake_game, history, half_life=10.0, market_spread=float(market_spread))
        if not features:
            continue

        X = pd.DataFrame([features])[FEATURE_COLS]
        pred_home_margin = float(model.predict(X)[0])
        pred_away_adj = -pred_home_margin

        away_margin = results[game_key]["away_margin"]
        actual_diff = away_margin + market_spread
        pred_diff = pred_away_adj + market_spread
        if actual_diff == 0:
            continue

        total += 1
        win = (actual_diff > 0) == (pred_diff > 0)
        correct += 1 if win else 0
        pred_line = -pred_away_adj
        actual_line = -away_margin
        away_abbr = abbr(away_team)
        lines.append(
            f"{game_key}: {'W' if win else 'L'} (Line(A) {away_abbr} {market_spread:+.1f}, PredLn(A) {away_abbr} {pred_line:+.1f}, ActualLn(A) {away_abbr} {actual_line:+.1f})"
        )

    print(f"Total: {total} | Wins: {correct} | Losses: {total - correct}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
