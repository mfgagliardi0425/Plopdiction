"""
Predict tonight's games vs market spreads and apply edge thresholds.
Uses ML model trained on historical data with market_spread feature.
"""
import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_api import get_espn_games_for_date
from data_fetching.odds_api import get_nba_odds, parse_odds_for_game
from ml.build_dataset_optimized import build_features_for_game, build_team_narrative_history
from models.injury_adjustment import apply_injury_adjustment, build_injury_adjustments
from models.matchup_model import build_team_history, DATA_DIR


FEATURE_COLS = [
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
]


def normalize(name: str) -> str:
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


def build_name_index(names: Dict[str, str]) -> Dict[str, str]:
    return {normalize(display): team_id for team_id, display in names.items()}


def find_team_id(team_name: str, name_index: Dict[str, str]) -> Optional[str]:
    norm = normalize(team_name)
    if norm in name_index:
        return name_index[norm]
    # fallback partial match
    for key, team_id in name_index.items():
        if key in norm or norm in key:
            return team_id
    return None


def build_fake_game(game_date: date, home_id: str, home_name: str, away_id: str, away_name: str) -> dict:
    return {
        "scheduled": f"{game_date.isoformat()}T00:00:00Z",
        "home": {"id": home_id, "market": home_name, "name": ""},
        "away": {"id": away_id, "market": away_name, "name": ""},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tonight's spreads with thresholds")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date to predict (YYYY-MM-DD)")
    parser.add_argument("--model", default="ml_data/best_model_with_spreads.joblib", help="Path to trained model")
    parser.add_argument("--thresholds", default="0,5,10", help="Comma-separated edge thresholds")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    model = joblib.load(args.model)

    history, names = build_team_history(DATA_DIR)
    narrative_history = build_team_narrative_history(DATA_DIR)
    name_index = build_name_index(names)

    games = get_espn_games_for_date(target_date)
    if not games:
        print(f"No games found for {target_date.isoformat()}")
        return

    odds_data = get_nba_odds(markets="spreads")
    odds_by_matchup = {}
    for game_odds in odds_data:
        parsed = parse_odds_for_game(game_odds)
        if not parsed:
            continue
        key = (parsed.get("away_team"), parsed.get("home_team"))
        odds_by_matchup[key] = parsed

    print(f"\nPredictions for {target_date.isoformat()}")
    print("Game".ljust(40), "Line(A)".rjust(10), "PredLn(A)".rjust(12), "AdjLn(A)".rjust(12), "Edge".rjust(8), "T0/T5/T10")

    injury_adjustments = build_injury_adjustments(target_date, use_cached_dataset=args.fast)

    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]

        home_id = find_team_id(home_team, name_index)
        away_id = find_team_id(away_team, name_index)
        if not home_id or not away_id:
            continue

        odds = odds_by_matchup.get((away_team, home_team))
        if not odds:
            continue

        market_spread = odds.get("away_spread")
        if market_spread is None:
            continue

        fake_game = build_fake_game(target_date, home_id, names[home_id], away_id, names[away_id])
        features = build_features_for_game(
            fake_game,
            history,
            half_life=10.0,
            market_spread=float(market_spread),
            narrative_history=narrative_history,
        )
        if not features:
            continue

        X = pd.DataFrame([features])[FEATURE_COLS]
        pred_home_margin = float(model.predict(X)[0])
        pred_away_margin = -pred_home_margin
        pred_away_adj = apply_injury_adjustment(away_team, home_team, pred_away_margin, injury_adjustments)
        edge = pred_away_adj + float(market_spread)

        picks = []
        for t in thresholds:
            if edge >= t:
                picks.append("AWAY")
            elif edge <= -t:
                picks.append("HOME")
            else:
                picks.append("PASS")

        game_label = f"{away_team} @ {home_team}"
        pred_line = -pred_away_margin
        adj_line = -pred_away_adj
        away_abbr = abbr(away_team)
        print(
            game_label.ljust(40),
            f"{away_abbr} {market_spread:+.1f}".rjust(10),
            f"{away_abbr} {pred_line:+.1f}".rjust(12),
            f"{away_abbr} {adj_line:+.1f}".rjust(12),
            f"{edge:+.1f}".rjust(8),
            "/".join(picks),
        )


if __name__ == "__main__":
    main()
