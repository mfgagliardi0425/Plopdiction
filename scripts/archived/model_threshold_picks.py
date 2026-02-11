"""Print threshold picks per model for a given date using ESPN closing spreads."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, "src")

import joblib
import pandas as pd

from data_fetching.espn_api import get_espn_games_for_date
from evaluation.tonight_spread_predictions_summary import (
    FEATURE_COLS,
    build_name_index,
    find_team_id,
    build_fake_game,
)
from evaluation.spread_utils import format_team_spread
from ml.build_dataset_optimized import (
    build_features_for_game,
    build_team_narrative_history,
    get_espn_spread_for_game,
    load_espn_spreads_cache,
    load_espn_spreads_detailed_cache,
)
from models.injury_adjustment import apply_injury_adjustment, build_injury_adjustments
from models.matchup_model import build_team_history, DATA_DIR

MODEL_PATHS = [
    Path("ml_data/best_model_with_spreads.joblib"),
    Path("ml_data/best_model.joblib"),
    Path("ml_data/best_model_optimized.joblib"),
    Path("ml_data/ridge_model.joblib"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model picks by threshold")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    model_map = {}
    for path in MODEL_PATHS:
        if path.exists():
            model_map[path.name] = joblib.load(path)

    if not model_map:
        print("No models found.")
        return

    history, names = build_team_history(DATA_DIR)
    narrative_history = build_team_narrative_history(DATA_DIR)
    name_index = build_name_index(names)

    espn_spreads = load_espn_spreads_cache()
    espn_detailed = load_espn_spreads_detailed_cache()

    injury_adjustments = build_injury_adjustments(target_date, use_cached_dataset=args.fast)

    games = get_espn_games_for_date(target_date)
    if not games:
        print("No games found for that date.")
        return

    rows = []
    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]

        home_id = find_team_id(home_team, name_index)
        away_id = find_team_id(away_team, name_index)
        if not home_id or not away_id:
            continue

        fake_game = build_fake_game(target_date, home_id, names[home_id], away_id, names[away_id])
        market_spread = get_espn_spread_for_game(fake_game, espn_spreads, espn_detailed)
        if not market_spread:
            continue

        features = build_features_for_game(
            fake_game,
            history,
            half_life=10.0,
            market_spread=float(market_spread),
            narrative_history=narrative_history,
        )
        if not features:
            continue

        X_full = pd.DataFrame([features])

        preds = {}
        for name, model in model_map.items():
            if hasattr(model, "feature_names_in_"):
                model_cols = list(model.feature_names_in_)
            else:
                model_cols = FEATURE_COLS
            for col in model_cols:
                if col not in X_full.columns:
                    X_full[col] = 0.0
            X = X_full[model_cols]

            pred_home_margin = float(model.predict(X)[0])
            pred_away_margin = -pred_home_margin
            pred_away_adj = apply_injury_adjustment(away_team, home_team, pred_away_margin, injury_adjustments)
            edge = pred_away_adj + float(market_spread)

            preds[name] = {
                "edge": edge,
                "market_spread": float(market_spread),
                "away_team": away_team,
                "home_team": home_team,
            }

        rows.append({
            "away_team": away_team,
            "home_team": home_team,
            "market_spread": float(market_spread),
            "preds": preds,
        })

    if not rows:
        print("No games with ESPN closing spreads found.")
        return

    thresholds = [0, 5, 10]
    print(f"\nModel picks for {target_date.isoformat()} (ESPN closing spreads)\n")

    for model_name in model_map:
        print(f"{model_name}:")
        for t in thresholds:
            picks = []
            for r in rows:
                p = r["preds"][model_name]
                edge = p["edge"]
                if edge >= t:
                    picks.append(format_team_spread(p["away_team"], p["market_spread"]))
                elif edge <= -t:
                    home_spread = -p["market_spread"]
                    picks.append(format_team_spread(p["home_team"], home_spread))
            print(f"  {t} threshold:")
            if picks:
                for pick in picks:
                    print(f"    {pick}")
            else:
                print("    (no picks)")
        print("")


if __name__ == "__main__":
    main()
