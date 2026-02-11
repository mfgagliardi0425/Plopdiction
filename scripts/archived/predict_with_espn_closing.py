"""Generate predictions for a past date using ESPN closing spreads cache."""
from __future__ import annotations

import argparse
from datetime import date
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
    summarize_rows,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predictions using ESPN closing spreads")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--model", default="ml_data/best_model_with_spreads.joblib")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    model = joblib.load(args.model)
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

        X = pd.DataFrame([features])[FEATURE_COLS]
        pred_home_margin = float(model.predict(X)[0])
        pred_away_margin = -pred_home_margin
        pred_away_adj = apply_injury_adjustment(away_team, home_team, pred_away_margin, injury_adjustments)
        edge = pred_away_adj + float(market_spread)

        pred_away_spread = -pred_away_margin
        pred_away_spread_adj = -pred_away_adj

        rows.append({
            "game": f"{away_team} @ {home_team}",
            "away_team": away_team,
            "home_team": home_team,
            "market_spread": float(market_spread),
            "pred_away_margin": pred_away_margin,
            "pred_away_adj": pred_away_adj,
            "pred_away_spread": pred_away_spread,
            "pred_away_spread_adj": pred_away_spread_adj,
            "edge": edge,
        })

    if not rows:
        print("No games with ESPN closing spreads found.")
        return

    print(f"\nPredictions for {target_date.isoformat()} (ESPN closing spreads)\n")
    print("Game".ljust(40), "Line(A)".rjust(10), "PredLn(A)".rjust(12), "AdjLn(A)".rjust(12), "Edge".rjust(8))
    for r in rows:
        pred_line = r.get("pred_away_spread", -r["pred_away_margin"])
        adj_line = r.get("pred_away_spread_adj", -r["pred_away_adj"])
        print(
            r["game"].ljust(40),
            format_team_spread(r["away_team"], r["market_spread"]).rjust(10),
            format_team_spread(r["away_team"], pred_line).rjust(12),
            format_team_spread(r["away_team"], adj_line).rjust(12),
            f"{r['edge']:+.1f}".rjust(8),
        )

    summaries = summarize_rows(rows)
    for t in [0, 5, 10]:
        picks = summaries[t]
        print(f"\n{t} threshold:")
        if picks:
            for p in picks:
                print(f"  {p}")
        else:
            print("  (no picks)")


if __name__ == "__main__":
    main()
