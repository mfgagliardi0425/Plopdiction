"""Generate a consistent JSON analysis report for a given date."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, "src")

import joblib
import pandas as pd

from data_fetching.espn_api import get_espn_games_for_date
from data_fetching.odds_api import get_nba_odds, parse_odds_for_game
from evaluation.tonight_spread_predictions_summary import (
    FEATURE_COLS,
    build_name_index,
    find_team_id,
    build_fake_game,
    summarize_rows,
    threshold_stats,
)
from ml.build_dataset_optimized import build_features_for_game, build_team_narrative_history
from models.injury_adjustment import (
    POINTS_PER_PPG,
    RANK_MULTIPLIER,
    apply_injury_adjustment,
    build_injury_adjustments,
    ppg_multiplier,
)
from models.matchup_model import build_team_history, DATA_DIR

INJURY_DATASET_PATH = Path("data/injuries/injury_dataset.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily JSON analysis report")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--model", default="ml_data/best_model_with_spreads.joblib")
    parser.add_argument("--out-root", default="reports", help="Root folder for reports")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    return parser.parse_args()


def load_injury_dataset(target_date: date) -> dict | None:
    if not INJURY_DATASET_PATH.exists():
        return None
    try:
        with open(INJURY_DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception:
        return None
    if dataset.get("date") != target_date.isoformat():
        return None
    return dataset


def build_injury_details(team_name: str, dataset: dict | None) -> tuple[float, list[dict]]:
    if not dataset:
        return 0.0, []

    penalty = 0.0
    details = []
    for inj in dataset.get("teams", {}).get(team_name, []):
        status = (inj.get("status") or "").lower()
        if status != "out":
            continue
        ppg = inj.get("ppg")
        if ppg is None:
            continue
        base = float(ppg) * POINTS_PER_PPG
        rank = inj.get("ppg_rank")
        mult = RANK_MULTIPLIER.get(rank, 1.0) if rank else ppg_multiplier(float(ppg))
        adj = base * mult
        penalty += adj
        details.append({
            "player": inj.get("player"),
            "status": inj.get("status"),
            "ppg": ppg,
            "ppg_rank": rank,
            "penalty": adj,
            "comment": inj.get("comment"),
        })

    return penalty, details


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    out_dir = Path(args.out_root) / target_date.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    history, names = build_team_history(DATA_DIR)
    narrative_history = build_team_narrative_history(DATA_DIR)
    name_index = build_name_index(names)

    games = get_espn_games_for_date(target_date)
    if not games:
        print("No games found.")
        return

    odds_data = get_nba_odds(markets="spreads")
    odds_by_matchup = {}
    for game_odds in odds_data:
        parsed = parse_odds_for_game(game_odds)
        if not parsed:
            continue
        key = (parsed.get("away_team"), parsed.get("home_team"))
        odds_by_matchup[key] = parsed

    injury_dataset = load_injury_dataset(target_date)
    injury_adjustments = build_injury_adjustments(target_date, use_cached_dataset=args.fast)

    rows = []
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

        pred_away_spread = -pred_away_margin
        pred_away_spread_adj = -pred_away_adj
        injury_adjustment = pred_away_adj - pred_away_margin

        away_penalty, away_injuries = build_injury_details(away_team, injury_dataset)
        home_penalty, home_injuries = build_injury_details(home_team, injury_dataset)

        rows.append({
            "game": f"{away_team} @ {home_team}",
            "away_team": away_team,
            "home_team": home_team,
            "market_spread": float(market_spread),
            "pred_away_margin": pred_away_margin,
            "pred_away_adj": pred_away_adj,
            "pred_away_spread": pred_away_spread,
            "pred_away_spread_adj": pred_away_spread_adj,
            "injury_adjustment": injury_adjustment,
            "injury_adjustment_spread": -injury_adjustment,
            "edge": edge,
            "injuries": {
                "away_penalty": away_penalty,
                "home_penalty": home_penalty,
                "away_out": away_injuries,
                "home_out": home_injuries,
            },
            "features": {col: float(features.get(col, 0.0)) for col in FEATURE_COLS},
        })

    summaries = summarize_rows(rows)
    stats = threshold_stats(
        model_path=args.model,
        data_path="ml_data/games_optimized.csv",
        cutoff=date(2026, 1, 16),
        thresholds=[0, 5, 10],
    )

    output = {
        "date": target_date.isoformat(),
        "model_path": args.model,
        "injury_dataset_date": injury_dataset.get("date") if injury_dataset else None,
        "rows": rows,
        "threshold_summaries": summaries,
        "threshold_stats": {
            str(t): {"bets": n, "ats_acc": acc}
            for t, (n, acc) in stats.items()
        },
    }

    out_path = out_dir / "analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved analysis report: {out_path}")


if __name__ == "__main__":
    main()
