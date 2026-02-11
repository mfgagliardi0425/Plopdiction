"""Compare ATS accuracy across models for a single date using ESPN closing spreads."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, "src")

import joblib
import pandas as pd

from ml.build_dataset_optimized import (
    build_features_for_game,
    build_team_narrative_history,
    get_espn_spread_for_game,
    load_espn_spreads_cache,
    load_espn_spreads_detailed_cache,
)
from models.injury_adjustment import apply_injury_adjustment, build_injury_adjustments
from models.matchup_model import DATA_DIR, build_team_history, extract_points, parse_team_display, parse_game_date

MODEL_PATHS = [
    Path("ml_data/best_model_with_spreads.joblib"),
    Path("ml_data/best_model.joblib"),
    Path("ml_data/best_model_optimized.joblib"),
    Path("ml_data/ridge_model.joblib"),
]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ATS accuracy for a date")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    return parser.parse_args()


def load_models() -> dict[str, object]:
    model_map = {}
    for path in MODEL_PATHS:
        if path.exists():
            model_map[path.name] = joblib.load(path)
    return model_map


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    model_map = load_models()
    if not model_map:
        print("No models found.")
        return

    history, _ = build_team_history(DATA_DIR)
    narrative_history = build_team_narrative_history(DATA_DIR)

    espn_spreads = load_espn_spreads_cache()
    espn_detailed = load_espn_spreads_detailed_cache()

    injury_adjustments = build_injury_adjustments(target_date, use_cached_dataset=args.fast)

    day_dir = DATA_DIR / target_date.isoformat()
    if not day_dir.exists():
        print("No data folder for that date.")
        return

    games = []
    for file_path in day_dir.glob("*.json"):
        try:
            game = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            import json
            game = json.loads(game)
        except Exception:
            continue

        status = (game.get("status") or "").lower()
        if status not in {"closed", "complete", "completed", "final"}:
            continue

        home = game.get("home", {})
        away = game.get("away", {})
        home_id, home_name = parse_team_display(home)
        away_id, away_name = parse_team_display(away)

        home_points = extract_points(home)
        away_points = extract_points(away)
        if home_points is None or away_points is None:
            continue

        games.append({
            "game": game,
            "home_id": home_id,
            "home_name": home_name,
            "away_id": away_id,
            "away_name": away_name,
            "home_points": home_points,
            "away_points": away_points,
        })

    if not games:
        print("No completed games found.")
        return

    results = {name: {"wins": 0, "losses": 0, "pushes": 0} for name in model_map}

    for g in games:
        market_spread = get_espn_spread_for_game(g["game"], espn_spreads, espn_detailed)
        if not market_spread:
            continue

        features = build_features_for_game(
            g["game"],
            history,
            half_life=10.0,
            market_spread=float(market_spread),
            narrative_history=narrative_history,
        )
        if not features:
            continue

        X_full = pd.DataFrame([features])

        away_actual_margin = float(g["away_points"] - g["home_points"])
        actual_diff = away_actual_margin + float(market_spread)

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
            pred_away_adj = apply_injury_adjustment(
                g["away_name"],
                g["home_name"],
                pred_away_margin,
                injury_adjustments,
            )
            edge = pred_away_adj + float(market_spread)

            if actual_diff == 0:
                results[name]["pushes"] += 1
            elif edge == 0:
                results[name]["pushes"] += 1
            else:
                pick_away = edge > 0
                actual_away = actual_diff > 0
                if pick_away == actual_away:
                    results[name]["wins"] += 1
                else:
                    results[name]["losses"] += 1

    print(f"ATS results for {target_date.isoformat()} (ESPN closing spreads)")
    for name, record in results.items():
        wins = record["wins"]
        losses = record["losses"]
        pushes = record["pushes"]
        total = wins + losses
        acc = (wins / total) if total else None
        acc_str = f"{acc*100:.1f}%" if acc is not None else "n/a"
        print(f"{name}: {wins}-{losses}-{pushes} (ATS acc: {acc_str})")


if __name__ == "__main__":
    main()
