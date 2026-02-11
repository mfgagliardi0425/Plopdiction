import argparse
import json
from datetime import date
from pathlib import Path
import sys

sys.path.append("src")

import joblib
import pandas as pd

from evaluation.ats_metrics import compute_ats_metrics, summarize_ats_metrics
from evaluation.live_ats_tracking import _load_game_results, TRACK_DIR
from evaluation.results_db import save_results
from evaluation.spread_utils import format_team_spread, away_margin_to_spread
from evaluation.tonight_spread_predictions_summary import FEATURE_COLS, build_fake_game, build_name_index, find_team_id
from ml.build_dataset_optimized import build_features_for_game, load_espn_spreads_cache, load_espn_spreads_detailed_cache
from models.matchup_model import build_team_history, DATA_DIR


def _normalize(name: str) -> str:
    return " ".join("".join(ch for ch in (name or "") if ch.isalnum() or ch.isspace()).lower().split())


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

    pred_path = TRACK_DIR / f"{target_date.isoformat()}_predictions.json"
    preds_payload = None
    if pred_path.exists():
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                preds_payload = json.load(f)
        except Exception:
            preds_payload = None

    history, names = build_team_history(DATA_DIR)
    name_index = build_name_index(names)
    espn_simple = load_espn_spreads_cache()
    espn_detailed = load_espn_spreads_detailed_cache()

    model = joblib.load("ml_data/best_model_with_spreads.joblib")

    metrics_rows = []
    games_out = []
    lines = []

    if preds_payload and preds_payload.get("games"):
        for game in preds_payload.get("games", []):
            game_key = game.get("game")
            if not game_key:
                continue
            away_team = game.get("away_team", "")
            home_team = game.get("home_team", "")
            line = game.get("market_spread")
            if line is None:
                continue
            line = float(line)

            pred_away_spread_adj = game.get("pred_away_spread_adj")
            if pred_away_spread_adj is None:
                pred_away_spread_adj = -float(game.get("pred_away_adj", 0.0))
            pred_away_adj = -float(pred_away_spread_adj)

            if game_key not in results:
                lines.append(
                    f"{game_key}: MISSING RESULT (Line(A) {format_team_spread(away_team, line)}, "
                    f"PredLn(A) {format_team_spread(away_team, float(pred_away_spread_adj))})"
                )
                games_out.append({
                    "game": game_key,
                    "line_away": line,
                    "pred_line_away": float(pred_away_spread_adj),
                    "actual_line_away": None,
                    "metrics": None,
                })
                continue

            away_margin = results[game_key]["away_margin"]
            metrics = compute_ats_metrics(away_margin, pred_away_adj, line)
            metrics_rows.append(metrics)
            pred_line = float(pred_away_spread_adj)
            actual_line = away_margin_to_spread(away_margin)
            lines.append(
                f"{game_key}: {metrics['result']} (Line(A) {format_team_spread(away_team, line)}, "
                f"PredLn(A) {format_team_spread(away_team, pred_line)}, "
                f"ActualLn(A) {format_team_spread(away_team, actual_line)})"
            )
            games_out.append({
                "game": game_key,
                "line_away": line,
                "pred_line_away": pred_line,
                "actual_line_away": actual_line,
                "metrics": metrics,
            })
    else:
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
            metrics = compute_ats_metrics(away_margin, pred_away_adj, market_spread)
            metrics_rows.append(metrics)
            pred_line = -pred_away_adj
            actual_line = away_margin_to_spread(away_margin)
            lines.append(
                f"{game_key}: {metrics['result']} (Line(A) {format_team_spread(away_team, market_spread)}, "
                f"PredLn(A) {format_team_spread(away_team, pred_line)}, "
                f"ActualLn(A) {format_team_spread(away_team, actual_line)})"
            )
            games_out.append({
                "game": game_key,
                "line_away": market_spread,
                "pred_line_away": pred_line,
                "actual_line_away": actual_line,
                "metrics": metrics,
            })

    summary = summarize_ats_metrics(metrics_rows)
    print(
        f"Total: {summary['graded_games']} | Wins: {summary['wins']} | "
        f"Losses: {summary['losses']} | Pushes: {summary['pushes']}"
    )
    save_results(target_date, summary, games_out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
