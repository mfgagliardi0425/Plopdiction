"""Send ATS results summary to Discord."""
import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ats_metrics import compute_ats_metrics, summarize_ats_metrics
from evaluation.discord_notifier import send_discord_message
from evaluation.live_ats_tracking import _load_game_results, save_predictions
from evaluation.results_db import save_results
from evaluation.spread_utils import away_margin_to_spread, format_team_spread, normalize_team_name
from evaluation.tonight_spread_predictions_summary import build_fake_game, build_name_index, find_team_id, FEATURE_COLS
from ml.build_dataset_optimized import build_features_for_game, load_espn_spreads_cache, load_espn_spreads_detailed_cache
from models.matchup_model import build_team_history, DATA_DIR
import pandas as pd
import joblib

TRACK_DIR = Path("tracking/live_ats")


def _get_cached_spread(target_date: date, away_team: str, home_team: str, espn_detailed: dict, espn_simple: dict) -> float | None:
    date_str = target_date.isoformat()
    away_norm = normalize_team_name(away_team)
    home_norm = normalize_team_name(home_team)

    # Prefer detailed cache
    for key, entry in espn_detailed.items():
        if not key.startswith(date_str):
            continue
        entry_home = _normalize(entry.get("home_team", ""))
        entry_away = _normalize(entry.get("away_team", ""))
        if (entry_home in home_norm or home_norm in entry_home) and (entry_away in away_norm or away_norm in entry_away):
            spread = entry.get("closing_spread_away")
            if spread is not None:
                return float(spread)
        if (entry_home in away_norm or away_norm in entry_home) and (entry_away in home_norm or home_norm in entry_away):
            spread = entry.get("closing_spread_away")
            if spread is not None:
                return -float(spread)

    # Fallback to simple cache (date + home token)
    home_token = home_team.split()[-1] if home_team else ""
    key = f"{date_str}_{home_token}"
    if key in espn_simple:
        spread = espn_simple[key]
        if spread is not None:
            return float(spread)

    away_token = away_team.split()[-1] if away_team else ""
    key = f"{date_str}_{away_token}"
    if key in espn_simple:
        spread = espn_simple[key]
        if spread is not None:
            return -float(spread)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Send ATS results to Discord")
    parser.add_argument("--date", help="Date to evaluate (YYYY-MM-DD); defaults to yesterday")
    return parser.parse_args()


def chunk_lines(lines: list[str], limit: int = 1900) -> list[str]:
    chunks = []
    current = ""
    for line in lines:
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _load_predictions_payload(pred_path: Path) -> dict | None:
    try:
        with open(pred_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def send_results_for_date(target_date: date) -> bool:
    pred_path = TRACK_DIR / f"{target_date.isoformat()}_predictions.json"
    payload = _load_predictions_payload(pred_path) if pred_path.exists() else None
    if not payload or not payload.get("games"):
        # Build predictions using ESPN closing spreads as fallback
        history, names = build_team_history(DATA_DIR)
        name_index = build_name_index(names)
        espn_simple = load_espn_spreads_cache()
        espn_detailed = load_espn_spreads_detailed_cache()
        model = joblib.load("ml_data/best_model_with_spreads.joblib")

        results = _load_game_results(target_date)
        games = results.get("by_key", {})
        if not games:
            print("Missing predictions file and results for that date.")
            return False

        rows = []
        for game_key, game_result in games.items():
            try:
                away_team, home_team = game_key.split(" @ ")
            except ValueError:
                continue

            home_id = find_team_id(home_team, name_index)
            away_id = find_team_id(away_team, name_index)
            if not home_id or not away_id:
                continue

            market_spread = _get_cached_spread(target_date, away_team, home_team, espn_detailed, espn_simple)
            if market_spread is None:
                continue

            fake_game = build_fake_game(target_date, home_id, names[home_id], away_id, names[away_id])
            features = build_features_for_game(fake_game, history, half_life=10.0, market_spread=float(market_spread))
            if not features:
                continue

            X = pd.DataFrame([features])[FEATURE_COLS]
            pred_home_margin = float(model.predict(X)[0])
            pred_away_margin = -pred_home_margin
            pred_away_spread = -pred_away_margin

            rows.append({
                "game": game_key,
                "away_team": away_team,
                "home_team": home_team,
                "away_id": game_result.get("away_id"),
                "home_id": game_result.get("home_id"),
                "market_spread": float(market_spread),
                "pred_away_margin": pred_away_margin,
                "pred_away_adj": pred_away_margin,
                "pred_away_spread": pred_away_spread,
                "pred_away_spread_adj": pred_away_spread,
            })

        if not rows:
            print("Unable to reconstruct predictions for that date.")
            return False

        save_predictions(target_date, rows)
        payload = {"date": target_date.isoformat(), "games": rows}

    results = _load_game_results(target_date)
    if not any(results.values()):
        print("Missing results for that date.")
        return False

    lines = [f"ATS Results for {target_date.isoformat()}:"]
    games_out = []
    metrics_rows = []

    for game in payload.get("games", []):
        key = game.get("game")
        result = None
        away_id = game.get("away_id")
        home_id = game.get("home_id")
        if away_id and home_id:
            result = results["by_id"].get(f"{away_id}@{home_id}")
        if result is None and key:
            result = results["by_key"].get(key)
        if result is None:
            away_team = game.get("away_team")
            home_team = game.get("home_team")
            if (not away_team or not home_team) and isinstance(key, str) and " @ " in key:
                away_team, home_team = key.split(" @ ", 1)
            if away_team and home_team:
                norm_key = f"{normalize_team_name(away_team)}@{normalize_team_name(home_team)}"
                result = results["by_norm"].get(norm_key)
        if result is None:
            continue
        line = float(game.get("market_spread", 0.0))
        if line == 0:
            continue
        away_margin = result["away_margin"]
        pred_away_spread_adj = game.get("pred_away_spread_adj")
        if pred_away_spread_adj is None:
            pred_away_spread_adj = -float(game.get("pred_away_adj", 0.0))
        pred_away_adj = -float(pred_away_spread_adj)

        metrics = compute_ats_metrics(away_margin, pred_away_adj, line)
        metrics_rows.append(metrics)
        result = metrics["result"]

        pred_line = float(pred_away_spread_adj)
        actual_line = away_margin_to_spread(away_margin)
        away_team = key.split(" @ ")[0] if isinstance(key, str) else ""
        lines.append(
            f"- {key} | Line(A) {format_team_spread(away_team, line)} | "
            f"PredLn(A) {format_team_spread(away_team, pred_line)} | "
            f"ActualLn(A) {format_team_spread(away_team, actual_line)} | {result}"
        )
        games_out.append({
            "game": key,
            "line_away": line,
            "pred_line_away": pred_line,
            "actual_line_away": actual_line,
            "metrics": metrics,
            "result": result,
        })

    summary = summarize_ats_metrics(metrics_rows)
    acc = summary.get("ats_accuracy")
    acc_str = f"{acc*100:.1f}%" if acc is not None else "n/a"
    lines.insert(1, f"Summary: {summary['wins']}/{summary['graded_games']} ATS ({acc_str})")

    save_results(target_date, summary, games_out)

    webhook = os.getenv("DISCORD_WEBHOOK_URL_RESULTS")
    if not webhook:
        print("DISCORD_WEBHOOK_URL_RESULTS not set.")
        return False

    chunks = chunk_lines(lines)
    for idx, chunk in enumerate(chunks, 1):
        header = f"ATS Results ({idx}/{len(chunks)}):"
        send_discord_message(f"{header}\n{chunk}", username="ATS Results", webhook_url=webhook)

    print("Sent ATS results.")
    return True


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=1))

    send_results_for_date(target_date)


if __name__ == "__main__":
    main()
