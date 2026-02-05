"""Track live ATS performance using injury-adjusted predictions."""
import json
from datetime import date
from pathlib import Path
from typing import Dict, List

from models.matchup_model import DATA_DIR, extract_points

TRACK_DIR = Path("tracking/live_ats")
TRACK_DIR.mkdir(parents=True, exist_ok=True)


def save_predictions(target_date: date, rows: List[dict]) -> Path:
    path = TRACK_DIR / f"{target_date.isoformat()}_predictions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": target_date.isoformat(), "games": rows}, f, indent=2)
    return path


def _load_game_results(target_date: date) -> Dict[str, dict]:
    day_dir = Path(DATA_DIR) / target_date.isoformat()
    results = {}
    if not day_dir.exists():
        return results
    for file_path in day_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
            home = game.get("home", {})
            away = game.get("away", {})
            home_points = extract_points(home)
            away_points = extract_points(away)
            if home_points is None or away_points is None:
                continue
            home_name = f"{home.get('market','')} {home.get('name','')}".strip()
            away_name = f"{away.get('market','')} {away.get('name','')}".strip()
            results[f"{away_name} @ {home_name}"] = {
                "home_points": home_points,
                "away_points": away_points,
                "away_margin": away_points - home_points,
            }
        except Exception:
            continue
    return results


def evaluate_date(target_date: date) -> dict:
    pred_path = TRACK_DIR / f"{target_date.isoformat()}_predictions.json"
    if not pred_path.exists():
        return {"date": target_date.isoformat(), "error": "missing predictions"}

    with open(pred_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = _load_game_results(target_date)
    if not results:
        return {"date": target_date.isoformat(), "error": "missing results"}

    total = 0
    correct = 0
    for game in payload.get("games", []):
        key = game.get("game")
        if key not in results:
            continue
        line = float(game.get("market_spread", 0.0))
        if line == 0:
            continue
        away_margin = results[key]["away_margin"]
        pred_away_adj = float(game.get("pred_away_adj", 0.0))

        actual_diff = away_margin + line
        pred_diff = pred_away_adj + line

        if actual_diff == 0:
            continue  # push

        total += 1
        if (actual_diff > 0) == (pred_diff > 0):
            correct += 1

    accuracy = (correct / total) if total else None
    summary = {
        "date": target_date.isoformat(),
        "total": total,
        "correct": correct,
        "ats_accuracy": accuracy,
    }

    out_path = TRACK_DIR / f"{target_date.isoformat()}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
