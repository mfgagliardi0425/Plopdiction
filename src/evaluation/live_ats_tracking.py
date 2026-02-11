"""Track live ATS performance using injury-adjusted predictions."""
import json
from datetime import date
from pathlib import Path
from typing import Dict, List

from evaluation.ats_metrics import compute_ats_metrics, summarize_ats_metrics
from evaluation.spread_utils import normalize_team_name
from evaluation.results_db import save_results
from models.matchup_model import DATA_DIR, extract_points

TRACK_DIR = Path("tracking/live_ats")
TRACK_DIR.mkdir(parents=True, exist_ok=True)


def save_predictions(target_date: date, rows: List[dict]) -> Path:
    path = TRACK_DIR / f"{target_date.isoformat()}_predictions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": target_date.isoformat(), "games": rows}, f, indent=2)
    return path


def _load_game_results(target_date: date) -> Dict[str, Dict[str, dict]]:
    day_dir = Path(DATA_DIR) / target_date.isoformat()
    results = {"by_key": {}, "by_id": {}, "by_norm": {}}
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
            home_id = home.get("id")
            away_id = away.get("id")
            payload = {
                "home_points": home_points,
                "away_points": away_points,
                "away_margin": away_points - home_points,
                "home_id": home_id,
                "away_id": away_id,
                "home_team": home_name,
                "away_team": away_name,
            }
            key = f"{away_name} @ {home_name}"
            results["by_key"][key] = payload
            if home_id and away_id:
                results["by_id"][f"{away_id}@{home_id}"] = payload
            away_norm = normalize_team_name(away_name)
            home_norm = normalize_team_name(home_name)
            if away_norm and home_norm:
                results["by_norm"][f"{away_norm}@{home_norm}"] = payload
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
    if not any(results.values()):
        return {"date": target_date.isoformat(), "error": "missing results"}

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

        metrics_rows.append(compute_ats_metrics(away_margin, pred_away_adj, line))

    summary = summarize_ats_metrics(metrics_rows)
    summary["date"] = target_date.isoformat()

    save_results(target_date, summary, metrics_rows)

    return summary
