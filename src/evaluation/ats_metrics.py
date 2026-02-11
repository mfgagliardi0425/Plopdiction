"""ATS metrics helpers using ESPN away-spread conventions."""
from __future__ import annotations

from typing import Any, Dict, Iterable

EDGE_THRESHOLD = 3.0


def compute_ats_metrics(away_margin: float, pred_away_margin: float, line: float) -> Dict[str, Any]:
    actual_diff = away_margin + line
    pred_diff = pred_away_margin + line

    if actual_diff == 0:
        result = "PUSH"
    else:
        result = "W" if (actual_diff > 0) == (pred_diff > 0) else "L"

    edge = pred_diff
    edge_opportunity = abs(edge) >= EDGE_THRESHOLD
    edge_pick = "AWAY" if edge > 0 else "HOME" if edge < 0 else "PUSH"
    edge_hit = None
    if actual_diff != 0 and edge_pick != "PUSH":
        edge_hit = (edge > 0) == (actual_diff > 0)

    return {
        "away_margin": away_margin,
        "pred_away_margin": pred_away_margin,
        "line": line,
        "actual_diff": actual_diff,
        "pred_diff": pred_diff,
        "result": result,
        "edge": edge,
        "edge_opportunity": edge_opportunity,
        "edge_pick": edge_pick,
        "edge_hit": edge_hit,
        "model_error": abs(pred_away_margin - away_margin),
        "market_error": abs(line - away_margin),
    }


def summarize_ats_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    wins = losses = pushes = 0
    edge_opps = edge_bets = edge_wins = 0
    model_errors = []
    market_errors = []

    for r in rows:
        result = r.get("result")
        if result == "W":
            wins += 1
        elif result == "L":
            losses += 1
        else:
            pushes += 1

        model_errors.append(r.get("model_error", 0.0))
        market_errors.append(r.get("market_error", 0.0))

        if r.get("edge_opportunity"):
            edge_opps += 1
            if r.get("result") != "PUSH":
                edge_bets += 1
                if r.get("edge_hit"):
                    edge_wins += 1

    graded = wins + losses
    ats_accuracy = (wins / graded) if graded else None
    edge_hit_rate = (edge_wins / edge_bets) if edge_bets else None

    model_mae = (sum(model_errors) / len(model_errors)) if model_errors else None
    market_mae = (sum(market_errors) / len(market_errors)) if market_errors else None

    return {
        "total_games": graded + pushes,
        "graded_games": graded,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_accuracy": ats_accuracy,
        "edge_opportunities": edge_opps,
        "edge_bets": edge_bets,
        "edge_wins": edge_wins,
        "edge_hit_rate": edge_hit_rate,
        "model_mae": model_mae,
        "market_mae": market_mae,
    }
