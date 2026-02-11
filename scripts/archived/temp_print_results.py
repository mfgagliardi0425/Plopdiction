import json
import sys
from datetime import date
from pathlib import Path

sys.path.append("src")

from evaluation.live_ats_tracking import _load_game_results
from evaluation.send_results_to_discord import TRACK_DIR
from evaluation.spread_utils import format_team_spread, away_margin_to_spread


def main() -> None:
    d = date(2026, 2, 4)
    pred_path = TRACK_DIR / f"{d.isoformat()}_predictions.json"
    if not pred_path.exists():
        print("Missing predictions file.")
        return

    payload = json.load(open(pred_path, "r", encoding="utf-8"))
    results = _load_game_results(d)

    total = 0
    correct = 0
    lines = []

    for game in payload.get("games", []):
        key = game.get("game")
        if key not in results:
            continue
        line = float(game.get("market_spread", 0.0))
        if line == 0:
            continue
        away_margin = results[key]["away_margin"]
        pred_away_spread_adj = game.get("pred_away_spread_adj")
        if pred_away_spread_adj is None:
            pred_away_spread_adj = -float(game.get("pred_away_adj", 0.0))

        actual_diff = away_margin + line
        pred_diff = -float(pred_away_spread_adj) + line
        if actual_diff == 0:
            continue

        total += 1
        win = (actual_diff > 0) == (pred_diff > 0)
        if win:
            correct += 1
        label = "W" if win else "L"
        pred_line = float(pred_away_spread_adj)
        actual_line = away_margin_to_spread(away_margin)
        away_team = key.split(" @ ")[0] if isinstance(key, str) else ""
        lines.append(
            f"{key}: {label} (Line(A) {format_team_spread(away_team, line)}, "
            f"PredLn(A) {format_team_spread(away_team, pred_line)}, "
            f"ActualLn(A) {format_team_spread(away_team, actual_line)})"
        )

    print(f"Total: {total} | Wins: {correct} | Losses: {total - correct}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
