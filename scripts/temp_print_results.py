import json
import sys
from datetime import date
from pathlib import Path

sys.path.append("src")

from evaluation.live_ats_tracking import _load_game_results
from evaluation.send_results_to_discord import TRACK_DIR


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
        pred_adj = float(game.get("pred_away_adj", 0.0))

        actual_diff = away_margin + line
        pred_diff = pred_adj + line
        if actual_diff == 0:
            continue

        total += 1
        win = (actual_diff > 0) == (pred_diff > 0)
        if win:
            correct += 1
        label = "W" if win else "L"
        lines.append(
            f"{key}: {label} (Line {line:+.1f}, PredAdj {pred_adj:+.1f}, Actual {away_margin:+.1f})"
        )

    print(f"Total: {total} | Wins: {correct} | Losses: {total - correct}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
