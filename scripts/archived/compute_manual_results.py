import sys
from datetime import date

sys.path.append("src")

from evaluation.live_ats_tracking import _load_game_results
from evaluation.spread_utils import format_team_spread, away_margin_to_spread

def main() -> None:
    d = date(2026, 2, 4)
    results = _load_game_results(d)

    preds = [
        ("Denver Nuggets", "New York Knicks", 4.5, 0.1),
        ("Minnesota Timberwolves", "Toronto Raptors", -1.5, 6.0),
        ("Boston Celtics", "Houston Rockets", 7.5, 6.8),
        ("New Orleans Pelicans", "Milwaukee Bucks", -5.5, 15.9),
        ("Oklahoma City Thunder", "San Antonio Spurs", 9.5, -14.5),
        ("Memphis Grizzlies", "Sacramento Kings", 1.5, -2.8),
    ]

    total = 0
    wins = 0
    details = []

    for away, home, line, adj in preds:
        key = f"{away} @ {home}"
        if key not in results:
            details.append(f"{key}: missing result")
            continue
        away_margin = results[key]["away_margin"]
        actual_diff = away_margin + line
        pred_diff = adj + line
        if actual_diff == 0:
            pred_spread = -adj
            actual_spread = away_margin_to_spread(away_margin)
            details.append(
                f"{key}: PUSH (Line(A) {format_team_spread(away, line)}, "
                f"PredLn(A) {format_team_spread(away, pred_spread)}, "
                f"ActualLn(A) {format_team_spread(away, actual_spread)})"
            )
            continue
        total += 1
        win = (actual_diff > 0) == (pred_diff > 0)
        if win:
            wins += 1
        label = "W" if win else "L"
        pred_line = -adj
        actual_line = away_margin_to_spread(away_margin)
        details.append(
            f"{key}: {label} (Line(A) {format_team_spread(away, line)}, "
            f"PredLn(A) {format_team_spread(away, pred_line)}, "
            f"ActualLn(A) {format_team_spread(away, actual_line)})"
        )

    print(f"Total graded: {total} | Wins: {wins} | Losses: {total - wins}")
    print("\n".join(details))


if __name__ == "__main__":
    main()
