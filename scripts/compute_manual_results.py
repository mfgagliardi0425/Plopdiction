import sys
from datetime import date

sys.path.append("src")

from evaluation.live_ats_tracking import _load_game_results

TEAM_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def abbr(team_name: str) -> str:
    return TEAM_ABBR.get(team_name, team_name)


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
            details.append(f"{key}: PUSH (Line {line:+.1f}, Adj {adj:+.1f}, Actual {away_margin:+.1f})")
            continue
        total += 1
        win = (actual_diff > 0) == (pred_diff > 0)
        if win:
            wins += 1
        label = "W" if win else "L"
        pred_line = -adj
        actual_line = -away_margin
        away_abbr = abbr(away)
        details.append(
            f"{key}: {label} (Line(A) {away_abbr} {line:+.1f}, PredLn(A) {away_abbr} {pred_line:+.1f}, ActualLn(A) {away_abbr} {actual_line:+.1f})"
        )

    print(f"Total graded: {total} | Wins: {wins} | Losses: {total - wins}")
    print("\n".join(details))


if __name__ == "__main__":
    main()
