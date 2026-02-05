"""Send ATS results summary to Discord."""
import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.discord_notifier import send_discord_message
from evaluation.live_ats_tracking import _load_game_results, save_predictions
from evaluation.results_db import save_results
from evaluation.tonight_spread_predictions_summary import build_fake_game, build_name_index, find_team_id, FEATURE_COLS
from ml.build_dataset_optimized import build_features_for_game, load_espn_spreads_cache, load_espn_spreads_detailed_cache
from models.matchup_model import build_team_history, DATA_DIR
import pandas as pd
import joblib

TRACK_DIR = Path("tracking/live_ats")


def _normalize(name: str) -> str:
    return " ".join("".join(ch for ch in (name or "") if ch.isalnum() or ch.isspace()).lower().split())


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


def _get_cached_spread(target_date: date, away_team: str, home_team: str, espn_detailed: dict, espn_simple: dict) -> float | None:
    date_str = target_date.isoformat()
    away_norm = _normalize(away_team)
    home_norm = _normalize(home_team)

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

    # Fallback to simple cache (date + home token)
    home_token = home_team.split()[-1] if home_team else ""
    key = f"{date_str}_{home_token}"
    if key in espn_simple:
        spread = espn_simple[key]
        if spread is not None:
            return float(spread)

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


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=1))

    pred_path = TRACK_DIR / f"{target_date.isoformat()}_predictions.json"
    if not pred_path.exists():
        # Build predictions using ESPN closing spreads as fallback
        history, names = build_team_history(DATA_DIR)
        name_index = build_name_index(names)
        espn_simple = load_espn_spreads_cache()
        espn_detailed = load_espn_spreads_detailed_cache()
        model = joblib.load("ml_data/best_model_with_spreads.joblib")

        games = _load_game_results(target_date)
        if not games:
            print("Missing predictions file and results for that date.")
            return

        rows = []
        for game_key in games.keys():
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

            rows.append({
                "game": game_key,
                "away_team": away_team,
                "home_team": home_team,
                "market_spread": float(market_spread),
                "pred_away_adj": pred_away_margin,
            })

        if not rows:
            print("Unable to reconstruct predictions for that date.")
            return

        save_predictions(target_date, rows)

    with open(pred_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = _load_game_results(target_date)
    if not results:
        print("Missing results for that date.")
        return

    total = 0
    correct = 0
    lines = [f"ATS Results for {target_date.isoformat()}:"]
    games_out = []

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

        result = "PUSH"
        if actual_diff != 0:
            total += 1
            result = "W" if (actual_diff > 0) == (pred_diff > 0) else "L"
            if result == "W":
                correct += 1

        pred_line = -pred_away_adj
        actual_line = -away_margin
        away_team = key.split(" @ ")[0] if isinstance(key, str) else ""
        away_abbr = abbr(away_team)
        lines.append(
            f"- {key} | Line(A) {away_abbr} {line:+.1f} | PredLn(A) {away_abbr} {pred_line:+.1f} | ActualLn(A) {away_abbr} {actual_line:+.1f} | {result}"
        )
        games_out.append({
            "game": key,
            "line_away": line,
            "pred_line_away": pred_line,
            "actual_line_away": actual_line,
            "result": result,
        })

    acc = f"{(correct/total)*100:.1f}%" if total else "n/a"
    lines.insert(1, f"Summary: {correct}/{total} ATS ({acc})")

    save_results(target_date, {"total": total, "correct": correct, "accuracy": acc}, games_out)

    webhook = os.getenv("DISCORD_WEBHOOK_URL_RESULTS")
    if not webhook:
        print("DISCORD_WEBHOOK_URL_RESULTS not set.")
        return

    chunks = chunk_lines(lines)
    for idx, chunk in enumerate(chunks, 1):
        header = f"ATS Results ({idx}/{len(chunks)}):"
        send_discord_message(f"{header}\n{chunk}", username="ATS Results", webhook_url=webhook)

    print("Sent ATS results.")


if __name__ == "__main__":
    main()
