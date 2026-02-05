"""
Print tonight's spread predictions as a table plus summaries for thresholds 0/5/10.
"""
import argparse
from datetime import date
from pathlib import Path
import sys

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_api import get_espn_games_for_date
from data_fetching.odds_api import get_nba_odds, parse_odds_for_game
from ml.build_dataset_optimized import build_features_for_game, build_team_narrative_history
from models.injury_adjustment import apply_injury_adjustment, build_injury_adjustments
from models.matchup_model import build_team_history, DATA_DIR
from evaluation.live_ats_tracking import save_predictions


FEATURE_COLS = [
    "home_weighted_margin",
    "away_weighted_margin",
    "home_weighted_win_pct",
    "away_weighted_win_pct",
    "home_recent_10_win_pct",
    "away_recent_10_win_pct",
    "home_weighted_points_for",
    "away_weighted_points_for",
    "home_weighted_points_against",
    "away_weighted_points_against",
    "home_weighted_point_diff",
    "away_weighted_point_diff",
    "home_recent_margin_3",
    "home_recent_margin_5",
    "home_recent_margin_10",
    "away_recent_margin_3",
    "away_recent_margin_5",
    "away_recent_margin_10",
    "home_recent_win_pct_3",
    "home_recent_win_pct_5",
    "away_recent_win_pct_3",
    "away_recent_win_pct_5",
    "home_recent_points_for_5",
    "home_recent_points_against_5",
    "away_recent_points_for_5",
    "away_recent_points_against_5",
    "home_blown_rate_10",
    "away_blown_rate_10",
    "home_clutch_margin_10",
    "away_clutch_margin_10",
    "home_max_lead_10",
    "away_max_lead_10",
    "home_h2h_margin_avg",
    "home_h2h_win_pct",
    "h2h_games_played",
    "rest_diff",
    "home_b2b",
    "away_b2b",
    "home_games_played",
    "away_games_played",
    "market_spread",
]


def normalize(name: str) -> str:
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


def build_name_index(names):
    return {normalize(display): team_id for team_id, display in names.items()}


def find_team_id(team_name: str, name_index):
    norm = normalize(team_name)
    if norm in name_index:
        return name_index[norm]
    for key, team_id in name_index.items():
        if key in norm or norm in key:
            return team_id
    return None


def build_fake_game(game_date, home_id, home_name, away_id, away_name):
    return {
        "scheduled": f"{game_date.isoformat()}T00:00:00Z",
        "home": {"id": home_id, "market": home_name, "name": ""},
        "away": {"id": away_id, "market": away_name, "name": ""},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tonight predictions with summary")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--model", default="ml_data/best_model_with_spreads.joblib")
    parser.add_argument("--fast", action="store_true", help="Use cached injury dataset only")
    return parser.parse_args()


def generate_rows(target_date: date, model_path: str, use_cached_dataset: bool = False):
    model = joblib.load(model_path)
    history, names = build_team_history(DATA_DIR)
    narrative_history = build_team_narrative_history(DATA_DIR)
    name_index = build_name_index(names)
    injury_adjustments = build_injury_adjustments(target_date, use_cached_dataset=use_cached_dataset)

    games = get_espn_games_for_date(target_date)
    if not games:
        return [], {}

    odds_data = get_nba_odds(markets="spreads")
    odds_by_matchup = {}
    for game_odds in odds_data:
        parsed = parse_odds_for_game(game_odds)
        if not parsed:
            continue
        key = (parsed.get("away_team"), parsed.get("home_team"))
        odds_by_matchup[key] = parsed

    rows = []
    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]

        home_id = find_team_id(home_team, name_index)
        away_id = find_team_id(away_team, name_index)
        if not home_id or not away_id:
            continue

        odds = odds_by_matchup.get((away_team, home_team))
        if not odds:
            continue

        market_spread = odds.get("away_spread")
        if market_spread is None:
            continue

        fake_game = build_fake_game(target_date, home_id, names[home_id], away_id, names[away_id])
        features = build_features_for_game(
            fake_game,
            history,
            half_life=10.0,
            market_spread=float(market_spread),
            narrative_history=narrative_history,
        )
        if not features:
            continue

        X = pd.DataFrame([features])[FEATURE_COLS]
        pred_home_margin = float(model.predict(X)[0])
        pred_away_margin = -pred_home_margin
        pred_away_adj = apply_injury_adjustment(away_team, home_team, pred_away_margin, injury_adjustments)
        edge = pred_away_adj + float(market_spread)

        rows.append({
            "game": f"{away_team} @ {home_team}",
            "away_team": away_team,
            "home_team": home_team,
            "market_spread": float(market_spread),
            "pred_away_margin": pred_away_margin,
            "pred_away_adj": pred_away_adj,
            "injury_adjustment": pred_away_adj - pred_away_margin,
            "edge": edge,
        })
    return rows, names


def summarize_rows(rows):
    def pick_for_threshold(t):
        picks = []
        for r in rows:
            if r["edge"] >= t:
                picks.append(f"{abbr(r['away_team'])} {r['market_spread']:+.1f}")
            elif r["edge"] <= -t:
                home_spread = -r["market_spread"]
                picks.append(f"{abbr(r['home_team'])} {home_spread:+.1f}")
        return picks

    summaries = {}
    for t in [0, 5, 10]:
        summaries[t] = pick_for_threshold(t)
    return summaries


def threshold_stats(model_path: str, data_path: str, cutoff: date, thresholds):
    df = pd.read_csv(data_path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    test_df = df[df["game_date"] >= cutoff]

    model = joblib.load(model_path)
    X = test_df[FEATURE_COLS]
    y = test_df["actual_margin"]
    preds = model.predict(X)

    away_actual_margin = -y
    away_pred_margin = -preds
    line = test_df["market_spread"]
    valid = line.abs() > 0

    away_actual_margin = away_actual_margin[valid]
    away_pred_margin = away_pred_margin[valid]
    line = line[valid]

    actual_diff = away_actual_margin + line

    results = {}
    for t in thresholds:
        take_away = away_pred_margin + line >= t
        take_home = away_pred_margin + line <= -t
        take = take_away | take_home

        if take.any():
            pred_diff = away_pred_margin[take] + line[take]
            actual_diff_take = actual_diff[take]
            non_push = actual_diff_take != 0
            if non_push.any():
                acc = (pred_diff[non_push] > 0).eq(actual_diff_take[non_push] > 0).mean()
                results[t] = (int(non_push.sum()), acc)
            else:
                results[t] = (0, None)
        else:
            results[t] = (0, None)

    return results


def format_summary(target_date: date, rows):
    thresholds = [0, 5, 10]
    stats = threshold_stats(
        model_path="ml_data/best_model_with_spreads.joblib",
        data_path="ml_data/games_optimized.csv",
        cutoff=date(2026, 1, 16),
        thresholds=thresholds,
    )
    lines = []
    lines.append(f"Predictions for {target_date.isoformat()}")
    lines.append("Game | Line(A) | PredLn(A) | AdjLn(A) | Edge")
    for r in rows:
        pred_line = -r["pred_away_margin"]
        adj_line = -r["pred_away_adj"]
        away_abbr = abbr(r["away_team"])
        lines.append(
            f"{r['game']} | {away_abbr} {r['market_spread']:+.1f} | {away_abbr} {pred_line:+.1f} | {away_abbr} {adj_line:+.1f} | {r['edge']:+.1f}"
        )

    summaries = summarize_rows(rows)
    for t in thresholds:
        n, acc = stats.get(t, (0, None))
        acc_str = f"{acc*100:.1f}%" if acc is not None else "n/a"
        lines.append("")
        lines.append(f"{t} threshold (ATS: {acc_str}, n={n}):")
        if summaries[t]:
            lines.extend([f"- {p}" for p in summaries[t]])
        else:
            lines.append("- (no picks)")

    return "\n".join(lines), summaries


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    rows, _ = generate_rows(target_date, args.model, use_cached_dataset=args.fast)
    if not rows:
        print("No games with odds and features found.")
        return

    save_predictions(target_date, rows)

    print(f"\nPredictions for {target_date.isoformat()}\n")
    print("Game".ljust(40), "Line(A)".rjust(10), "PredLn(A)".rjust(12), "AdjLn(A)".rjust(12), "Edge".rjust(8))
    for r in rows:
        pred_line = -r["pred_away_margin"]
        adj_line = -r["pred_away_adj"]
        away_abbr = abbr(r["away_team"])
        print(
            r["game"].ljust(40),
            f"{away_abbr} {r['market_spread']:+.1f}".rjust(10),
            f"{away_abbr} {pred_line:+.1f}".rjust(12),
            f"{away_abbr} {adj_line:+.1f}".rjust(12),
            f"{r['edge']:+.1f}".rjust(8),
        )

    summaries = summarize_rows(rows)
    stats = threshold_stats(
        model_path=args.model,
        data_path="ml_data/games_optimized.csv",
        cutoff=date(2026, 1, 16),
        thresholds=[0, 5, 10],
    )
    for t in [0, 5, 10]:
        n, acc = stats.get(t, (0, None))
        acc_str = f"{acc*100:.1f}%" if acc is not None else "n/a"
        picks = summaries[t]
        print(f"\n{t} threshold (ATS: {acc_str}, n={n}):")
        if picks:
            for p in picks:
                print(f"  {p}")
        else:
            print("  (no picks)")


if __name__ == "__main__":
    main()
