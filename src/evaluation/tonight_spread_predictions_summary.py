"""
Print tonight's spread predictions as a table plus summaries for thresholds 0/5/10.
"""
import argparse
from datetime import date, datetime, time
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_api import get_espn_games_for_date
from data_fetching.odds_api import (
    cache_odds_snapshot,
    get_nba_odds,
    get_opening_spreads_for_date,
    parse_odds_for_game,
)
from ml.build_dataset_optimized import build_features_for_game, build_team_narrative_history
from models.injury_adjustment import apply_injury_adjustment, build_injury_adjustments
from models.matchup_model import build_team_history, DATA_DIR
from evaluation.live_ats_tracking import save_predictions
from evaluation.spread_utils import format_team_spread, normalize_team_name


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
    "line_move",
]


def build_name_index(names):
    return {normalize_team_name(display): team_id for team_id, display in names.items()}


def find_team_id(team_name: str, name_index):
    norm = normalize_team_name(team_name)
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
    parser.add_argument(
        "--min-start-est",
        default=None,
        help="Only include games at/after this EST time (HH:MM)",
    )
    parser.add_argument(
        "--remaining",
        action="store_true",
        help="Only include games that are not in-progress or completed",
    )
    return parser.parse_args()


def _parse_est_cutoff(value: str | None) -> time | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError:
        raise ValueError("--min-start-est must be HH:MM (24h format)")


def _game_at_or_after_cutoff(game: dict, cutoff: time) -> bool:
    start_time = game.get("start_time_utc")
    if not start_time:
        return True
    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except ValueError:
        return True
    est_dt = start_dt.astimezone(ZoneInfo("America/New_York"))
    return est_dt.time() >= cutoff


def _is_remaining_game(game: dict) -> bool:
    status = game.get("status")
    if not isinstance(status, dict):
        return True
    if status.get("completed") is True:
        return False
    state = status.get("state")
    if state and state != "pre":
        return False
    return True


def generate_rows(
    target_date: date,
    model_path: str,
    use_cached_dataset: bool = False,
    min_start_est: str | None = None,
    remaining_only: bool = False,
):
    model = joblib.load(model_path)
    feature_cols = FEATURE_COLS
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    history, names = build_team_history(DATA_DIR)
    narrative_history = build_team_narrative_history(DATA_DIR)
    name_index = build_name_index(names)
    injury_adjustments = build_injury_adjustments(target_date, use_cached_dataset=use_cached_dataset)

    games = get_espn_games_for_date(target_date)
    cutoff = _parse_est_cutoff(min_start_est)
    if cutoff is not None:
        games = [g for g in games if _game_at_or_after_cutoff(g, cutoff)]
    if remaining_only:
        games = [g for g in games if _is_remaining_game(g)]
    if not games:
        return [], {}

    odds_data = get_nba_odds(markets="spreads")
    cache_odds_snapshot(odds_data, target_date.isoformat())
    opening_spreads = get_opening_spreads_for_date(target_date.isoformat())
    odds_by_matchup = {}
    odds_by_matchup_norm = {}
    for game_odds in odds_data:
        parsed = parse_odds_for_game(game_odds)
        if not parsed:
            continue
        key = (parsed.get("away_team"), parsed.get("home_team"))
        odds_by_matchup[key] = parsed
        away_norm = normalize_team_name(parsed.get("away_team"))
        home_norm = normalize_team_name(parsed.get("home_team"))
        if away_norm and home_norm:
            odds_by_matchup_norm[(away_norm, home_norm)] = parsed

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
            odds = odds_by_matchup_norm.get((normalize_team_name(away_team), normalize_team_name(home_team)))
        if not odds:
            continue

        market_spread = odds.get("away_spread")
        if market_spread is None:
            continue

        opening_spread = None
        open_key = f"{normalize_team_name(away_team)}@{normalize_team_name(home_team)}"
        if opening_spreads:
            opening_spread = opening_spreads.get(open_key)

        fake_game = build_fake_game(target_date, home_id, names[home_id], away_id, names[away_id])
        features = build_features_for_game(
            fake_game,
            history,
            half_life=10.0,
            market_spread=float(market_spread),
            opening_spread=opening_spread,
            narrative_history=narrative_history,
        )
        if not features:
            continue

        X = pd.DataFrame([features])[feature_cols]
        pred_home_margin = float(model.predict(X)[0])
        pred_away_margin = -pred_home_margin
        pred_away_adj = apply_injury_adjustment(away_team, home_team, pred_away_margin, injury_adjustments)
        edge = pred_away_adj + float(market_spread)

        pred_away_spread = -pred_away_margin
        pred_away_spread_adj = -pred_away_adj
        injury_adjustment = pred_away_adj - pred_away_margin

        line_move = float(market_spread - opening_spread) if opening_spread is not None else 0.0

        rows.append({
            "game": f"{away_team} @ {home_team}",
            "away_team": away_team,
            "home_team": home_team,
            "away_id": away_id,
            "home_id": home_id,
            "market_spread": float(market_spread),
            "opening_spread": opening_spread,
            "line_move": line_move,
            "pred_away_margin": pred_away_margin,
            "pred_away_adj": pred_away_adj,
            "pred_away_spread": pred_away_spread,
            "pred_away_spread_adj": pred_away_spread_adj,
            "injury_adjustment": injury_adjustment,
            "injury_adjustment_spread": -injury_adjustment,
            "edge": edge,
            "features": {
                "home_recent_margin_5": features.get("home_recent_margin_5", 0.0),
                "away_recent_margin_5": features.get("away_recent_margin_5", 0.0),
                "home_recent_win_pct_5": features.get("home_recent_win_pct_5", 0.0),
                "away_recent_win_pct_5": features.get("away_recent_win_pct_5", 0.0),
                "rest_diff": features.get("rest_diff", 0.0),
                "home_b2b": features.get("home_b2b", 0.0),
                "away_b2b": features.get("away_b2b", 0.0),
                "home_clutch_margin_10": features.get("home_clutch_margin_10", 0.0),
                "away_clutch_margin_10": features.get("away_clutch_margin_10", 0.0),
                "home_blown_rate_10": features.get("home_blown_rate_10", 0.0),
                "away_blown_rate_10": features.get("away_blown_rate_10", 0.0),
            },
        })
    return rows, names


def summarize_rows(rows):
    def pick_for_threshold(t):
        picks = []
        for r in rows:
            if r["edge"] >= t:
                picks.append(format_team_spread(r["away_team"], r["market_spread"]))
            elif r["edge"] <= -t:
                home_spread = -r["market_spread"]
                picks.append(format_team_spread(r["home_team"], home_spread))
        return picks

    summaries = {}
    for t in [0, 5, 10]:
        summaries[t] = pick_for_threshold(t)
    return summaries


def format_pick_rationale(rows, thresholds=None):
    if thresholds is None:
        thresholds = [0, 5, 10]

    lines = []
    lines.append("Pick rationale (edge = pred_away_adj + market_spread):")

    for t in thresholds:
        picks = []
        for r in rows:
            edge = r.get("edge", 0.0)
            if edge >= t:
                pred_adj = r.get("pred_away_adj", 0.0)
                market = r.get("market_spread", 0.0)
                picks.append(
                    f"- {format_team_spread(r['away_team'], market)} because edge {edge:+.1f} "
                    f"({pred_adj:+.1f} + {market:+.1f})"
                )
            elif edge <= -t:
                pred_adj = r.get("pred_away_adj", 0.0)
                market = r.get("market_spread", 0.0)
                home_spread = -market
                picks.append(
                    f"- {format_team_spread(r['home_team'], home_spread)} because edge {edge:+.1f} "
                    f"({pred_adj:+.1f} + {market:+.1f})"
                )

        lines.append("")
        lines.append(f"{t} threshold:")
        if picks:
            lines.extend(picks)
        else:
            lines.append("- (no picks)")

    return "\n".join(lines)


def _qualitative_factors(row: dict) -> list[str]:
    features = row.get("features") or {}
    notes = []

    rest_diff = float(features.get("rest_diff", 0.0))
    if abs(rest_diff) >= 1.0:
        if rest_diff > 0:
            notes.append(f"rest edge home by {rest_diff:.0f} day(s)")
        else:
            notes.append(f"rest edge away by {abs(rest_diff):.0f} day(s)")

    away_margin_5 = float(features.get("away_recent_margin_5", 0.0))
    home_margin_5 = float(features.get("home_recent_margin_5", 0.0))
    if abs(away_margin_5 - home_margin_5) >= 3.0:
        if away_margin_5 > home_margin_5:
            notes.append(f"recent margin favors away ({away_margin_5:+.1f} vs {home_margin_5:+.1f})")
        else:
            notes.append(f"recent margin favors home ({home_margin_5:+.1f} vs {away_margin_5:+.1f})")

    away_win_5 = float(features.get("away_recent_win_pct_5", 0.0))
    home_win_5 = float(features.get("home_recent_win_pct_5", 0.0))
    if abs(away_win_5 - home_win_5) >= 0.2:
        if away_win_5 > home_win_5:
            notes.append(f"recent win rate favors away ({away_win_5:.0%} vs {home_win_5:.0%})")
        else:
            notes.append(f"recent win rate favors home ({home_win_5:.0%} vs {away_win_5:.0%})")

    away_clutch = float(features.get("away_clutch_margin_10", 0.0))
    home_clutch = float(features.get("home_clutch_margin_10", 0.0))
    if abs(away_clutch - home_clutch) >= 2.0:
        if away_clutch > home_clutch:
            notes.append(f"clutch margin favors away ({away_clutch:+.1f} vs {home_clutch:+.1f})")
        else:
            notes.append(f"clutch margin favors home ({home_clutch:+.1f} vs {away_clutch:+.1f})")

    away_blown = float(features.get("away_blown_rate_10", 0.0))
    home_blown = float(features.get("home_blown_rate_10", 0.0))
    if abs(away_blown - home_blown) >= 0.2:
        if away_blown > home_blown:
            notes.append(f"away blown-lead risk higher ({away_blown:.0%} vs {home_blown:.0%})")
        else:
            notes.append(f"home blown-lead risk higher ({home_blown:.0%} vs {away_blown:.0%})")

    injury_adj = float(row.get("injury_adjustment", 0.0))
    if abs(injury_adj) >= 1.0:
        notes.append(f"injury adjustment {injury_adj:+.1f} to away margin")

    return notes


def format_prediction_narrative(target_date: date, rows: list[dict]) -> str:
    lines = [
        f"Prediction narratives for {target_date.isoformat()}",
        "", 
        "Edge formula: edge = pred_away_adj + market_spread",
        "",
    ]

    for row in rows:
        edge = float(row.get("edge", 0.0))
        market = float(row.get("market_spread", 0.0))
        pred_adj = float(row.get("pred_away_adj", 0.0))
        injury_adj = float(row.get("injury_adjustment", 0.0))

        if edge >= 0:
            pick = format_team_spread(row["away_team"], market)
        else:
            pick = format_team_spread(row["home_team"], -market)

        abs_edge = abs(edge)
        if abs_edge >= 10:
            tier = "10"
        elif abs_edge >= 5:
            tier = "5"
        else:
            tier = "0"

        lines.append(f"Pick: {pick} (threshold {tier})")
        lines.append(
            f"  Quant: edge {edge:+.1f} = pred_adj {pred_adj:+.1f} + market {market:+.1f}; "
            f"injury adj {injury_adj:+.1f}"
        )

        factors = _qualitative_factors(row)
        if factors:
            lines.append("  Qual: " + "; ".join(factors))
        else:
            lines.append("  Qual: no strong qualitative flags; edge driven by baseline model features")

        lines.append("")

    return "\n".join(lines)


def threshold_stats(model_path: str, data_path: str, cutoff: date, thresholds):
    df = pd.read_csv(data_path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    test_df = df[df["game_date"] >= cutoff]

    model = joblib.load(model_path)
    feature_cols = FEATURE_COLS
    if hasattr(model, "feature_names_in_"):
        feature_cols = [col for col in model.feature_names_in_ if col in test_df.columns]
    else:
        feature_cols = [col for col in FEATURE_COLS if col in test_df.columns]
    X = test_df[feature_cols]
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


def format_summary(
    target_date: date,
    rows,
    model_path: str = "ml_data/best_model_with_spreads.joblib",
    data_path: str = "ml_data/games_optimized.csv",
):
    thresholds = [0, 5, 10]
    stats = threshold_stats(
        model_path=model_path,
        data_path=data_path,
        cutoff=date(2026, 1, 16),
        thresholds=thresholds,
    )
    lines = []
    lines.append(f"Predictions for {target_date.isoformat()}")
    lines.append("Game | Line(A) | PredLn(A) | AdjLn(A) | Edge")
    for r in rows:
        pred_line = r.get("pred_away_spread", -r["pred_away_margin"])
        adj_line = r.get("pred_away_spread_adj", -r["pred_away_adj"])
        lines.append(
            f"{r['game']} | {format_team_spread(r['away_team'], r['market_spread'])} | "
            f"{format_team_spread(r['away_team'], pred_line)} | "
            f"{format_team_spread(r['away_team'], adj_line)} | {r['edge']:+.1f}"
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

    rows, _ = generate_rows(
        target_date,
        args.model,
        use_cached_dataset=args.fast,
        min_start_est=args.min_start_est,
        remaining_only=args.remaining,
    )
    if not rows:
        print("No games with odds and features found.")
        return

    save_predictions(target_date, rows)

    print(f"\nPredictions for {target_date.isoformat()}\n")
    print("Game".ljust(40), "Line(A)".rjust(10), "PredLn(A)".rjust(12), "AdjLn(A)".rjust(12), "Edge".rjust(8))
    for r in rows:
        pred_line = r.get("pred_away_spread", -r["pred_away_margin"])
        adj_line = r.get("pred_away_spread_adj", -r["pred_away_adj"])
        print(
            r["game"].ljust(40),
            format_team_spread(r["away_team"], r["market_spread"]).rjust(10),
            format_team_spread(r["away_team"], pred_line).rjust(12),
            format_team_spread(r["away_team"], adj_line).rjust(12),
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
