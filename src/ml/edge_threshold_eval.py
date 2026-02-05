"""
Evaluate ATS accuracy with edge thresholds on the test split.
Edge = (predicted away margin) - (market spread for away team)
If edge >= threshold -> bet away
If edge <= -threshold -> bet home
"""
import argparse
from datetime import date
from pathlib import Path

import joblib
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def split_by_date(df: pd.DataFrame, cutoff_date: date):
    train = df[df["game_date"] < cutoff_date]
    test = df[df["game_date"] >= cutoff_date]
    return train, test


def evaluate_thresholds(model, df: pd.DataFrame, feature_cols, thresholds):
    X = df[feature_cols]
    y = df["actual_margin"]
    preds = model.predict(X)

    # Away team margin and line are in away-team terms
    away_actual_margin = -y
    away_pred_margin = -preds
    line = df["market_spread"]

    # Exclude missing spreads and pushes
    valid = line.abs() > 0
    away_actual_margin = away_actual_margin[valid]
    away_pred_margin = away_pred_margin[valid]
    line = line[valid]

    actual_diff = away_actual_margin + line

    results = []
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
                results.append((t, int(non_push.sum()), acc))
            else:
                results.append((t, 0, None))
        else:
            results.append((t, 0, None))

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="ATS edge threshold evaluation")
    parser.add_argument("--data", default="ml_data/games_optimized.csv")
    parser.add_argument("--cutoff", default="2026-01-16")
    parser.add_argument("--model", default="ml_data/best_model_with_spreads.joblib")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_dataset(Path(args.data))
    cutoff = date.fromisoformat(args.cutoff)
    _, test_df = split_by_date(df, cutoff)

    model = joblib.load(args.model)

    feature_cols = [
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

    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    results = evaluate_thresholds(model, test_df, feature_cols, thresholds)

    print("ATS edge threshold results (test split)")
    print("threshold | bets | ATS acc")
    for t, n, acc in results:
        if acc is None:
            print(f"{t:>9} | {n:>4} | n/a")
        else:
            print(f"{t:>9} | {n:>4} | {acc*100:>6.1f}%")


if __name__ == "__main__":
    main()
