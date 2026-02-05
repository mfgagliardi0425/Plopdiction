"""
Train a baseline ML model to predict game margin.
"""
import argparse
from datetime import date
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


def load_model(model_path: Path):
    """Load a trained model from disk."""
    return joblib.load(model_path)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def split_by_date(df: pd.DataFrame, cutoff_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["game_date"] < cutoff_date]
    test = df[df["game_date"] >= cutoff_date]
    return train, test


def train_ridge_model(train_df: pd.DataFrame, feature_cols: List[str]) -> Ridge:
    X_train = train_df[feature_cols]
    y_train = train_df["actual_margin"]
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, df: pd.DataFrame, feature_cols: List[str]) -> dict:
    X = df[feature_cols]
    y = df["actual_margin"]
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)

    correct_winner = ((preds > 0) == (y > 0)).mean()
    within_3 = (abs(preds - y) <= 3).mean()
    within_5 = (abs(preds - y) <= 5).mean()
    within_7 = (abs(preds - y) <= 7).mean()

    # ATS (against the spread) accuracy using market_spread (away-team line)
    ats_acc = None
    ats_n = 0
    if "market_spread" in df.columns:
        valid = df["market_spread"].abs() > 0
        if valid.any():
            away_actual_margin = -y[valid]
            away_pred_margin = -preds[valid]
            line = df.loc[valid, "market_spread"]

            actual_diff = away_actual_margin + line
            pred_diff = away_pred_margin + line

            non_push = actual_diff != 0
            if non_push.any():
                ats_n = int(non_push.sum())
                ats_acc = (pred_diff[non_push] > 0).eq(actual_diff[non_push] > 0).mean()

    return {
        "mae": mae,
        "winner_accuracy": correct_winner,
        "within_3": within_3,
        "within_5": within_5,
        "within_7": within_7,
        "ats_accuracy": ats_acc,
        "ats_n": ats_n,
    }


def train_models(train_df: pd.DataFrame, feature_cols: List[str]) -> dict:
    models = {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "gbr": GradientBoostingRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
        ),
        "rf": RandomForestRegressor(
            random_state=42,
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            n_jobs=-1,
        ),
    }

    X_train = train_df[feature_cols]
    y_train = train_df["actual_margin"]
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline ML model")
    parser.add_argument("--data", default="ml_data/games.csv", help="Path to dataset CSV")
    parser.add_argument("--cutoff", required=True, help="Cutoff date for train/test split (YYYY-MM-DD)")
    parser.add_argument("--model-out", default="ml_data/ridge_model.joblib", help="Path to save model")
    return parser.parse_args()


def load_model(model_path: Path):
    """Load a trained model from disk."""
    return joblib.load(model_path)


def main() -> None:
    args = parse_args()
    df = load_dataset(Path(args.data))

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

    cutoff_date = date.fromisoformat(args.cutoff)
    train_df, test_df = split_by_date(df, cutoff_date)

    if train_df.empty or test_df.empty:
        raise RuntimeError("Train/test split is empty. Adjust cutoff date.")

    models = train_models(train_df, feature_cols)

    best_name = None
    best_test_mae = float("inf")
    best_model = None

    for name, model in models.items():
        train_metrics = evaluate_model(model, train_df, feature_cols)
        test_metrics = evaluate_model(model, test_df, feature_cols)

        print(f"\n{name.upper()} TRAIN METRICS:")
        print(f"  MAE: {train_metrics['mae']:.2f}")
        print(f"  Winner Acc: {train_metrics['winner_accuracy']*100:.1f}%")
        print(f"  Within 3: {train_metrics['within_3']*100:.1f}%")
        print(f"  Within 5: {train_metrics['within_5']*100:.1f}%")
        print(f"  Within 7: {train_metrics['within_7']*100:.1f}%")

        print(f"\n{name.upper()} TEST METRICS:")
        print(f"  MAE: {test_metrics['mae']:.2f}")
        print(f"  Winner Acc: {test_metrics['winner_accuracy']*100:.1f}%")
        print(f"  Within 3: {test_metrics['within_3']*100:.1f}%")
        print(f"  Within 5: {test_metrics['within_5']*100:.1f}%")
        print(f"  Within 7: {test_metrics['within_7']*100:.1f}%")
        if test_metrics["ats_accuracy"] is not None and test_metrics["ats_n"] > 0:
            print(f"  ATS Acc: {test_metrics['ats_accuracy']*100:.1f}% (n={test_metrics['ats_n']})")

        if test_metrics["mae"] < best_test_mae:
            best_test_mae = test_metrics["mae"]
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model trained successfully.")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_out)
    print(f"\nBEST MODEL: {best_name.upper()} | Saved to {args.model_out}")


if __name__ == "__main__":
    main()
