"""
Daily update script:
1) Download yesterday's final game data (SportRadar)
2) Scrape ESPN closing spreads for yesterday
3) Optionally rebuild dataset and retrain model
"""
import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_spread_scraper import scrape_espn_spreads_for_date_range
from data_fetching.download_season_data import download_season_data
from ml.build_dataset_optimized import build_dataset
from ml.train_model import train_models, evaluate_model
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib


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


def parse_args():
    parser = argparse.ArgumentParser(description="Daily update for NBA data")
    parser.add_argument("--date", help="Date to update (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between API calls")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild ML dataset")
    parser.add_argument("--retrain", action="store_true", help="Retrain model after rebuild")
    parser.add_argument("--model-out", default="ml_data/best_model_with_spreads.joblib")
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def split_by_date(df: pd.DataFrame, cutoff: date):
    train = df[df["game_date"] < cutoff]
    test = df[df["game_date"] >= cutoff]
    return train, test


def retrain_model(data_path: Path, model_out: Path, cutoff: date):
    df = load_dataset(data_path)
    train_df, test_df = split_by_date(df, cutoff)

    models = {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "gbr": GradientBoostingRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=3),
        "rf": RandomForestRegressor(random_state=42, n_estimators=400, max_depth=8, min_samples_leaf=5, n_jobs=-1),
    }

    best_name = None
    best_mae = float("inf")
    best_model = None

    for name, model in models.items():
        model.fit(train_df[FEATURE_COLS], train_df["actual_margin"])
        metrics = evaluate_model(model, test_df, FEATURE_COLS)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model trained")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_out)
    print(f"Saved model: {model_out} (best: {best_name})")


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=1))

    print(f"Updating data for {target_date.isoformat()}...")

    # 1) Download SportRadar game data for yesterday
    download_season_data(target_date, target_date, delay=args.delay)

    # 2) Scrape ESPN closing spreads for yesterday
    scrape_espn_spreads_for_date_range(target_date, target_date, delay_seconds=0.5)

    # 3) Optional: rebuild dataset
    if args.rebuild_dataset:
        from ml.build_dataset_optimized import OUTPUT_DIR
        train_start = date(2025, 12, 1)
        test_end = date(2026, 1, 31)
        output_path = OUTPUT_DIR / "games_optimized.csv"
        build_dataset(train_start, test_end, half_life=10.0, output_path=output_path)

    # 4) Optional: retrain model
    if args.retrain:
        data_path = Path("ml_data/games_optimized.csv")
        retrain_model(data_path, Path(args.model_out), cutoff=date(2026, 1, 16))


if __name__ == "__main__":
    main()
