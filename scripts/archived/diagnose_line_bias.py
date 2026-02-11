from datetime import date
import sys

sys.path.insert(0, "src")

import joblib
import pandas as pd

MODEL_PATH = "ml_data/best_model_with_spreads.joblib"
DATA_PATH = "ml_data/games_optimized.csv"
CUTOFF = date(2026, 1, 16)

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


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    test_df = df[df["game_date"] >= CUTOFF].copy()

    valid = test_df["market_spread"].abs() > 0
    test_df = test_df[valid].copy()

    model = joblib.load(MODEL_PATH)
    X = test_df[FEATURE_COLS]
    preds = model.predict(X)

    away_actual_margin = -test_df["actual_margin"].values
    away_pred_margin = -preds
    line = test_df["market_spread"].values

    edge = away_pred_margin + line
    actual_diff = away_actual_margin + line

    buckets = {
        "away_underdog": line > 0,
        "away_favorite": line < 0,
    }

    summary = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            continue
        e = edge[mask]
        ad = actual_diff[mask]
        ap = away_pred_margin[mask]
        aa = away_actual_margin[mask]

        take = e != 0
        if take.any():
            correct = (e[take] > 0) == (ad[take] > 0)
            ats = correct.mean()
            n = int(take.sum())
        else:
            ats = None
            n = 0

        summary[name] = {
            "n": int(mask.sum()),
            "edge_mean": float(e.mean()),
            "edge_median": float(pd.Series(e).median()),
            "pred_margin_mean": float(ap.mean()),
            "actual_margin_mean": float(aa.mean()),
            "actual_diff_mean": float(ad.mean()),
            "ats_acc": None if ats is None else float(ats),
        }

    calibration = {
        "pred_margin_mean": float(away_pred_margin.mean()),
        "actual_margin_mean": float(away_actual_margin.mean()),
        "actual_diff_mean": float(actual_diff.mean()),
        "edge_mean": float(edge.mean()),
        "edge_median": float(pd.Series(edge).median()),
    }

    print({
        "cutoff": CUTOFF.isoformat(),
        "n_test": int(len(test_df)),
        "summary_by_line_sign": summary,
        "overall": calibration,
    })


if __name__ == "__main__":
    main()
