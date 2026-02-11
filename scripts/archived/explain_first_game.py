from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")

import joblib
import pandas as pd

from ml.build_dataset_optimized import build_features_for_game, build_team_narrative_history
from models.matchup_model import build_team_history, DATA_DIR
from evaluation.tonight_spread_predictions_summary import FEATURE_COLS, build_name_index, find_team_id, build_fake_game

TARGET_DATE = date(2026, 2, 5)
AWAY = "Washington Wizards"
HOME = "Detroit Pistons"
MARKET_SPREAD = 15.5  # Away line from tonight's output

model = joblib.load("ml_data/best_model_with_spreads.joblib")
history, names = build_team_history(DATA_DIR)
narrative_history = build_team_narrative_history(DATA_DIR)
name_index = build_name_index(names)

home_id = find_team_id(HOME, name_index)
away_id = find_team_id(AWAY, name_index)

fake_game = build_fake_game(TARGET_DATE, home_id, names[home_id], away_id, names[away_id])
features = build_features_for_game(
    fake_game,
    history,
    half_life=10.0,
    market_spread=MARKET_SPREAD,
    narrative_history=narrative_history,
)

X = pd.DataFrame([features])[FEATURE_COLS]
pred_home_margin = float(model.predict(X)[0])
pred_away_margin = -pred_home_margin

injury_dataset_path = Path("data/injuries/injury_dataset.json")
injury_dataset = None
if injury_dataset_path.exists():
    with open(injury_dataset_path, "r", encoding="utf-8") as f:
        injury_dataset = json.load(f)

espn_injuries_path = Path("odds_cache/espn_injuries.json")
espn_injuries = None
if espn_injuries_path.exists():
    with open(espn_injuries_path, "r", encoding="utf-8") as f:
        espn_injuries = json.load(f)

POINTS_PER_PPG = 0.15
RANK_MULTIPLIER = {1: 1.8, 2: 1.5, 3: 1.3, 4: 1.15, 5: 1.1}

def ppg_multiplier(ppg: float) -> float:
    if ppg >= 25:
        return 1.8
    if ppg >= 20:
        return 1.5
    if ppg >= 15:
        return 1.3
    if ppg >= 10:
        return 1.15
    return 1.0

away_penalty = 0.0
home_penalty = 0.0
injuries_detail = {"away": [], "home": []}

if injury_dataset and injury_dataset.get("date") == TARGET_DATE.isoformat():
    teams = injury_dataset.get("teams", {})
    for team_name, side in [(AWAY, "away"), (HOME, "home")]:
        for inj in teams.get(team_name, []):
            status = (inj.get("status") or "").lower()
            if status != "out":
                continue
            ppg = inj.get("ppg")
            if ppg is None:
                continue
            base = float(ppg) * POINTS_PER_PPG
            rank = inj.get("ppg_rank")
            mult = RANK_MULTIPLIER.get(rank, 1.0) if rank else ppg_multiplier(float(ppg))
            penalty = base * mult
            injuries_detail[side].append({
                "player": inj.get("player"),
                "status": inj.get("status"),
                "ppg": ppg,
                "ppg_rank": rank,
                "penalty": penalty,
                "comment": inj.get("comment"),
            })
            if side == "away":
                away_penalty += penalty
            else:
                home_penalty += penalty

injury_adjustment = -away_penalty + home_penalty
pred_away_adj = pred_away_margin + injury_adjustment
pred_away_spread = -pred_away_margin
pred_away_spread_adj = -pred_away_adj

key_features = {
    "rest_diff": features.get("rest_diff"),
    "home_b2b": features.get("home_b2b"),
    "away_b2b": features.get("away_b2b"),
    "home_weighted_margin": features.get("home_weighted_margin"),
    "away_weighted_margin": features.get("away_weighted_margin"),
    "home_recent_margin_5": features.get("home_recent_margin_5"),
    "away_recent_margin_5": features.get("away_recent_margin_5"),
    "home_recent_win_pct_5": features.get("home_recent_win_pct_5"),
    "away_recent_win_pct_5": features.get("away_recent_win_pct_5"),
    "home_recent_10_win_pct": features.get("home_recent_10_win_pct"),
    "away_recent_10_win_pct": features.get("away_recent_10_win_pct"),
    "home_weighted_points_for": features.get("home_weighted_points_for"),
    "away_weighted_points_for": features.get("away_weighted_points_for"),
    "home_weighted_points_against": features.get("home_weighted_points_against"),
    "away_weighted_points_against": features.get("away_weighted_points_against"),
    "home_blown_rate_10": features.get("home_blown_rate_10"),
    "away_blown_rate_10": features.get("away_blown_rate_10"),
    "home_clutch_margin_10": features.get("home_clutch_margin_10"),
    "away_clutch_margin_10": features.get("away_clutch_margin_10"),
    "home_max_lead_10": features.get("home_max_lead_10"),
    "away_max_lead_10": features.get("away_max_lead_10"),
    "home_h2h_margin_avg": features.get("home_h2h_margin_avg"),
    "home_h2h_win_pct": features.get("home_h2h_win_pct"),
    "h2h_games_played": features.get("h2h_games_played"),
    "home_games_played": features.get("home_games_played"),
    "away_games_played": features.get("away_games_played"),
}

result = {
    "market_spread_away": MARKET_SPREAD,
    "pred_away_margin": pred_away_margin,
    "pred_away_adj": pred_away_adj,
    "pred_away_spread": pred_away_spread,
    "pred_away_spread_adj": pred_away_spread_adj,
    "injury_adjustment": injury_adjustment,
    "injury_adjustment_spread": -injury_adjustment,
    "away_penalty": away_penalty,
    "home_penalty": home_penalty,
    "injuries_detail": injuries_detail,
    "injury_dataset_date": injury_dataset.get("date") if injury_dataset else None,
    "espn_injuries_date": espn_injuries.get("date") if espn_injuries else None,
    "espn_injuries_away": (espn_injuries.get("teams", {}).get(AWAY, []) if espn_injuries else []),
    "espn_injuries_home": (espn_injuries.get("teams", {}).get(HOME, []) if espn_injuries else []),
    "key_features": key_features,
}

print(json.dumps(result, indent=2, sort_keys=True))
