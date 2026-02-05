"""
Blend market spreads with model predictions.

Strategy: 60% market spread + 40% model prediction
This combines the strength of professional oddsmakers with our ML model.
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.train_model import load_model
from models.improved_matchup_model import compute_enhanced_stats, predict_game_enhanced
from models.matchup_model import (
    build_team_history,
    get_upcoming_games,
    parse_team_display,
    DATA_DIR,
)
from evaluation.tracking import TRACKING_DIR


def blend_predictions(
    model_margin: float,
    market_spread: float,
    market_weight: float = 0.6,
) -> float:
    """
    Blend market spread and model prediction.
    
    Args:
        model_margin: Predicted margin from Ridge model
        market_spread: Market spread (positive = home favored)
        market_weight: Weight for market spread (default 0.6 = 60%)
    
    Returns:
        Blended prediction
    """
    model_weight = 1.0 - market_weight
    
    # If no market spread, use model prediction only
    if market_spread == 0.0:
        return model_margin
    
    blended = (market_weight * market_spread) + (model_weight * model_margin)
    return blended


def evaluate_blending(target_date: date, model_path: Path = Path("ml_data/best_model.joblib")) -> None:
    """
    Evaluate blending strategy on a specific date.
    """
    # Load tracking data
    tracking_file = TRACKING_DIR / f"{target_date.isoformat()}.json"
    if not tracking_file.exists():
        print(f"No tracking data for {target_date}")
        return
    
    with open(tracking_file, "r", encoding="utf-8") as f:
        tracking_data = json.load(f)
    
    # Load model
    model = load_model(model_path)
    
    # Build team history
    history, names = build_team_history(DATA_DIR)
    
    print(f"\n{'='*90}")
    print(f"BLENDING EVALUATION FOR {target_date.isoformat()}")
    print(f"{'='*90}\n")
    print(f"{'Matchup':<40} | {'Model':<8} | {'Market':<8} | {'Blended':<8} | {'Actual':<8}")
    print("-" * 100)
    
    model_correct = 0
    market_correct = 0
    blended_correct = 0
    
    model_total_error = 0
    market_total_error = 0
    blended_total_error = 0
    
    for game in tracking_data.get("games", []):
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        
        # Get actual result
        actual = game.get("actual", {})
        if not actual:
            continue
        
        actual_margin = actual.get("home_margin", 0)
        
        # Get predictions
        pred = game.get("prediction", {})
        model_pred = pred.get("home_margin", 0)
        
        market_data = game.get("market", {})
        market_spread = market_data.get("spread") if market_data else None
        if market_spread is None:
            market_spread = 0.0
        
        # Blend
        blended_pred = blend_predictions(model_pred, market_spread)
        
        # Evaluate
        model_winner_correct = (model_pred > 0) == (actual_margin > 0)
        market_winner_correct = (market_spread > 0) == (actual_margin > 0)
        blended_winner_correct = (blended_pred > 0) == (actual_margin > 0)
        
        model_error = abs(model_pred - actual_margin)
        market_error = abs(market_spread - actual_margin)
        blended_error = abs(blended_pred - actual_margin)
        
        if model_winner_correct:
            model_correct += 1
        if market_winner_correct:
            market_correct += 1
        if blended_winner_correct:
            blended_correct += 1
        
        model_total_error += model_error
        market_total_error += market_error
        blended_total_error += blended_error
        
        matchup_str = f"{home_team[:18]} vs {away_team[:18]}"
        print(f"{matchup_str:<40} | {model_pred:>6.1f} | {market_spread:>6.1f} | {blended_pred:>6.1f} | {actual_margin:>6.1f}")
    
    total_games = len([g for g in tracking_data.get("games", []) if g.get("actual")])
    
    if total_games > 0:
        print("\n" + "="*90)
        print("SUMMARY:")
        print(f"  Model:   {model_correct}/{total_games} ({100*model_correct/total_games:.1f}%) | MAE: {model_total_error/total_games:.2f}")
        print(f"  Market:  {market_correct}/{total_games} ({100*market_correct/total_games:.1f}%) | MAE: {market_total_error/total_games:.2f}")
        print(f"  Blended: {blended_correct}/{total_games} ({100*blended_correct/total_games:.1f}%) | MAE: {blended_total_error/total_games:.2f}")
        print("="*90)


if __name__ == "__main__":
    # Test on 2/2 games
    evaluate_blending(date(2026, 2, 2))
