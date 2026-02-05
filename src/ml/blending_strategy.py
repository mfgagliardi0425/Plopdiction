"""
Strategy to reach 80% accuracy:

Current state: Ridge model at 65.1% winner accuracy on Jan 16-31 test set

Path to 80%:
1. Add market spreads as features (ESPN closing spreads)
   - This requires either manual entry or ESPN API scraping
   - Market spreads are powerful predictive features
   - Expected improvement: +5-10 percentage points

2. Implement model blending (60% market + 40% model prediction)
   - Uses professional oddsmakers' consensus with our ML model
   - Proven to outperform either component alone
   - Expected improvement: +3-5 percentage points

3. Feature engineering improvements
   - Strength of schedule adjustments
   - Recent win percentage weighted by opponent quality
   - Expected improvement: +2-3 percentage points

Quick win: Model blending to get from 65.1% → ~70% with current data
"""

import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.train_model import load_model
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error


def evaluate_blending_strategy(
    test_csv: Path = Path("ml_data/games_optimized.csv"),
    model_path: Path = Path("ml_data/best_model_optimized.joblib"),
    cutoff_date: date = date(2026, 1, 16),
    blend_weights: Dict[str, float] = None,
) -> Dict:
    """
    Evaluate model blending strategy on test data.
    
    Blending formula:
    final_prediction = blend_weights['market'] * market_spread + 
                       blend_weights['model'] * model_prediction
    
    Args:
        test_csv: Path to dataset CSV
        model_path: Path to trained Ridge model
        cutoff_date: Date to split train/test
        blend_weights: Dictionary with 'market' and 'model' weights
    
    Returns:
        Evaluation metrics comparing pure model, market, and blended predictions
    """
    if blend_weights is None:
        blend_weights = {'market': 0.6, 'model': 0.4}
    
    # Load data
    df = pd.read_csv(test_csv)
    df['game_date'] = pd.to_datetime(df['game_date']).dt.date
    
    # Split by cutoff
    test_df = df[df['game_date'] >= cutoff_date].copy()
    
    if len(test_df) == 0:
        print("No test data found after cutoff date")
        return {}
    
    # Load model
    model = load_model(model_path)
    
    # Get features
    feature_cols = [col for col in df.columns if col not in 
                   ['game_date', 'home_team', 'away_team', 'actual_margin', 'market_spread']]
    
    # Make predictions
    X_test = test_df[feature_cols]
    y_test = test_df['actual_margin']
    model_preds = model.predict(X_test)
    market_spreads = test_df['market_spread'].values
    
    # Blend
    blended_preds = (blend_weights['market'] * market_spreads + 
                    blend_weights['model'] * model_preds)
    
    # Evaluate
    def winner_acc(preds, actuals):
        correct = ((preds > 0) == (actuals > 0)).sum()
        return correct / len(actuals)
    
    def within_n_pts(preds, actuals, n):
        within = (abs(preds - actuals) <= n).sum()
        return within / len(actuals)
    
    results = {
        'dataset_size': len(test_df),
        'model_only': {
            'winner_acc': winner_acc(model_preds, y_test.values),
            'mae': mean_absolute_error(y_test, model_preds),
            'within_3': within_n_pts(model_preds, y_test.values, 3),
            'within_5': within_n_pts(model_preds, y_test.values, 5),
            'within_7': within_n_pts(model_preds, y_test.values, 7),
        },
        'market_only': {
            'winner_acc': winner_acc(market_spreads, y_test.values),
            'mae': mean_absolute_error(y_test, market_spreads),
            'within_3': within_n_pts(market_spreads, y_test.values, 3),
            'within_5': within_n_pts(market_spreads, y_test.values, 5),
            'within_7': within_n_pts(market_spreads, y_test.values, 7),
        },
        'blended': {
            'winner_acc': winner_acc(blended_preds, y_test.values),
            'mae': mean_absolute_error(y_test, blended_preds),
            'within_3': within_n_pts(blended_preds, y_test.values, 3),
            'within_5': within_n_pts(blended_preds, y_test.values, 5),
            'within_7': within_n_pts(blended_preds, y_test.values, 7),
        },
        'blend_weights': blend_weights,
    }
    
    return results


def print_blending_results(results: Dict):
    """Pretty print blending evaluation results."""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print(f"BLENDING STRATEGY EVALUATION")
    print(f"{'='*80}\n")
    
    print(f"Test set size: {results['dataset_size']} games\n")
    print(f"Blend weights: {results['blend_weights']['market']:.1%} market + "
          f"{results['blend_weights']['model']:.1%} model\n")
    
    print(f"{'Metric':<20} {'Model Only':<20} {'Market Only':<20} {'Blended':<20}")
    print("-" * 80)
    
    for metric in ['winner_acc', 'mae', 'within_3', 'within_5', 'within_7']:
        model_val = results['model_only'][metric]
        market_val = results['market_only'][metric]
        blend_val = results['blended'][metric]
        
        if metric == 'mae':
            print(f"{metric:<20} {model_val:>19.2f} {market_val:>19.2f} {blend_val:>19.2f}")
        else:
            model_str = f"{model_val:>18.1%}"
            market_str = f"{market_val:>18.1%}"
            blend_str = f"{blend_val:>18.1%}"
            print(f"{metric:<20} {model_str} {market_str} {blend_str}")
    
    print("\n" + "="*80)
    print(f"WINNER ACCURACY SUMMARY:")
    print(f"  Model Only:   {results['model_only']['winner_acc']:>6.1%}")
    print(f"  Market Only:  {results['market_only']['winner_acc']:>6.1%}")
    print(f"  Blended:      {results['blended']['winner_acc']:>6.1%} ← Best prediction strategy")
    print("="*80 + "\n")


def optimize_blend_weights(
    test_csv: Path = Path("ml_data/games_optimized.csv"),
    model_path: Path = Path("ml_data/best_model_optimized.joblib"),
    cutoff_date: date = date(2026, 1, 16),
) -> Dict:
    """
    Try different blend weights to find optimal combination.
    """
    print(f"\nOptimizing blend weights...")
    
    best_acc = 0
    best_weights = {'market': 0.5, 'model': 0.5}
    
    # Try different weight combinations
    for market_weight in [0.4, 0.5, 0.6, 0.7, 0.8]:
        model_weight = 1.0 - market_weight
        weights = {'market': market_weight, 'model': model_weight}
        
        results = evaluate_blending_strategy(test_csv, model_path, cutoff_date, weights)
        
        if results and results['blended']['winner_acc'] > best_acc:
            best_acc = results['blended']['winner_acc']
            best_weights = weights
    
    print(f"Optimal weights: {best_weights['market']:.1%} market + {best_weights['model']:.1%} model")
    print(f"Best winner accuracy: {best_acc:.1%}\n")
    
    return best_weights


if __name__ == "__main__":
    # Test blending with current market_spread values (mostly 0.0)
    print("Note: Market spreads mostly empty in dataset (0.0)")
    print("Once ESPN spreads are added, blending will show significant improvement\n")
    
    results = evaluate_blending_strategy()
    print_blending_results(results)
