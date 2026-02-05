"""
Optimize model parameters by testing on historical data.
"""
import itertools
from datetime import date, timedelta
from typing import List, Dict, Tuple
import json

from evaluate_spreads import evaluate_predictions, load_actual_games


def find_optimal_parameters(
    start_date: date,
    end_date: date,
    param_ranges: Dict,
) -> Dict:
    """
    Grid search to find optimal parameters.
    
    Args:
        start_date: First date to evaluate
        end_date: Last date to evaluate
        param_ranges: Dictionary of parameter names to lists of values to try
    
    Returns:
        Best parameters and their performance
    """
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    
    best_score = float('-inf')
    best_params = None
    best_details = None
    
    total_combos = 1
    for values in param_values:
        total_combos *= len(values)
    
    print(f"Testing {total_combos} parameter combinations...")
    print(f"Date range: {start_date} to {end_date}\n")
    
    combo_num = 0
    for combo in itertools.product(*param_values):
        combo_num += 1
        params = dict(zip(param_names, combo))
        
        # Evaluate across all dates
        total_games = 0
        total_correct_winners = 0
        total_favorites_covered = 0
        total_abs_error = 0.0
        
        current_date = start_date
        while current_date <= end_date:
            games = load_actual_games(current_date)
            if games:
                results, summary = evaluate_predictions(current_date, **params)
                total_games += summary['total_games']
                total_correct_winners += summary['correct_winners']
                total_favorites_covered += summary['favorites_covered']
                total_abs_error += summary['avg_abs_margin_error'] * summary['total_games']
            
            current_date += timedelta(days=1)
        
        if total_games == 0:
            continue
        
        win_acc = total_correct_winners / total_games
        cover_rate = total_favorites_covered / total_games
        avg_error = total_abs_error / total_games
        
        # Scoring: prioritize cover rate, then minimize margin error, then win accuracy
        # We want high cover rate and low margin error
        score = (cover_rate * 100) + (win_acc * 50) - (avg_error * 2)
        
        if combo_num % 10 == 0 or score > best_score:
            print(f"[{combo_num}/{total_combos}] Games: {total_games}, "
                  f"Win%: {win_acc*100:.1f}%, Cover%: {cover_rate*100:.1f}%, "
                  f"AvgErr: {avg_error:.1f}, Score: {score:.2f}")
            if score > best_score:
                print(f"  ^ NEW BEST! Params: {params}")
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_details = {
                'score': score,
                'games': total_games,
                'win_accuracy': win_acc,
                'cover_rate': cover_rate,
                'avg_margin_error': avg_error,
            }
    
    return {
        'params': best_params,
        'performance': best_details,
    }


def quick_optimization():
    """Run a quick optimization on recent games."""
    # Use last 7 days as test set
    end_date = date(2026, 2, 1)  # Day before our target
    start_date = end_date - timedelta(days=6)  # 7 days total
    
    print("QUICK PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Test ranges for key parameters
    param_ranges = {
        'half_life': [7.0, 10.0, 15.0],
        'home_advantage': [1.5, 2.5, 3.5],
        'margin_scale': [7.0, 8.5, 10.0],
        'rest_weight': [0.3, 0.5, 0.8],
        'b2b_weight': [0.5, 1.0, 1.5],
    }
    
    result = find_optimal_parameters(start_date, end_date, param_ranges)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nBest Parameters:")
    for param, value in result['params'].items():
        print(f"  {param}: {value}")
    
    print("\nPerformance on test set:")
    perf = result['performance']
    print(f"  Games evaluated: {perf['games']}")
    print(f"  Win accuracy: {perf['win_accuracy']*100:.1f}%")
    print(f"  Cover rate: {perf['cover_rate']*100:.1f}%")
    print(f"  Avg margin error: {perf['avg_margin_error']:.1f} points")
    print(f"  Overall score: {perf['score']:.2f}")
    
    return result


def comprehensive_optimization():
    """Run a more comprehensive optimization."""
    # Use last 14 days
    end_date = date(2026, 2, 1)
    start_date = end_date - timedelta(days=13)
    
    print("COMPREHENSIVE PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    param_ranges = {
        'half_life': [5.0, 7.0, 10.0, 12.0, 15.0],
        'home_advantage': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'margin_scale': [6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0],
        'rest_weight': [0.0, 0.3, 0.5, 0.8, 1.0],
        'b2b_weight': [0.0, 0.5, 1.0, 1.5, 2.0],
    }
    
    result = find_optimal_parameters(start_date, end_date, param_ranges)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nBest Parameters:")
    for param, value in result['params'].items():
        print(f"  {param}: {value}")
    
    print("\nPerformance on test set:")
    perf = result['performance']
    print(f"  Games evaluated: {perf['games']}")
    print(f"  Win accuracy: {perf['win_accuracy']*100:.1f}%")
    print(f"  Cover rate: {perf['cover_rate']*100:.1f}%")
    print(f"  Avg margin error: {perf['avg_margin_error']:.1f} points")
    print(f"  Overall score: {perf['score']:.2f}")
    
    # Save results
    with open("optimization_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to optimization_results.json")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize model parameters")
    parser.add_argument("--mode", choices=["quick", "comprehensive"], default="quick",
                       help="Optimization mode")
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_optimization()
    else:
        comprehensive_optimization()
