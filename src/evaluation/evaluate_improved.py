"""
Evaluate the improved model on historical data.
"""
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

from improved_matchup_model import (
    compute_enhanced_stats,
    predict_game_enhanced,
    EnhancedTeamStats,
)
from matchup_model import build_team_history, parse_team_display, extract_points, DATA_DIR


def evaluate_improved_model(
    target_date: date,
    half_life: float = 10.0,
    home_advantage: float = 2.5,
    momentum_weight: float = 0.15,
    regression_factor: float = 0.10,
) -> Tuple[List[dict], Dict]:
    """Evaluate improved model on a specific date."""
    # Build history up to target date
    history, names = build_team_history(DATA_DIR)
    
    for team_id in history:
        history[team_id] = [g for g in history[team_id] if g.game_date < target_date]
    
    # Compute enhanced stats
    team_stats = {}
    for team_id, games in history.items():
        if games:
            team_stats[team_id] = compute_enhanced_stats(team_id, games, names.get(team_id, team_id), half_life)
    
    # Load actual games
    date_dir = DATA_DIR / target_date.isoformat()
    if not date_dir.exists():
        return [], {}
    
    results = []
    for file_path in date_dir.glob("*.json"):
        try:
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
            
            status = (game.get("status") or "").lower()
            if status not in {"closed", "complete", "completed", "final"}:
                continue
            
            home = game.get("home", {})
            away = game.get("away", {})
            home_id, home_name = parse_team_display(home)
            away_id, away_name = parse_team_display(away)
            
            home_points = extract_points(home)
            away_points = extract_points(away)
            
            if home_points is None or away_points is None:
                continue
            
            actual_margin = home_points - away_points
            
            # Make prediction
            _, _, pred_margin = predict_game_enhanced(
                game,
                team_stats,
                home_advantage,
                momentum_weight,
                regression_factor,
            )
            
            margin_error = pred_margin - actual_margin
            abs_margin_error = abs(margin_error)
            
            pred_winner = "home" if pred_margin > 0 else "away"
            actual_winner = "home" if actual_margin > 0 else "away"
            correct_winner = pred_winner == actual_winner
            
            # Spread coverage (using our prediction as the spread)
            if pred_margin > 0:
                favorite_covered = actual_margin >= pred_margin
                favorite_team = home_name
            else:
                favorite_covered = actual_margin <= pred_margin
                favorite_team = away_name
            
            results.append({
                "away_team": away_name,
                "home_team": home_name,
                "actual_score": f"{away_points}-{home_points}",
                "actual_margin": actual_margin,
                "pred_margin": pred_margin,
                "margin_error": margin_error,
                "abs_margin_error": abs_margin_error,
                "correct_winner": correct_winner,
                "favorite_team": favorite_team,
                "favorite_covered": favorite_covered,
            })
        except Exception as e:
            continue
    
    # Summary
    if results:
        total = len(results)
        correct_winners = sum(1 for r in results if r["correct_winner"])
        favorites_covered = sum(1 for r in results if r["favorite_covered"])
        avg_error = sum(r["abs_margin_error"] for r in results) / total
        
        summary = {
            "total_games": total,
            "correct_winners": correct_winners,
            "win_accuracy": correct_winners / total,
            "favorites_covered": favorites_covered,
            "cover_rate": favorites_covered / total,
            "avg_abs_margin_error": avg_error,
        }
    else:
        summary = {}
    
    return results, summary


def print_comparison(target_date: date):
    """Print comparison between old and new model."""
    print(f"\n{'='*90}")
    print(f"MODEL COMPARISON for {target_date.isoformat()}")
    print(f"{'='*90}\n")
    
    # Test with default parameters
    results, summary = evaluate_improved_model(target_date)
    
    if not results:
        print("No games found.")
        return
    
    print("IMPROVED MODEL RESULTS:")
    print("-" * 90)
    for r in results:
        winner_mark = "✓" if r["correct_winner"] else "✗"
        cover_mark = "✓" if r["favorite_covered"] else "✗"
        
        print(f"{r['away_team']} @ {r['home_team']}")
        print(f"   Actual: {r['actual_score']} (margin: {r['actual_margin']:+d})")
        print(f"   Predicted: {r['pred_margin']:+.1f} ({r['favorite_team']} favored)")
        print(f"   {winner_mark} Winner | {cover_mark} Cover | Error: {r['abs_margin_error']:.1f} pts")
        print()
    
    print("=" * 90)
    print("SUMMARY:")
    print(f"  Win accuracy: {summary['correct_winners']}/{summary['total_games']} ({summary['win_accuracy']*100:.1f}%)")
    print(f"  Cover rate: {summary['favorites_covered']}/{summary['total_games']} ({summary['cover_rate']*100:.1f}%)")
    print(f"  Avg margin error: {summary['avg_abs_margin_error']:.1f} points")
    print("=" * 90)
    
    # Compare to old model results (from previous run: 50% win, 25% cover, 11.2 avg error)
    print("\nCOMPARISON TO OLD MODEL:")
    print(f"  Win accuracy: {summary['win_accuracy']*100:.1f}% (was 50.0%)")
    print(f"  Cover rate: {summary['cover_rate']*100:.1f}% (was 25.0%)")
    print(f"  Avg error: {summary['avg_abs_margin_error']:.1f} pts (was 11.2 pts)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate improved model")
    parser.add_argument("--date", required=True, help="Date to evaluate (YYYY-MM-DD)")
    parser.add_argument("--half-life", type=float, default=10.0)
    parser.add_argument("--home-adv", type=float, default=2.5)
    parser.add_argument("--momentum-weight", type=float, default=0.15)
    parser.add_argument("--regression", type=float, default=0.10)
    args = parser.parse_args()
    
    print_comparison(date.fromisoformat(args.date))
