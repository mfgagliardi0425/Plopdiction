"""
Evaluate prediction model accuracy against actual game results.
"""
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

from matchup_model import (
    build_team_history,
    compute_team_stats,
    predict_game,
    build_player_averages,
    parse_game_date,
    parse_team_display,
    extract_points,
    DATA_DIR,
)


def load_actual_games(target_date: date) -> List[dict]:
    """Load all completed games from a specific date."""
    date_dir = DATA_DIR / target_date.isoformat()
    if not date_dir.exists():
        return []
    
    games = []
    for file_path in date_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
            status = (game.get("status") or "").lower()
            if status in {"closed", "complete", "completed", "final"}:
                games.append(game)
        except Exception:
            continue
    
    return games


def evaluate_predictions(
    target_date: date,
    half_life: float = 10.0,
    home_advantage: float = 2.5,
    margin_scale: float = 8.5,
    rest_weight: float = 0.5,
    time_weight: float = 0.2,
    roster_weight: float = 0.1,
    b2b_weight: float = 1.0,
    time_zone_weight: float = 0.1,
    preferred_local_hour: int = 19,
    player_points_weight: float = 1.0,
    player_pm_weight: float = 0.25,
    player_impact_scale: float = 0.1,
    include_rosters: bool = False,
    include_player_impact: bool = False,
) -> Tuple[List[dict], Dict]:
    """
    Evaluate predictions for games on a specific date.
    
    Returns:
        Tuple of (results list, summary dict)
    """
    # Build historical data up to (but not including) the target date
    history, names = build_team_history(DATA_DIR)
    
    # Filter history to only include games before target date
    for team_id in history:
        history[team_id] = [g for g in history[team_id] if g.game_date < target_date]
    
    # Compute team stats
    team_stats = {}
    for team_id, games in history.items():
        if games:
            team_stats[team_id] = compute_team_stats(team_id, games, names.get(team_id, team_id), half_life)
    
    # Build player averages if needed
    player_averages = {}
    if include_player_impact:
        for team_id, games in history.items():
            if games:
                player_averages[team_id] = build_player_averages(team_id, games, half_life, max_games=10)
    
    # Load actual games from target date
    actual_games = load_actual_games(target_date)
    
    results = []
    for game in actual_games:
        home = game.get("home", {})
        away = game.get("away", {})
        home_id, home_name = parse_team_display(home)
        away_id, away_name = parse_team_display(away)
        
        # Get actual scores
        home_points = extract_points(home)
        away_points = extract_points(away)
        
        if home_points is None or away_points is None:
            continue
        
        actual_margin = home_points - away_points
        
        # Make prediction
        _, _, pred_margin, home_prob, away_prob = predict_game(
            game,
            team_stats,
            player_averages,
            include_rosters,
            include_player_impact,
            home_advantage,
            margin_scale,
            rest_weight,
            time_weight,
            roster_weight,
            b2b_weight,
            time_zone_weight,
            preferred_local_hour,
            player_points_weight,
            player_pm_weight,
            player_impact_scale,
        )
        
        # Determine predicted winner
        pred_winner = "home" if pred_margin > 0 else "away"
        actual_winner = "home" if actual_margin > 0 else "away"
        correct_winner = pred_winner == actual_winner
        
        # Calculate margin error
        margin_error = abs(pred_margin - actual_margin)
        
        results.append({
            "game_id": game.get("id"),
            "away_team": away_name,
            "home_team": home_name,
            "actual_score": f"{away_points}-{home_points}",
            "actual_margin": actual_margin,
            "pred_margin": pred_margin,
            "margin_error": margin_error,
            "home_prob": home_prob,
            "away_prob": away_prob,
            "pred_winner": pred_winner,
            "actual_winner": actual_winner,
            "correct_winner": correct_winner,
        })
    
    # Calculate summary statistics
    if results:
        total_games = len(results)
        correct_winners = sum(1 for r in results if r["correct_winner"])
        avg_margin_error = sum(r["margin_error"] for r in results) / total_games
        
        summary = {
            "total_games": total_games,
            "correct_winners": correct_winners,
            "win_accuracy": correct_winners / total_games,
            "avg_margin_error": avg_margin_error,
        }
    else:
        summary = {
            "total_games": 0,
            "correct_winners": 0,
            "win_accuracy": 0.0,
            "avg_margin_error": 0.0,
        }
    
    return results, summary


def print_evaluation(target_date: date, **kwargs):
    """Print evaluation results for a specific date."""
    print(f"\n{'='*80}")
    print(f"Evaluating predictions for {target_date.isoformat()}")
    print(f"{'='*80}\n")
    
    results, summary = evaluate_predictions(target_date, **kwargs)
    
    if not results:
        print("No completed games found for this date.")
        return
    
    # Print individual game results
    print("Game Results:")
    print("-" * 80)
    for r in results:
        status = "✓" if r["correct_winner"] else "✗"
        print(f"{status} {r['away_team']} @ {r['home_team']}")
        print(f"   Actual: {r['actual_score']} (margin: {r['actual_margin']:+.0f})")
        print(f"   Predicted margin: {r['pred_margin']:+.1f} (error: {r['margin_error']:.1f})")
        print(f"   Win prob: {r['home_team']} {r['home_prob']*100:.1f}% / {r['away_team']} {r['away_prob']*100:.1f}%")
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY:")
    print(f"  Total games: {summary['total_games']}")
    print(f"  Correct winner predictions: {summary['correct_winners']}/{summary['total_games']} ({summary['win_accuracy']*100:.1f}%)")
    print(f"  Average margin error: {summary['avg_margin_error']:.1f} points")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate prediction model performance")
    parser.add_argument("--date", required=True, help="Date to evaluate (YYYY-MM-DD)")
    parser.add_argument("--half-life", type=float, default=10.0, help="Half-life in games")
    parser.add_argument("--home-adv", type=float, default=2.5, help="Home court advantage")
    parser.add_argument("--margin-scale", type=float, default=8.5, help="Margin scale for win prob")
    parser.add_argument("--rest-weight", type=float, default=0.5, help="Rest weight")
    parser.add_argument("--b2b-weight", type=float, default=1.0, help="Back-to-back penalty")
    parser.add_argument("--include-rosters", action="store_true", help="Include roster adjustments")
    parser.add_argument("--include-player-impact", action="store_true", help="Include player impact")
    args = parser.parse_args()
    
    target_date = date.fromisoformat(args.date)
    
    print_evaluation(
        target_date,
        half_life=args.half_life,
        home_advantage=args.home_adv,
        margin_scale=args.margin_scale,
        rest_weight=args.rest_weight,
        b2b_weight=args.b2b_weight,
        include_rosters=args.include_rosters,
        include_player_impact=args.include_player_impact,
    )
