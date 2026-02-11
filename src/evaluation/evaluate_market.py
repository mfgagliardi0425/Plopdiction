"""
Evaluate model predictions against real market spreads from The Odds API.
"""
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.odds_api import get_odds_for_date, match_odds_to_sportradar_game
from models.matchup_model import build_team_history, parse_team_display, extract_points, DATA_DIR
from models.improved_matchup_model import compute_enhanced_stats, predict_game_enhanced
from evaluation.spread_utils import format_team_spread


def evaluate_against_market(
    target_date: date,
    half_life: float = 10.0,
    home_advantage: float = 2.5,
    momentum_weight: float = 0.15,
    regression_factor: float = 0.10,
    fetch_odds: bool = True,
) -> Tuple[List[dict], Dict]:
    """
    Evaluate model predictions against actual market spreads.
    
    Returns:
        (results_list, summary_dict)
    """
    # Build historical data up to target date
    history, names = build_team_history(DATA_DIR)
    
    for team_id in history:
        history[team_id] = [g for g in history[team_id] if g.game_date < target_date]
    
    # Compute team stats
    team_stats = {}
    for team_id, games in history.items():
        if games:
            team_stats[team_id] = compute_enhanced_stats(team_id, games, names.get(team_id, team_id), half_life)
    
    # Get odds for target date
    odds_games = get_odds_for_date(target_date, fetch_if_missing=fetch_odds)
    
    # Load actual game results
    date_dir = DATA_DIR / target_date.isoformat()
    if not date_dir.exists():
        return [], {}
    
    results = []
    for file_path in date_dir.glob("*.json"):
        try:
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
            
            # Make prediction with our model
            _, _, pred_margin = predict_game_enhanced(
                game, team_stats, home_advantage, momentum_weight, regression_factor
            )
            
            # Match to odds
            odds = match_odds_to_sportradar_game(game, odds_games)
            
            result = {
                "away_team": away_name,
                "home_team": home_name,
                "actual_score": f"{away_points}-{home_points}",
                "actual_margin": actual_margin,
                "pred_margin": pred_margin,
                "model_error": abs(pred_margin - actual_margin),
            }
            
            if odds and 'home_spread' in odds:
                market_spread = odds['home_spread']  # e.g., -1.5 means home favored
                result['market_spread'] = market_spread
                result['market_error'] = abs(market_spread - actual_margin)
                
                # Did favorite cover the market spread?
                if market_spread < 0:  # Home favored
                    market_favorite = home_name
                    market_covered = actual_margin >= market_spread
                else:  # Away favored
                    market_favorite = away_name
                    market_covered = actual_margin <= market_spread
                
                result['market_favorite'] = market_favorite
                result['market_covered'] = market_covered
                
                # Did favorite cover our predicted spread?
                if pred_margin > 0:  # Home favored
                    model_favorite = home_name
                    model_covered = actual_margin >= pred_margin
                else:  # Away favored
                    model_favorite = away_name
                    model_covered = actual_margin <= pred_margin
                
                result['model_favorite'] = model_favorite
                result['model_covered'] = model_covered
                
                # Would we have made a profitable bet?
                # If our prediction differs significantly from market, that's a potential edge
                spread_diff = abs(pred_margin - market_spread)
                result['spread_diff'] = spread_diff
                
                # Edge opportunity: if we think margin will be different by 3+ points
                if spread_diff >= 3.0:
                    # We think the favorite won't cover as much
                    if abs(pred_margin) < abs(market_spread):
                        result['edge_bet'] = f"Take {market_favorite} opponent"
                        # Did the underdog cover?
                        result['edge_hit'] = not market_covered
                    else:
                        result['edge_bet'] = f"Take {market_favorite}"
                        result['edge_hit'] = market_covered
                else:
                    result['edge_bet'] = None
                    result['edge_hit'] = None
            else:
                result['market_spread'] = None
                result['has_odds'] = False
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing game: {e}")
            continue
    
    # Calculate summary with focus on correctness percentages
    if results:
        total = len(results)
        
        # Winner prediction accuracy (all games)
        correct_winners = sum(1 for r in results if (r['actual_margin'] > 0) == (r['pred_margin'] > 0))
        winner_accuracy = correct_winners / total if total > 0 else 0.0
        
        # Margin accuracy tiers
        within_3 = sum(1 for r in results if r['model_error'] <= 3.0)
        within_5 = sum(1 for r in results if r['model_error'] <= 5.0)
        within_7 = sum(1 for r in results if r['model_error'] <= 7.0)
        
        games_with_odds = [r for r in results if r.get('market_spread') is not None]
        
        summary = {
            "total_games": total,
            "correct_winners": correct_winners,
            "winner_accuracy": winner_accuracy,
            "within_3_pts": within_3,
            "within_3_pct": within_3 / total,
            "within_5_pts": within_5,
            "within_5_pct": within_5 / total,
            "within_7_pts": within_7,
            "within_7_pct": within_7 / total,
            "avg_model_error": sum(r['model_error'] for r in results) / total,
        }
        
        if games_with_odds:
            market_covered = sum(1 for r in games_with_odds if r.get('market_covered', False))
            model_covered = sum(1 for r in games_with_odds if r.get('model_covered', False))
            
            avg_market_error = sum(r['market_error'] for r in games_with_odds) / len(games_with_odds)
            
            edge_bets = [r for r in games_with_odds if r.get('edge_bet')]
            edge_hits = sum(1 for r in edge_bets if r.get('edge_hit', False))
            
            summary.update({
                "games_with_odds": len(games_with_odds),
                "market_covered": market_covered,
                "market_cover_rate": market_covered / len(games_with_odds),
                "model_covered": model_covered,
                "model_cover_rate": model_covered / len(games_with_odds),
                "avg_market_error": avg_market_error,
                "edge_bets": len(edge_bets),
                "edge_hits": edge_hits,
                "edge_hit_rate": edge_hits / len(edge_bets) if edge_bets else 0.0,
            })
        else:
            summary["games_with_odds"] = 0
    else:
        summary = {}
    
    return results, summary


def print_market_evaluation(target_date: date, **kwargs):
    """Print evaluation against market spreads."""
    print(f"\n{'='*100}")
    print(f"MODEL vs MARKET SPREADS - {target_date.isoformat()}")
    print(f"{'='*100}\n")
    
    results, summary = evaluate_against_market(target_date, **kwargs)
    
    if not results:
        print("No games found.")
        return
    
    print("GAME RESULTS:")
    print("-" * 100)
    
    for r in results:
        print(f"\n{r['away_team']} @ {r['home_team']}")
        print(f"  Actual: {r['actual_score']} (margin: {r['actual_margin']:+d})")
        
        if r.get('market_spread') is not None:
            market_fav = "HOME" if r['market_spread'] < 0 else "AWAY"
            model_fav = "HOME" if r['pred_margin'] > 0 else "AWAY"
            
            market_away_spread = -float(r["market_spread"])
            print(
                f"  Market spread: {format_team_spread(r['away_team'], market_away_spread)} "
                f"({r['market_favorite']} favored)"
            )
            print(
                f"  Our prediction: {format_team_spread(r['away_team'], float(r['pred_margin']))} "
                f"({r['model_favorite']} favored)"
            )
            print(f"  Spread difference: {r['spread_diff']:.1f} pts")
            
            market_mark = "✓" if r['market_covered'] else "✗"
            model_mark = "✓" if r['model_covered'] else "✗"
            
            print(f"  {market_mark} Market favorite covered | {model_mark} Our pick covered")
            print(f"  Market error: {r['market_error']:.1f} | Our error: {r['model_error']:.1f}")
            
            if r.get('edge_bet'):
                edge_mark = "✓ WIN" if r['edge_hit'] else "✗ LOSS"
                print(f"  🎯 EDGE BET: {r['edge_bet']} - {edge_mark}")
        else:
            print(f"  Our prediction: {format_team_spread(r['away_team'], float(r['pred_margin']))}")
            print(f"  Error: {r['model_error']:.1f} pts")
            print(f"  (No market odds available)")
    
    print("\n" + "="*100)
    print("SUMMARY:")
    print("="*100)
    
    print(f"  Total games: {summary['total_games']}")
    print(f"\n  WINNER PREDICTION:")
    print(f"    Correct: {summary['correct_winners']}/{summary['total_games']} ({summary['winner_accuracy']*100:.1f}%)")
    
    print(f"\n  MARGIN ACCURACY:")
    print(f"    Within 3 pts: {summary['within_3_pts']}/{summary['total_games']} ({summary['within_3_pct']*100:.1f}%)")
    print(f"    Within 5 pts: {summary['within_5_pts']}/{summary['total_games']} ({summary['within_5_pct']*100:.1f}%)")
    print(f"    Within 7 pts: {summary['within_7_pts']}/{summary['total_games']} ({summary['within_7_pct']*100:.1f}%)")
    print(f"    Average error: {summary['avg_model_error']:.1f} pts")
    
    if summary.get('games_with_odds', 0) > 0:
        print(f"\n  AGAINST THE SPREAD (vs Market):")
        print(f"    Games with odds: {summary['games_with_odds']}")
        print(f"    Market favorites covered: {summary['market_covered']}/{summary['games_with_odds']} ({summary['market_cover_rate']*100:.1f}%)")
        print(f"    Our picks covered: {summary['model_covered']}/{summary['games_with_odds']} ({summary['model_cover_rate']*100:.1f}%)")
        print(f"    Market avg error: {summary['avg_market_error']:.1f} pts")
        
        if summary['edge_bets'] > 0:
            print(f"\n  EDGE BETTING (3+ pt disagreements):")
            print(f"    Opportunities found: {summary['edge_bets']}")
            print(f"    Edge bets won: {summary['edge_hits']}/{summary['edge_bets']} ({summary['edge_hit_rate']*100:.1f}%)")
        
        # Overall comparison
        if summary['model_cover_rate'] > summary['market_cover_rate']:
            diff = (summary['model_cover_rate'] - summary['market_cover_rate']) * 100
            print(f"\n  ✓ Our model covers {diff:+.1f}% better than market!")
        elif summary['model_cover_rate'] < summary['market_cover_rate']:
            diff = (summary['market_cover_rate'] - summary['model_cover_rate']) * 100
            print(f"\n  Market covers {diff:.1f}% better than our model")
        else:
            print(f"\n  Our model and market have equal cover rates")
    
    print("="*100)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model against market spreads")
    parser.add_argument("--date", required=True, help="Date to evaluate (YYYY-MM-DD)")
    parser.add_argument("--half-life", type=float, default=10.0)
    parser.add_argument("--home-adv", type=float, default=2.5)
    parser.add_argument("--momentum-weight", type=float, default=0.15)
    parser.add_argument("--regression", type=float, default=0.10)
    parser.add_argument("--no-fetch", action="store_true", help="Don't fetch odds, use cache only")
    args = parser.parse_args()
    
    print_market_evaluation(
        date.fromisoformat(args.date),
        half_life=args.half_life,
        home_advantage=args.home_adv,
        momentum_weight=args.momentum_weight,
        regression_factor=args.regression,
        fetch_odds=not args.no_fetch,
    )
