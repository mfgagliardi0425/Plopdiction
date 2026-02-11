"""
Compare model predictions to current market spreads.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.odds_api import get_nba_odds, parse_odds_for_game
from models.matchup_model import build_team_history, get_upcoming_games, DATA_DIR
from models.improved_matchup_model import compute_enhanced_stats, predict_game_enhanced
from evaluation.spread_utils import format_team_spread


def compare_predictions_to_market(
    target_date: date,
    half_life: float = 10.0,
    home_advantage: float = 2.5,
    momentum_weight: float = 0.15,
    regression_factor: float = 0.10,
):
    """Show predictions vs market spreads side by side."""
    
    print(f"\n{'='*110}")
    print(f"MODEL vs MARKET COMPARISON - {target_date.isoformat()}")
    print(f"{'='*110}\n")
    
    # Build team history
    history, names = build_team_history(DATA_DIR)
    
    # Compute stats
    team_stats = {}
    for team_id, games in history.items():
        if games:
            team_stats[team_id] = compute_enhanced_stats(
                team_id, games, names.get(team_id, team_id), half_life
            )
    
    # Get upcoming games
    games = get_upcoming_games(target_date, target_date)
    
    # Get market odds
    print("Fetching market odds...\n")
    odds_data = get_nba_odds(markets='spreads')
    
    print(f"{'Game':<45} {'Market Spread':<20} {'Our Prediction':<20} {'Difference':>10}")
    print("-" * 110)
    
    edge_opportunities = []
    
    for game in games:
        home_name, away_name, pred_margin = predict_game_enhanced(
            game, team_stats, home_advantage, momentum_weight, regression_factor
        )
        
        # Find matching odds
        market_spread = None
        market_book = None
        
        for odds_game in odds_data:
            odds_home = odds_game.get('home_team', '').upper()
            odds_away = odds_game.get('away_team', '').upper()
            
            if (away_name.upper() in odds_away or odds_away in away_name.upper()) and \
               (home_name.upper() in odds_home or odds_home in home_name.upper()):
                parsed = parse_odds_for_game(odds_game)
                if parsed and 'home_spread' in parsed:
                    market_spread = parsed['home_spread']
                    market_book = parsed['sportsbook']
                break
        
        # Format output
        matchup = f"{away_name} @ {home_name}"
        
        our_pick = format_team_spread(away_name, pred_margin)
        
        if market_spread is not None:
            market_away_spread = -float(market_spread)
            market_str = format_team_spread(away_name, market_away_spread)
            
            diff = abs(pred_margin - market_spread)
            
            # Identify edge opportunities (3+ point difference)
            if diff >= 3.0:
                edge_opportunities.append({
                    'matchup': matchup,
                    'market_spread': market_spread,
                    'our_prediction': pred_margin,
                    'difference': diff,
                    'away_team': away_name,
                })
                print(f"{matchup:<45} {market_str:<20} {our_pick:<20} {diff:>9.1f} 🎯")
            else:
                print(f"{matchup:<45} {market_str:<20} {our_pick:<20} {diff:>9.1f}")
        else:
            print(f"{matchup:<45} {'N/A':<20} {our_pick:<20} {'N/A':>10}")
    
    # Show edge opportunities
    if edge_opportunities:
        print("\n" + "="*110)
        print(f"🎯 EDGE OPPORTUNITIES (3+ point disagreement with market):")
        print("="*110)
        for edge in edge_opportunities:
            print(f"\n{edge['matchup']}")
            away_team = edge.get("away_team") or ""
            print(f"  Market spread: {format_team_spread(away_team, -float(edge['market_spread']))}")
            print(f"  Our prediction: {format_team_spread(away_team, float(edge['our_prediction']))}")
            print(f"  Difference: {edge['difference']:.1f} points")
            
            # Recommend which side to take
            if abs(edge['our_prediction']) < abs(edge['market_spread']):
                print(f"  💡 SUGGESTION: Market is overvaluing the favorite - consider taking the UNDERDOG")
            else:
                print(f"  💡 SUGGESTION: Market is undervaluing the favorite - consider taking the FAVORITE")
    
    print("\n" + "="*110)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare predictions to market spreads")
    parser.add_argument("--date", help="Date (YYYY-MM-DD), defaults to tomorrow")
    parser.add_argument("--half-life", type=float, default=10.0)
    parser.add_argument("--home-adv", type=float, default=2.5)
    parser.add_argument("--momentum-weight", type=float, default=0.15)
    parser.add_argument("--regression", type=float, default=0.10)
    args = parser.parse_args()
    
    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = date.today() + timedelta(days=1)
    
    compare_predictions_to_market(
        target_date,
        args.half_life,
        args.home_adv,
        args.momentum_weight,
        args.regression,
    )
