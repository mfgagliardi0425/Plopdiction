"""
Track predictions and odds for NBA games.
Stores market spreads, our predictions, and actual results.
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.odds_api import get_nba_odds, parse_odds_for_game
from models.matchup_model import (
    build_team_history,
    get_upcoming_games,
    parse_team_display,
    extract_points,
    DATA_DIR,
)
from models.improved_matchup_model import compute_enhanced_stats, predict_game_enhanced
from evaluation.spread_utils import format_team_spread

TRACKING_DIR = Path("tracking/daily")
TRACKING_DIR.mkdir(parents=True, exist_ok=True)


def save_predictions_and_odds(
    target_date: date,
    half_life: float = 10.0,
    home_advantage: float = 2.5,
    momentum_weight: float = 0.15,
    regression_factor: float = 0.10,
) -> str:
    """
    Fetch odds and make predictions for a date, save to tracking file.
    
    Returns:
        Path to saved tracking file
    """
    print(f"\n{'='*90}")
    print(f"TRACKING PREDICTIONS FOR {target_date.isoformat()}")
    print(f"{'='*90}\n")
    
    # Build team history
    history, names = build_team_history(DATA_DIR)
    
    # Compute stats
    team_stats = {}
    for team_id, games in history.items():
        if games:
            team_stats[team_id] = compute_enhanced_stats(
                team_id, games, names.get(team_id, team_id), half_life
            )
    
    # Get games for this date (including completed)
    date_dir = DATA_DIR / target_date.isoformat()
    if not date_dir.exists():
        print(f"No game data found for {target_date.isoformat()}")
        return None
    
    games = []
    for game_file in date_dir.glob("*.json"):
        try:
            with open(game_file, 'r', encoding='utf-8') as f:
                game = json.load(f)
            games.append(game)
        except Exception:
            continue
    
    if not games:
        print(f"No games found for {target_date.isoformat()}")
        return None
    
    # Get market odds
    print("Fetching market odds...")
    try:
        odds_data = get_nba_odds(markets='spreads,totals,h2h')
        print(f"Found odds for {len(odds_data)} games\n")
    except Exception as e:
        print(f"Error fetching odds: {e}")
        odds_data = []
    
    tracked_games = []
    
    for game in games:
        home = game.get("home", {})
        away = game.get("away", {})
        home_id, home_name = parse_team_display(home)
        away_id, away_name = parse_team_display(away)
        game_id = game.get("id")
        scheduled = game.get("scheduled")
        
        # Make prediction
        _, _, pred_margin = predict_game_enhanced(
            game, team_stats, home_advantage, momentum_weight, regression_factor
        )
        
        # Find matching market odds
        market_data = None
        for odds_game in odds_data:
            odds_home = odds_game.get('home_team', '').upper()
            odds_away = odds_game.get('away_team', '').upper()
            
            if (away_name.upper() in odds_away or odds_away in away_name.upper()) and \
               (home_name.upper() in odds_home or odds_home in home_name.upper()):
                market_data = parse_odds_for_game(odds_game)
                break
        
        # Determine our pick
        if pred_margin > 0:
            our_favorite = home_name
            our_spread = pred_margin
        else:
            our_favorite = away_name
            our_spread = abs(pred_margin)
        
        tracked_game = {
            "game_id": game_id,
            "date": target_date.isoformat(),
            "scheduled": scheduled,
            "away_team": away_name,
            "away_id": away_id,
            "home_team": home_name,
            "home_id": home_id,
            "prediction": {
                "home_margin": pred_margin,
                "favorite": our_favorite,
                "spread": our_spread,
                "parameters": {
                    "half_life": half_life,
                    "home_advantage": home_advantage,
                    "momentum_weight": momentum_weight,
                    "regression_factor": regression_factor,
                }
            },
            "market": None,
            "actual": None,
            "evaluation": None,
        }
        
        if market_data:
            market_spread = market_data.get('home_spread')
            if market_spread is not None:
                if market_spread < 0:
                    market_favorite = home_name
                    market_spread_value = abs(market_spread)
                else:
                    market_favorite = away_name
                    market_spread_value = market_spread
                
                tracked_game["market"] = {
                    "home_spread": market_data.get('home_spread'),
                    "away_spread": market_data.get('away_spread'),
                    "favorite": market_favorite,
                    "spread": market_spread_value,
                    "sportsbook": market_data.get('sportsbook'),
                    "over_under": market_data.get('over'),
                    "home_moneyline": market_data.get('home_moneyline'),
                    "away_moneyline": market_data.get('away_moneyline'),
                }
                
                # Calculate edge
                spread_diff = abs(pred_margin - market_data.get('home_spread'))
                tracked_game["edge_opportunity"] = spread_diff >= 3.0
                tracked_game["spread_difference"] = spread_diff
        
        tracked_games.append(tracked_game)
        
        # Print summary
        print(f"{away_name} @ {home_name}")
        print(f"  Our pick: {format_team_spread(away_name, pred_margin)}")
        if market_data and tracked_game["market"]:
            market_info = tracked_game["market"]
            market_away_spread = market_info.get("away_spread")
            if market_away_spread is None and market_info.get("home_spread") is not None:
                market_away_spread = -float(market_info["home_spread"])
            print(
                f"  Market: {format_team_spread(away_name, market_away_spread)} "
                f"({market_info['sportsbook']})"
            )
            if tracked_game.get("edge_opportunity"):
                print(f"  🎯 EDGE: {tracked_game['spread_difference']:.1f} pt difference")
        else:
            print(f"  Market: No odds available")
        print()
    
    # Save to file
    tracking_file = TRACKING_DIR / f"{target_date.isoformat()}.json"
    tracking_data = {
        "date": target_date.isoformat(),
        "tracked_at": datetime.now().isoformat(),
        "total_games": len(tracked_games),
        "games_with_odds": sum(1 for g in tracked_games if g["market"] is not None),
        "edge_opportunities": sum(1 for g in tracked_games if g.get("edge_opportunity", False)),
        "games": tracked_games,
    }
    
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2)
    
    print("="*90)
    print(f"SAVED: {tracking_file}")
    print(f"  Total games: {len(tracked_games)}")
    print(f"  Games with odds: {tracking_data['games_with_odds']}")
    print(f"  Edge opportunities: {tracking_data['edge_opportunities']}")
    print("="*90)
    
    return str(tracking_file)


def update_actual_results(target_date: date) -> Dict:
    """
    Update tracking file with actual game results.
    
    Returns:
        Summary of results
    """
    tracking_file = TRACKING_DIR / f"{target_date.isoformat()}.json"
    
    if not tracking_file.exists():
        print(f"No tracking file found for {target_date.isoformat()}")
        return None
    
    with open(tracking_file, 'r', encoding='utf-8') as f:
        tracking_data = json.load(f)
    
    print(f"\n{'='*90}")
    print(f"UPDATING RESULTS FOR {target_date.isoformat()}")
    print(f"{'='*90}\n")
    
    # Load actual game data
    date_dir = DATA_DIR / target_date.isoformat()
    if not date_dir.exists():
        print(f"No game data found for {target_date.isoformat()}")
        return None
    
    updated_count = 0
    
    for tracked_game in tracking_data["games"]:
        game_id = tracked_game["game_id"]
        game_file = date_dir / f"{game_id}.json"
        
        if not game_file.exists():
            continue
        
        with open(game_file, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        status = (game_data.get("status") or "").lower()
        if status not in {"closed", "complete", "completed", "final"}:
            continue
        
        home = game_data.get("home", {})
        away = game_data.get("away", {})
        
        home_points = extract_points(home)
        away_points = extract_points(away)
        
        if home_points is None or away_points is None:
            continue
        
        actual_margin = home_points - away_points
        
        # Update actual results
        tracked_game["actual"] = {
            "home_points": home_points,
            "away_points": away_points,
            "home_margin": actual_margin,
            "winner": tracked_game["home_team"] if actual_margin > 0 else tracked_game["away_team"],
        }
        
        # Evaluate predictions
        pred_margin = tracked_game["prediction"]["home_margin"]
        
        # Did we predict the correct winner?
        correct_winner = (pred_margin > 0) == (actual_margin > 0)
        
        # Margin error
        margin_error = abs(pred_margin - actual_margin)
        
        # Did our pick cover?
        if pred_margin > 0:  # We picked home
            our_covered = actual_margin >= pred_margin
        else:  # We picked away
            our_covered = actual_margin <= pred_margin
        
        evaluation = {
            "correct_winner": correct_winner,
            "margin_error": margin_error,
            "our_pick_covered": our_covered,
            "within_3_pts": margin_error <= 3.0,
            "within_5_pts": margin_error <= 5.0,
            "within_7_pts": margin_error <= 7.0,
        }
        
        # Market evaluation if available
        if tracked_game["market"]:
            market_spread = tracked_game["market"]["home_spread"]
            
            # Did market favorite cover?
            if market_spread < 0:  # Home favored
                market_covered = actual_margin >= market_spread
            else:  # Away favored
                market_covered = actual_margin <= market_spread
            
            market_error = abs(market_spread - actual_margin)
            
            evaluation["market_favorite_covered"] = market_covered
            evaluation["market_error"] = market_error
            evaluation["beat_market"] = margin_error < market_error
            
            # Did our edge bet hit?
            if tracked_game.get("edge_opportunity"):
                # We identified an edge - did it pay off?
                evaluation["edge_bet_result"] = our_covered
        
        tracked_game["evaluation"] = evaluation
        updated_count += 1
    
    # Calculate summary statistics
    evaluated_games = [g for g in tracking_data["games"] if g.get("evaluation")]
    
    if evaluated_games:
        total = len(evaluated_games)
        
        summary = {
            "total_games": total,
            "correct_winners": sum(1 for g in evaluated_games if g["evaluation"]["correct_winner"]),
            "our_picks_covered": sum(1 for g in evaluated_games if g["evaluation"]["our_pick_covered"]),
            "within_3_pts": sum(1 for g in evaluated_games if g["evaluation"]["within_3_pts"]),
            "within_5_pts": sum(1 for g in evaluated_games if g["evaluation"]["within_5_pts"]),
            "within_7_pts": sum(1 for g in evaluated_games if g["evaluation"]["within_7_pts"]),
            "avg_margin_error": sum(g["evaluation"]["margin_error"] for g in evaluated_games) / total,
        }
        
        # Market comparison
        games_with_market = [g for g in evaluated_games if g["market"] and g["evaluation"].get("market_error")]
        if games_with_market:
            summary["games_with_market"] = len(games_with_market)
            summary["market_favorites_covered"] = sum(
                1 for g in games_with_market if g["evaluation"]["market_favorite_covered"]
            )
            summary["beat_market_count"] = sum(
                1 for g in games_with_market if g["evaluation"]["beat_market"]
            )
            summary["avg_market_error"] = sum(
                g["evaluation"]["market_error"] for g in games_with_market
            ) / len(games_with_market)
            
            # Edge bets
            edge_bets = [g for g in games_with_market if g.get("edge_opportunity")]
            if edge_bets:
                summary["edge_bets"] = len(edge_bets)
                summary["edge_bets_won"] = sum(
                    1 for g in edge_bets if g["evaluation"].get("edge_bet_result")
                )
        
        tracking_data["summary"] = summary
    
    tracking_data["updated_at"] = datetime.now().isoformat()
    
    # Save updated file
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"Updated {updated_count} games with actual results")
    print(f"SAVED: {tracking_file}\n")
    
    # Print summary
    if evaluated_games:
        print_tracking_summary(tracking_data)
    
    return tracking_data.get("summary")


def print_tracking_summary(tracking_data: Dict):
    """Print summary of tracking results."""
    summary = tracking_data.get("summary")
    if not summary:
        print("No evaluated games yet")
        return
    
    print("="*90)
    print("PERFORMANCE SUMMARY")
    print("="*90)
    print(f"  Winner Accuracy: {summary['correct_winners']}/{summary['total_games']} ({summary['correct_winners']/summary['total_games']*100:.1f}%)")
    print(f"  Our Picks Covered: {summary['our_picks_covered']}/{summary['total_games']} ({summary['our_picks_covered']/summary['total_games']*100:.1f}%)")
    print(f"  Within 3 pts: {summary['within_3_pts']}/{summary['total_games']} ({summary['within_3_pts']/summary['total_games']*100:.1f}%)")
    print(f"  Within 5 pts: {summary['within_5_pts']}/{summary['total_games']} ({summary['within_5_pts']/summary['total_games']*100:.1f}%)")
    print(f"  Avg error: {summary['avg_margin_error']:.1f} pts")
    
    if summary.get("games_with_market"):
        print(f"\n  VS MARKET:")
        print(f"    Games compared: {summary['games_with_market']}")
        print(f"    Market favorites covered: {summary['market_favorites_covered']}/{summary['games_with_market']} ({summary['market_favorites_covered']/summary['games_with_market']*100:.1f}%)")
        print(f"    Beat market: {summary['beat_market_count']}/{summary['games_with_market']} ({summary['beat_market_count']/summary['games_with_market']*100:.1f}%)")
        print(f"    Market avg error: {summary['avg_market_error']:.1f} pts")
        
        if summary.get("edge_bets"):
            print(f"\n  EDGE BETS:")
            print(f"    Opportunities: {summary['edge_bets']}")
            print(f"    Won: {summary['edge_bets_won']}/{summary['edge_bets']} ({summary['edge_bets_won']/summary['edge_bets']*100:.1f}%)")
    
    print("="*90)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Track predictions and odds")
    parser.add_argument("action", choices=["save", "update", "show"], 
                       help="save=store predictions, update=add results, show=display summary")
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    target_date = date.fromisoformat(args.date)
    
    if args.action == "save":
        save_predictions_and_odds(target_date)
    elif args.action == "update":
        update_actual_results(target_date)
    elif args.action == "show":
        tracking_file = TRACKING_DIR / f"{target_date.isoformat()}.json"
        if tracking_file.exists():
            with open(tracking_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print_tracking_summary(data)
        else:
            print(f"No tracking file found for {target_date.isoformat()}")
