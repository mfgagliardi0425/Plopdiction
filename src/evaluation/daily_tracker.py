"""
Automated daily tracking of NBA predictions and market odds.
Runs daily to:
1. Fetch upcoming games from ESPN
2. Get our model predictions
3. Get market odds
4. Track results after games complete
5. Build training data with real odds
"""
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.odds_api import get_nba_odds
from data_fetching.espn_api import get_espn_games_for_date, scrape_espn_gamecast_spreads
from models.matchup_model import (
    build_team_history,
    parse_team_display,
    extract_points,
    DATA_DIR,
)
from models.improved_matchup_model import compute_enhanced_stats, predict_game_enhanced
from evaluation.tracking import TRACKING_DIR


def run_daily_tracking(target_date: date = None, save_espn_spreads: bool = True):
    """
    Run daily prediction and tracking.
    
    Args:
        target_date: Date to track (defaults to today)
        save_espn_spreads: Whether to fetch and save ESPN closing spreads
    """
    if target_date is None:
        target_date = date.today()
    
    print(f"\n{'='*90}")
    print(f"DAILY TRACKING: {target_date.isoformat()}")
    print(f"{'='*90}\n")
    
    # Build team history from local data
    print("Building team history...")
    history, names = build_team_history(DATA_DIR)
    
    # Get games for today
    print(f"Fetching games from ESPN for {target_date.isoformat()}...")
    games = get_espn_games_for_date(target_date)
    print(f"Found {len(games)} games\n")
    
    if not games:
        print("No games found for this date.")
        return
    
    # Compute team stats
    team_stats = {}
    for team_id, team_games in history.items():
        if team_games:
            team_stats[team_id] = compute_enhanced_stats(
                team_id, team_games, names.get(team_id, team_id), half_life=10.0
            )
    
    # Get market odds
    print("Fetching market odds from The Odds API...")
    try:
        market_odds_data = get_nba_odds(markets='spreads,h2h,totals')
        print(f"Retrieved odds for {len(market_odds_data)} games\n")
    except Exception as e:
        print(f"Error fetching odds: {e}")
        market_odds_data = []
    
    # Track each game
    tracked_games = []
    
    for game in games:
        espn_game_id = game['id']
        home_team = game['home_team']
        away_team = game['away_team']
        game_date = game['date']
        
        print(f"{away_team:25} @ {home_team:25}", end=" | ", flush=True)
        
        # Match to SportRadar game (using names)
        home_id = None
        away_id = None
        for team_id, team_name in names.items():
            if team_name == home_team:
                home_id = team_id
            if team_name == away_team:
                away_id = team_id
        
        if not home_id or not away_id:
            print("Could not match teams")
            continue
        
        # Make prediction
        try:
            # Find matching SportRadar game in data directory
            day_dir = Path(DATA_DIR) / game_date
            if not day_dir.exists():
                print("No local game data")
                continue
            
            sr_game = None
            for game_file in day_dir.glob("*.json"):
                with open(game_file, 'r', encoding='utf-8') as f:
                    g = json.load(f)
                h = g.get("home", {})
                a = g.get("away", {})
                h_id, _ = parse_team_display(h)
                a_id, _ = parse_team_display(a)
                if h_id == home_id and a_id == away_id:
                    sr_game = g
                    break
            
            if not sr_game:
                print("Could not find SportRadar game")
                continue
            
            # Predict
            _, _, pred_margin = predict_game_enhanced(
                sr_game, team_stats, home_advantage=2.5, 
                momentum_weight=0.15, regression_factor=0.10
            )
            
            # Get ESPN closing spread (attempt to scrape)
            espn_spread = None
            if save_espn_spreads:
                espn_spread = scrape_espn_gamecast_spreads(espn_game_id)
            
            # Get Odds API spread
            odds_api_spread = None
            for odds_game in market_odds_data:
                if odds_game.get('home_team') == home_team and odds_game.get('away_team') == away_team:
                    odds_api_spread = odds_game.get('spread')
                    break
            
            # Use ESPN spread if available, otherwise Odds API
            market_spread = espn_spread or odds_api_spread
            
            prediction = {
                'espn_game_id': espn_game_id,
                'home_margin': pred_margin,
                'espn_spread': espn_spread,
                'odds_api_spread': odds_api_spread,
                'market_spread': market_spread,
            }
            
            tracked_games.append({
                'date': game_date,
                'away_team': away_team,
                'home_team': home_team,
                'prediction': prediction,
            })
            
            spread_str = f"{market_spread:+.1f}" if market_spread else "N/A"
            print(f"Prediction: {pred_margin:+6.1f} | Market: {spread_str}")
        
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Save tracking data
    tracking_data = {
        'date': target_date.isoformat(),
        'tracked_at': datetime.now().isoformat(),
        'total_games': len(games),
        'games_tracked': len(tracked_games),
        'games': tracked_games,
    }
    
    tracking_file = TRACKING_DIR / f"{target_date.isoformat()}_automated.json"
    TRACKING_DIR.mkdir(exist_ok=True)
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"\n{'='*90}")
    print(f"Tracked {len(tracked_games)} games")
    print(f"Saved to: {tracking_file}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    # Run for today
    run_daily_tracking()
