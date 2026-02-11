"""
Fetch NBA games and odds from ESPN using their public data endpoints.
"""
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_espn_games_for_date(game_date: date) -> List[Dict]:
    """
    Fetch games from ESPN for a specific date.
    Uses ESPN's public scoreboard API.
    """
    try:
        # ESPN scoreboard URL format
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={game_date.strftime('%Y%m%d')}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        games = []
        for event in data.get('events', []):
            game_info = {
                'id': event.get('id'),
                'date': game_date.isoformat(),
                'home_team': None,
                'away_team': None,
                'home_id': None,
                'away_id': None,
                'status': event.get('status', {}).get('type'),
                'start_time_utc': event.get('date'),
            }
            
            # Extract team info
            competitions = event.get('competitions', [])
            if competitions:
                comp = competitions[0]
                for team_data in comp.get('competitors', []):
                    team = team_data.get('team', {})
                    team_name = team.get('displayName')
                    is_home = team_data.get('homeAway') == 'home'
                    
                    if is_home:
                        game_info['home_team'] = team_name
                        game_info['home_id'] = team.get('id')
                    else:
                        game_info['away_team'] = team_name
                        game_info['away_id'] = team.get('id')
            
            if game_info['home_team'] and game_info['away_team']:
                games.append(game_info)
        
        return games
    
    except Exception as e:
        print(f"Error fetching ESPN games for {game_date}: {e}")
        return []


def scrape_espn_gamecast_spreads(espn_game_id: str) -> Optional[float]:
    """
    Scrape closing spread from ESPN gamecast page.
    
    Attempts to extract the closing spread from the HTML.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("BeautifulSoup4 required: pip install beautifulsoup4")
        return None
    
    try:
        url = f"https://www.espn.com/nba/game?gameId={espn_game_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for the odds section in the page
        # ESPN typically has a section with betting odds
        # We need to find text that shows spreads like "-6.5" or "+6.5"
        
        # Strategy: Find all text containing spread patterns and odds
        text = soup.get_text()
        
        # Look for DraftKings closing spread patterns
        # DraftKings spreads appear in format: "TEAM -6.5 -110" or similar
        lines = text.split('\n')
        
        closing_found = False
        for i, line in enumerate(lines):
            if 'closing' in line.lower() or 'final odds' in line.lower():
                closing_found = True
                # Look in the next 10 lines for a spread value
                for j in range(i, min(i+10, len(lines))):
                    match = re.search(r'([+-]?\d+\.5?)\s*(?:-\d+|\+\d+|−\d+)?', lines[j])
                    if match:
                        try:
                            spread = float(match.group(1))
                            # Valid NBA spreads are typically between -20 and 20
                            if -20 < spread < 20:
                                return spread
                        except ValueError:
                            continue
        
        # If not found with keyword search, try to extract from meta tags or JSON
        # Look for structured data
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                data = json.loads(script.string)
                # Check for event data with odds
                if isinstance(data, dict) and data.get('@type') == 'SportsEvent':
                    # Look for spread information
                    if 'potentialAction' in data:
                        action = data['potentialAction']
                        if isinstance(action, dict) and 'offers' in action:
                            for offer in action['offers']:
                                if 'priceCurrency' in offer:
                                    try:
                                        price = float(offer.get('price', 0))
                                        if -20 < price < 20:
                                            return price
                                    except (ValueError, TypeError):
                                        pass
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        
        return None
    
    except Exception as e:
        print(f"Error scraping gamecast for {espn_game_id}: {e}")
        return None


def build_espn_spreads_db(start_date: date, end_date: date, cache_file: Path = Path("odds_cache/espn_spreads_db.json")) -> Dict[str, float]:
    """
    Build a database of ESPN closing spreads for date range.
    
    Args:
        start_date: Start date for games
        end_date: End date for games
        cache_file: File to cache results
    
    Returns:
        Dictionary mapping "GAME_ID" -> closing_spread
    """
    # Load existing cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    spreads_db = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                spreads_db = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    print(f"\nFetching ESPN games from {start_date} to {end_date}")
    print(f"Already cached: {len(spreads_db)} games\n")
    
    current = start_date
    games_processed = 0
    spreads_found = 0
    
    while current <= end_date:
        print(f"Fetching games for {current.isoformat()}...", end=" ", flush=True)
        games = get_espn_games_for_date(current)
        print(f"Found {len(games)} games")
        
        for game in games:
            game_id = game['id']
            
            # Skip if already cached
            if game_id in spreads_db:
                continue
            
            games_processed += 1
            
            # Scrape gamecast for closing spread
            spread = scrape_espn_gamecast_spreads(game_id)
            
            if spread is not None:
                spreads_db[game_id] = spread
                spreads_found += 1
                print(f"  {game['away_team']} @ {game['home_team']}: {spread:+.1f}")
            else:
                spreads_db[game_id] = None  # Mark as attempted but not found
        
        current += timedelta(days=1)
    
    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(spreads_db, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Processed: {games_processed} games")
    print(f"Found spreads: {spreads_found} games")
    print(f"Cache saved to: {cache_file}")
    print(f"{'='*70}\n")
    
    return spreads_db


def get_spread_for_game(espn_game_id: str, spreads_db: Dict[str, float]) -> Optional[float]:
    """Lookup closing spread for a game."""
    return spreads_db.get(espn_game_id)


if __name__ == "__main__":
    import re
    
    # Test: Fetch games for a sample date
    test_date = date(2026, 1, 6)
    games = get_espn_games_for_date(test_date)
    print(f"Games on {test_date}: {len(games)}")
    for game in games[:3]:
        print(f"  {game['away_team']} @ {game['home_team']} (ID: {game['id']})")
