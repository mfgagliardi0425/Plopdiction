"""
Module for fetching and managing betting odds from The Odds API.
"""
import json
import os
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv('ODDS_API_KEY', 'f67349ded55af918272d9a3ea220510a')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
ODDS_CACHE_DIR = Path('odds_cache')
ODDS_CACHE_DIR.mkdir(exist_ok=True)


def get_nba_odds(
    regions: str = 'us',
    markets: str = 'spreads,totals',
    oddsFormat: str = 'american',
    retries: int = 3,
    delay: float = 1.0,
) -> dict:
    """
    Fetch current NBA odds.
    
    Args:
        regions: Comma-separated regions (us, uk, eu, au)
        markets: Comma-separated markets (spreads, totals, h2h)
        oddsFormat: american, decimal, or fractional
        retries: Number of retry attempts
        delay: Delay between retries
    
    Returns:
        Dict with odds data
    """
    url = f'{ODDS_API_BASE}/sports/basketball_nba/odds/'
    
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': regions,
        'markets': markets,
        'oddsFormat': oddsFormat,
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Check remaining quota
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')
            if remaining:
                print(f"Odds API: {remaining} requests remaining (used: {used})")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = delay * (attempt + 1)
                print(f"Rate limited. Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
    
    raise RuntimeError("Failed to fetch odds after retries")


def cache_odds(odds_data: dict, date_str: Optional[str] = None) -> Path:
    """
    Cache odds data to disk.
    
    Args:
        odds_data: Odds data from API
        date_str: Optional date string (YYYY-MM-DD), defaults to today
    
    Returns:
        Path to cached file
    """
    if not date_str:
        date_str = date.today().isoformat()
    
    cache_file = ODDS_CACHE_DIR / f'nba_odds_{date_str}.json'
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({
            'fetched_at': datetime.utcnow().isoformat(),
            'date': date_str,
            'games': odds_data,
        }, f, indent=2)
    
    print(f"Cached odds to: {cache_file}")
    return cache_file


def load_cached_odds(date_str: str) -> Optional[dict]:
    """Load cached odds for a specific date."""
    cache_file = ODDS_CACHE_DIR / f'nba_odds_{date_str}.json'
    
    if not cache_file.exists():
        return None
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_odds_for_game(game_odds: dict, sportsbook: str = 'draftkings') -> Optional[Dict]:
    """
    Parse odds data for a single game.
    
    Args:
        game_odds: Single game from odds API response
        sportsbook: Preferred sportsbook (draftkings, fanduel, betmgm, etc.)
    
    Returns:
        Dict with spread, total, and moneyline info, or None
    """
    if not game_odds:
        return None
    
    home_team = game_odds.get('home_team')
    away_team = game_odds.get('away_team')
    commence_time = game_odds.get('commence_time')
    
    # Find bookmaker odds
    bookmakers = game_odds.get('bookmakers', [])
    
    # Try preferred sportsbook first
    selected_book = None
    for book in bookmakers:
        if book.get('key', '').lower() == sportsbook.lower():
            selected_book = book
            break
    
    # Fall back to first available
    if not selected_book and bookmakers:
        selected_book = bookmakers[0]
    
    if not selected_book:
        return None
    
    result = {
        'home_team': home_team,
        'away_team': away_team,
        'commence_time': commence_time,
        'sportsbook': selected_book.get('key'),
    }
    
    # Parse markets
    for market in selected_book.get('markets', []):
        market_key = market.get('key')
        outcomes = market.get('outcomes', [])
        
        if market_key == 'spreads':
            # Find home and away spreads
            for outcome in outcomes:
                team = outcome.get('name')
                point = outcome.get('point')
                price = outcome.get('price')
                
                if team == home_team:
                    result['home_spread'] = point
                    result['home_spread_odds'] = price
                elif team == away_team:
                    result['away_spread'] = point
                    result['away_spread_odds'] = price
        
        elif market_key == 'totals':
            # Over/under
            for outcome in outcomes:
                name = outcome.get('name')
                point = outcome.get('point')
                price = outcome.get('price')
                
                if name == 'Over':
                    result['over'] = point
                    result['over_odds'] = price
                elif name == 'Under':
                    result['under'] = point
                    result['under_odds'] = price
        
        elif market_key == 'h2h':
            # Moneyline
            for outcome in outcomes:
                team = outcome.get('name')
                price = outcome.get('price')
                
                if team == home_team:
                    result['home_moneyline'] = price
                elif team == away_team:
                    result['away_moneyline'] = price
    
    return result


def get_odds_for_date(target_date: date, fetch_if_missing: bool = True) -> List[Dict]:
    """
    Get odds for all games on a specific date.
    
    Args:
        target_date: Date to get odds for
        fetch_if_missing: If True, fetch from API if not cached
    
    Returns:
        List of parsed odds dicts
    """
    date_str = target_date.isoformat()
    
    # Try cache first
    cached = load_cached_odds(date_str)
    if cached:
        print(f"Loaded odds from cache for {date_str}")
        games_data = cached.get('games', [])
    elif fetch_if_missing:
        print(f"Fetching odds from API for {date_str}...")
        games_data = get_nba_odds()
        cache_odds(games_data, date_str)
    else:
        return []
    
    # Parse each game
    parsed_games = []
    for game in games_data:
        parsed = parse_odds_for_game(game)
        if parsed:
            parsed_games.append(parsed)
    
    return parsed_games


def match_odds_to_sportradar_game(
    sportradar_game: dict,
    odds_games: List[Dict],
) -> Optional[Dict]:
    """
    Match a SportRadar game to odds data.
    
    Args:
        sportradar_game: Game dict from SportRadar API
        odds_games: List of parsed odds dicts
    
    Returns:
        Matched odds dict or None
    """
    sr_home = sportradar_game.get('home', {})
    sr_away = sportradar_game.get('away', {})
    
    sr_home_alias = sr_home.get('alias', '').upper()
    sr_away_alias = sr_away.get('alias', '').upper()
    sr_home_name = sr_home.get('name', '').upper()
    sr_away_name = sr_away.get('name', '').upper()
    
    # Try to match by team names/aliases
    for odds in odds_games:
        odds_home = odds.get('home_team', '').upper()
        odds_away = odds.get('away_team', '').upper()
        
        # Check various matching strategies
        if (sr_home_alias in odds_home or sr_home_name in odds_home) and \
           (sr_away_alias in odds_away or sr_away_name in odds_away):
            return odds
    
    return None


if __name__ == '__main__':
    """Test the odds API integration."""
    print("Testing The Odds API integration\n")
    
    # Fetch current odds
    print("Fetching current NBA odds...")
    odds = get_nba_odds(markets='spreads,totals,h2h')
    
    print(f"\nFound {len(odds)} games with odds:\n")
    
    for game in odds[:5]:  # Show first 5
        parsed = parse_odds_for_game(game)
        if parsed:
            print(f"{parsed['away_team']} @ {parsed['home_team']}")
            if 'home_spread' in parsed:
                print(f"  Spread: {parsed['home_team']} {parsed['home_spread']:+.1f}")
            if 'over' in parsed:
                print(f"  Total: {parsed['over']} (O/U)")
            if 'home_moneyline' in parsed and 'away_moneyline' in parsed:
                print(f"  Moneyline: {parsed['home_team']} {parsed['home_moneyline']:+d} / {parsed['away_team']} {parsed['away_moneyline']:+d}")
            print()
