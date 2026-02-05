"""
Scrape closing pregame spreads from ESPN gamecast pages.
"""
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.matchup_model import DATA_DIR


def get_espn_game_url(espn_game_id: str) -> str:
    """Construct ESPN gamecast URL from game ID."""
    return f"https://www.espn.com/nba/game/_/gameId/{espn_game_id}"


def scrape_closing_spread(html_content: str) -> Optional[float]:
    """
    Extract closing spread from ESPN gamecast HTML.
    
    ESPN shows closing spreads in the betting odds section.
    We look for patterns like "-4.5" or "+6.5" in the odds section.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for the odds section on ESPN gamecast pages
        # ESPN typically shows spreads in tables or structured data
        # We'll search for the spread pattern in the page
        
        # Method 1: Look for structured data (JSON-LD)
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                data = json.loads(script.string)
                # Check if this is event data with spread info
                if isinstance(data, dict) and 'offers' in data:
                    for offer in data.get('offers', []):
                        if 'priceCurrency' in offer:  # Betting odds
                            price = offer.get('price')
                            if price:
                                return float(price)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        
        # Method 2: Look for text patterns in the page
        # ESPN shows spreads like "-6.5" or "+6.5"
        text = soup.get_text()
        
        # Look for spread patterns near keywords like "closing", "final", "draftkings"
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'closing' in line.lower() or 'final' in line.lower():
                # Check nearby lines for spread pattern
                for j in range(max(0, i-3), min(len(lines), i+4)):
                    match = re.search(r'([+-]?\d+\.?\d*)\s*(?:-\d+|\+\d+)?', lines[j])
                    if match:
                        try:
                            spread = float(match.group(1))
                            if -15 < spread < 15:  # Reasonable NBA spread range
                                return spread
                        except ValueError:
                            continue
        
        return None
    except Exception as e:
        print(f"Error scraping HTML: {e}")
        return None


def find_game_by_teams(home_team: str, away_team: str, game_date: date) -> Optional[str]:
    """
    Find ESPN game ID by matching teams and date.
    Uses the SportRadar data to build a mapping.
    """
    # This would require either:
    # 1. ESPN's game ID in SportRadar data
    # 2. A manual mapping file
    # 3. Querying ESPN's API or search
    
    # For now, return None - we'll need to handle this differently
    return None


def load_espn_spreads_from_file(spreads_cache_file: Path) -> Dict[str, float]:
    """Load previously scraped spreads from cache file."""
    spreads = {}
    if spreads_cache_file.exists():
        try:
            with open(spreads_cache_file, 'r') as f:
                spreads = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return spreads


def save_espn_spreads_to_file(spreads: Dict[str, float], spreads_cache_file: Path) -> None:
    """Save scraped spreads to cache file."""
    spreads_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(spreads_cache_file, 'w') as f:
        json.dump(spreads, f, indent=2)


def scrape_espn_odds(start_date: date, end_date: date, spreads_cache_file: Path = Path("odds_cache/espn_spreads.json")) -> Dict[str, float]:
    """
    Attempt to scrape ESPN closing spreads for games in date range.
    
    Returns:
        Dictionary mapping "YYYY-MM-DD_TEAM1_vs_TEAM2" -> closing_spread
    """
    spreads = load_espn_spreads_from_file(spreads_cache_file)
    
    print(f"\nAttempting to scrape ESPN spreads from {start_date} to {end_date}")
    print(f"Already cached: {len(spreads)} games\n")
    
    # Iterate through game files
    games_to_process = 0
    games_scraped = 0
    
    current = start_date
    while current <= end_date:
        day_dir = DATA_DIR / current.isoformat()
        if day_dir.exists():
            for game_file in day_dir.glob("*.json"):
                try:
                    with open(game_file, 'r', encoding='utf-8') as f:
                        game = json.load(f)
                    
                    # Extract game info
                    home = game.get("home", {})
                    away = game.get("away", {})
                    home_name = f"{home.get('market','')} {home.get('name','')}".strip()
                    away_name = f"{away.get('market','')} {away.get('name','')}".strip()
                    
                    # Create cache key
                    cache_key = f"{current.isoformat()}_{away_name}_vs_{home_name}"
                    
                    # Skip if already cached
                    if cache_key in spreads:
                        games_scraped += 1
                        continue
                    
                    games_to_process += 1
                    
                    # Note: ESPN scraping would require ESPN game IDs which we don't have
                    # For now, this is a placeholder structure
                    # In practice, you'd need to:
                    # 1. Query ESPN's search API or schedule
                    # 2. Match by date and team names
                    # 3. Scrape the gamecast page
                    
                except Exception:
                    continue
        
        current += timedelta(days=1)
    
    print(f"Games already cached: {games_scraped}")
    print(f"Games to process: {games_to_process}")
    print("\nNote: ESPN scraping requires ESPN game IDs which are not in SportRadar data.")
    print("To implement this fully, you'd need to:")
    print("  1. Query ESPN's API/schedule for game IDs")
    print("  2. Match games by date and team names")
    print("  3. Scrape each gamecast page for closing spreads")
    print("  4. Cache the results")
    
    save_espn_spreads_to_file(spreads, spreads_cache_file)
    
    return spreads


if __name__ == "__main__":
    # Example usage
    start = date(2025, 12, 1)
    end = date(2026, 1, 15)
    spreads = scrape_espn_odds(start, end)
    print(f"\nScraped {len(spreads)} games with closing spreads")
