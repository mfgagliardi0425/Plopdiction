"""
Scrape ESPN NBA scoreboard to extract closing spreads for historical games.
"""
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.matchup_model import DATA_DIR


SPREADS_CACHE = Path("odds_cache/espn_closing_spreads.json")
DETAILED_SPREADS_CACHE = Path("odds_cache/espn_closing_spreads_detailed.json")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def load_spreads_cache() -> Dict:
    """Load cached spreads."""
    if SPREADS_CACHE.exists():
        try:
            with open(SPREADS_CACHE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_spreads_cache(spreads: Dict):
    """Save spreads to cache."""
    SPREADS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPREADS_CACHE, 'w') as f:
        json.dump(spreads, f, indent=2)


def load_detailed_spreads_cache() -> Dict:
    """Load detailed cached spreads (with matchup info)."""
    if DETAILED_SPREADS_CACHE.exists():
        try:
            with open(DETAILED_SPREADS_CACHE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_detailed_spreads_cache(spreads: Dict):
    """Save detailed spreads to cache."""
    DETAILED_SPREADS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(DETAILED_SPREADS_CACHE, 'w') as f:
        json.dump(spreads, f, indent=2)


def get_espn_scoreboard_html(target_date: date) -> Optional[str]:
    """
    Fetch ESPN scoreboard HTML for a date.
    
    Args:
        target_date: Date to fetch (YYYYMMDD format for ESPN)
    
    Returns:
        HTML content or None if request fails
    """
    try:
        # ESPN scoreboard URL format: dates=YYYYMMDD
        date_str = target_date.strftime("%Y%m%d")
        url = f"https://www.espn.com/nba/scoreboard?dates={date_str}"
        
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.text
    
    except Exception as e:
        print(f"  Error fetching scoreboard for {target_date}: {e}")
        return None


def extract_game_links_from_scoreboard(html: str) -> List[str]:
    """
    Extract gamecast links from scoreboard HTML.
    
    Returns:
        List of ESPN game IDs
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all game containers and extract game IDs
        game_ids = []
        
        # Look for links to gamecast pages
        # ESPN gamecast URLs look like: /nba/game/_/gameId/401810365/cavaliers-pacers
        links = soup.find_all('a', href=re.compile(r'/nba/game/_/gameId/\d+'))
        
        for link in links:
            href = link.get('href', '')
            match = re.search(r'gameId/(\d+)', href)
            if match:
                game_id = match.group(1)
                if game_id not in game_ids:  # Avoid duplicates
                    game_ids.append(game_id)
        
        return game_ids
    
    except Exception as e:
        print(f"  Error extracting game links: {e}")
        return []


def scrape_closing_spread_from_gamecast(game_id: str) -> Optional[Tuple[str, str, float]]:
    """
    Scrape closing spread from a gamecast page.
    
    Returns:
        (home_team, away_team, closing_spread) or None if not found
    """
    try:
        url = f"https://www.espn.com/nba/game/_/gameId/{game_id}"
        headers = {"User-Agent": USER_AGENT}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract team names from page title or header
        home_team = None
        away_team = None
        
        # Try to find in page title
        title = soup.find('title')
        if title:
            title_text = title.string
            # Format is usually "Away Team Score-Home Score (Date) Final Score - ESPN"
            # Example: "Cavaliers 120-116 Pacers (Jan 6, 2026) Final Score - ESPN"
            if 'Final Score' in title_text:
                # Remove " Final Score - ESPN" part
                title_clean = title_text.replace(' Final Score - ESPN', '').replace('(', '').split(')')[0]
                # Format is now like "Cavaliers 120-116 Pacers Jan 6, 2026"
                # Extract teams and score
                # Match pattern: TeamName number-number TeamName
                match = re.search(r'([A-Za-z0-9\s]+?)\s+(\d+)-(\d+)\s+([A-Za-z0-9\s]+?)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', title_clean)
                if match:
                    away_team = match.group(1).strip()
                    home_team = match.group(4).strip()
        
        # Look for odds/spread information
        # ESPN displays the odds table with team lines:
        # "OpenSpreadTotalML[AwayTeam]-Opening-OpenOdds-Closing-ClosingOdds..."
        # "[HomeTeam]+Closing+ClosingOdds..."
        # We want the closing spread from the AWAY team line
        closing_spread = None
        
        text = soup.get_text()
        
        # Extract the odds block around "OpenSpreadTotal" to avoid matching unrelated numbers
        odds_start = text.find("OpenSpreadTotal")
        if odds_start != -1:
            odds_end = text.find("See More Odds", odds_start)
            if odds_end == -1:
                odds_end = text.find("Data is currently unavailable", odds_start)
            if odds_end == -1:
                odds_end = min(len(text), odds_start + 800)
            odds_block = text[odds_start:odds_end]

            def extract_team_closing_spread(team_name: Optional[str]) -> Optional[float]:
                if not team_name:
                    return None
                pattern = rf"{re.escape(team_name)}.*?([+-]?\d+\.5?)-\d+([+-]?\d+\.5?)-\d+"
                match = re.search(pattern, odds_block)
                if match:
                    try:
                        return float(match.group(2))
                    except (ValueError, IndexError):
                        return None
                return None

            away_closing = extract_team_closing_spread(away_team)
            home_closing = extract_team_closing_spread(home_team)

            if away_closing is None and home_closing is not None:
                away_closing = -home_closing

            if away_closing is not None and -20 <= away_closing <= 20:
                closing_spread = away_closing
        
        # Fallback patterns if main pattern not found
        if closing_spread is None:
            spread_patterns = [
                r'Closing[:\s]+([+-]?\d+\.5?)',
                r'Final[:\s]+Spread[:\s]+([+-]?\d+\.5?)',
                r'Spread[:\s]+([+-]?\d+\.5?)',
            ]
            
            for pattern in spread_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        closing_spread = float(matches[-1])
                        # Validate it's in reasonable range
                        if -20 <= closing_spread <= 20:
                            break
                    except (ValueError, IndexError):
                        continue
        
        # Method 2: Look for betting information in page structure
        if closing_spread is None:
            # Find sections with betting info
            betting_sections = soup.find_all(['div', 'section'], 
                                           class_=re.compile('odds|betting|spread', re.IGNORECASE))
            
            for section in betting_sections[:3]:  # Check first 3 sections
                section_text = section.get_text()
                matches = re.findall(r'([+-]?\d+\.5?)', section_text)
                if matches:
                    try:
                        # Try the last value found
                        closing_spread = float(matches[-1])
                        if -20 < closing_spread < 20:
                            break
                    except ValueError:
                        continue
        
        if home_team and away_team and closing_spread is not None:
            return (home_team, away_team, closing_spread)
        
        return None
    
    except Exception as e:
        print(f"    Error scraping gamecast {game_id}: {e}")
        return None


def scrape_espn_spreads_for_date_range(
    start_date: date,
    end_date: date,
    delay_seconds: float = 0.5,
) -> Dict[str, float]:
    """
    Scrape closing spreads for all games in date range.
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        delay_seconds: Delay between requests to avoid hammering ESPN
    
    Returns:
        Dictionary mapping "DATE_HOME_TEAM" -> closing_spread
    """
    spreads = load_spreads_cache()
    detailed_spreads = load_detailed_spreads_cache()
    
    print(f"\n{'='*80}")
    print(f"SCRAPING ESPN CLOSING SPREADS")
    print(f"Range: {start_date} to {end_date}")
    print(f"Already cached: {len(spreads)} spreads")
    print(f"{'='*80}\n")
    
    current = start_date
    total_games = 0
    spreads_found = 0
    
    while current <= end_date:
        print(f"[{current.isoformat()}] Fetching scoreboard...", end=" ", flush=True)
        
        # Get scoreboard
        html = get_espn_scoreboard_html(current)
        if not html:
            print("SKIP")
            current += timedelta(days=1)
            time.sleep(delay_seconds)
            continue
        
        # Extract game IDs
        game_ids = extract_game_links_from_scoreboard(html)
        print(f"Found {len(game_ids)} games")
        
        for game_id in game_ids:
            total_games += 1
            
            # Check if already cached
            cache_key = f"{current.isoformat()}_{game_id}"
            if cache_key in spreads:
                continue
            
            # Scrape gamecast
            print(f"  [{game_id}] Scraping...", end=" ", flush=True)
            
            result = scrape_closing_spread_from_gamecast(game_id)
            
            if result:
                home_team, away_team, closing_spread = result
                # Store with date and home team as key
                storage_key = f"{current.isoformat()}_{home_team}"
                spreads[storage_key] = closing_spread
                detailed_key = f"{current.isoformat()}_{away_team}_@_{home_team}"
                detailed_spreads[detailed_key] = {
                    "date": current.isoformat(),
                    "away_team": away_team,
                    "home_team": home_team,
                    "closing_spread_away": closing_spread,
                    "closing_spread_away_str": f"{closing_spread:+.1f}",
                    "spread_basis": "away_team",
                }
                spreads_found += 1
                print(f"[OK] {away_team} @ {home_team}: {closing_spread:+.1f}")
            else:
                print("[X] No spread found")
            
            # Delay to avoid hammering
            time.sleep(delay_seconds)
        
        current += timedelta(days=1)
        time.sleep(delay_seconds)
    
    # Save cache
    save_spreads_cache(spreads)
    save_detailed_spreads_cache(detailed_spreads)
    
    print(f"\n{'='*80}")
    print(f"SCRAPING COMPLETE")
    print(f"Total games scanned: {total_games}")
    print(f"Spreads found: {spreads_found}")
    print(f"Total in cache: {len(spreads)}")
    print(f"Cache saved: {SPREADS_CACHE}")
    print(f"{'='*80}\n")
    
    return spreads


def map_spreads_to_dataset(spreads: Dict[str, float]) -> Dict[str, float]:
    """
    Map ESPN spreads to our dataset format.
    
    Matches by date and home team name.
    """
    mapped = {}
    
    for key, spread in spreads.items():
        # Key format: "YYYY-MM-DD_HOME_TEAM"
        if spread is not None:
            mapped[key] = spread
    
    return mapped


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape ESPN closing spreads")
    parser.add_argument("--start", default="2025-12-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    
    args = parser.parse_args()
    
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    
    spreads = scrape_espn_spreads_for_date_range(start_date, end_date, args.delay)
    
    print("Sample spreads found:")
    for key, value in list(spreads.items())[-10:]:
        if value is not None:
            print(f"  {key}: {value:+.1f}")
