import os
import json
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.getenv('SPORTRADAR_API_KEY')
ACCESS_LEVEL = os.getenv('SPORTRADAR_ACCESS_LEVEL', 'trial')
LANGUAGE = os.getenv('SPORTRADAR_LANGUAGE', 'en')
VERSION = os.getenv('SPORTRADAR_VERSION', 'v8')
BASE_URL = f"https://api.sportradar.com/nba/{ACCESS_LEVEL}/{VERSION}/{LANGUAGE}"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_test_games():
    """Download a few games as a test."""
    
    # Test with season opener (Oct 22, 2025)
    test_date = date(2025, 10, 22)
    
    print("Downloading test games from season opener...")
    print(f"Date: {test_date.isoformat()}\n")
    
    # Get schedule
    url = f"{BASE_URL}/games/{test_date.year}/{test_date.month:02d}/{test_date.day:02d}/schedule.json"
    response = requests.get(url, params={"api_key": API_KEY}, timeout=30)
    response.raise_for_status()
    
    schedule = response.json()
    games = schedule.get('games', [])[:3]  # Just first 3 games
    
    print(f"Found {len(games)} games to download\n")
    
    for i, game in enumerate(games, 1):
        game_id = game.get('id')
        away = game.get('away', {}).get('alias')
        home = game.get('home', {}).get('alias')
        
        print(f"[{i}/{len(games)}] {away} @ {home}")
        print(f"  Game ID: {game_id}")
        
        # Download play-by-play
        pbp_url = f"{BASE_URL}/games/{game_id}/pbp.json"
        pbp_response = requests.get(pbp_url, params={"api_key": API_KEY}, timeout=30)
        pbp_response.raise_for_status()
        
        pbp_data = pbp_response.json()
        
        # Save to file
        date_dir = DATA_DIR / test_date.isoformat()
        date_dir.mkdir(exist_ok=True)
        
        file_path = date_dir / f"{game_id}.json"
        with open(file_path, 'w') as f:
            json.dump(pbp_data, f, indent=2)
        
        size_kb = file_path.stat().st_size / 1024
        
        # Quick stats
        num_periods = len(pbp_data.get('periods', []))
        total_events = sum(len(p.get('events', [])) for p in pbp_data.get('periods', []))
        
        print(f"  Saved: {size_kb:.1f} KB")
        print(f"  Periods: {num_periods}, Events: {total_events}")
        print()
        
        time.sleep(2.0)  # Rate limiting (trial key)
    
    print(f"Test complete! Files saved to: {DATA_DIR.absolute()}")
    print(f"\nYou can now run download_season_data.py to download all games.")


if __name__ == "__main__":
    download_test_games()
