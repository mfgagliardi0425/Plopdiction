import os
import json
import time
from datetime import date, timedelta
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

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Season start date
SEASON_START = date(2025, 10, 22)


def get_schedule(day: date) -> dict:
    """Get the schedule for a specific date."""
    url = f"{BASE_URL}/games/{day.year}/{day.month:02d}/{day.day:02d}/schedule.json"
    params = {"api_key": API_KEY}
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_play_by_play(game_id: str) -> dict:
    """Get play-by-play data for a specific game."""
    url = f"{BASE_URL}/games/{game_id}/pbp.json"
    params = {"api_key": API_KEY}
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_game_data(game_date: date, game_id: str, data: dict):
    """Save game data to disk."""
    date_dir = DATA_DIR / game_date.isoformat()
    date_dir.mkdir(exist_ok=True)
    
    file_path = date_dir / f"{game_id}.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return file_path


def download_season_data(start_date: date, end_date: date, delay: float = 2.0):
    """
    Download all game data for the season.
    
    Args:
        start_date: First date to download
        end_date: Last date to download
        delay: Seconds to wait between API calls (respect rate limits)
    """
    total_games = 0
    errors = []
    
    current_date = start_date
    while current_date <= end_date:
        print(f"\n{'='*60}")
        print(f"Processing date: {current_date.isoformat()}")
        print(f"{'='*60}")
        
        try:
            # Get schedule for the date
            schedule = get_schedule(current_date)
            games = schedule.get('games', [])
            
            print(f"Found {len(games)} games")
            
            if not games:
                current_date += timedelta(days=1)
                time.sleep(delay)
                continue
            
            # Download each game
            for i, game in enumerate(games, 1):
                game_id = game.get('id')
                away_team = game.get('away', {}).get('alias', 'UNK')
                home_team = game.get('home', {}).get('alias', 'UNK')
                status = game.get('status', 'unknown')
                
                # Check if already downloaded
                file_path = DATA_DIR / current_date.isoformat() / f"{game_id}.json"
                if file_path.exists():
                    print(f"  [{i}/{len(games)}] SKIP: {away_team} @ {home_team} (already downloaded)")
                    continue
                
                print(f"  [{i}/{len(games)}] Downloading: {away_team} @ {home_team} ({status})")
                
                try:
                    # Download play-by-play data
                    pbp_data = get_play_by_play(game_id)
                    
                    # Save to disk
                    saved_path = save_game_data(current_date, game_id, pbp_data)
                    
                    # Get size for reporting
                    size_kb = saved_path.stat().st_size / 1024
                    print(f"       Saved: {size_kb:.1f} KB")
                    
                    total_games += 1
                    
                    # Rate limiting
                    time.sleep(delay)
                    
                except requests.exceptions.HTTPError as e:
                    error_msg = f"Failed to download {game_id} ({away_team} @ {home_team}): {e}"
                    print(f"       ERROR: {error_msg}")
                    errors.append(error_msg)
                    time.sleep(delay)
                    
                except Exception as e:
                    error_msg = f"Unexpected error for {game_id}: {e}"
                    print(f"       ERROR: {error_msg}")
                    errors.append(error_msg)
                    time.sleep(delay)
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Failed to get schedule for {current_date}: {e}"
            print(f"ERROR: {error_msg}")
            errors.append(error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error for date {current_date}: {e}"
            print(f"ERROR: {error_msg}")
            errors.append(error_msg)
        
        # Move to next date
        current_date += timedelta(days=1)
        time.sleep(delay)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total games downloaded: {total_games}")
    print(f"Errors encountered: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print(f"\nData saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download NBA game data")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD), defaults to season start")
    parser.add_argument("--end", help="End date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls in seconds")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    print("NBA Season Data Downloader")
    
    # Parse dates
    start_date = date.fromisoformat(args.start) if args.start else SEASON_START
    end_date = date.fromisoformat(args.end) if args.end else date.today()
    
    print(f"Start date: {start_date.isoformat()}")
    print(f"End date: {end_date.isoformat()}")
    
    # Calculate approximate number of days
    days = (end_date - start_date).days + 1
    print(f"\nWill check {days} days of schedules")
    print(f"Estimated time: ~{days * args.delay / 60:.1f} minutes (with {args.delay}s delay)")
    print(f"Note: Trial API keys have rate limits. Download may take longer.")
    
    if not args.yes:
        response = input("\nStart download? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Download cancelled")
            exit(0)
    
    download_season_data(start_date, end_date, delay=args.delay)
