"""
Fetch historical closing spreads from Basketball Reference or other sources.
Alternative: Manual entry from ESPN for key games, or use available APIs.
"""
import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional

# For now, we'll create a manual lookup for Dec 2025 - Jan 2026
# This can be populated from ESPN gamecast pages manually or via API integration

HISTORICAL_SPREADS = {
    # Format: "YYYY-MM-DD_HOME_TEAM": closing_spread
    # Positive = home team favored, negative = away team favored
    
    # To populate this:
    # 1. Go to ESPN gamecast for each game
    # 2. Find the closing spread (last odds before game start)
    # 3. Add entry: "date_home_team": spread_value
    # 
    # Example from your CLE vs IND game (1/6/26):
    # "2026-01-06_Indiana Pacers": 6.5  (CLE favored by 6.5, so IND +6.5)
    # "2026-01-06_Cleveland Cavaliers": -6.5  (CLE -6.5)
}


def load_manual_spreads() -> Dict[str, float]:
    """Load manually entered ESPN spreads."""
    return HISTORICAL_SPREADS.copy()


def get_closing_spread(home_team: str, game_date: date) -> Optional[float]:
    """
    Lookup closing spread for a game.
    
    Args:
        home_team: Home team name
        game_date: Game date
    
    Returns:
        Closing spread (positive = home favored) or None
    """
    spreads = load_manual_spreads()
    key = f"{game_date.isoformat()}_{home_team}"
    return spreads.get(key)


def add_closing_spread(home_team: str, game_date: date, spread: float) -> None:
    """Add a manually entered closing spread."""
    HISTORICAL_SPREADS[f"{game_date.isoformat()}_{home_team}"] = spread


if __name__ == "__main__":
    print("Historical spreads module loaded")
    print(f"Current spreads in database: {len(HISTORICAL_SPREADS)}")
