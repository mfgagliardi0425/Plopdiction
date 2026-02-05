"""
Guide for populating ESPN closing spreads into the dataset.

Steps:
1. For each game, go to ESPN gamecast page (Dec 1 - Jan 31 games)
2. Find the closing spread (last odds before game start)
3. Record: game date, home team, spread value
4. Add to the HISTORICAL_ESPN_SPREADS dictionary below
5. Run the rebuild dataset script

This will integrate market spreads into training data for 80% accuracy target.
"""

# Historical ESPN closing spreads (manually populated from gamecast pages)
# Format: "YYYY-MM-DD_HOME_TEAM": spread (positive = home favored)
HISTORICAL_ESPN_SPREADS = {
    # December 2025 games (populate from ESPN gamecast pages)
    # Example: "2025-12-01_Team Name": 3.5
    
    # January 2026 games
    # "2026-01-06_Indiana Pacers": 6.5,  # Cleveland was -6.5
    
    # Add more as you populate from ESPN...
}


def add_espn_spreads_to_dataset():
    """
    Once HISTORICAL_ESPN_SPREADS is populated, integrate into dataset.
    
    Instructions:
    1. Go to each game's ESPN gamecast page
       Example: https://www.espn.com/nba/game?gameId=401810365
    
    2. Look for the betting odds section (usually right side of page)
    
    3. Find "CLOSING ODDS" or similar section showing final pregame lines
    
    4. Record the spread (e.g., "-6.5", "+3.5")
       - Positive = home team favored
       - Negative = away team favored
    
    5. Add to dictionary:
       "2026-01-06_Indiana Pacers": 6.5  # Positive = home favored
       "2026-01-06_Cleveland Cavaliers": -6.5  # Would be away favored but we use home perspective
    
    6. After populating ~100+ games, run:
       python src/ml/build_dataset_optimized.py
       python src/ml/train_model.py --data ml_data/games_optimized.csv --cutoff 2026-01-16
    
    Expected improvement:
    - With ESPN spreads as features: 65% → 70%+ winner accuracy
    - With 60/40 blending: 70% → 75%+
    - Combined optimizations: 75%+ → 80%+
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print(f"\nCurrently populated spreads: {len(HISTORICAL_ESPN_SPREADS)}")
    print("Goal: Populate 100+ games from Dec 1 - Jan 31")
