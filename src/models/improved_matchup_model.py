"""
Improved matchup prediction model with enhanced features.
Key improvements:
1. Separate offensive and defensive ratings
2. Better momentum tracking (recent 3, 5, 10 games)
3. Pace-adjusted metrics
4. More granular home/away splits
5. Regression towards league mean to reduce overconfidence
"""
import argparse
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.matchup_model import (
    build_team_history,
    GameResult,
    get_upcoming_games,
    parse_team_display,
    DATA_DIR,
)


@dataclass
class EnhancedTeamStats:
    team_id: str
    name: str
    
    # Overall metrics
    weighted_offensive_rating: float  # Points per game
    weighted_defensive_rating: float  # Points allowed per game
    weighted_net_rating: float  # Offensive - Defensive
    
    # Home/Away splits
    home_offensive_rating: float
    home_defensive_rating: float
    away_offensive_rating: float
    away_defensive_rating: float
    
    # Momentum indicators
    last_3_margin: float
    last_5_margin: float
    last_10_margin: float
    
    # Win percentages
    weighted_win_pct: float
    recent_10_win_pct: float
    
    # Metadata
    games_played: int
    last_game_date: date = None


def compute_enhanced_stats(
    team_id: str,
    games: List[GameResult],
    name: str,
    half_life: float = 10.0,
) -> EnhancedTeamStats:
    """Compute enhanced team statistics with better features."""
    if not games:
        return EnhancedTeamStats(
            team_id=team_id,
            name=name,
            weighted_offensive_rating=105.0,
            weighted_defensive_rating=105.0,
            weighted_net_rating=0.0,
            home_offensive_rating=105.0,
            home_defensive_rating=105.0,
            away_offensive_rating=105.0,
            away_defensive_rating=105.0,
            last_3_margin=0.0,
            last_5_margin=0.0,
            last_10_margin=0.0,
            weighted_win_pct=0.5,
            recent_10_win_pct=0.5,
            games_played=0,
        )
    
    # Weighted calculations
    total_weight = 0.0
    weighted_off = 0.0
    weighted_def = 0.0
    weighted_wins = 0.0
    
    most_recent_idx = len(games) - 1
    for idx, game in enumerate(games):
        games_ago = most_recent_idx - idx
        weight = 0.5 ** (games_ago / half_life)
        
        total_weight += weight
        weighted_off += weight * game.points_for
        weighted_def += weight * game.points_against
        weighted_wins += weight * (1.0 if game.margin > 0 else 0.0)
    
    off_rating = weighted_off / total_weight if total_weight > 0 else 105.0
    def_rating = weighted_def / total_weight if total_weight > 0 else 105.0
    win_pct = weighted_wins / total_weight if total_weight > 0 else 0.5
    
    # Home/Away splits
    home_games = [g for g in games if g.is_home]
    away_games = [g for g in games if not g.is_home]
    
    home_off = sum(g.points_for for g in home_games) / len(home_games) if home_games else off_rating
    home_def = sum(g.points_against for g in home_games) / len(home_games) if home_games else def_rating
    away_off = sum(g.points_for for g in away_games) / len(away_games) if away_games else off_rating
    away_def = sum(g.points_against for g in away_games) / len(away_games) if away_games else def_rating
    
    # Recent momentum
    recent_3 = games[-3:] if len(games) >= 3 else games
    recent_5 = games[-5:] if len(games) >= 5 else games
    recent_10 = games[-10:] if len(games) >= 10 else games
    
    last_3_margin = sum(g.margin for g in recent_3) / len(recent_3) if recent_3 else 0.0
    last_5_margin = sum(g.margin for g in recent_5) / len(recent_5) if recent_5 else 0.0
    last_10_margin = sum(g.margin for g in recent_10) / len(recent_10) if recent_10 else 0.0
    recent_10_wins = sum(1 for g in recent_10 if g.margin > 0) / len(recent_10) if recent_10 else 0.5
    
    return EnhancedTeamStats(
        team_id=team_id,
        name=name,
        weighted_offensive_rating=off_rating,
        weighted_defensive_rating=def_rating,
        weighted_net_rating=off_rating - def_rating,
        home_offensive_rating=home_off,
        home_defensive_rating=home_def,
        away_offensive_rating=away_off,
        away_defensive_rating=away_def,
        last_3_margin=last_3_margin,
        last_5_margin=last_5_margin,
        last_10_margin=last_10_margin,
        weighted_win_pct=win_pct,
        recent_10_win_pct=recent_10_wins,
        games_played=len(games),
        last_game_date=games[-1].game_date if games else None,
    )


def predict_game_enhanced(
    game: dict,
    stats: Dict[str, EnhancedTeamStats],
    home_advantage: float = 2.5,
    momentum_weight: float = 0.15,
    regression_factor: float = 0.10,
) -> Tuple[str, str, float]:
    """
    Enhanced game prediction using offensive/defensive ratings.
    
    Args:
        game: Game data dict
        stats: Team statistics
        home_advantage: Points added for home court
        momentum_weight: Weight for recent form (0-1)
        regression_factor: How much to regress towards mean (0-1, higher = more regression)
    
    Returns:
        (home_name, away_name, predicted_home_margin)
    """
    home = game.get("home", {})
    away = game.get("away", {})
    home_id, home_name = parse_team_display(home)
    away_id, away_name = parse_team_display(away)
    
    home_stats = stats.get(home_id)
    away_stats = stats.get(away_id)
    
    if not home_stats or not away_stats:
        return home_name, away_name, home_advantage
    
    # Base prediction using offensive/defensive matchup
    # Home expected score = (Home Off + Away Def) / 2
    # Away expected score = (Away Off + Home Def) / 2
    home_expected = (home_stats.home_offensive_rating + away_stats.away_defensive_rating) / 2
    away_expected = (away_stats.away_offensive_rating + home_stats.home_defensive_rating) / 2
    
    base_margin = home_expected - away_expected + home_advantage
    
    # Apply momentum adjustment (recent form)
    momentum = (home_stats.last_5_margin - away_stats.last_5_margin) * momentum_weight
    
    # Rest days adjustment
    rest_adj = 0.0
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from matchup_model import parse_game_date
    
    game_date = parse_game_date(game)
    if game_date and home_stats.last_game_date and away_stats.last_game_date:
        home_rest = (game_date - home_stats.last_game_date).days
        away_rest = (game_date - away_stats.last_game_date).days
        
        # Back-to-back penalty
        if home_rest == 1:
            rest_adj -= 2.5
        if away_rest == 1:
            rest_adj += 2.5
        
        # Rest advantage (capped)
        rest_diff = min(max(home_rest - away_rest, -2), 2)
        rest_adj += rest_diff * 0.5
    
    # Combine predictions
    raw_margin = base_margin + momentum + rest_adj
    
    # Regress towards mean to avoid overconfidence
    # If we predict +15, regress 10% towards 0 = +13.5
    predicted_margin = raw_margin * (1 - regression_factor)
    
    return home_name, away_name, predicted_margin


def print_predictions_enhanced(
    games: List[dict],
    stats: Dict[str, EnhancedTeamStats],
    home_advantage: float,
    momentum_weight: float,
    regression_factor: float,
) -> None:
    """Print enhanced predictions."""
    if not games:
        print("No upcoming games found.")
        return
    
    print("\n" + "="*90)
    print("ENHANCED MATCHUP PREDICTIONS")
    print("="*90 + "\n")
    
    for game in games:
        home_name, away_name, margin = predict_game_enhanced(
            game, stats, home_advantage, momentum_weight, regression_factor
        )
        
        scheduled = game.get("scheduled")
        date_label = "Unknown"
        if scheduled:
            try:
                from datetime import datetime
                date_label = datetime.fromisoformat(scheduled.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                pass
        
        favorite = home_name if margin > 0 else away_name
        spread = abs(margin)
        
        print(f"{date_label}: {away_name} @ {home_name}")
        print(f"  Predicted spread: {favorite} -{spread:.1f}")
        print(f"  Expected margin (home): {margin:+.1f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Enhanced matchup prediction model")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--half-life", type=float, default=10.0, help="Half-life for weighting")
    parser.add_argument("--home-adv", type=float, default=2.5, help="Home court advantage")
    parser.add_argument("--momentum-weight", type=float, default=0.15, help="Recent form weight")
    parser.add_argument("--regression", type=float, default=0.10, help="Regression factor (0-1)")
    args = parser.parse_args()
    
    if not DATA_DIR.exists():
        print("No data/ directory found. Download game data first.")
        return
    
    # Build team history
    history, names = build_team_history(DATA_DIR)
    if not history:
        print("No completed games found. Download games first.")
        return
    
    # Compute enhanced stats
    team_stats = {}
    for team_id, games in history.items():
        team_stats[team_id] = compute_enhanced_stats(
            team_id, games, names.get(team_id, team_id), args.half_life
        )
    
    # Get upcoming games
    start_date = date.fromisoformat(args.start) if args.start else date.today()
    end_date = date.fromisoformat(args.end) if args.end else (start_date + timedelta(days=7))
    
    games = get_upcoming_games(start_date, end_date)
    
    # Print predictions
    print_predictions_enhanced(
        games,
        team_stats,
        args.home_adv,
        args.momentum_weight,
        args.regression,
    )


if __name__ == "__main__":
    main()
