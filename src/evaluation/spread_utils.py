"""Shared helpers for ESPN-style away spread formatting."""
from __future__ import annotations

from typing import Optional


TEAM_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def normalize_team_name(name: str) -> str:
    cleaned = " ".join("".join(ch for ch in (name or "") if ch.isalnum() or ch.isspace()).lower().split())
    if cleaned.startswith("la "):
        return cleaned.replace("la ", "los angeles ", 1)
    if cleaned.startswith("ny "):
        return cleaned.replace("ny ", "new york ", 1)
    return cleaned


def abbr(team_name: str) -> str:
    name = (team_name or "").strip()
    return TEAM_ABBR.get(name, name)


def clean_signed(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.{decimals}f}"


def format_team_spread(team_name: str, spread: Optional[float], decimals: int = 1) -> str:
    return f"{abbr(team_name)} {clean_signed(spread, decimals)}"


def away_margin_to_spread(away_margin: float) -> float:
    return -away_margin


def spread_to_away_margin(spread: float) -> float:
    return -spread
