import requests
from bs4 import BeautifulSoup
import re

game_id = "401810365"  # Cavs vs Pacers on 1/6/2026
url = f"https://www.espn.com/nba/game/_/gameId/{game_id}"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

text = soup.get_text()

# Get title to identify teams
title = soup.find('title')
if title:
    title_text = title.string
    print(f"Title: {title_text}")
    # Format: "Cavaliers 120-116 Pacers (Jan 6, 2026) Final Score - ESPN"
    # So away_team is first, home_team is second

# Find the full odds line with team info
idx = text.find('OpenSpreadTotal')
if idx > 0:
    # Get context before and after
    context_before = text[max(0, idx - 200):idx]
    context_after = text[idx:idx + 300]
    
    print("\n=== BEFORE OpenSpreadTotal ===")
    print(repr(context_before[-100:]))
    
    print("\n=== AFTER OpenSpreadTotal ===")
    print(repr(context_after[:150]))
    
    # Look for team abbreviations
    # Format should be: "...TeamName...OpenSpreadTotal...Spread-Odds Spread-Odds..."
    
    # Let's find which team's line has the -6.5
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'OpenSpreadTotal' in line:
            print(f"\n=== Line {i} with OpenSpreadTotal ===")
            print(repr(line))
            
            # Check lines before and after for team names
            if i > 0:
                print(f"Line {i-1}: {repr(lines[i-1])}")
            if i < len(lines) - 1:
                print(f"Line {i+1}: {repr(lines[i+1])}")
    
    # Extract which team the -6.5 belongs to
    # The pattern should be: "TeamName...OpenSpreadTotal...-4.5-115-6.5-112..."
    # Let's find the team name before OpenSpreadTotal in the same logical section
    
    pattern = r'([A-Z]{2,3}).*?OpenSpreadTotal.*?([+-]?\d+\.5?)[-](\d+)([+-]?\d+\.5?)[-](\d+)'
    match = re.search(pattern, text)
    if match:
        print(f"\n=== Extracted ===")
        print(f"Team code: {match.group(1)}")
        print(f"Opening spread: {match.group(2)}")
        print(f"Closing spread: {match.group(4)}")
