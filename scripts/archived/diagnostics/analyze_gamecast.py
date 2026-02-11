"""Analyze ESPN gamecast page structure."""
import requests
from bs4 import BeautifulSoup
import re

game_id = "401810365"  # Cavaliers vs Pacers from Jan 6
url = f"https://www.espn.com/nba/game/_/gameId/{game_id}"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

try:
    r = requests.get(url, headers=headers, timeout=10)
    print(f'Status: {r.status_code}')
    print(f'Content length: {len(r.text)} chars\n')
    
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # Get title for teams
    title = soup.find('title')
    if title:
        print(f'Page title: {title.string}\n')
    
    # Look for team names
    print('Looking for team names...')
    team_headers = soup.find_all(['h1', 'h2', 'span'], class_=lambda x: 'team' in str(x).lower() if x else False)
    for h in team_headers[:5]:
        print(f'  {h.get_text(strip=True)[:100]}')
    
    # Look for betting/odds information
    print('\nLooking for betting/odds sections...')
    odds_sections = soup.find_all(['div', 'section'], class_=lambda x: 'odds' in str(x).lower() or 'betting' in str(x).lower() if x else False)
    print(f'Found {len(odds_sections)} odds/betting sections')
    
    for section in odds_sections[:2]:
        print(f'  Text: {section.get_text(strip=True)[:200]}...')
    
    # Look for spread patterns in full text
    print('\nSearching for spread patterns...')
    text = soup.get_text()
    
    spread_patterns = [
        (r'Closing[:\s]+([+-]?\d+\.5?)', 'Closing'),
        (r'Final[:\s]+Spread[:\s]+([+-]?\d+\.5?)', 'Final Spread'),
        (r'Spread[:\s]+([+-]?\d+\.5?)', 'Spread'),
        (r'([+-]\d+\.5?)\s*(?:EV|-\d+|\+\d+)', 'Spread with odds'),
        (r'DraftKings[:\s]*([+-]?\d+\.5?)', 'DraftKings'),
    ]
    
    for pattern, name in spread_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            print(f'  {name}: {matches}')
    
    # Look for links related to betting
    print('\nLinks with "odds" or "bet":')
    links = soup.find_all('a', href=lambda x: x and ('odds' in x.lower() or 'bet' in x.lower()) if x else False)
    for link in links[:5]:
        print(f'  {link.get("href")}')
    
    # Look for scripts with data
    print('\nJSON data scripts:')
    scripts = soup.find_all('script', type='application/json')
    print(f'  Total: {len(scripts)}')
    
    for script in scripts[:2]:
        data = script.string
        if data:
            # Check if it contains relevant info
            if 'spread' in data.lower() or 'odds' in data.lower():
                print(f'  Found odds/spread data in script')
                # Print first 300 chars
                print(f'    {data[:300]}...')
    
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
