"""Test ESPN page structure."""
import requests
from bs4 import BeautifulSoup

url = 'https://www.espn.com/nba/scoreboard?dates=20260106'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

try:
    r = requests.get(url, headers=headers, timeout=10)
    print(f'Status: {r.status_code}')
    print(f'Content length: {len(r.text)} chars')
    
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # Look for game containers
    print('\nSearching for game elements...')
    
    # Try different selectors
    divs = soup.find_all('div', class_=lambda x: 'Card' in str(x) if x else False)
    print(f'Card divs: {len(divs)}')
    
    # Look for links
    links = soup.find_all('a', href=True)
    game_links = [l for l in links if '/nba/game' in l.get('href', '')]
    print(f'Game links: {len(game_links)}')
    
    if game_links:
        print('\nFirst 5 game links:')
        for link in game_links[:5]:
            print(f'  {link.get("href")}')
    
    # Look for spread info
    print('\nSearching for spread text...')
    text = soup.get_text()
    if 'Spread' in text or 'spread' in text:
        print('Found "Spread" in page')
    
    # Check for scripts with data
    scripts = soup.find_all('script', type='application/json')
    print(f'JSON scripts: {len(scripts)}')
    
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
