import requests
from bs4 import BeautifulSoup
import re

game_id = "401810365"
url = f"https://www.espn.com/nba/game/_/gameId/{game_id}"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')
text = soup.get_text()

# Search for the exact pattern
pattern = r'OpenSpreadTotal.*?([+-]?\d+\.5?)[-](\d+)([+-]?\d+\.5?)[-](\d+)'
match = re.search(pattern, text)

print(f"Pattern found: {match is not None}")

if match:
    print(f"Group 1 (opening): {match.group(1)}")
    print(f"Group 3 (closing): {match.group(3)}")
else:
    # Try simpler pattern
    print("\nTrying simpler search...")
    idx = text.find('OpenSpreadTotal')
    if idx >= 0:
        print(f"Found 'OpenSpreadTotal' at {idx}")
        context = text[idx:idx + 200]
        print(f"Context: {repr(context)}")
        
        # Try to extract spreads with a simpler regex
        spreads = re.findall(r'([+-]?\d+\.5?)', context)
        print(f"Found spreads in context: {spreads}")
