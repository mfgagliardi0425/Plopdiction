import requests
from bs4 import BeautifulSoup
import re

game_id = "401810365"
url = f"https://www.espn.com/nba/game/_/gameId/{game_id}"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

text = soup.get_text()

# The pattern is in something like:
# "OpenSpreadTotalMLCleveland CavaliersCavaliersCLE(21-17)(21-17, 8-8 Away)-4.5-115-6.5-110..."
# Where we have:
# - "-4.5" is Cleveland (home) spread
# - "-115" is the odds
# - "-6.5" is Indiana (away) spread
# - "-110" is the odds for away

# Find "OpenSpreadTotal" and extract the spreads after it
pattern = r'OpenSpreadTotal.*?([+-]?\d+\.5?)[-](\d+)([+-]?\d+\.5?)[-](\d+)'
match = re.search(pattern, text)

if match:
    print(f"Found spreads!")
    print(f"Home spread: {match.group(1)} (odds: -{match.group(2)})")
    print(f"Away spread: {match.group(3)} (odds: -{match.group(4)})")
    # The closing spread is typically the one for the away team (or the one with different sign)
    home_spread = float(match.group(1))
    away_spread = float(match.group(3))
    print(f"Using away spread: {away_spread}")
else:
    print("Pattern not found")
    # Try alternative: just look for the closing spread line
    idx = text.find("OpenSpreadTotal")
    if idx > 0:
        context = text[idx:idx + 200]
        print(f"Context: {repr(context)}")
        # Try to extract all numbers with decimals
        numbers = re.findall(r'[+-]?\d+\.5?', context)
        print(f"Numbers found: {numbers}")
