import requests
from bs4 import BeautifulSoup
import re

game_id = "401810365"
url = f"https://www.espn.com/nba/game/_/gameId/{game_id}"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

text = soup.get_text()

# Find any lines containing spread-related info
print("=" * 80)
print("Looking for spread-related lines...")
print("=" * 80)

lines = text.split('\n')
for i, line in enumerate(lines):
    line_clean = line.strip()
    if line_clean and any(keyword in line_clean.lower() for keyword in ['spread', '-6.5', '-4.5', '+6.5', 'odds', 'line']):
        if len(line_clean) < 150:
            print(f"{i}: {repr(line_clean)}")

print("\n" + "=" * 80)
print("Looking for -6.5 or +6.5...")
print("=" * 80)
if '-6.5' in text:
    print("Found -6.5 in text")
    idx = text.find('-6.5')
    start = max(0, idx - 100)
    end = min(len(text), idx + 100)
    print(f"Context:\n{repr(text[start:end])}")
else:
    print("Not found")

print("\n" + "=" * 80)
print("All unique lines with numbers...")
print("=" * 80)
for i, line in enumerate(lines):
    line_clean = line.strip()
    if line_clean and re.search(r'\d', line_clean):
        if len(line_clean) < 80 and any(x in line_clean for x in ['.5', '-', '+']):
            print(f"{i}: {line_clean}")
